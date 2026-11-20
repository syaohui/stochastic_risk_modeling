"""Counterparty credit risk: exposure simulation, CVA and wrong-way risk.

Rates: one-factor Hull-White model with mean reversion a (1/year) and
volatility sigma (1/year^{3/2}, a short-rate volatility), fitted to a flat
initial curve with continuously compounded rate r0, P^M(0, t) = exp(-r0 t).
Write r(t) = x(t) + phi(t) with dx = -a x dt + sigma dW, x(0) = 0, and
phi(t) = r0 + sigma^2 / (2 a^2) (1 - e^{-a t})^2. Zero-coupon bonds
(Brigo and Mercurio, G1++ form):

    P(t, T) = P^M(0,T) / P^M(0,t) * exp( [V(t,T) - V(0,T) + V(0,t)] / 2 - B(t,T) x(t) ),
    B(t, T) = (1 - e^{-a (T - t)}) / a,
    V(t, T) = sigma^2 / a^2 [ T - t + (2/a) e^{-a(T-t)} - (1/(2a)) e^{-2a(T-t)} - 3/(2a) ].

x and its time integral are simulated exactly (jointly Gaussian), so the bank
account discount D(0, t) = exp(-int_0^t r) has E[D(0, t)] = P^M(0, t) exactly.

Exposure of a payer interest rate swap (pay fixed K, receive floating, unit
notional, payment dates T_1 < ... < T_n with accrual delta). Inside
(T_{k-1}, T_k] the floating rate L_k is already fixed, and

    V(t) = P(t, T_k)(1 + delta L_k) - P(t, T_n) - K delta sum_{i>=k} P(t, T_i).

Exposure metrics (Basel conventions, undiscounted): EE(t) = E[V(t)^+];
PFE_q(t) = q-quantile of V(t)^+; EPE = time average of EE over the first year;
effective EE is the running maximum of EE; effective EPE (EEPE) averages it
over the first year; the internal-model EAD is alpha * EEPE with alpha = 1.4.

Independent check: at a reset date T_k, the discounted expected exposure
E[D(0, T_k) V(T_k)^+] is the price of a payer swaption, given in closed form
by Jamshidian's decomposition into zero-coupon bond puts.

CVA with independent default: CVA = LGD sum_i E[D(0,t_i) V(t_i)^+] (S(t_{i-1}) - S(t_i)),
S(t) = exp(-lambda t). Wrong-way risk (Hull and White 2012): the hazard rate on
(t_{i-1}, t_i] is exp(a_i + b V(t_i)), with a_i calibrated so that the
expected survival matches exp(-lambda t_i); b > 0 is wrong-way risk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class HullWhite:
    a: float = 0.05        # mean reversion (1/year)
    sigma: float = 0.01    # short-rate volatility
    r0: float = 0.03       # flat initial curve (continuous compounding)

    def p0(self, t):
        return np.exp(-self.r0 * np.asarray(t, float))

    def B(self, t, T):
        return (1.0 - np.exp(-self.a * (T - t))) / self.a

    def V(self, t, T):
        a, s, tau = self.a, self.sigma, T - t
        return s * s / (a * a) * (tau + 2.0 / a * np.exp(-a * tau)
                                  - 0.5 / a * np.exp(-2 * a * tau) - 1.5 / a)

    def bond(self, t, T, x):
        """P(t, T) given the state x(t) (vectorized in x)."""
        lnA = np.log(self.p0(T) / self.p0(t)) + 0.5 * (self.V(t, T) - self.V(0, T) + self.V(0, t))
        return np.exp(lnA - self.B(t, T) * np.asarray(x, float))

    def int_phi(self, t):
        a, s = self.a, self.sigma
        return self.r0 * t + s * s / (2 * a * a) * (t - 2 * (1 - np.exp(-a * t)) / a
                                                    + (1 - np.exp(-2 * a * t)) / (2 * a))

    def simulate(self, times, n_paths, rng=None):
        """Exact joint simulation of x(t) and D(0, t) on a time grid starting at 0.

        Returns (x, D), both of shape (n_paths, len(times)).
        """
        rng = np.random.default_rng(rng)
        a, s = self.a, self.sigma
        times = np.asarray(times, float)
        x = np.zeros((n_paths, len(times)))
        I = np.zeros((n_paths, len(times)))        # int_0^t x ds
        for k in range(1, len(times)):
            d = times[k] - times[k - 1]
            e1, e2 = np.exp(-a * d), np.exp(-2 * a * d)
            bd = (1 - e1) / a
            vx = s * s * (1 - e2) / (2 * a)
            vi = s * s / (a * a) * (d - 2 * bd + (1 - e2) / (2 * a))
            cxi = s * s / (2 * a * a) * (1 - e1) ** 2
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            dx = np.sqrt(vx) * z1
            di = cxi / np.sqrt(vx) * z1 + np.sqrt(max(vi - cxi * cxi / vx, 0.0)) * z2
            x[:, k] = x[:, k - 1] * e1 + dx
            I[:, k] = I[:, k - 1] + x[:, k - 1] * bd + di
        D = np.exp(-self.int_phi(times)[None, :] - I)
        return x, D


def par_rate(hw, pay_times, delta):
    p = hw.p0(np.asarray(pay_times))
    return float((1.0 - p[-1]) / (delta * p.sum()))


def swap_values(hw, times, x, pay_times, delta, K):
    """Payer swap values V(t) on the grid (paths x times), unit notional.

    The grid must contain every payment date. At a payment date the coupon is
    paid and the next floating rate fixed, so V(T_k) is the value just after.
    """
    times = np.asarray(times, float)
    pay = np.asarray(pay_times, float)
    n_paths = x.shape[0]
    V = np.zeros((n_paths, len(times)))
    L = np.full(n_paths, np.nan)
    reset = np.concatenate([[0.0], pay[:-1]])
    for j, t in enumerate(times):
        k = int(np.searchsorted(pay, t, side="right"))   # index of the next payment
        if k >= len(pay):
            continue                                       # swap has matured
        if np.isclose(t, reset[k]):                        # fix L_k at its reset date
            L = (1.0 / hw.bond(t, pay[k], x[:, j]) - 1.0) / delta
        nxt = pay[k:]
        bonds = np.stack([hw.bond(t, T, x[:, j]) for T in nxt], axis=1)
        V[:, j] = bonds[:, 0] * (1 + delta * L) - bonds[:, -1] - K * delta * bonds.sum(axis=1)
    return V


def exposure_metrics(times, V, q=0.975):
    """EE, PFE_q, EPE, effective EE, EEPE and EAD = 1.4 EEPE (first year)."""
    times = np.asarray(times, float)
    pos = np.maximum(V, 0.0)
    ee = pos.mean(axis=0)
    pfe = np.quantile(pos, q, axis=0)
    eff = np.maximum.accumulate(ee)
    yr = times <= 1.0 + 1e-12
    dt = np.diff(times[yr])
    epe = float(np.sum(0.5 * (ee[yr][1:] + ee[yr][:-1]) * dt) / times[yr][-1])
    eepe = float(np.sum(0.5 * (eff[yr][1:] + eff[yr][:-1]) * dt) / times[yr][-1])
    return {"EE": ee, "PFE": pfe, "EffEE": eff, "EPE": epe, "EEPE": eepe, "EAD": 1.4 * eepe}


def payer_swaption(hw, expiry, pay_times, delta, K):
    """Payer swaption price by Jamshidian's decomposition (Hull-White, flat curve)."""
    pay = np.asarray(pay_times, float)
    pay = pay[pay > expiry + 1e-12]
    c = np.full(len(pay), K * delta)
    c[-1] += 1.0

    def f(xs):
        return np.sum(c * hw.bond(expiry, pay, xs)) - 1.0

    xstar = optimize.brentq(f, -1.0, 1.0)
    X = hw.bond(expiry, pay, xstar)
    a, s = hw.a, hw.sigma
    sp = s * np.sqrt((1 - np.exp(-2 * a * expiry)) / (2 * a)) * hw.B(expiry, pay)
    pT, pS = hw.p0(expiry), hw.p0(pay)
    h = np.log(pS / (pT * X)) / sp + sp / 2
    zbp = X * pT * stats.norm.cdf(-h + sp) - pS * stats.norm.cdf(-h)
    return float(np.sum(c * zbp))


def cva_independent(times, D, V, lgd, lam):
    """CVA with default independent of exposure; times start at 0."""
    times = np.asarray(times, float)
    dee = (D * np.maximum(V, 0.0)).mean(axis=0)
    S = np.exp(-lam * times)
    return float(lgd * np.sum(dee[1:] * (S[:-1] - S[1:])))


def cva_wrong_way(times, D, V, lgd, lam, b, scale=0.01):
    """Hull-White (2012) wrong-way CVA: hazard exp(a_i + b V(t_i)/scale) on (t_{i-1}, t_i].

    a_i is calibrated so that the expected survival at t_i equals exp(-lam t_i).
    scale sets the units of V in the hazard (0.01 = per 1% of notional).
    Returns (CVA, a_i).
    """
    times = np.asarray(times, float)
    n_paths = V.shape[0]
    surv = np.ones(n_paths)
    cva = 0.0
    a_all = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        target = np.exp(-lam * times[i])
        z = b * V[:, i] / scale

        def gap(ai):
            return np.mean(surv * np.exp(-np.exp(ai + z) * dt)) - target

        ai = optimize.brentq(gap, -40.0, 10.0)
        new = surv * np.exp(-np.exp(ai + z) * dt)
        cva += lgd * np.mean(D[:, i] * np.maximum(V[:, i], 0.0) * (surv - new))
        surv = new
        a_all.append(ai)
    return float(cva), np.array(a_all)
