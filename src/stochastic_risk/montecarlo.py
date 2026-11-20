"""Monte Carlo methods for risk: nested simulation, importance sampling, quasi-Monte Carlo.

Nested simulation (Gordy and Juneja 2010). Portfolio risk at a horizon tau
needs the portfolio value V_tau in each outer scenario, which for derivatives
is itself an expectation. With m outer scenarios and n inner samples each,
the inner estimate V_hat = V + noise of variance s^2 / n biases a tail
probability alpha = P(L > c) by about theta / n, while the outer sampling
error has variance alpha (1 - alpha) / m. For a budget k = m n the mean squared
error is about theta^2 / n^2 + alpha (1 - alpha) n / k, minimized at n of
order k^{1/3}, so the root mean squared error falls like k^{-1/3}, not k^{-1/2}.

The test portfolio here is a long European call on a geometric Brownian motion
(Black-Scholes), so the exact inner value is known and the bias can be
measured: L = V_0 - V_tau(S_tau), the value lost over the horizon.

Importance sampling for a Gaussian tail: P(X > c), X ~ N(0, 1), sampled under
N(c, 1) with likelihood ratio exp(-c X + c^2 / 2). Relative error stays bounded
as c grows, while plain Monte Carlo needs about 1 / P(X > c) samples per hit.

Quasi-Monte Carlo: scrambled Sobol points (randomized QMC, independent
replications give an error estimate) for an arithmetic basket call.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import qmc


# ---------------------------------------------------------------------- Black-Scholes
def bs_call(S, K, r, sigma, T):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def call_loss_exact(S_tau, S0, K, r, sigma, T, tau):
    """Exact loss L = V_0 - V_tau(S_tau) of a long call (no discounting over tau)."""
    return bs_call(S0, K, r, sigma, T) - bs_call(S_tau, K, r, sigma, T - tau)


def loss_threshold(alpha, S0, K, r, sigma, T, tau, mu=None):
    """Loss level c with P(L > c) = alpha exactly (L decreases in S_tau)."""
    mu = r if mu is None else mu
    z = stats.norm.ppf(alpha)                        # low S_tau quantile
    S_q = S0 * np.exp((mu - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * z)
    return float(call_loss_exact(S_q, S0, K, r, sigma, T, tau))


def nested_tail_probability(m, n, c, S0, K, r, sigma, T, tau, rng=None, mu=None, batch=None):
    """Nested estimate of P(L > c): m outer scenarios, n inner payoffs per scenario."""
    rng = np.random.default_rng(rng)
    mu = r if mu is None else mu
    S_tau = S0 * np.exp((mu - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * rng.standard_normal(m))
    V0 = bs_call(S0, K, r, sigma, T)
    h = T - tau
    batch = batch or max(1, 2_000_000 // n)
    hits = 0
    for s in range(0, m, batch):
        sb = S_tau[s:s + batch]
        z = rng.standard_normal((len(sb), n))
        ST = sb[:, None] * np.exp((r - 0.5 * sigma ** 2) * h + sigma * np.sqrt(h) * z)
        v = np.exp(-r * h) * np.maximum(ST - K, 0.0).mean(axis=1)
        hits += int(np.sum(V0 - v > c))
    return hits / m


# ---------------------------------------------------------------------- importance sampling
def gaussian_tail_mc(c, n, rng=None):
    rng = np.random.default_rng(rng)
    hit = rng.standard_normal(n) > c
    p = hit.mean()
    return float(p), float(np.sqrt(p * (1 - p) / n))


def gaussian_tail_is(c, n, rng=None):
    rng = np.random.default_rng(rng)
    x = c + rng.standard_normal(n)
    w = np.exp(-c * x + 0.5 * c * c) * (x > c)
    return float(w.mean()), float(w.std(ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------------- quasi-Monte Carlo
def basket_call_payoffs(u, S0, K, r, sigma, corr_chol, T, weights):
    """Discounted payoffs of an arithmetic basket call from uniforms u (n x d)."""
    z = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)) @ corr_chol.T
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    return np.exp(-r * T) * np.maximum(ST @ weights - K, 0.0)


def basket_estimates(n, reps, method, S0, K, r, sigma, corr_chol, T, weights, seed=0):
    """reps independent estimates of the basket price with n points each ('mc' or 'sobol')."""
    d = len(weights)
    out = np.empty(reps)
    for k in range(reps):
        if method == "mc":
            u = np.random.default_rng(seed + k).random((n, d))
        else:
            u = qmc.Sobol(d, scramble=True, seed=seed + k).random(n)
        out[k] = basket_call_payoffs(u, S0, K, r, sigma, corr_chol, T, weights).mean()
    return out
