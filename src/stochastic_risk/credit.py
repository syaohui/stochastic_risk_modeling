"""Portfolio credit risk: the one-factor model, Basel IRB capital, copulas, granularity, importance sampling.

One-factor Gaussian model (Vasicek). Obligor i defaults when
sqrt(rho) Y + sqrt(1 - rho) e_i > Phi^{-1}(1 - p_i), with systematic factor Y
and idiosyncratic e_i independent standard normal (large Y = bad state),
default probability p_i (one year, dimensionless) and asset correlation rho.
Given Y = y the defaults are independent with

    p_i(y) = Phi( (Phi^{-1}(p_i) + sqrt(rho) y) / sqrt(1 - rho) ).

For an infinitely granular homogeneous portfolio the loss fraction is p(Y)
(times LGD), so (Vasicek; Gordy 2003)

    P(L <= x) = Phi( (sqrt(1 - rho) Phi^{-1}(x) - Phi^{-1}(p)) / sqrt(rho) ),
    VaR_q(L)  = Phi( (Phi^{-1}(p) + sqrt(rho) Phi^{-1}(q)) / sqrt(1 - rho) ).

Basel IRB capital for corporates (BCBS 2005 explanatory note; CRE31), per
unit of exposure at default, with PD and LGD as decimals and maturity M in years:

    R = 0.12 (1 - e^{-50 PD}) / (1 - e^{-50}) + 0.24 [1 - (1 - e^{-50 PD}) / (1 - e^{-50})],
    b = (0.11852 - 0.05478 ln PD)^2,
    K = [LGD Phi( (1-R)^{-1/2} Phi^{-1}(PD) + (R/(1-R))^{1/2} Phi^{-1}(0.999) ) - PD LGD]
        * (1 + (M - 2.5) b) / (1 - 1.5 b),
    RWA = 12.5 K EAD.

Granularity (second-order expansion, derived for this package). With
X = E[L | Y] = m(Y) and Var(L | Y) = v(Y), the q-quantile of L is approximately
m(y_q) + Delta, y_q = Phi^{-1}(q), and

    Delta = -(1 / (2 h(y))) d/dy [ h(y) v(y) / m'(y) ]  at y = y_q,

with h the standard normal density. This is the derivative-of-quantile
expansion written in the factor variable. For N equal names with LGD 1,
m(y) = p(y) and v(y) = p(y)(1 - p(y)) / N, so Delta is of order 1/N.

Importance sampling (Glasserman and Li 2005): shift the factor mean to mu and
exponentially twist the conditional default probabilities so that the
conditional mean loss reaches the threshold x:

    q_i(theta, y) = p_i(y) e^{theta c_i} / (1 + p_i(y)(e^{theta c_i} - 1)),
    psi(theta, y) = sum_i ln(1 + p_i(y)(e^{theta c_i} - 1)),

with theta = theta_x(y) >= 0 solving d psi / d theta = x (0 if E[L|y] >= x),
and mu maximizing psi(theta_x(y), y) - theta_x(y) x - y^2 / 2. The estimator of
P(L > x) is 1{L > x} exp(-theta L + psi(theta, Y)) exp(-mu Y + mu^2 / 2), Y ~ N(mu, 1).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

N_ = stats.norm


# ---------------------------------------------------------------------- Vasicek and IRB
def cond_pd(p, rho, y):
    """Default probability given the systematic factor y (large y = bad state)."""
    return N_.cdf((N_.ppf(p) + np.sqrt(rho) * np.asarray(y, float)) / np.sqrt(1.0 - rho))


def vasicek_cdf(x, p, rho):
    return N_.cdf((np.sqrt(1.0 - rho) * N_.ppf(x) - N_.ppf(p)) / np.sqrt(rho))


def vasicek_quantile(q, p, rho):
    return N_.cdf((N_.ppf(p) + np.sqrt(rho) * N_.ppf(q)) / np.sqrt(1.0 - rho))


def irb_correlation(pd):
    w = (1.0 - np.exp(-50.0 * pd)) / (1.0 - np.exp(-50.0))
    return 0.12 * w + 0.24 * (1.0 - w)


def irb_maturity_b(pd):
    return (0.11852 - 0.05478 * np.log(pd)) ** 2


def irb_capital(pd, lgd=0.45, m=2.5):
    """Basel IRB capital requirement K per unit EAD for corporate exposures."""
    pd = np.asarray(pd, float)
    r = irb_correlation(pd)
    b = irb_maturity_b(pd)
    cond = N_.cdf((N_.ppf(pd) + np.sqrt(r) * N_.ppf(0.999)) / np.sqrt(1.0 - r))
    return (lgd * cond - pd * lgd) * (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)


# ---------------------------------------------------------------------- simulation
def simulate_losses(p, c, rho, n_sims, rng=None, copula="gaussian", nu=None, batch=2000):
    """Portfolio losses sum_i c_i 1{default_i} in the one-factor model.

    p, c: arrays of default probabilities and loss-given-default exposures.
    copula = "gaussian" or "t" (Student t with nu dof: the latent variables
    are divided by a common sqrt(W / nu), W ~ chi2(nu), which adds tail dependence).
    """
    rng = np.random.default_rng(rng)
    p = np.asarray(p, float)
    c = np.asarray(c, float)
    out = np.empty(n_sims)
    if copula == "gaussian":
        thr = N_.ppf(1.0 - p)
    else:
        thr = stats.t.ppf(1.0 - p, nu)
    sr, s1 = np.sqrt(rho), np.sqrt(1.0 - rho)
    for start in range(0, n_sims, batch):
        m = min(batch, n_sims - start)
        y = rng.standard_normal((m, 1))
        lat = sr * y + s1 * rng.standard_normal((m, len(p)))
        if copula != "gaussian":
            lat = lat * np.sqrt(nu / rng.chisquare(nu, (m, 1)))
        out[start:start + m] = (lat > thr) @ c
    return out


def factor_grid(n_grid=4001, lim=10.0):
    """Grid and normalized weights for integrals over the standard normal factor."""
    y = np.linspace(-lim, lim, n_grid)
    w = N_.pdf(y)
    return y, w / w.sum()


def homogeneous_loss_pmf(n, p, rho, n_grid=4001):
    """Exact distribution of the number of defaults among n equal names (binomial mixture)."""
    y, w = factor_grid(n_grid)
    py = cond_pd(p, rho, y)
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k[:, None], n, py[None, :]) @ w
    return pmf / pmf.sum()


def granularity_adjustment(q, p, rho, n, h=1e-4):
    """Second-order correction Delta to the loss-fraction quantile of n equal names (LGD 1)."""
    def g(y):
        m = cond_pd(p, rho, y)
        dm = (cond_pd(p, rho, y + h) - cond_pd(p, rho, y - h)) / (2 * h)
        return N_.pdf(y) * m * (1 - m) / n / dm

    y = N_.ppf(q)
    deriv = (g(y + h) - g(y - h)) / (2 * h)
    return float(-deriv / (2 * N_.pdf(y)))


# ---------------------------------------------------------------------- importance sampling
def _psi(theta, py, c):
    return np.sum(np.log1p(py * np.expm1(theta * c)))


def _theta_x(py, c, x):
    if np.sum(py * c) >= x:
        return 0.0

    def dpsi(t):
        e = np.exp(t * c)
        return np.sum(c * py * e / (1 + py * (e - 1))) - x

    hi = 1.0
    while dpsi(hi) < 0:
        hi *= 2.0
    return optimize.brentq(dpsi, 0.0, hi)


def is_shift(p, c, rho, x):
    """Optimal factor mean shift mu for P(L > x) (Glasserman and Li 2005)."""
    def neg_f(y):
        py = cond_pd(p, rho, y)
        th = _theta_x(py, c, x)
        return -(_psi(th, py, c) - th * x - 0.5 * y * y)

    res = optimize.minimize_scalar(neg_f, bounds=(-2.0, 10.0), method="bounded")
    return float(res.x)


def is_tail_probability(p, c, rho, x, n_sims, rng=None, mu=None):
    """Importance-sampling estimate of P(L > x); returns (estimate, standard error, mu)."""
    rng = np.random.default_rng(rng)
    p = np.asarray(p, float)
    c = np.asarray(c, float)
    if mu is None:
        mu = is_shift(p, c, rho, x)
    est = np.empty(n_sims)
    for k in range(n_sims):
        y = mu + rng.standard_normal()
        py = cond_pd(p, rho, y)
        th = _theta_x(py, c, x)
        e = np.exp(th * c)
        q = py * e / (1 + py * (e - 1))
        loss = c @ (rng.random(len(p)) < q)
        lr = np.exp(-th * loss + _psi(th, py, c) - mu * y + 0.5 * mu * mu)
        est[k] = lr * (loss > x)
    return float(est.mean()), float(est.std(ddof=1) / np.sqrt(n_sims)), mu
