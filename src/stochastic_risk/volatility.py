"""Volatility models for conditional risk: GARCH(1,1), EWMA and filtered historical simulation.

GARCH(1,1) for daily returns r_t (fraction of value):

    r_t = sigma_t z_t,   sigma_t^2 = omega + a r_{t-1}^2 + b sigma_{t-1}^2,

with omega > 0 (variance per day), a, b >= 0, a + b < 1, and z_t i.i.d. with
mean 0 and variance 1: standard normal, or Student t with nu > 2 degrees of
freedom rescaled to unit variance. The unconditional variance is
omega / (1 - a - b). Code names: a = alpha_g, b = beta_g (alpha is kept for
confidence levels elsewhere in the package).

EWMA (RiskMetrics): sigma_t^2 = lam sigma_{t-1}^2 + (1 - lam) r_{t-1}^2, lam = 0.94.

Filtered historical simulation (FHS): standardize past returns by their
fitted sigma_t, take empirical quantiles of the standardized losses, and
rescale by the one-day-ahead sigma forecast. GARCH-EVT (McNeil and Frey 2000)
replaces the empirical tail of the standardized losses by a generalized
Pareto fit (see evt.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal, stats

from .measures import std_t_scale


@dataclass
class Garch:
    omega: float          # variance intercept (per day)
    alpha_g: float        # reaction to the last squared return
    beta_g: float         # persistence
    nu: float | None = None   # Student t dof of the innovations; None = normal

    @property
    def persistence(self):
        return self.alpha_g + self.beta_g

    @property
    def uncond_var(self):
        return self.omega / (1.0 - self.persistence)

    def innovations(self, size, rng):
        if self.nu is None:
            return rng.standard_normal(size)
        return rng.standard_t(self.nu, size) * std_t_scale(self.nu)

    def simulate(self, n, rng=None, burn=1000, sigma2_0=None):
        """Returns (r, sigma) of length n; sigma[t] is the conditional sd of r[t]."""
        rng = np.random.default_rng(rng)
        z = self.innovations(n + burn, rng)
        r = np.empty(n + burn)
        s2 = np.empty(n + burn)
        s2[0] = self.uncond_var if sigma2_0 is None else sigma2_0
        r[0] = np.sqrt(s2[0]) * z[0]
        for t in range(1, n + burn):
            s2[t] = self.omega + self.alpha_g * r[t - 1] ** 2 + self.beta_g * s2[t - 1]
            r[t] = np.sqrt(s2[t]) * z[t]
        return r[burn:], np.sqrt(s2[burn:])

    def simulate_paths(self, sigma2_next, horizon, n_paths, rng=None):
        """Cumulative return over `horizon` days on n_paths paths, starting from a
        known next-day variance sigma2_next. Used for multi-day risk."""
        rng = np.random.default_rng(rng)
        s2 = np.full(n_paths, float(sigma2_next))
        cum = np.zeros(n_paths)
        for _ in range(horizon):
            r = np.sqrt(s2) * self.innovations(n_paths, rng)
            cum += r
            s2 = self.omega + self.alpha_g * r * r + self.beta_g * s2
        return cum


def garch_filter(r, omega, alpha_g, beta_g, sigma2_0=None):
    """Conditional variances sigma_t^2 for t = 0..n (the last entry is the forecast).

    The recursion sigma_t^2 - b sigma_{t-1}^2 = omega + a r_{t-1}^2 is a linear
    filter, evaluated with scipy.signal.lfilter.
    """
    r = np.asarray(r, float)
    s0 = np.var(r) if sigma2_0 is None else sigma2_0
    drive = omega + alpha_g * r * r                 # input for t = 1..n
    zi = np.array([beta_g * s0])
    out, _ = signal.lfilter([1.0], [1.0, -beta_g], drive, zi=zi)
    return np.concatenate([[s0], out])


def _negloglik(theta, r, dist):
    omega, a, b = np.exp(theta[0]), theta[1], theta[2]
    if a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10
    s2 = garch_filter(r, omega, a, b)[:-1]
    if dist == "normal":
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + r * r / s2)
    nu = 2.0 + np.exp(theta[3])
    c = std_t_scale(nu)
    z = r / np.sqrt(s2)
    return -np.sum(stats.t.logpdf(z / c, nu) - np.log(c) - 0.5 * np.log(s2))


def fit_garch(r, dist="t"):
    """Maximum-likelihood GARCH(1,1) fit with normal or Student t innovations."""
    r = np.asarray(r, float)
    v = np.var(r)
    x0 = [np.log(0.05 * v), 0.08, 0.88] + ([np.log(6.0)] if dist == "t" else [])
    res = optimize.minimize(_negloglik, x0, args=(r, dist), method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
    th = res.x
    nu = 2.0 + float(np.exp(th[3])) if dist == "t" else None
    return Garch(float(np.exp(th[0])), float(th[1]), float(th[2]), nu)


def ewma_variance(r, lam=0.94, sigma2_0=None):
    """EWMA variances for t = 0..n (last entry is the forecast)."""
    r = np.asarray(r, float)
    s0 = np.var(r) if sigma2_0 is None else sigma2_0
    out, _ = signal.lfilter([1.0 - lam], [1.0, -lam], r * r, zi=np.array([lam * s0]))
    return np.concatenate([[s0], out])
