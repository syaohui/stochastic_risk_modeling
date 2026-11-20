"""Operational risk: the loss distribution approach (LDA).

Annual loss S = X_1 + ... + X_N with N ~ Poisson(lam) losses per year and
i.i.d. severities X (currency) independent of N.

Single-loss approximation (Boecker and Klueppelberg 2005). For subexponential
(heavy-tailed) severities with distribution function F,

    VaR_alpha(S) ~ F^{-1}( 1 - (1 - alpha) / lam )     as alpha -> 1:

at very high levels one large loss drives the annual total.

Exact reference: the compound Poisson distribution of a discretized severity
(span h) by the fast Fourier transform, f_S = IFFT( exp( lam (FFT(f_X) - 1) ) ),
with enough zero padding to avoid wrap-around; Panjer's recursion
f_S(k) = (lam / k) sum_{j=1}^{k} j f_X(j) f_S(k - j), f_S(0) = exp(-lam (1 - f_X(0))),
gives the same numbers and checks the FFT.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def simulate_annual_losses(lam, sampler, n_years, rng=None):
    """Annual losses for n_years; sampler(size, rng) draws severities."""
    rng = np.random.default_rng(rng)
    n = rng.poisson(lam, n_years)
    x = sampler(int(n.sum()), rng)
    idx = np.repeat(np.arange(n_years), n)
    return np.bincount(idx, weights=x, minlength=n_years)


def lognormal_sampler(mu, s):
    return lambda size, rng: rng.lognormal(mu, s, size)


def sla_var(alpha, lam, severity_ppf):
    return float(severity_ppf(1.0 - (1.0 - alpha) / lam))


def discretize(severity_cdf, h, n):
    """Severity probabilities on 0, h, 2h, ... by the rounding method (mass at nearest point)."""
    edges = (np.arange(n + 1) - 0.5) * h
    edges[0] = 0.0
    F = severity_cdf(edges)
    f = np.diff(F)
    f[-1] += 1.0 - F[-1]          # tail mass beyond the grid goes to the last point
    return f


def compound_poisson_fft(lam, fx, pad=4):
    n = len(fx)
    m = int(2 ** np.ceil(np.log2(pad * n)))
    phi = np.fft.rfft(fx, m)
    fs = np.fft.irfft(np.exp(lam * (phi - 1.0)), m)[:n]
    return np.clip(fs, 0.0, None)


def compound_poisson_panjer(lam, fx):
    n = len(fx)
    fs = np.zeros(n)
    fs[0] = np.exp(-lam * (1.0 - fx[0]))
    j = np.arange(n)
    for k in range(1, n):
        fs[k] = lam / k * np.sum(j[1:k + 1] * fx[1:k + 1] * fs[k - 1::-1][:k])
    return fs


def quantile_from_pmf(pmf, h, alpha):
    cdf = np.cumsum(pmf)
    return float(np.searchsorted(cdf, alpha) * h)
