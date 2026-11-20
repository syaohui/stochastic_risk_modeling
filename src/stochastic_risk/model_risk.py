"""Model risk and estimation risk in risk measures.

Estimation risk. A risk model estimated from n observations and then used to
forecast is itself random. For i.i.d. normal losses with unknown mean and
variance, plug-in VaR = mean + s z_alpha (s with ddof = 1, z_alpha the normal
quantile) is exceeded with probability

    P(L_{n+1} > mean + s z_alpha) = 1 - T_{n-1}( z_alpha / sqrt(1 + 1/n) ),

because (L_{n+1} - mean) / (s sqrt(1 + 1/n)) has a Student t distribution with
n - 1 degrees of freedom. This exceeds the nominal 1 - alpha. The predictive
VaR  mean + s sqrt(1 + 1/n) t_{n-1}^{-1}(alpha)  has exact coverage.

Historical simulation with the k-th smallest of n losses as VaR is exceeded
with probability (n + 1 - k) / (n + 1) on average, for any continuous loss
distribution (order statistics of the uniform distribution).

Model uncertainty (Glasserman and Xu 2014). Among all models Q within
relative entropy eta of the nominal model P, the worst case of E_Q[V] is
attained by the exponential tilt dQ/dP = exp(theta V) / E_P[exp(theta V)],
with theta >= 0 such that KL(Q || P) = theta E_Q[V] - ln E_P[exp(theta V)] = eta.
For V standard normal the tilt is N(theta, 1) with KL = theta^2 / 2, so the
worst-case mean is sqrt(2 eta).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .measures import var_es_empirical


# ---------------------------------------------------------------------- estimation risk
def plugin_normal_exceedance(n, alpha):
    """Exact exceedance probability of the plug-in normal VaR from n i.i.d. normal losses."""
    z = stats.norm.ppf(alpha)
    return float(stats.t.sf(z / np.sqrt(1.0 + 1.0 / n), n - 1))


def predictive_normal_var(losses, alpha):
    """VaR with exact unconditional coverage for i.i.d. normal losses."""
    x = np.asarray(losses, float)
    n = len(x)
    return float(x.mean() + x.std(ddof=1) * np.sqrt(1.0 + 1.0 / n) * stats.t.ppf(alpha, n - 1))


def hs_expected_exceedance(n, k):
    """Mean exceedance probability of the k-th smallest of n losses used as VaR."""
    return (n + 1.0 - k) / (n + 1.0)


def bootstrap_es_interval(losses, alpha, level=0.95, n_boot=2000, rng=None):
    """Percentile bootstrap interval for the empirical ES."""
    rng = np.random.default_rng(rng)
    x = np.asarray(losses, float)
    n = len(x)
    es = np.array([var_es_empirical(x[rng.integers(0, n, n)], alpha)[1] for _ in range(n_boot)])
    lo, hi = np.quantile(es, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------- model uncertainty
def worst_case_tilt(v, eta):
    """Worst-case reweighting of samples v under a relative entropy budget eta.

    Returns (theta, weights, worst-case mean of v). Weights sum to one.
    """
    v = np.asarray(v, float)
    vc = v - v.max()

    def kl(theta):
        w = np.exp(theta * vc)
        w /= w.sum()
        return float(np.sum(w * np.log(np.maximum(w * len(v), 1e-300)))) - eta

    if eta <= 0:
        w = np.full(len(v), 1.0 / len(v))
        return 0.0, w, float(v.mean())
    hi = 1.0
    while kl(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            break
    theta = optimize.brentq(kl, 0.0, hi, xtol=1e-12)
    w = np.exp(theta * vc)
    w /= w.sum()
    return float(theta), w, float(np.sum(w * v))


def weighted_var_es(losses, weights, alpha):
    """VaR and ES of a weighted sample (weights sum to one)."""
    x = np.asarray(losses, float)
    o = np.argsort(x)
    cw = np.cumsum(weights[o])
    i = min(int(np.searchsorted(cw, alpha)), len(x) - 1)
    var = x[o][i]
    tail = x >= var
    return float(var), float(np.sum(weights[tail] * x[tail]) / np.sum(weights[tail]))


def worst_case_es(losses, alpha, eta):
    """ES under the model within relative entropy eta that maximizes the expected loss.

    The worst-case model is the exponential tilt of the loss itself; its ES is
    computed from the reweighted scenarios. For N(0, 1) losses the tilt is
    N(theta, 1) with theta = sqrt(2 eta), so the ES rises by exactly theta.
    For heavy tails E[exp(theta L)] is infinite, the tilt does not exist, and
    any sample answer is driven by the largest simulated losses.
    """
    x = np.asarray(losses, float)
    _, w, _ = worst_case_tilt(x, eta)
    return weighted_var_es(x, w, alpha)[1]
