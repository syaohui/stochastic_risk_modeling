"""Stress testing: reverse stress tests, regime switching, and the FRTB stressed calibration.

Reverse stress test for a linear portfolio. Risk factor changes x ~ N(0, Sigma),
P&L = w^T x, loss = -w^T x. Among all scenarios with loss at least l, the most
likely one (highest density, smallest Mahalanobis distance) is

    x* = -l Sigma w / (w^T Sigma w),   at Mahalanobis distance d = l / sqrt(w^T Sigma w),

and P(loss >= l) = 1 - Phi(d). For Gaussian x it is also the conditional mean
E[x | loss = l], which a Monte Carlo check confirms.

Regime switching: a two-state Markov chain s_t in {calm, stress} with daily
switching probabilities; returns are N(0, Sigma_s) given the state, with
higher volatilities and correlations in stress.

FRTB indirect stress calibration (MAR33.5): ES on the full set of risk factors
in a stress period is approximated by

    ES_F,S ~ ES_R,S * ES_F,C / ES_R,C,

using a reduced set of factors R with a long history, where C is the current
period and S the stress period. The reduced set must explain at least 75% of
the full current ES (ES_R,C / ES_F,C >= 0.75 on average over 12 weeks).
The approximation assumes the ratio of full to reduced ES is the same in
calm and stress periods; it fails when correlations change in stress.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .measures import var_es_empirical


def reverse_stress_point(w, Sigma, loss):
    w = np.asarray(w, float)
    Sigma = np.asarray(Sigma, float)
    s2 = float(w @ Sigma @ w)
    x = -loss * (Sigma @ w) / s2
    d = loss / np.sqrt(s2)
    return x, float(d), float(stats.norm.sf(d))


def conditional_scenario_mc(w, Sigma, loss, n, band=0.02, rng=None):
    """Monte Carlo mean of x over draws whose loss lies within loss * (1 +- band)."""
    rng = np.random.default_rng(rng)
    L = np.linalg.cholesky(Sigma)
    x = rng.standard_normal((n, len(w))) @ L.T
    lo = -(x @ w)
    sel = np.abs(lo - loss) <= band * loss
    return x[sel].mean(axis=0), int(sel.sum())


def corr_matrix(n, rho):
    return np.full((n, n), rho) + (1 - rho) * np.eye(n)


def simulate_regimes(n_days, p_cs, p_sc, cov_calm, cov_stress, rng=None, start=0):
    """Markov-switching Gaussian returns. p_cs: P(calm -> stress); p_sc: P(stress -> calm)."""
    rng = np.random.default_rng(rng)
    d = cov_calm.shape[0]
    Lc, Ls = np.linalg.cholesky(cov_calm), np.linalg.cholesky(cov_stress)
    s = np.empty(n_days, dtype=int)
    s[0] = start
    u = rng.random(n_days)
    for t in range(1, n_days):
        if s[t - 1] == 0:
            s[t] = 1 if u[t] < p_cs else 0
        else:
            s[t] = 0 if u[t] < p_sc else 1
    z = rng.standard_normal((n_days, d))
    r = np.where(s[:, None] == 0, z @ Lc.T, z @ Ls.T)
    return r, s


def portfolio_es(returns, w, alpha=0.975):
    return var_es_empirical(-(returns @ w), alpha)[1]


def indirect_stress_es(r_current, r_stress, w, reduced, alpha=0.975):
    """FRTB indirect stressed ES and the reference ES on the full set in stress.

    reduced: indices of the reduced risk factor set. Returns dict with
    ES_F,C, ES_R,C, ES_R,S, the indirect estimate and the full stressed ES_F,S.
    """
    w = np.asarray(w, float)
    wr = np.zeros_like(w)
    wr[reduced] = w[reduced]
    es_fc = portfolio_es(r_current, w, alpha)
    es_rc = portfolio_es(r_current, wr, alpha)
    es_rs = portfolio_es(r_stress, wr, alpha)
    es_fs = portfolio_es(r_stress, w, alpha)
    return {"ES_FC": es_fc, "ES_RC": es_rc, "ES_RS": es_rs,
            "indirect": es_rs * es_fc / es_rc, "ES_FS": es_fs, "ratio_RC_FC": es_rc / es_fc}
