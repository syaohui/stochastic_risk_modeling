"""Validation of probability-of-default (PD) models: discrimination, calibration, stability.

Discrimination. With risk scores where higher means riskier, the area under
the ROC curve AUC = P(score of a defaulter > score of a non-defaulter) + 0.5
P(tie) (Mann-Whitney form). Gini = 2 AUC - 1. The DeLong, DeLong and
Clarke-Pearson (1988) variance uses the structural components
V10_i = mean_j psi(X_i, Y_j) over non-defaulters j for each defaulter i, and
V01_j likewise; Var(AUC) = s10^2 / n_D + s01^2 / n_N.

Calibration of a grade with N obligors, D defaults and assigned PD p:

- binomial test: p-value P(Binomial(N, p) >= D); assumes independent defaults;
- Jeffreys test (ECB 2019 validation reporting instructions): with the
  Jeffreys prior the posterior of the default rate is Beta(D + 1/2, N - D + 1/2);
  the p-value is its distribution function at p. A small value says p is too low;
- correlation-adjusted test: p-value P(D' >= D) when D' follows the
  one-factor binomial mixture with asset correlation rho (see credit.py).

Defaults in one year share the systematic factor, so the first two tests
reject a correct PD far more often than their nominal size when rho > 0.

Hosmer-Lemeshow over G grades: sum_g (D_g - N_g p_g)^2 / (N_g p_g (1 - p_g)) ~ chi2(G).

Population stability index between expected proportions e_k and actual a_k
over K bins: PSI = sum_k (a_k - e_k) ln(a_k / e_k). Common rules of thumb
(industry practice, not regulation) read PSI < 0.1 as stable and > 0.25 as a shift.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .credit import cond_pd, factor_grid


# ---------------------------------------------------------------------- discrimination
def auc(scores_default, scores_nondefault):
    """AUC by the Mann-Whitney statistic (higher score = riskier)."""
    x = np.asarray(scores_default, float)
    y = np.asarray(scores_nondefault, float)
    allv = np.concatenate([x, y])
    ranks = stats.rankdata(allv)
    r_x = ranks[: len(x)].sum()
    u = r_x - len(x) * (len(x) + 1) / 2.0
    return float(u / (len(x) * len(y)))


def delong_se(scores_default, scores_nondefault):
    """DeLong standard error of the AUC."""
    x = np.asarray(scores_default, float)
    y = np.asarray(scores_nondefault, float)
    # psi(x, y) = 1 if x > y, 0.5 if equal, 0 otherwise, via ranks
    all_r = stats.rankdata(np.concatenate([x, y]))
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    m, n = len(x), len(y)
    v10 = (all_r[:m] - rx) / n              # for each defaulter: share of non-defaulters below
    v01 = 1.0 - (all_r[m:] - ry) / m        # for each non-defaulter: share of defaulters above
    return float(np.sqrt(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n))


# ---------------------------------------------------------------------- calibration
def binomial_pvalue(d, n, pd):
    return float(stats.binom.sf(d - 1, n, pd))


def jeffreys_pvalue(d, n, pd):
    return float(stats.beta.cdf(pd, d + 0.5, n - d + 0.5))


def vasicek_pvalue(d, n, pd, rho):
    """P(D' >= d) under the one-factor binomial mixture (exact up to quadrature)."""
    y, w = factor_grid()
    py = cond_pd(pd, rho, y)
    return float(np.sum(w * stats.binom.sf(d - 1, n, py)))


def hosmer_lemeshow(d, n, pd):
    d, n, pd = (np.asarray(v, float) for v in (d, n, pd))
    hl = np.sum((d - n * pd) ** 2 / (n * pd * (1 - pd)))
    return float(hl), float(stats.chi2.sf(hl, len(d)))


def simulate_defaults(pd, n, rho, n_years, rng=None):
    """Default counts per grade and year in the one-factor model; shape (n_years, grades)."""
    rng = np.random.default_rng(rng)
    pd = np.atleast_1d(np.asarray(pd, float))
    n = np.atleast_1d(np.asarray(n, int))
    y = rng.standard_normal((n_years, 1))
    py = cond_pd(pd[None, :], rho, y)
    return rng.binomial(n[None, :], py)


# ---------------------------------------------------------------------- stability
def psi(expected_counts, actual_counts, eps=1e-6):
    e = np.asarray(expected_counts, float)
    a = np.asarray(actual_counts, float)
    e = np.maximum(e / e.sum(), eps)
    a = np.maximum(a / a.sum(), eps)
    return float(np.sum((a - e) * np.log(a / e)))
