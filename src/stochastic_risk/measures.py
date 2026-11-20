"""Risk measures: value-at-risk (VaR) and expected shortfall (ES).

Conventions used throughout the package. L is a loss in currency units
(positive = money lost), so a return r gives L = -r times the position value.
For a confidence level alpha in (0, 1), for example alpha = 0.99,

    VaR_alpha(L) = inf { x : P(L <= x) >= alpha },
    ES_alpha(L)  = (1 / (1 - alpha)) * int_alpha^1 VaR_u(L) du,

which for a continuous loss equals E[L | L >= VaR_alpha]. ES is coherent and
sub-additive; VaR is not. Under the Basel FRTB rules (MAR33.3) ES is computed
at alpha = 0.975, and backtesting uses VaR at 0.99 and 0.975 (MAR32).

Closed forms used as references:

    normal (mu, sigma):   VaR = mu + sigma z,  ES = mu + sigma phi(z) / (1 - alpha),
                          z = Phi^{-1}(alpha);
    Student t (nu dof, location 0, scale s):
                          VaR = s q,  ES = s * g_nu(q) / (1 - alpha) * (nu + q^2) / (nu - 1),
                          q = t_nu^{-1}(alpha), g_nu the t density.

The FRTB liquidity-horizon formula (MAR33.4) scales a 10-day ES:

    ES = sqrt( ES_T(P)^2 + sum_{j>=2} ( ES_T(P, j) sqrt((LH_j - LH_{j-1}) / T) )^2 ),

with T = 10 days, LH = (10, 20, 40, 60, 120) days, and ES_T(P, j) the ES of the
shocks to the risk factors whose liquidity horizon is at least LH_j. For
independent risk factors with Gaussian P&L that scales with the square root
of time, the formula is exact (derived in lh_adjusted_es; checked in tests).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

FRTB_LIQUIDITY_HORIZONS = (10, 20, 40, 60, 120)   # days, MAR33.4 Table 1


# ---------------------------------------------------------------------- empirical
def var_es_empirical(losses, alpha):
    """VaR and ES of the empirical distribution of a loss sample.

    VaR is the ceil(n alpha)-th smallest loss. ES is the exact ES of the
    empirical measure: the average of the largest n(1 - alpha) losses, with a
    fractional weight on the boundary loss (Acerbi and Tasche 2002).
    """
    x = np.sort(np.asarray(losses, float))
    n = len(x)
    i = int(np.ceil(n * alpha)) - 1
    var = x[i]
    k = n * (1.0 - alpha)                  # tail mass, in observations
    m = int(np.floor(k))
    tail = x[n - m:].sum() if m > 0 else 0.0
    es = (tail + (k - m) * x[n - m - 1]) / k
    return float(var), float(es)


# ---------------------------------------------------------------------- closed forms
def normal_var_es(alpha, mu=0.0, sigma=1.0):
    z = stats.norm.ppf(alpha)
    return mu + sigma * z, mu + sigma * stats.norm.pdf(z) / (1.0 - alpha)


def t_var_es(alpha, nu, scale=1.0, loc=0.0):
    """VaR and ES of loc + scale * T_nu (nu > 1)."""
    q = stats.t.ppf(alpha, nu)
    es = stats.t.pdf(q, nu) / (1.0 - alpha) * (nu + q * q) / (nu - 1.0)
    return loc + scale * q, loc + scale * es


def std_t_scale(nu):
    """Scale that gives a Student t with nu > 2 degrees of freedom unit variance."""
    return np.sqrt((nu - 2.0) / nu)


# ---------------------------------------------------------------------- estimators
def parametric_normal(losses, alpha):
    x = np.asarray(losses, float)
    return normal_var_es(alpha, x.mean(), x.std(ddof=1))


def parametric_t(losses, alpha):
    """Maximum-likelihood Student t fit (dof, location, scale), then closed forms."""
    nu, loc, scale = stats.t.fit(np.asarray(losses, float))
    nu = max(nu, 2.05)
    return t_var_es(alpha, nu, scale, loc)


# ---------------------------------------------------------------------- horizons
def sqrt_time(risk_1d, horizon):
    """Square-root-of-time scaling: exact only for i.i.d. zero-mean Gaussian returns."""
    return risk_1d * np.sqrt(horizon)


def lh_adjusted_es(es_base, es_subsets, lh=FRTB_LIQUIDITY_HORIZONS, T=10.0):
    """FRTB liquidity-adjusted ES (MAR33.4).

    es_base: ES_T(P) for shocks to all risk factors (10-day, currency).
    es_subsets: ES_T(P, j) for j = 2..len(lh), the ES for shocks to risk factors
    with liquidity horizon at least lh[j-1] (same units).

    Why it is exact in one case. Let risk factor class k have horizon LH_k and
    independent Gaussian P&L with 10-day standard deviation s_k. Then
    ES_T(P, j)^2 = c^2 sum_{k: LH_k >= LH_j} s_k^2 with c = phi(z)/(1-alpha), and
    the formula gives c^2 sum_k s_k^2 (1 + (LH_k - LH_1)/T) = c^2 sum_k s_k^2 LH_k / T,
    the ES of independent P&Ls each scaled to its own horizon.
    """
    total = es_base ** 2
    for j, es_j in enumerate(es_subsets, start=1):
        total += es_j ** 2 * (lh[j] - lh[j - 1]) / T
    return float(np.sqrt(total))
