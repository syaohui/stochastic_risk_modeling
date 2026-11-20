"""Backtesting VaR and ES, and the FRTB profit-and-loss attribution (PLA) test.

VaR backtests count exceptions: days on which the loss exceeds the VaR
forecast. Under a correct VaR at level alpha, the hit sequence I_t is i.i.d.
Bernoulli(p) with p = 1 - alpha.

- Kupiec (1995) proportion-of-failures test: with x exceptions in n days,
  LR_pof = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x]  ~ chi2(1).
- Christoffersen (1998) independence test: a two-state Markov chain for I_t
  against independence, LR_ind ~ chi2(1); conditional coverage
  LR_cc = LR_pof + LR_ind ~ chi2(2).
- Basel traffic light (MAR32.9 Table 1), 250 days of 99% VaR: green 0 to 4
  exceptions (multiplier 1.50), amber 5 to 9 (1.70, 1.76, 1.83, 1.88, 1.92),
  red 10 or more (2.00). Desk level (MAR32.19): more than 12 exceptions at 99%
  or 30 at 97.5% in 12 months sends the desk to the standardised approach.

ES backtest of Acerbi and Szekely (2014), "Test 2". With returns X_t (gains
positive), VaR_t and ES_t (positive numbers) at tail probability p = 1 - alpha,
and I_t = 1{X_t + VaR_t < 0}:

    Z2 = sum_t X_t I_t / (T p ES_t) + 1,     E_H0[Z2] = 0,

negative when risk is underestimated. The authors report 5% and 0.01% critical
values close to -0.7 and -1.8 for T = 250, stable across Student t models.

PLA test (MAR32.34 to MAR32.42), 250 days of hypothetical P&L (HPL) and
risk-theoretical P&L (RTPL): Spearman rank correlation and the two-sample
Kolmogorov-Smirnov (KS) distance. Green if correlation > 0.80 and KS < 0.09;
red if correlation < 0.70 or KS > 0.12; amber otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

TRAFFIC_LIGHT_MULTIPLIER = {0: 1.50, 1: 1.50, 2: 1.50, 3: 1.50, 4: 1.50,
                            5: 1.70, 6: 1.76, 7: 1.83, 8: 1.88, 9: 1.92}


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


# ---------------------------------------------------------------------- VaR tests
def kupiec_pof(x, n, p):
    """Kupiec likelihood ratio and its chi2(1) p-value for x exceptions in n days."""
    phat = x / n
    ll0 = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2.0 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(hits, p):
    """Independence and conditional-coverage tests for a 0/1 hit sequence.

    Returns dict with LR_ind, p_ind, LR_cc, p_cc. Transition counts n_ij count
    days with I_{t-1} = i and I_t = j.
    """
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind = (_xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1))
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = -2.0 * (ll_null - ll_ind)
    lr_pof, _ = kupiec_pof(int(h.sum()), len(h), p)
    lr_cc = lr_pof + lr_ind
    return {"LR_ind": float(lr_ind), "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "LR_cc": float(lr_cc), "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def traffic_light(x):
    """Zone and backtesting multiplier for x exceptions in 250 days (MAR32.9)."""
    if x <= 4:
        return "green", TRAFFIC_LIGHT_MULTIPLIER[x]
    if x <= 9:
        return "amber", TRAFFIC_LIGHT_MULTIPLIER[x]
    return "red", 2.00


def zone_probabilities(p, n=250):
    """P(green), P(amber), P(red) when exceptions are Binomial(n, p)."""
    cdf = stats.binom.cdf
    g = cdf(4, n, p)
    a = cdf(9, n, p) - g
    return float(g), float(a), float(1.0 - g - a)


def desk_ineligible(exc_99, exc_975):
    """MAR32.19: more than 12 exceptions at 99% or more than 30 at 97.5% in 12 months."""
    return bool(exc_99 > 12 or exc_975 > 30)


# ---------------------------------------------------------------------- ES test
def acerbi_szekely_z2(returns, var, es, alpha):
    """Z2 statistic. returns: gains positive; var, es: positive forecasts at level alpha."""
    x = np.asarray(returns, float)
    var = np.broadcast_to(np.asarray(var, float), x.shape)
    es = np.broadcast_to(np.asarray(es, float), x.shape)
    p = 1.0 - alpha
    hit = (x + var) < 0
    return float(np.sum(x * hit / es) / (len(x) * p) + 1.0)


# ---------------------------------------------------------------------- PLA test
def pla_metrics(hpl, rtpl):
    """Spearman correlation, KS distance and PLA zone for two P&L series."""
    rho = float(stats.spearmanr(hpl, rtpl)[0])
    ks = float(stats.ks_2samp(hpl, rtpl).statistic)
    if rho > 0.80 and ks < 0.09:
        zone = "green"
    elif rho < 0.70 or ks > 0.12:
        zone = "red"
    else:
        zone = "amber"
    return rho, ks, zone
