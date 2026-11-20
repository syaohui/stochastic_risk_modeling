"""Extreme value theory for tail risk: peaks over threshold and the Hill estimator.

Peaks over threshold (POT). For a loss L and a high threshold u, the excesses
Y = L - u given L > u are approximately generalized Pareto (GPD) with shape xi
(dimensionless; xi > 0 for heavy tails) and scale beta (loss units):

    P(Y <= y) = 1 - (1 + xi y / beta)^(-1/xi).

With n observations of which N_u exceed u, the tail estimators are
(McNeil and Frey 2000; McNeil, Frey and Embrechts 2015):

    VaR_alpha = u + (beta / xi) * ( ( n (1 - alpha) / N_u )^(-xi) - 1 ),
    ES_alpha  = ( VaR_alpha + beta - xi u ) / (1 - xi),     xi < 1.

Hill estimator of the tail index xi from the k largest losses (Pareto-type
tail, xi > 0):  xi_hat = (1/k) sum_{i=1}^{k} ln X_(n-i+1) - ln X_(n-k).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def gpd_fit(excesses):
    """Maximum-likelihood (xi, beta) of a generalized Pareto distribution with location 0."""
    xi, _, beta = stats.genpareto.fit(np.asarray(excesses, float), floc=0.0)
    return float(xi), float(beta)


def pot_var_es(losses, alpha, threshold_quantile=0.9, u=None):
    """POT estimates of VaR and ES at level alpha.

    The threshold is the empirical threshold_quantile of the losses unless u is
    given. Returns (VaR, ES, xi, beta, N_u).
    """
    x = np.asarray(losses, float)
    n = len(x)
    if u is None:
        u = float(np.quantile(x, threshold_quantile))
    exc = x[x > u] - u
    nu_ = len(exc)
    xi, beta = gpd_fit(exc)
    ratio = n * (1.0 - alpha) / nu_
    if abs(xi) < 1e-8:
        var = u - beta * np.log(ratio)
    else:
        var = u + beta / xi * (ratio ** (-xi) - 1.0)
    es = (var + beta - xi * u) / (1.0 - xi) if xi < 1 else np.inf
    return float(var), float(es), xi, beta, nu_


def hill(losses, k):
    """Hill estimate of the tail index xi from the k largest positive losses."""
    x = np.sort(np.asarray(losses, float))
    top = x[-k:]
    ref = x[-k - 1]
    return float(np.mean(np.log(top)) - np.log(ref))


def gpd_quantile_true(alpha, xi, beta, u, p_exceed):
    """Exact VaR of a loss whose excesses over u are GPD(xi, beta) with P(L > u) = p_exceed."""
    return u + beta / xi * (((1.0 - alpha) / p_exceed) ** (-xi) - 1.0)
