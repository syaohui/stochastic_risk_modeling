"""A few calls from each module, with the numbers they print.

Run from the repository root after `pip install -e .`:

    python examples/quickstart.py

Everything here is a closed form or a small exact computation, so it runs in
about a second and the printed values do not depend on a random seed.
"""
import numpy as np
from scipy import stats

from stochastic_risk import (backtest, ccr, credit, measures, model_risk, oprisk,
                             pd_validation, stress)

# Market risk: 97.5% VaR and ES of a standard normal loss and a unit-variance t4 loss.
print("normal VaR, ES at 97.5%:", np.round(measures.normal_var_es(0.975), 4))
s4 = measures.std_t_scale(4)
print("t4 (unit variance) VaR, ES at 97.5%:", np.round(measures.t_var_es(0.975, 4, scale=s4), 4))

# Backtesting: Basel traffic-light probabilities for a correct 99% VaR over 250 days.
print("green, amber, red for a correct model:", np.round(backtest.zone_probabilities(0.01), 4))
print("zone and multiplier for 6 exceptions:", backtest.traffic_light(6))

# Estimation risk: exceedance probability of a plug-in normal 99% VaR fitted to 250 days,
# and of historical simulation using the third-largest of 250 losses (k = 248).
print("plug-in normal VaR exceedance:", round(model_risk.plugin_normal_exceedance(250, 0.99), 5))
print("HS order-statistic exceedance:", round(model_risk.hs_expected_exceedance(250, 248), 5))

# Credit: Basel IRB capital per unit EAD for PD 1%, LGD 45%, M = 2.5 years.
print("IRB capital K:", round(credit.irb_capital(0.01, lgd=0.45, m=2.5), 5))
# 99.9% default rate of an infinitely granular portfolio, PD 1%, asset correlation 0.12.
print("ASRF 99.9% default rate:", round(credit.vasicek_quantile(0.999, 0.01, 0.12), 5))

# PD validation: 15 defaults among 1,000 obligors with an assigned PD of 1%.
print("binomial p-value:", round(pd_validation.binomial_pvalue(15, 1000, 0.01), 4))
print("Jeffreys p-value:", round(pd_validation.jeffreys_pvalue(15, 1000, 0.01), 4))
print("correlation-adjusted p-value (rho = 0.12):",
      round(pd_validation.vasicek_pvalue(15, 1000, 0.01, 0.12), 4))

# Counterparty risk: two-year expiry payer swaption into a three-year annual swap,
# Hull-White with a = 0.05 per year, sigma = 1%, flat 3% curve.
hw = ccr.HullWhite(a=0.05, sigma=0.01, r0=0.03)
pay = np.array([3.0, 4.0, 5.0])
K = float((hw.p0(2.0) - hw.p0(5.0)) / hw.p0(pay).sum())   # forward par rate, annual accrual
print("forward par rate:", round(K, 5))
print("ATM payer swaption (Jamshidian):", round(ccr.payer_swaption(hw, 2.0, pay, 1.0, K), 6))

# Operational risk: single-loss approximation of the 99.9% annual loss,
# 25 losses a year, lognormal severities with mu = 10 and sigma = 2.
sev = stats.lognorm(s=2.0, scale=np.exp(10.0))
print("SLA 99.9% VaR:", round(oprisk.sla_var(0.999, 25, sev.ppf) / 1e6, 2), "million")

# Stress testing: most likely factor move behind a loss of 4 units.
w = np.array([1.0, -1.0, 0.5])
Sigma = np.array([[1.0, 0.3, -0.5], [0.3, 1.0, -0.2], [-0.5, -0.2, 1.0]])
x_star, d, prob = stress.reverse_stress_point(w, Sigma, 4.0)
print("reverse stress scenario:", np.round(x_star, 3), "Mahalanobis distance:", round(d, 3))
