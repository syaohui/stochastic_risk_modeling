"""Stochastic simulation for risk modeling and model validation.

Modules
-------
measures       VaR and ES: empirical, closed forms, FRTB liquidity-horizon scaling
volatility     GARCH(1,1), EWMA, filtered historical simulation inputs
evt            peaks over threshold, generalized Pareto, Hill estimator
backtest       Kupiec, Christoffersen, Basel traffic light, Acerbi-Szekely ES test, FRTB PLA test
model_risk     estimation risk of VaR, bootstrap intervals, relative-entropy worst case
credit         one-factor credit model, Basel IRB capital, copulas, granularity, importance sampling
pd_validation  AUC and DeLong errors, binomial, Jeffreys and correlation-adjusted tests, PSI
ccr            Hull-White exposure simulation, EE/PFE/EEPE, swaptions, CVA, wrong-way risk
montecarlo     nested simulation, importance sampling, quasi-Monte Carlo
oprisk         loss distribution approach, single-loss approximation, FFT and Panjer
stress         reverse stress tests, regime switching, FRTB stressed calibration
plotting       journal figure style
"""
from . import (backtest, ccr, credit, evt, measures, model_risk, montecarlo, oprisk,
               pd_validation, stress, volatility)

__all__ = ["backtest", "ccr", "credit", "evt", "measures", "model_risk", "montecarlo",
           "oprisk", "pd_validation", "stress", "volatility"]
__version__ = "0.1.0"
