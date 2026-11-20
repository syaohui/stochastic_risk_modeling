"""Checks of every module against an independent result: a closed form, an exact
computation, a known limit, or Monte Carlo with a stated tolerance.

Run with  pytest  or, without pytest,  python tests/test_risk.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochastic_risk import backtest as bt  # noqa: E402
from stochastic_risk import ccr, credit as cr, evt  # noqa: E402
from stochastic_risk import measures as ms, model_risk as mr, montecarlo as mc  # noqa: E402
from stochastic_risk import oprisk as op, pd_validation as pv, stress as st  # noqa: E402
from stochastic_risk import volatility as vol  # noqa: E402


# ---------------------------------------------------------------- measures
def test_empirical_matches_closed_forms():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2_000_000)
    v, e = ms.var_es_empirical(x, 0.975)
    v0, e0 = ms.normal_var_es(0.975)
    assert abs(v - v0) < 0.01 and abs(e - e0) < 0.01
    y = rng.standard_t(4, 2_000_000)
    v, e = ms.var_es_empirical(y, 0.99)
    v0, e0 = ms.t_var_es(0.99, 4)
    assert abs(v / v0 - 1) < 0.01 and abs(e / e0 - 1) < 0.02


def test_empirical_es_of_small_sample_by_hand():
    x = np.arange(1, 101, dtype=float)              # losses 1..100
    v, e = ms.var_es_empirical(x, 0.975)            # tail mass 2.5 observations
    assert v == 98.0 and abs(e - (100 + 99 + 0.5 * 98) / 2.5) < 1e-12


def test_lh_cascade_exact_for_independent_gaussian():
    c = ms.normal_var_es(0.975)[1]
    s = np.array([1.0, 0.5, 0.7, 0.3, 0.2])          # one class per liquidity horizon
    lh = np.array(ms.FRTB_LIQUIDITY_HORIZONS)
    sub = [c * np.sqrt(np.sum(s[k:] ** 2)) for k in range(1, 5)]
    got = ms.lh_adjusted_es(c * np.sqrt(np.sum(s ** 2)), sub)
    assert abs(got - c * np.sqrt(np.sum(s ** 2 * lh / 10))) < 1e-12


# ---------------------------------------------------------------- volatility and EVT
def test_garch_filter_reproduces_simulation_and_fit_recovers():
    g = vol.Garch(2e-6, 0.08, 0.90, 6.0)
    r, sig = g.simulate(4000, rng=2)
    s2 = vol.garch_filter(r, g.omega, g.alpha_g, g.beta_g, sig[0] ** 2)
    assert np.max(np.abs(np.sqrt(s2[:-1]) - sig)) < 1e-12
    f = vol.fit_garch(r)
    assert abs(f.persistence - g.persistence) < 0.03


def test_pot_recovers_pareto_tail():
    rng = np.random.default_rng(3)
    xi, beta = 0.3, 1.0
    x = stats.genpareto.rvs(xi, scale=beta, size=200_000, random_state=rng)
    v, e, xh, bh, _ = evt.pot_var_es(x, 0.999, u=0.0)
    true_v = beta / xi * ((1 - 0.999) ** (-xi) - 1)
    assert abs(xh - xi) < 0.02 and abs(v / true_v - 1) < 0.05
    assert abs(evt.hill(rng.pareto(3.0, 400_000) + 1.0, 4000) - 1 / 3) < 0.02


# ---------------------------------------------------------------- backtesting
def test_traffic_light_table_and_zone_probabilities():
    assert bt.traffic_light(4) == ("green", 1.50)
    assert bt.traffic_light(5) == ("amber", 1.70) and bt.traffic_light(9) == ("amber", 1.92)
    assert bt.traffic_light(10) == ("red", 2.00)
    g, a, r = bt.zone_probabilities(0.01)
    assert abs(g - stats.binom.cdf(4, 250, 0.01)) < 1e-12 and abs(g + a + r - 1) < 1e-12


def test_kupiec_and_christoffersen_sizes():
    lr, p = bt.kupiec_pof(3, 250, 0.01)
    assert lr >= 0 and 0 < p <= 1
    # The chi2 approximation is poor in short samples (see e3); check its size
    # where the asymptotics apply.
    rng = np.random.default_rng(4)
    rej = 0
    for _ in range(2000):
        h = rng.random(20000) < 0.05
        rej += bt.christoffersen(h, 0.05)["p_ind"] < 0.05
    assert abs(rej / 2000 - 0.05) < 0.015


def test_z2_has_zero_mean_under_correct_model():
    rng = np.random.default_rng(5)
    v, e = ms.t_var_es(0.975, 5)
    z = [bt.acerbi_szekely_z2(-rng.standard_t(5, 250), v, e, 0.975) for _ in range(4000)]
    assert abs(np.mean(z)) < 0.02


def test_pla_zones():
    rng = np.random.default_rng(6)
    h = rng.standard_normal(250)
    assert bt.pla_metrics(h, h + 0.05 * rng.standard_normal(250))[2] == "green"
    assert bt.pla_metrics(h, rng.standard_normal(250))[2] == "red"
    rho, ks, _ = bt.pla_metrics(h, 1.5 * h)          # pure scale error
    assert rho > 0.999 and ks > 0.09


# ---------------------------------------------------------------- model risk
def test_plugin_exceedance_formula_by_simulation():
    rng = np.random.default_rng(7)
    n = 100
    s = rng.standard_normal((300_000, n + 1))
    m, sd = s[:, :n].mean(1), s[:, :n].std(1, ddof=1)
    emp = np.mean(s[:, n] > m + sd * stats.norm.ppf(0.99))
    th = mr.plugin_normal_exceedance(n, 0.99)
    assert abs(emp - th) < 4 * np.sqrt(th * (1 - th) / 300_000)
    pred = m + sd * np.sqrt(1 + 1 / n) * stats.t.ppf(0.99, n - 1)
    assert abs(np.mean(s[:, n] > pred) - 0.01) < 4 * np.sqrt(0.01 * 0.99 / 300_000)


def test_hs_order_statistic_exceedance_is_distribution_free():
    rng = np.random.default_rng(8)
    n, k = 250, 248
    for draw in (rng.standard_normal, lambda size: rng.standard_t(3, size)):
        s = draw((100_000, n + 1))
        q = np.sort(s[:, :n], axis=1)[:, k - 1]
        assert abs(np.mean(s[:, n] > q) - mr.hs_expected_exceedance(n, k)) < 0.0015


def test_worst_case_tilt_gaussian_closed_form():
    v = np.random.default_rng(9).standard_normal(400_000)
    for eta in (0.1, 0.5):
        assert abs(mr.worst_case_tilt(v, eta)[2] - np.sqrt(2 * eta)) < 0.01
    # worst-case ES: the tilted model is N(theta, 1), so ES rises by theta = sqrt(2 eta)
    es0 = ms.var_es_empirical(v, 0.975)[1]
    assert abs(mr.worst_case_es(v, 0.975, 0.1) - es0 - np.sqrt(0.2)) < 0.02


# ---------------------------------------------------------------- credit
def test_irb_capital_reference_value():
    # PD 1%, LGD 45%, M 2.5: K = 7.385% (risk weight 92.3%)
    assert abs(cr.irb_capital(0.01) - 0.073853) < 1e-5
    assert abs(cr.irb_correlation(1e-9) - 0.24) < 1e-6


def test_vasicek_quantile_inverts_cdf_and_finite_n_converges():
    q = cr.vasicek_quantile(0.999, 0.01, 0.12)
    assert abs(cr.vasicek_cdf(q, 0.01, 0.12) - 0.999) < 1e-10
    pmf = cr.homogeneous_loss_pmf(5000, 0.01, 0.12)
    exact = np.searchsorted(np.cumsum(pmf), 0.999) / 5000
    ga = q + cr.granularity_adjustment(0.999, 0.01, 0.12, 5000)
    assert abs(exact - ga) < 0.002 and abs(exact - q) > abs(exact - ga)


def test_simulated_losses_match_binomial_mixture():
    n = 200
    L = cr.simulate_losses(np.full(n, 0.01), np.ones(n), 0.12, 200_000, rng=10)
    pmf = cr.homogeneous_loss_pmf(n, 0.01, 0.12)
    emp = np.bincount(L.astype(int), minlength=n + 1)[:10] / len(L)
    assert np.max(np.abs(emp - pmf[:10])) < 0.004


def test_importance_sampling_unbiased():
    p, c = np.full(100, 0.01), np.ones(100)
    x = 10.0
    est, se, mu = cr.is_tail_probability(p, c, 0.2, x, 4000, rng=11)
    pmf = cr.homogeneous_loss_pmf(100, 0.01, 0.2)
    exact = pmf[11:].sum()                      # P(L > 10)
    assert abs(est - exact) < 4 * se and se / est < 0.05


# ---------------------------------------------------------------- PD validation
def test_auc_and_delong():
    rng = np.random.default_rng(12)
    d, nd = rng.normal(1.0, 1, 300), rng.normal(0.0, 1, 3000)
    a = pv.auc(d, nd)
    assert abs(a - stats.norm.cdf(1 / np.sqrt(2))) < 0.03
    assert 0.005 < pv.delong_se(d, nd) < 0.03


def test_calibration_tests_reduce_to_each_other():
    assert abs(pv.vasicek_pvalue(15, 1000, 0.01, 1e-9) - pv.binomial_pvalue(15, 1000, 0.01)) < 1e-6
    # Jeffreys p-value is close to the binomial one for large N
    assert abs(pv.jeffreys_pvalue(15, 1000, 0.01) - pv.binomial_pvalue(15, 1000, 0.01)) < 0.03
    assert pv.psi([10, 20, 30], [10, 20, 30]) == 0.0


# ---------------------------------------------------------------- counterparty
def test_hull_white_martingales_and_swaption():
    hw = ccr.HullWhite(0.05, 0.01, 0.03)
    t = np.linspace(0, 5, 61)
    x, D = hw.simulate(t, 200_000, rng=13)
    assert abs(D[:, -1].mean() / hw.p0(5.0) - 1) < 2e-4
    assert abs((D[:, 24] * hw.bond(2.0, 5.0, x[:, 24])).mean() / hw.p0(5.0) - 1) < 2e-4
    pay = np.arange(1, 6.0)
    K = ccr.par_rate(hw, pay, 1.0)
    V = ccr.swap_values(hw, t, x, pay, 1.0, K)
    assert abs(V[:, 0].mean()) < 1e-12
    pv_ = D[:, 24] * np.maximum(V[:, 24], 0)
    assert abs(pv_.mean() - ccr.payer_swaption(hw, 2.0, pay, 1.0, K)) < 4 * pv_.std() / np.sqrt(len(pv_))


def test_wrong_way_cva_reduces_to_independent():
    hw = ccr.HullWhite(0.05, 0.01, 0.03)
    t = np.linspace(0, 5, 21)
    x, D = hw.simulate(t, 20_000, rng=14)
    pay = np.arange(1, 6.0)
    V = ccr.swap_values(hw, t, x, pay, 1.0, ccr.par_rate(hw, pay, 1.0))
    c0 = ccr.cva_independent(t, D, V, 0.6, 0.02)
    cw, _ = ccr.cva_wrong_way(t, D, V, 0.6, 0.02, 0.0)
    assert abs(cw / c0 - 1) < 0.01


# ---------------------------------------------------------------- Monte Carlo, op risk, stress
def test_nested_converges_to_exact_tail_probability():
    c = mc.loss_threshold(0.05, 100, 100, 0.02, 0.25, 1.0, 1 / 52)
    p = mc.nested_tail_probability(40_000, 8192, c, 100, 100, 0.02, 0.25, 1.0, 1 / 52, rng=15)
    assert abs(p - 0.05) < 0.006


def test_gaussian_tail_importance_sampling():
    est, se = mc.gaussian_tail_is(4.0, 200_000, rng=16)
    assert abs(est - stats.norm.sf(4.0)) < 4 * se and se / est < 0.01


def test_fft_matches_panjer_and_monte_carlo():
    s, mu, lam, h = 1.0, 0.0, 5.0, 0.05
    fx = op.discretize(lambda v: stats.lognorm.cdf(v, s, scale=np.exp(mu)), h, 3000)
    f1 = op.compound_poisson_fft(lam, fx)
    f2 = op.compound_poisson_panjer(lam, fx)
    assert np.max(np.abs(f1 - f2)) < 1e-10
    S = op.simulate_annual_losses(lam, op.lognormal_sampler(mu, s), 400_000, rng=17)
    assert abs(op.quantile_from_pmf(f1, h, 0.99) / np.quantile(S, 0.99) - 1) < 0.02


def test_reverse_stress_point_is_conditional_mean():
    w, Sig = np.array([1.0, 0.5, -0.3]), st.corr_matrix(3, 0.4)
    x, d, _ = st.reverse_stress_point(w, Sig, 2.0)
    m, k = st.conditional_scenario_mc(w, Sig, 2.0, 2_000_000, band=0.01, rng=18)
    assert k > 1000 and np.max(np.abs(m - x)) < 0.05


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
