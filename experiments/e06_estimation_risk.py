"""E06. Estimation risk: a correct model with estimated parameters is not a correct forecast.

(a) Exceedance probability of 99% VaR estimated from n past observations:
    plug-in normal VaR (exact formula, Monte Carlo check), predictive normal
    VaR (exact 1%), and historical simulation with the ceil(0.99 n)-th order
    statistic, whose mean exceedance (n + 1 - k)/(n + 1) does not depend on the
    loss distribution.
(b) Sampling error of the empirical ES_97.5 from 250 days, and the coverage of
    95% percentile-bootstrap intervals, for normal and Student t (nu = 3) losses.

Output: figures/fig06_estimation_risk.png, results/e06_estimation_risk.{json,tex}
"""
import numpy as np
from scipy import stats

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import measures as ms, model_risk as mr
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e06_estimation_risk"
N_GRID = np.unique(np.geomspace(20, 1000, 40).astype(int))
N_MC = np.array([25, 50, 100, 250, 500])
R_MC = 200_000
R_ES = 2000
B = 1000
N_ES = 250


def compute():
    rng = np.random.default_rng(601)
    plug = np.array([mr.plugin_normal_exceedance(n, 0.99) for n in N_GRID])
    hs = np.array([mr.hs_expected_exceedance(n, int(np.ceil(0.99 * n))) for n in N_GRID])
    mc = np.zeros(len(N_MC))
    pred = np.zeros(len(N_MC))
    for i, n in enumerate(N_MC):
        hits = hp = 0
        for _ in range(R_MC // 20_000):
            s = rng.standard_normal((20_000, n + 1))
            m, sd = s[:, :n].mean(1), s[:, :n].std(1, ddof=1)
            hits += np.sum(s[:, n] > m + sd * stats.norm.ppf(0.99))
            hp += np.sum(s[:, n] > m + sd * np.sqrt(1 + 1 / n) * stats.t.ppf(0.99, n - 1))
        mc[i], pred[i] = hits / R_MC, hp / R_MC
    rel = np.zeros((2, R_ES))
    cover = np.zeros(2)
    width = np.zeros(2)
    for j, draw in enumerate((lambda size: rng.standard_normal(size),
                              lambda size: rng.standard_t(3, size))):
        true_es = ms.normal_var_es(0.975)[1] if j == 0 else ms.t_var_es(0.975, 3)[1]
        hit = 0
        w = []
        for k in range(R_ES):
            x = draw(N_ES)
            rel[j, k] = ms.var_es_empirical(x, 0.975)[1] / true_es - 1
            idx = rng.integers(0, N_ES, (B, N_ES))
            xs = np.sort(x[idx], axis=1)
            kt = N_ES * 0.025                      # tail mass, 6.25 observations
            mt = int(np.floor(kt))
            es_b = (xs[:, N_ES - mt:].sum(axis=1) + (kt - mt) * xs[:, N_ES - mt - 1]) / kt
            lo, hi = np.quantile(es_b, [0.025, 0.975])
            hit += lo <= true_es <= hi
            w.append((hi - lo) / true_es)
        cover[j] = hit / R_ES
        width[j] = np.mean(w)
    save_raw(NAME, N=N_GRID, plug=plug, hs=hs, NMC=N_MC, mc=mc, pred=pred, rel=rel, cover=cover, width=width)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.semilogx(d["N"], 100 * d["hs"], "-", lw=0.9, label="Historical Simulation")
    a1.semilogx(d["N"], 100 * d["plug"], "--", label="Plug-In Normal, Exact")
    a1.semilogx(d["NMC"], 100 * d["mc"], "o", ms=4, mfc="white", label="Plug-In Normal, Monte Carlo")
    a1.semilogx(d["NMC"], 100 * d["pred"], "s", ms=3.5, label="Predictive Normal, Monte Carlo")
    a1.axhline(1.0, color="0.3", ls=":")
    a1.set_xlabel(r"Estimation Window $n$ (days)")
    a1.set_ylabel(r"Exceedance Probability of VaR$_{99}$ (%)")
    a1.set_ylim(0.6, 3.6)
    a1.legend(loc="upper right", fontsize=7)
    panel_label(a1, "(a)")
    bins = np.linspace(-0.8, 1.6, 61)
    a2.hist(d["rel"][0], bins=bins, histtype="step", density=True, lw=1.2, label="Normal Losses")
    a2.hist(d["rel"][1], bins=bins, histtype="stepfilled", alpha=0.35, density=True, label=r"Student $t_3$ Losses")
    a2.axvline(0, color="0.3", ls=":")
    a2.set_xlabel(r"Relative Error of ES$_{97.5}$ ($n = 250$)")
    a2.set_ylabel("Density")
    a2.set_ylim(0, 1.5 * a2.get_ylim()[1])
    a2.legend(loc="upper right", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig06_estimation_risk", FIG)

    i250 = int(np.argmin(np.abs(d["N"] - 250)))
    j250 = int(np.flatnonzero(d["NMC"] == 250)[0])
    write_numbers(NAME, {
        "ESixPlugTwoFifty": pct(mr.plugin_normal_exceedance(250, 0.99), 2),
        "ESixPlugTwentyFive": pct(mr.plugin_normal_exceedance(25, 0.99), 2),
        "ESixMcTwoFifty": pct(float(d["mc"][j250]), 2),
        "ESixPredTwoFifty": pct(float(d["pred"][j250]), 2),
        "ESixHsTwoFifty": pct(mr.hs_expected_exceedance(250, 248), 2),
        "ESixHsTwoHundred": pct(mr.hs_expected_exceedance(200, 198), 2),
        "ESixSdNorm": pct(float(np.std(d["rel"][0])), 0),
        "ESixSdT": pct(float(np.std(d["rel"][1])), 0),
        "ESixBiasT": pct(float(np.mean(d["rel"][1])), 0),
        "ESixCoverNorm": pct(float(d["cover"][0]), 0),
        "ESixCoverT": pct(float(d["cover"][1]), 0),
    })


if __name__ == "__main__":
    run(compute, plot)
