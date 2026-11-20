"""E07. Model uncertainty: relative-entropy worst cases and challenger dispersion.

(a) Normal losses: ES_97.5 under the model within relative entropy eta that
    maximizes the expected loss (Glasserman and Xu 2014), from 2 million
    reweighted scenarios, against the exact value ES + sqrt(2 eta).
(b) Student t losses (nu = 4, unit variance): the same worst case at
    eta = 0.05 against the number of scenarios, five seeds each. The
    exponential tilt does not exist for heavy tails, so the answer grows with
    the sample and depends on its largest draws.
(c) Challenger dispersion: for each of the 1,000 windows of E01, the ratio of
    the largest to the smallest ES_97.5 among the five estimators (reads
    results/raw/e01_estimators.npz).

Output: figures/fig07_model_uncertainty.png, results/e07_model_uncertainty.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import measures as ms, model_risk as mr
from stochastic_risk.plotting import panel_label, plt, save, use_style

NAME = "e07_model_uncertainty"
ETA = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
N_NORM = 2_000_000
N_T = np.array([10_000, 100_000, 1_000_000, 4_000_000])
SEEDS = 5
ETA_T = 0.05


def compute():
    x = np.random.default_rng(701).standard_normal(N_NORM)
    nominal = ms.var_es_empirical(x, 0.975)[1]
    wc = np.array([mr.worst_case_es(x, 0.975, e) if e > 0 else nominal for e in ETA])
    tr = np.zeros((len(N_T), SEEDS))
    for i, n in enumerate(N_T):
        for s in range(SEEDS):
            y = np.random.default_rng(7100 + 10 * i + s).standard_t(4, n) * ms.std_t_scale(4)
            tr[i, s] = mr.worst_case_es(y, 0.975, ETA_T) / ms.var_es_empirical(y, 0.975)[1]
    save_raw(NAME, eta=ETA, wc=wc, nominal=nominal, NT=N_T, tr=tr)


def plot():
    use_style()
    d = load_raw(NAME)
    r1 = load_raw("e01_estimators")["ratio"][:, :, 1]      # estimate / true ES, by method
    spread = r1.max(axis=1) / r1.min(axis=1)
    es0 = ms.normal_var_es(0.975)[1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.5))
    eg = np.linspace(0, d["eta"].max(), 100)
    a1.plot(eg, es0 + np.sqrt(2 * eg), "-", color="0.3", label=r"Exact, ES $+ \sqrt{2\eta}$")
    a1.plot(d["eta"], d["wc"], "o", ms=4, mfc="white", label="Reweighted Scenarios")
    a1.set_xlabel(r"Relative Entropy Budget $\eta$")
    a1.set_ylabel(r"Worst-Case ES$_{97.5}$ (std. units)")
    a1.set_ylim(2.2, 3.4)
    a1.legend(loc="upper left", fontsize=6.5)
    panel_label(a1, "(a)")
    for s in range(d["tr"].shape[1]):
        a2.semilogx(d["NT"], d["tr"][:, s], "o-", ms=2.5, lw=0.8, color="#0C5DA5", alpha=0.8)
    a2.plot([], [], "o-", color="#0C5DA5", ms=2.5, label=r"$t_4$ Losses, Five Seeds")
    a2.set_xlabel(r"Number of Scenarios")
    a2.set_ylabel(rf"Worst-Case / Nominal ES, $\eta = {ETA_T}$")
    a2.set_ylim(0, 1.45 * d["tr"].max())
    a2.legend(loc="upper left", fontsize=6.5)
    panel_label(a2, "(b)")
    a3.hist(spread, bins=np.linspace(1, 3, 41), color="#cfe0f0", edgecolor="0.5", lw=0.3)
    a3.axvline(np.median(spread), color="#c0392b", ls="--", label=f"Median {np.median(spread):.2f}")
    a3.set_xlabel(r"ES Ratio, Largest / Smallest")
    a3.set_ylabel("Number of Windows")
    a3.set_ylim(0, 1.4 * a3.get_ylim()[1])
    a3.legend(loc="upper right", fontsize=6.5)
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig07_model_uncertainty", FIG, width_in=7.2)
    i1 = int(np.flatnonzero(np.isclose(d["eta"], 0.1))[0])
    err = np.max(np.abs(d["wc"] - (es0 + np.sqrt(2 * d["eta"]))))
    write_numbers(NAME, {
        "ESevenWcNormTenth": fmt(float(d["wc"][i1] / es0), 2),
        "ESevenWcErr": fmt(float(err), 3),
        "ESevenTSmallLo": fmt(float(d["tr"][0].min()), 1),
        "ESevenTSmallHi": fmt(float(d["tr"][0].max()), 1),
        "ESevenTLargeLo": fmt(float(d["tr"][-1].min()), 1),
        "ESevenTLargeHi": fmt(float(d["tr"][-1].max()), 1),
        "ESevenSpreadMed": fmt(float(np.median(spread)), 2),
        "ESevenSpreadNinety": fmt(float(np.quantile(spread, 0.9)), 2),
        "ESevenSpreadOver": pct(float(np.mean(spread > 1.5)), 0),
    })


if __name__ == "__main__":
    run(compute, plot)
