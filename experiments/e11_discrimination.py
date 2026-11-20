"""E11. Discrimination and stability statistics in small samples.

(a) Coverage of the 95% DeLong interval for the AUC against the number of
    defaults n_D (binormal scores, defaulters N(1, 1), non-defaulters N(0, 1),
    true AUC = Phi(1/sqrt 2) = 0.760, 50 non-defaulters per default).
(b) Population stability index when nothing has changed: 10 equal-probability
    bins fixed on a development sample of 10,000; a new sample of size n from
    the same distribution. Share of false alarms at the common 0.10 and 0.25
    thresholds, against the chi-square approximation
    PSI ~ (1/n + 1/n_dev) chi2(9).

Output: figures/fig11_discrimination.png, results/e11_discrimination.{json,tex}
"""
import numpy as np
from scipy import stats

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import pd_validation as pv
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e11_discrimination"
ND_GRID = np.array([5, 10, 20, 50, 100, 200, 500])
R = 4000
N_DEV = 10_000
N_NEW = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
R_PSI = 4000


def compute():
    rng = np.random.default_rng(1101)
    true_auc = stats.norm.cdf(1 / np.sqrt(2))
    cover = np.zeros(len(ND_GRID))
    sd = np.zeros(len(ND_GRID))
    for i, nd in enumerate(ND_GRID):
        hit = 0
        a = []
        for _ in range(R):
            x = rng.normal(1.0, 1.0, nd)
            y = rng.normal(0.0, 1.0, 50 * nd)
            au = pv.auc(x, y)
            se = pv.delong_se(x, y)
            hit += abs(au - true_auc) <= 1.96 * se
            a.append(au)
        cover[i] = hit / R
        sd[i] = np.std(a)
    edges_q = np.linspace(0, 1, 11)
    psi_vals = np.zeros((len(N_NEW), R_PSI))
    for i, n in enumerate(N_NEW):
        for k in range(R_PSI):
            dev = rng.standard_normal(N_DEV)
            edges = np.quantile(dev, edges_q)
            edges[0], edges[-1] = -np.inf, np.inf
            e = np.histogram(dev, edges)[0]
            a = np.histogram(rng.standard_normal(n), edges)[0]
            psi_vals[i, k] = pv.psi(e, a)
    save_raw(NAME, ND=ND_GRID, cover=cover, sd=sd, NN=N_NEW, psi=psi_vals)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.semilogx(d["ND"], 100 * d["cover"], "o-", ms=3.5, label="DeLong 95% Interval")
    a1.axhline(95, color="0.4", ls=":")
    a1.set_xlabel(r"Number of Defaults $n_D$")
    a1.set_ylabel("Coverage (%)")
    a1.set_ylim(80, 100)
    a1.legend(loc="lower right", fontsize=7.5)
    panel_label(a1, "(a)")
    fa10 = (d["psi"] > 0.10).mean(axis=1)
    fa25 = (d["psi"] > 0.25).mean(axis=1)
    nn = d["NN"].astype(float)
    a2.loglog(nn, 100 * np.maximum(fa10, 1e-4), "o-", ms=3.5, label=r"PSI $> 0.10$, Simulated")
    a2.loglog(nn, 100 * np.maximum(fa25, 1e-4), "s-", ms=3.5, label=r"PSI $> 0.25$, Simulated")
    x = np.geomspace(nn[0], nn[-1], 100)
    for thr, ls in ((0.10, "--"), (0.25, ":")):
        a2.loglog(x, 100 * stats.chi2.sf(thr / (1 / x + 1 / N_DEV), 9), ls, color="0.35", lw=0.9)
    a2.plot([], [], "--", color="0.35", label=r"$\chi^2_9$ Approximation")
    a2.set_xlabel(r"New Sample Size $n$")
    a2.set_ylabel(r"False Alarm Rate, No Shift (%)")
    a2.set_ylim(0.01, 3000)
    a2.legend(loc="upper right", fontsize=7)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig11_discrimination", FIG)
    i10 = int(np.flatnonzero(d["ND"] == 10)[0])
    i100 = int(np.flatnonzero(d["ND"] == 100)[0])
    j100 = int(np.flatnonzero(d["NN"] == 100)[0])
    j200 = int(np.flatnonzero(d["NN"] == 200)[0])
    write_numbers(NAME, {
        "EElevenCoverTen": pct(float(d["cover"][i10]), 1),
        "EElevenCoverHundred": pct(float(d["cover"][i100]), 1),
        "EElevenSdTen": fmt(float(d["sd"][i10]), 3),
        "EElevenFaHundred": pct(float(fa10[j100]), 0),
        "EElevenFaTwoHundred": pct(float(fa10[j200]), 0),
        "EElevenMedHundred": fmt(float(np.median(d["psi"][j100])), 3),
    })


if __name__ == "__main__":
    run(compute, plot)
