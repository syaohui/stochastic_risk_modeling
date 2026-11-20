"""E17. Estimating a 99.9% quantile from a few years of data.

Losses are Student t with nu = 3 (tail index xi = 1/3). Target: VaR_99.9.
Estimators: the empirical quantile, a normal fit, and peaks over threshold
(POT) with the threshold at the 90% or 97.5% sample quantile. Root mean
squared error of ln(estimate / truth) over 1,000 samples of each size n.
A Hill plot for one sample of n = 2,500 shows how the tail index estimate
depends on the number k of upper order statistics.

Output: figures/fig17_evt.png, results/e17_evt.{json,tex}
"""
import numpy as np
from scipy import stats

from common import FIG, fmt, load_raw, run, save_raw, write_numbers
from stochastic_risk import evt, measures as ms
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e17_evt"
NU = 3.0
N_GRID = np.array([250, 500, 1000, 2500, 10000])
REPS = 1000
METHODS = ("Empirical", "Normal", "POT 90%", "POT 97.5%")


def compute():
    rng = np.random.default_rng(1701)
    true = stats.t.ppf(0.999, NU)
    err = np.full((len(N_GRID), len(METHODS)), np.nan)
    for i, n in enumerate(N_GRID):
        e = np.zeros((REPS, len(METHODS)))
        for k in range(REPS):
            x = rng.standard_t(NU, n)
            e[k, 0] = ms.var_es_empirical(x, 0.999)[0]
            e[k, 1] = ms.parametric_normal(x, 0.999)[0]
            e[k, 2] = evt.pot_var_es(x, 0.999, 0.90)[0]
            e[k, 3] = evt.pot_var_es(x, 0.999, 0.975)[0] if n * 0.025 >= 15 else np.nan
        with np.errstate(invalid="ignore"):
            err[i] = np.sqrt(np.nanmean(np.log(np.abs(e) / true) ** 2, axis=0))
    x = np.abs(rng.standard_t(NU, 2500))
    ks = np.arange(10, 1001, 10)
    hill = np.array([evt.hill(x, int(k)) for k in ks])
    save_raw(NAME, N=N_GRID, err=err, ks=ks, hill=hill, true=true)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    for j, (m, mk) in enumerate(zip(METHODS, ("o-", "s--", "^-", "v:"))):
        a1.loglog(d["N"], d["err"][:, j], mk, ms=3, label=m)
    a1.set_xlabel(r"Sample Size $n$")
    a1.set_ylabel(r"RMSE of $\ln$(VaR$_{99.9}$ Estimate / Truth)")
    a1.set_ylim(0.03, 3)
    a1.legend(loc="lower left", fontsize=7.5)
    panel_label(a1, "(a)")
    a2.plot(d["ks"], d["hill"], "-", label=r"Hill Estimate $\hat\xi(k)$")
    a2.axhline(1 / NU, color="0.3", ls="--", label=r"True $\xi = 1/3$")
    a2.set_xlabel(r"Upper Order Statistics Used $k$")
    a2.set_ylabel(r"Tail Index $\xi$")
    a2.set_ylim(0, 0.8)
    a2.legend(loc="upper right", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig17_evt", FIG)
    i1000 = int(np.flatnonzero(d["N"] == 1000)[0])
    i250 = int(np.flatnonzero(d["N"] == 250)[0])
    write_numbers(NAME, {
        "ESeventeenEmpThousand": fmt(float(d["err"][i1000, 0]), 2),
        "ESeventeenNormThousand": fmt(float(d["err"][i1000, 1]), 2),
        "ESeventeenPotThousand": fmt(float(d["err"][i1000, 2]), 2),
        "ESeventeenEmpQuarter": fmt(float(d["err"][i250, 0]), 2),
        "ESeventeenPotQuarter": fmt(float(d["err"][i250, 2]), 2),
        "ESeventeenHillSmall": fmt(float(d["hill"][2]), 2),
        "ESeventeenHillLarge": fmt(float(d["hill"][-1]), 2),
    })


if __name__ == "__main__":
    run(compute, plot)
