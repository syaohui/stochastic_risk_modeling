"""E08. The asymptotic single risk factor model behind Basel IRB capital.

(a) IRB capital K per unit EAD (LGD 45%) against PD, for maturities of 1, 2.5
    and 5 years, with the Basel correlation R(PD).
(b) Finite portfolios: 99.9% quantile of the default rate among N equal names
    (PD 1%, rho = 0.12), exact from the binomial mixture, against the
    infinitely granular (Vasicek) value and the granularity-adjusted value.
(c) Monte Carlo check: simulated default counts for N = 500 against the exact
    binomial mixture.

Output: figures/fig08_asrf.png, results/e08_asrf.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_int, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import credit as cr
from stochastic_risk.plotting import panel_label, plt, save, use_style

NAME = "e08_asrf"
PD_GRID = np.geomspace(3e-4, 0.2, 60)
N_GRID = np.unique(np.geomspace(20, 10_000, 30).astype(int))
PD, RHO, Q = 0.01, 0.12, 0.999


def compute():
    K = np.array([[cr.irb_capital(p, 0.45, m) for p in PD_GRID] for m in (1.0, 2.5, 5.0)])
    exact = np.array([np.searchsorted(np.cumsum(cr.homogeneous_loss_pmf(n, PD, RHO)), Q) / n
                      for n in N_GRID])
    ga = np.array([cr.granularity_adjustment(Q, PD, RHO, n) for n in N_GRID])
    n_mc = 500
    L = cr.simulate_losses(np.full(n_mc, PD), np.ones(n_mc), RHO, 1_000_000, rng=801)
    emp = np.bincount(L.astype(int), minlength=n_mc + 1)[:40] / len(L)
    pmf = cr.homogeneous_loss_pmf(n_mc, PD, RHO)[:40]
    save_raw(NAME, PD=PD_GRID, K=K, N=N_GRID, exact=exact, ga=ga, emp=emp, pmf=pmf)


def plot():
    use_style()
    d = load_raw(NAME)
    asrf = cr.vasicek_quantile(Q, PD, RHO)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    a1, a2, a3 = axs
    for m, ls, row in ((1.0, ":", 0), (2.5, "-", 1), (5.0, "--", 2)):
        a1.semilogx(100 * d["PD"], 100 * d["K"][row], ls, label=rf"$M = {m:g}$ y")
    a1.set_xlabel(r"Probability of Default PD (%)")
    a1.set_ylabel(r"Capital $K$ (% of EAD)")
    a1.set_ylim(0, 1.45 * 100 * d["K"].max())
    a1.legend(loc="upper left", fontsize=7)
    panel_label(a1, "(a)")
    a2.semilogx(d["N"], 100 * d["exact"], "o", ms=3, mfc="white", label="Exact, $N$ Names")
    a2.semilogx(d["N"], 100 * (asrf + d["ga"]), "-", label="ASRF + Granularity")
    a2.axhline(100 * asrf, color="0.3", ls="--", label="ASRF ($N \\to \\infty$)")
    a2.set_xlabel(r"Number of Names $N$")
    a2.set_ylabel(r"99.9% Default Rate (%)")
    a2.set_ylim(0, 1.5 * 100 * d["exact"].max())
    a2.legend(loc="upper right", fontsize=7)
    panel_label(a2, "(b)")
    k = np.arange(len(d["pmf"]))
    a3.semilogy(k, d["pmf"], "-", lw=1.6, label="Binomial Mixture")
    a3.semilogy(k[::2], np.where(d["emp"] > 0, d["emp"], np.nan)[::2], "o", ms=3, mfc="white", color="C1",
                label="Monte Carlo")
    a3.set_xlabel(r"Defaults Among 500 Names")
    a3.set_ylabel("Probability")
    a3.set_ylim(1e-6, 10)
    a3.legend(loc="upper right", fontsize=7)
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig08_asrf", FIG, width_in=7.2)

    i100 = int(np.argmin(np.abs(d["N"] - 100)))
    i1000 = int(np.argmin(np.abs(d["N"] - 1000)))
    write_numbers(NAME, {
        "EEightKOne": pct(cr.irb_capital(0.01), 2),
        "EEightRWOne": fmt(12.5 * 100 * cr.irb_capital(0.01), 1),
        "EEightAsrf": pct(asrf, 2),
        "EEightExactHundred": pct(float(d["exact"][i100]), 1),
        "EEightGAHundred": pct(asrf + float(d["ga"][i100]), 1),
        "EEightNHundred": str(int(d["N"][i100])),
        "EEightExactThousand": pct(float(d["exact"][i1000]), 2),
        "EEightGAThousand": pct(asrf + float(d["ga"][i1000]), 2),
        "EEightNThousand": fmt_int(int(d["N"][i1000])),
        "EEightMcMax": fmt(float(np.nanmax(np.abs(d["emp"] - d["pmf"]))), 4),
    })


if __name__ == "__main__":
    run(compute, plot)
