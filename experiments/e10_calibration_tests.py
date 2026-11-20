"""E10. Calibration tests for PD models when defaults are correlated.

One rating grade with N = 1,000 obligors and assigned PD 1%. Defaults in a
year follow the one-factor model with asset correlation rho. All
probabilities below are exact (sums over the binomial-mixture distribution).

(a) Rejection rate at 5% of the binomial test, the Jeffreys test (ECB) and the
    correlation-adjusted test (assuming rho = 0.12), when the PD is correct,
    against the true rho.
(b) Power when the true PD is m times the assigned PD (rho = 0.12).

Output: figures/fig10_calibration_tests.png, results/e10_calibration_tests.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import credit as cr, pd_validation as pv
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e10_calibration_tests"
N, PD = 1000, 0.01
RHO_GRID = np.linspace(0.0, 0.24, 25)
M_GRID = np.linspace(1.0, 3.0, 21)
RHO_ADJ = 0.12


def rejection_regions():
    d = np.arange(N + 1)
    rb = np.array([pv.binomial_pvalue(int(k), N, PD) < 0.05 for k in d])
    rj = np.array([pv.jeffreys_pvalue(int(k), N, PD) < 0.05 for k in d])
    rv = np.array([pv.vasicek_pvalue(int(k), N, PD, RHO_ADJ) < 0.05 for k in d])
    return rb, rj, rv


def pmf_mixture(pd, rho):
    if rho < 1e-9:
        from scipy import stats
        return stats.binom.pmf(np.arange(N + 1), N, pd)
    return cr.homogeneous_loss_pmf(N, pd, rho)


def compute():
    rb, rj, rv = rejection_regions()
    size = np.array([[pmf_mixture(PD, r)[reg].sum() for reg in (rb, rj, rv)] for r in RHO_GRID])
    power = np.array([[pmf_mixture(min(m * PD, 0.5), RHO_ADJ)[reg].sum() for reg in (rb, rj, rv)]
                      for m in M_GRID])
    dcrit = np.array([np.argmax(reg) for reg in (rb, rj, rv)])
    save_raw(NAME, RHO=RHO_GRID, M=M_GRID, size=size, power=power, dcrit=dcrit)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    labels = ("Binomial", "Jeffreys (ECB)", rf"Adjusted, $\rho = {RHO_ADJ}$")
    for j, (lab, mk) in enumerate(zip(labels, ("o-", "s--", "^:"))):
        a1.plot(d["RHO"], 100 * d["size"][:, j], mk, ms=3, label=lab)
        a2.plot(d["M"], 100 * d["power"][:, j], mk, ms=3, label=lab)
    a1.axhline(5, color="0.4", lw=0.8)
    a1.set_xlabel(r"True Asset Correlation $\rho$")
    a1.set_ylabel(r"Rejection Rate, Correct PD (%)")
    a1.set_ylim(0, 1.45 * 100 * d["size"].max())
    a1.legend(loc="upper left", fontsize=7.5)
    panel_label(a1, "(a)")
    a2.axhline(5, color="0.4", lw=0.8)
    a2.set_xlabel(r"True PD / Assigned PD $m$")
    a2.set_ylabel(r"Rejection Rate, $\rho = 0.12$ (%)")
    a2.set_ylim(0, 140)
    a2.legend(loc="upper left", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig10_calibration_tests", FIG)
    i12 = int(np.argmin(np.abs(d["RHO"] - 0.12)))
    i2 = int(np.argmin(np.abs(d["M"] - 2.0)))
    write_numbers(NAME, {
        "ETenSizeBinZero": pct(float(d["size"][0, 0])),
        "ETenSizeBin": pct(float(d["size"][i12, 0]), 0),
        "ETenSizeJef": pct(float(d["size"][i12, 1]), 0),
        "ETenSizeAdj": pct(float(d["size"][i12, 2])),
        "ETenPowBin": pct(float(d["power"][i2, 0]), 0),
        "ETenPowAdj": pct(float(d["power"][i2, 2]), 0),
        "ETenDcritBin": str(int(d["dcrit"][0])),
        "ETenDcritAdj": str(int(d["dcrit"][2])),
    })


if __name__ == "__main__":
    run(compute, plot)
