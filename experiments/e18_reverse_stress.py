"""E18. Reverse stress testing a linear portfolio.

Three daily risk factor changes, in standard deviations: equity index,
10-year rate, credit spread. Calm correlations (equity, rate) = 0.3,
(equity, spread) = -0.5, (rate, spread) = -0.2; stressed correlations are 1.6
times larger in magnitude (0.48, -0.80, -0.32). Portfolio P&L = w^T x with w = (1.0, 0.6, -0.8)
(long equity, long rates exposure that gains when rates rise, short spread
protection that loses when spreads widen).

(a) The most likely scenario x* behind a loss of l, against l (closed form),
    with Monte Carlo conditional means at three loss levels.
(b) Probability of a loss of at least l (closed form, 1 - Phi(d)), calm and
    stressed correlations: the same loss is far more plausible in stress.

Output: figures/fig18_reverse_stress.png, results/e18_reverse_stress.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, run, save_raw, write_numbers
from stochastic_risk import stress as st
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e18_reverse_stress"
W = np.array([1.0, 0.6, -0.8])
LOSS = np.linspace(0.5, 6.0, 23)
CHECK = np.array([2.0, 3.0, 4.0])


def cov(scale):
    c = np.eye(3)
    c[0, 1] = c[1, 0] = 0.3 * scale
    c[0, 2] = c[2, 0] = -0.5 * scale
    c[1, 2] = c[2, 1] = -0.2 * scale
    return c


def compute():
    calm, stressed = cov(1.0), cov(1.6)
    xs = np.array([st.reverse_stress_point(W, calm, l)[0] for l in LOSS])
    pc = np.array([st.reverse_stress_point(W, calm, l)[2] for l in LOSS])
    ps = np.array([st.reverse_stress_point(W, stressed, l)[2] for l in LOSS])
    mcm = np.array([st.conditional_scenario_mc(W, calm, l, 4_000_000, band=0.01, rng=1801 + i)[0]
                    for i, l in enumerate(CHECK)])
    sd = np.sqrt(W @ calm @ W), np.sqrt(W @ stressed @ W)
    save_raw(NAME, loss=LOSS, xs=xs, pc=pc, ps=ps, check=CHECK, mcm=mcm, sd=np.array(sd))


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    names = ("Equity", "Rate", "Spread")
    for j, (nm, mk) in enumerate(zip(names, ("o", "s", "^"))):
        line, = a1.plot(d["loss"], d["xs"][:, j], "-", label=f"{nm}, Closed Form")
        a1.plot(d["check"], d["mcm"][:, j], mk, ms=4.5, mfc="white", color=line.get_color())
    a1.plot([], [], "o", color="0.3", mfc="white", label="Monte Carlo Means")
    a1.axhline(0, color="0.5", lw=0.6)
    a1.set_xlabel(r"Portfolio Loss $l$ (P&L standard units)")
    a1.set_ylabel(r"Factor Move in $x^*$ (standard deviations)")
    a1.set_ylim(-6, 8.5)
    a1.legend(loc="upper left", fontsize=7, ncol=2)
    panel_label(a1, "(a)")
    a2.semilogy(d["loss"], d["pc"], "-", label="Calm Correlations")
    a2.semilogy(d["loss"], d["ps"], "--", label="Stressed Correlations")
    a2.set_xlabel(r"Portfolio Loss $l$ (P&L standard units)")
    a2.set_ylabel(r"Probability of Loss $\geq l$")
    a2.set_ylim(1e-5, 10)
    a2.legend(loc="upper right", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig18_reverse_stress", FIG)
    i4 = int(np.argmin(np.abs(d["loss"] - 4.0)))
    i6 = int(np.argmin(np.abs(d["loss"] - 6.0)))
    err = np.max(np.abs(d["mcm"] - np.array([st.reverse_stress_point(W, cov(1.0), l)[0] for l in d["check"]])))
    write_numbers(NAME, {
        "EEighteenSdCalm": fmt(float(d["sd"][0]), 2),
        "EEighteenSdStress": fmt(float(d["sd"][1]), 2),
        "EEighteenPCalm": f"{float(d['pc'][i6]):.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}",
        "EEighteenPStress": f"{float(d['ps'][i6]):.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}",
        "EEighteenRatio": fmt(float(d["ps"][i6] / d["pc"][i6]), 1),
        "EEighteenMcErr": fmt(float(err), 3),
        "EEighteenEqFour": fmt(float(d["xs"][i4, 0]), 2),
        "EEighteenRateFour": fmt(float(d["xs"][i4, 1]), 2),
        "EEighteenSpreadFour": fmt(float(d["xs"][i4, 2]), 2),
    })


if __name__ == "__main__":
    run(compute, plot)
