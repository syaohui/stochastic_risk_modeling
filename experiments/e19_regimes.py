"""E19. Regimes, procyclical risk measures and the FRTB stressed calibration.

Ten risk factors with daily returns N(0, Sigma_s): calm regime volatility 1%
and pairwise correlation 0.2; stress regime volatility 2.5%. The regime is a
Markov chain (calm to stress 0.4% a day, stress to calm 2% a day).
Portfolio: equal weights.

(a) Rolling 250-day ES_97.5 over 20 simulated years against the regime path:
    the risk measure reacts late and stays high after the stress ends.
(b) FRTB indirect stressed ES (MAR33.5): ES_R,S * ES_F,C / ES_R,C using a
    reduced set of k factors, against the full-set stressed ES, from 250
    calm and 250 stressed days (500 repetitions). Stress correlation 0.2
    (unchanged) or 0.6 (correlations rise in stress).

Output: figures/fig19_regimes.png, results/e19_regimes.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import measures as ms, stress as st
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e19_regimes"
D = 10
W = np.full(D, 1.0 / D)
K_GRID = np.arange(1, D + 1)
REPS = 500


def cov(vol, rho):
    return vol * vol * st.corr_matrix(D, rho)


def compute():
    r, s = st.simulate_regimes(5000, 0.004, 0.02, cov(0.01, 0.2), cov(0.025, 0.6), rng=1901)
    pnl = r @ W
    roll = np.array([ms.var_es_empirical(-pnl[t - 250:t], 0.975)[1] for t in range(250, len(pnl))])
    rng = np.random.default_rng(1902)
    ratio = np.zeros((2, len(K_GRID)))
    cover = np.zeros((2, len(K_GRID)))
    for a, rho_s in enumerate((0.2, 0.6)):
        Lc = np.linalg.cholesky(cov(0.01, 0.2))
        Ls = np.linalg.cholesky(cov(0.025, rho_s))
        acc_r = np.zeros(len(K_GRID))
        acc_c = np.zeros(len(K_GRID))
        for _ in range(REPS):
            rc = rng.standard_normal((250, D)) @ Lc.T
            rs = rng.standard_normal((250, D)) @ Ls.T
            for j, k in enumerate(K_GRID):
                out = st.indirect_stress_es(rc, rs, W, np.arange(k))
                acc_r[j] += out["indirect"] / out["ES_FS"]
                acc_c[j] += out["ratio_RC_FC"]
        ratio[a] = acc_r / REPS
        cover[a] = acc_c / REPS
    save_raw(NAME, pnl_es=roll, regime=s[250:], K=K_GRID, ratio=ratio, cover=cover)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    t = np.arange(len(d["pnl_es"])) / 250.0
    a1.fill_between(t, 0, d["regime"] * 100 * d["pnl_es"].max() * 1.1, color="#f3c4bf", lw=0,
                    label="Stress Regime")
    a1.plot(t, 100 * d["pnl_es"], "-", color="#0C5DA5", label=r"Rolling 250-Day ES$_{97.5}$")
    a1.set_xlabel(r"Time (years)")
    a1.set_ylabel(r"ES$_{97.5}$ (% of value)")
    a1.set_ylim(0, 1.5 * 100 * d["pnl_es"].max())
    a1.legend(loc="upper left", fontsize=7.5)
    panel_label(a1, "(a)")
    a2.plot(100 * d["cover"][0], d["ratio"][0], "o-", ms=3, label=r"Stress $\rho = 0.2$ (Unchanged)")
    a2.plot(100 * d["cover"][1], d["ratio"][1], "s--", ms=3, label=r"Stress $\rho = 0.6$")
    a2.axhline(1, color="0.4", ls=":")
    a2.axvline(75, color="0.4", ls="--", lw=0.8)
    a2.text(76, 0.62, "75% Rule", fontsize=7)
    a2.set_xlabel(r"Coverage $ES_{R,C} / ES_{F,C}$ (%)")
    a2.set_ylabel("Indirect / True Stressed ES")
    a2.set_ylim(0.55, 1.25)
    a2.legend(loc="upper left", fontsize=7)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig19_regimes", FIG)
    c6 = d["cover"][1]
    j75 = int(np.flatnonzero(c6 >= 0.75)[0])
    write_numbers(NAME, {
        "ENineteenShareStress": pct(float(d["regime"].mean()), 0),
        "ENineteenKSeventyFive": str(int(d["K"][j75])),
        "ENineteenRatioSeventyFive": fmt(float(d["ratio"][1][j75]), 2),
        "ENineteenRatioOne": fmt(float(d["ratio"][1][0]), 2),
        "ENineteenRatioOneSame": fmt(float(d["ratio"][0][0]), 2),
        "ENineteenCoverOne": pct(float(d["cover"][1][0]), 0),
    })


if __name__ == "__main__":
    run(compute, plot)
