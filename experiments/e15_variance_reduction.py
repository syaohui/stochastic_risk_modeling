"""E15. Variance reduction: importance sampling for tails, quasi-Monte Carlo for expectations.

(a) P(X > c), X standard normal: relative error with 100,000 samples, plain
    Monte Carlo against importance sampling under N(c, 1); 50 replications.
(b) Five-asset arithmetic basket call (S0 = 100, weights 0.2, sigma = 25%,
    pairwise correlation 0.3, r = 2%, T = 1, strike 100): RMSE against n for
    plain Monte Carlo and scrambled Sobol points (30 replications each); the
    reference price uses 10 scrambled Sobol runs of 2^22 points.

Output: figures/fig15_variance_reduction.png, results/e15_variance_reduction.{json,tex}
"""
import numpy as np
from scipy import stats

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import montecarlo as mc
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e15_variance_reduction"
C_GRID = np.linspace(1.0, 6.0, 11)
N_TAIL = 100_000
REPS_TAIL = 50
D = 5
BASKET = dict(S0=100.0, K=100.0, r=0.02, sigma=0.25, T=1.0, weights=np.full(D, 0.2))
N_GRID = 2 ** np.arange(8, 17)
REPS_Q = 30


def chol():
    C = np.full((D, D), 0.3) + 0.7 * np.eye(D)
    return np.linalg.cholesky(C)


def compute():
    rel_mc = np.zeros(len(C_GRID))
    rel_is = np.zeros(len(C_GRID))
    for i, c in enumerate(C_GRID):
        true = stats.norm.sf(c)
        e_mc = np.array([mc.gaussian_tail_mc(c, N_TAIL, rng=1500 + 100 * i + k)[0] for k in range(REPS_TAIL)])
        e_is = np.array([mc.gaussian_tail_is(c, N_TAIL, rng=1600 + 100 * i + k)[0] for k in range(REPS_TAIL)])
        rel_mc[i] = np.sqrt(np.mean((e_mc - true) ** 2)) / true
        rel_is[i] = np.sqrt(np.mean((e_is - true) ** 2)) / true
    L = chol()
    ref = mc.basket_estimates(2 ** 22, 10, "sobol", corr_chol=L, seed=9000, **BASKET).mean()
    rm = np.zeros(len(N_GRID))
    rq = np.zeros(len(N_GRID))
    for i, n in enumerate(N_GRID):
        em = mc.basket_estimates(int(n), REPS_Q, "mc", corr_chol=L, seed=10_000 * (i + 1), **BASKET)
        eq = mc.basket_estimates(int(n), REPS_Q, "sobol", corr_chol=L, seed=20_000 * (i + 1), **BASKET)
        rm[i] = np.sqrt(np.mean((em - ref) ** 2))
        rq[i] = np.sqrt(np.mean((eq - ref) ** 2))
    save_raw(NAME, C=C_GRID, rel_mc=rel_mc, rel_is=rel_is, N=N_GRID, rm=rm, rq=rq, ref=ref)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    # show plain Monte Carlo only where at least 3 hits are expected
    ok = N_TAIL * stats.norm.sf(d["C"]) >= 3
    a1.semilogy(d["C"][ok], 100 * d["rel_mc"][ok], "o-", ms=3, label="Plain Monte Carlo")
    a1.semilogy(d["C"], 100 * d["rel_is"], "s-", ms=3, label="Importance Sampling")
    a1.set_xlabel(r"Threshold $c$ (standard deviations)")
    a1.set_ylabel(r"Relative Error, $10^5$ Samples (%)")
    a1.set_ylim(0.1, 1e4)
    a1.legend(loc="upper left", fontsize=7.5)
    panel_label(a1, "(a)")
    n = d["N"].astype(float)
    a2.loglog(n, d["rm"], "o-", ms=3, label="Monte Carlo")
    a2.loglog(n, d["rq"], "s-", ms=3, label="Scrambled Sobol")
    a2.loglog(n, d["rm"][0] * (n / n[0]) ** -0.5, ":", color="0.35", label=r"$n^{-1/2}$")
    a2.loglog(n, d["rq"][0] * (n / n[0]) ** -1.0, "--", color="0.35", label=r"$n^{-1}$")
    a2.set_xlabel(r"Number of Points $n$")
    a2.set_ylabel(r"RMSE of Basket Price")
    a2.set_ylim(1e-4, 10)
    a2.legend(loc="upper right", fontsize=7, ncol=2)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig15_variance_reduction", FIG)
    i5 = int(np.argmin(np.abs(d["C"] - 4.0)))
    slope_q = np.polyfit(np.log(n), np.log(d["rq"]), 1)[0]
    slope_m = np.polyfit(np.log(n), np.log(d["rm"]), 1)[0]
    write_numbers(NAME, {
        "EFifteenRelMcFour": pct(float(d["rel_mc"][i5]), 0),
        "EFifteenRelIsFour": pct(float(d["rel_is"][i5]), 1),
        "EFifteenRelIsSix": pct(float(d["rel_is"][-1]), 1),
        "EFifteenSlopeQ": fmt(float(slope_q), 2),
        "EFifteenSlopeM": fmt(float(slope_m), 2),
        "EFifteenGain": fmt(float(d["rm"][-1] / d["rq"][-1]), 0),
        "EFifteenRef": fmt(float(d["ref"]), 4),
    })


if __name__ == "__main__":
    run(compute, plot)
