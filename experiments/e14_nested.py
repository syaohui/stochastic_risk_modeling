"""E14. Nested simulation: how to split a fixed budget between outer and inner samples.

Portfolio: a long one-year at-the-money call (S0 = K = 100, r = 2%, sigma = 25%);
risk horizon one week. The target is P(L > c) = 1%, with c the exact loss
level from Black-Scholes. For budgets k = m n of 2^17, 2^19 and 2^21 payoff
draws, the inner sample size n runs over powers of two (m = k / n outer
scenarios). Mean squared error over 100 replications, against the exact 1%.

Gordy and Juneja (2010): bias ~ theta / n and variance ~ alpha(1 - alpha) n / k,
so the best n grows like k^{1/3} and the best RMSE falls like k^{-1/3}.

Output: figures/fig14_nested.png, results/e14_nested.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import montecarlo as mc
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e14_nested"
ARGS = dict(S0=100.0, K=100.0, r=0.02, sigma=0.25, T=1.0, tau=1.0 / 52)
ALPHA = 0.01
BUDGETS = np.array([2 ** 17, 2 ** 19, 2 ** 21])
REPS = 100


def compute():
    c = mc.loss_threshold(ALPHA, **ARGS)
    rng = np.random.default_rng(1401)
    ns = [2 ** j for j in range(3, 15)]
    mse = np.full((len(BUDGETS), len(ns)), np.nan)
    bias = np.full_like(mse, np.nan)
    for i, k in enumerate(BUDGETS):
        for j, n in enumerate(ns):
            m = int(k // n)
            if m < 200:
                continue
            est = np.array([mc.nested_tail_probability(m, n, c, rng=rng, **ARGS) for _ in range(REPS)])
            mse[i, j] = np.mean((est - ALPHA) ** 2)
            bias[i, j] = est.mean() - ALPHA
    save_raw(NAME, ns=np.array(ns), budgets=BUDGETS, mse=mse, bias=bias, c=c)


def plot():
    use_style()
    d = load_raw(NAME)
    ns = d["ns"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    best_n, best_rmse = [], []
    for i, (k, mk) in enumerate(zip(d["budgets"], ("o-", "s-", "^-"))):
        ok = np.isfinite(d["mse"][i])
        rm = np.sqrt(d["mse"][i][ok])
        a1.loglog(ns[ok], 100 * rm, mk, ms=3, label=rf"Budget $k = 2^{{{int(np.log2(k))}}}$")
        j = int(np.argmin(rm))
        best_n.append(ns[ok][j])
        best_rmse.append(rm[j])
    a1.set_xlabel(r"Inner Samples per Scenario $n$")
    a1.set_ylabel(r"RMSE of $\hat P(L > c)$ (% points)")
    a1.set_ylim(0.03, 30)
    a1.legend(loc="upper right", fontsize=7.5)
    panel_label(a1, "(a)")
    k = d["budgets"].astype(float)
    a2.loglog(k, 100 * np.array(best_rmse), "o", ms=4, label="Best RMSE (Simulated)")
    a2.loglog(k, 100 * best_rmse[0] * (k / k[0]) ** (-1 / 3), "--", color="0.3", label=r"Slope $-1/3$")
    a2.loglog(k, 100 * best_rmse[0] * (k / k[0]) ** (-1 / 2), ":", color="0.3", label=r"Slope $-1/2$")
    a2.set_xlabel(r"Total Budget $k = m\,n$")
    a2.set_ylabel(r"Best RMSE (% points)")
    a2.set_ylim(0.03, 3.0)
    a2.legend(loc="lower left", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig14_nested", FIG)
    slope = np.polyfit(np.log(k), np.log(best_rmse), 1)[0]
    row = d["mse"][-1]
    j8 = int(np.flatnonzero(ns == 8)[0])
    write_numbers(NAME, {
        "EFourteenC": fmt(float(d["c"]), 2),
        "EFourteenBestNSmall": str(int(best_n[0])),
        "EFourteenBestNLarge": str(int(best_n[-1])),
        "EFourteenSlope": fmt(float(slope), 2),
        "EFourteenBiasEight": pct(float(d["bias"][-1][j8]), 2),
        "EFourteenRmseBest": pct(float(best_rmse[-1]), 3),
        "EFourteenRmseEight": pct(float(np.sqrt(row[j8])), 2),
    })


if __name__ == "__main__":
    run(compute, plot)
