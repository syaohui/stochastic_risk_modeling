"""E03. What VaR backtests can and cannot detect.

(a) Basel traffic light over 250 days of 99% VaR: probability of each zone
    when the true exceedance probability is p (exact binomial), with the
    Kupiec test's rejection rate at 5% size.
(b) Size of the Christoffersen independence and conditional-coverage tests
    (chi2 critical values) under a correct model, against sample length:
    the asymptotic approximation is poor at 250 days.
(c) Clustering: GARCH-t returns, VaR fixed at the true unconditional 99%
    quantile, so the exception rate is right but exceptions cluster. Rejection
    rates of the Kupiec, independence and conditional-coverage tests.

Output: figures/fig03_var_backtests.png, results/e03_var_backtests.{json,tex}
"""
import numpy as np
from scipy import stats

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import backtest as bt, volatility as vol
from stochastic_risk.plotting import panel_label, plt, save, use_style

NAME = "e03_var_backtests"
P_GRID = np.linspace(0.002, 0.04, 60)
N_GRID = np.array([250, 500, 1000, 2500, 10000])
R_SIZE = 4000
R_POW = 2000
G = vol.Garch(1.44e-4 * 0.02, 0.08, 0.90, 5.0)


def kupiec_rejection(p_true, n=250, p0=0.01, level=0.05):
    x = np.arange(n + 1)
    rej = np.array([bt.kupiec_pof(int(k), n, p0)[1] < level for k in x])
    return float(stats.binom.pmf(x, n, p_true)[rej].sum())


def compute():
    zones = np.array([bt.zone_probabilities(p) for p in P_GRID])
    kup = np.array([kupiec_rejection(p) for p in P_GRID])
    rng = np.random.default_rng(301)
    size = np.zeros((len(N_GRID), 3))
    for i, n in enumerate(N_GRID):
        res = [bt.christoffersen(rng.random(n) < 0.01, 0.01) for _ in range(R_SIZE)]
        size[i, 0] = np.mean([bt.kupiec_pof(int(h), n, 0.01)[1] < 0.05 for h in
                              rng.binomial(n, 0.01, R_SIZE)])
        size[i, 1] = np.mean([q["p_ind"] < 0.05 for q in res])
        size[i, 2] = np.mean([q["p_cc"] < 0.05 for q in res])
    # unconditional 99% quantile of GARCH-t losses by a long simulation
    r_long, _ = G.simulate(4_000_000, rng=302)
    q99 = float(np.quantile(-r_long, 0.99))
    power = np.zeros((len(N_GRID), 3))
    rate = np.zeros(len(N_GRID))
    for i, n in enumerate(N_GRID):
        rk = rp = ri = 0
        hits_total = 0
        for k in range(R_POW):
            r, _ = G.simulate(int(n), rng)
            h = (-r > q99)
            hits_total += h.sum()
            c = bt.christoffersen(h, 0.01)
            rk += bt.kupiec_pof(int(h.sum()), int(n), 0.01)[1] < 0.05
            ri += c["p_ind"] < 0.05
            rp += c["p_cc"] < 0.05
        power[i] = (rk / R_POW, ri / R_POW, rp / R_POW)
        rate[i] = hits_total / (R_POW * n)
    save_raw(NAME, P=P_GRID, zones=zones, kup=kup, N=N_GRID, size=size, power=power, rate=rate, q99=q99)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    a1, a2, a3 = axs
    P = 100 * d["P"]
    a1.plot(P, d["zones"][:, 0], "-", color="#2a9d4b", label="Green")
    a1.plot(P, d["zones"][:, 1], "--", color="#e0a100", label="Amber")
    a1.plot(P, d["zones"][:, 2], "-.", color="#c0392b", label="Red")
    a1.plot(P, d["kup"], ":", color="0.2", label="Kupiec Rejects")
    a1.axvline(1.0, color="0.5", lw=0.7)
    a1.set_xlabel(r"True Exceedance Rate $p$ (%)")
    a1.set_ylabel("Probability")
    a1.set_ylim(0, 1.45)
    a1.legend(loc="upper center", ncol=2, fontsize=6.5)
    panel_label(a1, "(a)")

    for j, (lab, mk) in enumerate((("Kupiec", "o"), ("Independence", "s"), ("Cond. Coverage", "^"))):
        a2.semilogx(d["N"], 100 * d["size"][:, j], mk + "-", ms=3, label=lab)
        a3.semilogx(d["N"], 100 * d["power"][:, j], mk + "-", ms=3, label=lab)
    a2.axhline(5, color="0.4", ls=":")
    a2.set_xlabel(r"Backtest Length $n$ (days)")
    a2.set_ylabel(r"Rejection Rate, Correct Model (%)")
    a2.set_ylim(0, 12)
    a2.legend(loc="upper right", fontsize=6.5)
    panel_label(a2, "(b)")
    a3.set_xlabel(r"Backtest Length $n$ (days)")
    a3.set_ylabel(r"Rejection Rate, Clustered Hits (%)")
    a3.set_ylim(0, 140)
    a3.legend(loc="upper left", fontsize=6.5)
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig03_var_backtests", FIG, width_in=7.2)

    z = d["zones"]
    i1 = int(np.argmin(np.abs(d["P"] - 0.01)))
    i2 = int(np.argmin(np.abs(d["P"] - 0.02)))
    write_numbers(NAME, {
        "EThreeGreenOne": pct(bt.zone_probabilities(0.01)[0]),
        "EThreeAmberOne": pct(bt.zone_probabilities(0.01)[1]),
        "EThreeRedOne": pct(bt.zone_probabilities(0.01)[2], 3),
        "EThreeGreenTwo": pct(bt.zone_probabilities(0.02)[0]),
        "EThreeRedTwo": pct(bt.zone_probabilities(0.02)[2]),
        "EThreeKupTwo": pct(float(d["kup"][i2])),
        "EThreeKupOne": pct(float(d["kup"][i1])),
        "EThreeSizeIndShort": pct(float(d["size"][0, 1])),
        "EThreeSizeIndLong": pct(float(d["size"][-1, 1])),
        "EThreeSizeKupShort": pct(float(d["size"][0, 0])),
        "EThreePowKupShort": pct(float(d["power"][0, 0])),
        "EThreePowIndShort": pct(float(d["power"][0, 1])),
        "EThreePowIndLong": pct(float(d["power"][-1, 1])),
        "EThreePowCcLong": pct(float(d["power"][-1, 2])),
        "EThreeRate": pct(float(d["rate"].mean()), 2),
    })


if __name__ == "__main__":
    run(compute, plot)
