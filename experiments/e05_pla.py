"""E05. What the FRTB P&L attribution test sees.

Daily hypothetical P&L HPL = sqrt(1 - f) A + sqrt(f) B over 250 days, with A
and B independent Student t (nu = 4) factor P&Ls of unit variance. Three ways
the risk model's RTPL can differ:

(a) a missing risk factor: RTPL = sqrt(1 - f) A, share f of variance missing;
(b) a scale error (wrong sensitivity or volatility): RTPL = s HPL;
(c) valuation noise: RTPL = HPL + sqrt(g) e, e independent standard normal.

Zone probabilities (MAR32.42: green if Spearman > 0.80 and KS < 0.09; red if
Spearman < 0.70 or KS > 0.12) from 4,000 simulated years per point. A pure
scale error leaves the rank correlation at one and is caught only by KS.

Output: figures/fig05_pla.png, results/e05_pla.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import backtest as bt, measures as ms
from stochastic_risk.plotting import panel_label, plt, save, use_style

NAME = "e05_pla"
T = 250
R = 4000
F_GRID = np.linspace(0.0, 0.5, 21)
S_GRID = np.linspace(0.6, 1.4, 33)
G_GRID = np.linspace(0.0, 0.6, 25)


def tdraw(rng, size):
    return rng.standard_t(4, size) * ms.std_t_scale(4)


def zone_probs(make, grid, rng):
    out = np.zeros((len(grid), 3))
    rho_mean = np.zeros(len(grid))
    ks_mean = np.zeros(len(grid))
    for i, v in enumerate(grid):
        cnt = {"green": 0, "amber": 0, "red": 0}
        rs, ks = [], []
        for _ in range(R):
            h, r = make(v, rng)
            rho, k, z = bt.pla_metrics(h, r)
            cnt[z] += 1
            rs.append(rho)
            ks.append(k)
        out[i] = (cnt["green"] / R, cnt["amber"] / R, cnt["red"] / R)
        rho_mean[i], ks_mean[i] = np.mean(rs), np.mean(ks)
    return out, rho_mean, ks_mean


def compute():
    rng = np.random.default_rng(501)

    def missing(f, rng):
        a, b = tdraw(rng, T), tdraw(rng, T)
        return np.sqrt(1 - f) * a + np.sqrt(f) * b, np.sqrt(1 - f) * a

    def scale(s, rng):
        h = tdraw(rng, T)
        return h, s * h

    def noise(g, rng):
        h = tdraw(rng, T)
        return h, h + np.sqrt(g) * rng.standard_normal(T)

    zf, rf, kf = zone_probs(missing, F_GRID, rng)
    zs, rs, ks = zone_probs(scale, S_GRID, rng)
    zg, rg, kg = zone_probs(noise, G_GRID, rng)
    save_raw(NAME, F=F_GRID, S=S_GRID, G=G_GRID, zf=zf, zs=zs, zg=zg,
             rf=rf, kf=kf, rs=rs, ks=ks, rg=rg, kg=kg)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, x, z, lab, tag in ((axs[0], 100 * d["F"], d["zf"], r"Missing Variance Share $f$ (%)", "(a)"),
                               (axs[1], d["S"], d["zs"], r"Scale Error $s$ (RTPL / HPL)", "(b)"),
                               (axs[2], 100 * d["G"], d["zg"], r"Noise Variance $g$ (%)", "(c)")):
        ax.plot(x, z[:, 0], "-", color="#2a9d4b", label="Green")
        ax.plot(x, z[:, 1], "--", color="#e0a100", label="Amber")
        ax.plot(x, z[:, 2], "-.", color="#c0392b", label="Red")
        ax.set_xlabel(lab)
        ax.set_ylabel("Zone Probability")
        ax.set_ylim(0, 1.4)
        ax.legend(loc="upper center", ncol=3, fontsize=6.5)
        panel_label(ax, tag)
    fig.tight_layout()
    save(fig, "fig05_pla", FIG, width_in=7.2)

    def first_below(x, z, col=0, thr=0.5):
        k = np.flatnonzero(z[:, col] < thr)
        return float(x[k[0]]) if k.size else float("nan")

    i0 = int(np.argmin(np.abs(d["S"] - 1.0)))
    up = d["S"] >= 1.0
    lo = d["S"] <= 1.0
    write_numbers(NAME, {
        "EFiveGreenZero": pct(float(d["zf"][0, 0]), 0),
        "EFiveHalfGreenF": fmt(100 * first_below(d["F"], d["zf"]), 0),
        "EFiveHalfGreenSUp": fmt(first_below(d["S"][up], d["zs"][up]), 2),
        "EFiveHalfGreenSLo": fmt(first_below(d["S"][lo][::-1], d["zs"][lo][::-1]), 2),
        "EFiveHalfGreenG": fmt(100 * first_below(d["G"], d["zg"]), 0),
        "EFiveRhoScale": fmt(float(d["rs"].min()), 3),
        "EFiveGreenSOne": pct(float(d["zs"][i0, 0]), 0),
    })


if __name__ == "__main__":
    run(compute, plot)
