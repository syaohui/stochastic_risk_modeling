"""E12. Exposure profiles of an interest rate swap under Hull-White.

Five-year payer swap, annual payments, unit notional, struck at par; Hull-White
with a = 0.05/year, sigma = 1%, flat 3% curve; 50,000 paths on a monthly grid.

(a) Expected exposure EE, effective EE and the 97.5% PFE over the life of the
    swap, for the payer and the receiver side.
(b) Check: discounted expected exposure at reset dates against the payer
    swaption price from Jamshidian's decomposition.

Output: figures/fig12_exposure.png, results/e12_exposure.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_int, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import ccr
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e12_exposure"
HW = ccr.HullWhite(0.05, 0.01, 0.03)
PAY = np.arange(1.0, 6.0)
TIMES = np.round(np.arange(0, 61) / 12.0, 12)
N_PATHS = 50_000


def compute():
    x, D = HW.simulate(TIMES, N_PATHS, rng=1201)
    K = ccr.par_rate(HW, PAY, 1.0)
    V = ccr.swap_values(HW, TIMES, x, PAY, 1.0, K)
    pay_m = ccr.exposure_metrics(TIMES, V)
    rec_m = ccr.exposure_metrics(TIMES, -V)
    idx = [int(np.argmin(np.abs(TIMES - t))) for t in (1.0, 2.0, 3.0, 4.0)]
    dee = np.array([(D[:, j] * np.maximum(V[:, j], 0)).mean() for j in idx])
    dse = np.array([(D[:, j] * np.maximum(V[:, j], 0)).std() / np.sqrt(N_PATHS) for j in idx])
    jam = np.array([ccr.payer_swaption(HW, t, PAY, 1.0, K) for t in (1.0, 2.0, 3.0, 4.0)])
    save_raw(NAME, t=TIMES, K=K, pay_EE=pay_m["EE"], pay_PFE=pay_m["PFE"], pay_Eff=pay_m["EffEE"],
             rec_EE=rec_m["EE"], rec_PFE=rec_m["PFE"],
             scal=np.array([pay_m["EPE"], pay_m["EEPE"], pay_m["EAD"], rec_m["EPE"], rec_m["EEPE"]]),
             dee=dee, dse=dse, jam=jam)


def plot():
    use_style()
    d = load_raw(NAME)
    t = d["t"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.plot(t, 100 * d["pay_EE"], "-", label="Payer EE")
    a1.plot(t, 100 * d["pay_PFE"], "--", label="Payer PFE$_{97.5}$")
    a1.plot(t, 100 * d["rec_EE"], "-", label="Receiver EE")
    a1.plot(t, 100 * d["rec_PFE"], "--", label="Receiver PFE$_{97.5}$")
    a1.plot(t[t <= 1], 100 * d["pay_Eff"][t <= 1], ":", color="0.2", lw=1.4, label="Payer Effective EE")
    a1.set_xlabel(r"Time $t$ (years)")
    a1.set_ylabel(r"Exposure (% of Notional)")
    a1.set_ylim(0, 1.5 * 100 * max(d["pay_PFE"].max(), d["rec_PFE"].max()))
    a1.legend(loc="upper right", ncol=2, fontsize=6.5)
    panel_label(a1, "(a)")
    tt = np.array([1, 2, 3, 4])
    a2.errorbar(tt, 100 * d["dee"], yerr=200 * d["dse"], fmt="o", ms=4, mfc="white",
                label="Monte Carlo, Discounted EE")
    a2.plot(tt, 100 * d["jam"], "s", ms=4, label="Payer Swaption (Jamshidian)")
    a2.set_xlabel(r"Reset Date $T_k$ (years)")
    a2.set_ylabel(r"Value (% of Notional)")
    a2.set_ylim(0, 1.5 * 100 * d["jam"].max())
    a2.legend(loc="upper right", fontsize=7)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig12_exposure", FIG)
    z = np.max(np.abs(d["dee"] - d["jam"]) / d["dse"])
    s = d["scal"]
    write_numbers(NAME, {
        "ETwelveK": pct(float(d["K"]), 3),
        "ETwelvePeakEE": pct(float(d["pay_EE"].max()), 2),
        "ETwelvePeakPFE": pct(float(d["pay_PFE"].max()), 2),
        "ETwelveTPeak": fmt(float(d["t"][np.argmax(d["pay_EE"])]), 1),
        "ETwelveEPE": pct(float(s[0]), 3),
        "ETwelveEEPE": pct(float(s[1]), 3),
        "ETwelveEAD": pct(float(s[2]), 3),
        "ETwelveRecPeakEE": pct(float(d["rec_EE"].max()), 2),
        "ETwelveMaxZ": fmt(float(z), 1),
        "ETwelvePaths": fmt_int(N_PATHS),
    })


if __name__ == "__main__":
    run(compute, plot)
