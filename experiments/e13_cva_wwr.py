"""E13. CVA and wrong-way risk.

The swaps of E12 (quarterly grid, 40,000 paths). Counterparty hazard 2% a
year on average, LGD 60%. Wrong-way risk follows Hull and White (2012): on
each interval the hazard is exp(a_i + b V(t_i) / 1%), with a_i calibrated so
that the expected survival curve stays exp(-0.02 t). b > 0: default becomes
more likely when our exposure to the counterparty is high (wrong-way risk);
b < 0: right-way risk.

(a) CVA of the payer and the receiver swap against b, relative to the
    independent CVA.
(b) The share of CVA from each time bucket for b = 0 and b = 0.3.

Output: figures/fig13_cva_wwr.png, results/e13_cva_wwr.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import ccr
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e13_cva_wwr"
HW = ccr.HullWhite(0.05, 0.01, 0.03)
PAY = np.arange(1.0, 6.0)
TIMES = np.round(np.arange(0, 21) / 4.0, 12)
N_PATHS = 40_000
B_GRID = np.linspace(-0.6, 0.6, 13)
LGD, LAM = 0.6, 0.02


def compute():
    x, D = HW.simulate(TIMES, N_PATHS, rng=1301)
    K = ccr.par_rate(HW, PAY, 1.0)
    V = ccr.swap_values(HW, TIMES, x, PAY, 1.0, K)
    cva0 = {s: ccr.cva_independent(TIMES, D, sgn * V, LGD, LAM) for s, sgn in (("pay", 1), ("rec", -1))}
    cva = {s: np.array([ccr.cva_wrong_way(TIMES, D, sgn * V, LGD, LAM, b)[0] for b in B_GRID])
           for s, sgn in (("pay", 1), ("rec", -1))}
    # contribution by bucket, payer, b = 0 and 0.3
    contrib = []
    for b in (0.0, 0.3):
        surv = np.ones(N_PATHS)
        parts = []
        _, a_i = ccr.cva_wrong_way(TIMES, D, V, LGD, LAM, b)
        for i in range(1, len(TIMES)):
            dt = TIMES[i] - TIMES[i - 1]
            new = surv * np.exp(-np.exp(a_i[i - 1] + b * V[:, i] / 0.01) * dt)
            parts.append(LGD * np.mean(D[:, i] * np.maximum(V[:, i], 0) * (surv - new)))
            surv = new
        contrib.append(parts)
    save_raw(NAME, B=B_GRID, t=TIMES, cva_pay=cva["pay"], cva_rec=cva["rec"],
             cva0=np.array([cva0["pay"], cva0["rec"]]), contrib=np.array(contrib))


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.plot(d["B"], d["cva_pay"] / d["cva0"][0], "o-", ms=3, label="Payer Swap")
    a1.plot(d["B"], d["cva_rec"] / d["cva0"][1], "s--", ms=3, label="Receiver Swap")
    a1.axhline(1, color="0.4", ls=":")
    a1.axvline(0, color="0.4", ls=":")
    a1.set_xlabel(r"Wrong-Way Parameter $b$ (per 1% of Notional)")
    a1.set_ylabel("CVA / Independent CVA")
    a1.set_ylim(0, 1.4 * max(d["cva_pay"].max() / d["cva0"][0], d["cva_rec"].max() / d["cva0"][1]))
    a1.legend(loc="upper left", fontsize=7.5)
    panel_label(a1, "(a)")
    t = d["t"][1:]
    w = 0.1
    a2.bar(t - w / 2, 1e4 * d["contrib"][0], w, label=r"$b = 0$ (Independent)", color="#cfe0f0", edgecolor="0.4")
    a2.bar(t + w / 2, 1e4 * d["contrib"][1], w, label=r"$b = 0.3$ (Wrong Way)", color="#c0392b")
    a2.set_xlabel(r"Default Time Bucket End $t_i$ (years)")
    a2.set_ylabel(r"CVA Contribution (bp of Notional)")
    a2.set_ylim(0, 1.45 * 1e4 * d["contrib"].max())
    a2.legend(loc="upper right", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig13_cva_wwr", FIG)
    i3 = int(np.argmin(np.abs(d["B"] - 0.3)))
    im3 = int(np.argmin(np.abs(d["B"] + 0.3)))
    write_numbers(NAME, {
        "EThirteenCvaPay": fmt(1e4 * float(d["cva0"][0]), 1),
        "EThirteenCvaRec": fmt(1e4 * float(d["cva0"][1]), 1),
        "EThirteenWwrPay": fmt(float(d["cva_pay"][i3] / d["cva0"][0]), 2),
        "EThirteenRwrPay": fmt(float(d["cva_pay"][im3] / d["cva0"][0]), 2),
        "EThirteenWwrRec": fmt(float(d["cva_rec"][i3] / d["cva0"][1]), 2),
    })


if __name__ == "__main__":
    run(compute, plot)
