"""E02. Square-root-of-time scaling of ES from one day to ten days.

Ten-day ES_97.5 by Monte Carlo against sqrt(10) times the one-day ES, as a
function of today's volatility state k = sigma_next / sigma_bar (sigma_bar the
unconditional daily volatility), for three return models with the same
unconditional variance:

  GARCH-t   GARCH(1,1), a = 0.08, b = 0.90, Student t innovations (nu = 5);
  GARCH-N   same GARCH, normal innovations;
  i.i.d. t  constant volatility, Student t (nu = 5).

Volatility mean-reverts over the horizon, so scaling overstates risk when
volatility is high and understates it when low. Fat tails thin out under
aggregation, so for i.i.d. t returns scaling overstates ten-day ES.

Output: figures/fig02_scaling.png, results/e02_scaling.{json,tex}
"""
import numpy as np

from common import FIG, fmt, load_raw, run, save_raw, write_numbers
from stochastic_risk import measures as ms, volatility as vol
from stochastic_risk.plotting import SINGLE, plt, save, use_style

NAME = "e02_scaling"
VBAR = 1.44e-4
MODELS = {
    "GARCH-t": vol.Garch(VBAR * 0.02, 0.08, 0.90, 5.0),
    "GARCH-N": vol.Garch(VBAR * 0.02, 0.08, 0.90, None),
    "iid t": vol.Garch(VBAR, 0.0, 0.0, 5.0),
}
K = np.geomspace(0.4, 3.0, 12)
N_PATHS = 400_000
H = 10


def compute():
    out = {}
    for j, (name, g) in enumerate(MODELS.items()):
        ratio = []
        for i, k in enumerate(K):
            s2 = (k * np.sqrt(VBAR)) ** 2 if name != "iid t" else VBAR
            cum = g.simulate_paths(s2, H, N_PATHS, rng=2000 + 100 * j + i)
            es10 = ms.var_es_empirical(-cum, 0.975)[1]
            one = ms.t_var_es(0.975, g.nu, ms.std_t_scale(g.nu))[1] if g.nu else ms.normal_var_es(0.975)[1]
            es1 = np.sqrt(s2) * one
            ratio.append(es10 / (np.sqrt(H) * es1))
        out[name.replace(" ", "_").replace("-", "_")] = np.array(ratio)
    save_raw(NAME, K=K, **out)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, ax = plt.subplots(figsize=(SINGLE[0] * 1.15, SINGLE[1] * 1.1))
    ax.semilogx(d["K"], d["GARCH_t"], "o-", ms=3, label="GARCH-t")
    ax.semilogx(d["K"], d["GARCH_N"], "s--", ms=3, label="GARCH-N")
    ax.semilogx(d["K"], d["iid_t"], "^:", ms=3, label="i.i.d. $t_5$")
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set_xlabel(r"Volatility State $\sigma_{t+1}/\bar\sigma$")
    ax.set_ylabel(r"True / Scaled 10-Day ES$_{97.5}$")
    ax.set_xticks([0.5, 1, 2, 3])
    ax.set_xticklabels(["0.5", "1", "2", "3"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0.4, 1.9)
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, "fig02_scaling", FIG)

    def at(arr, k):
        return float(np.interp(np.log(k), np.log(d["K"]), arr))

    write_numbers(NAME, {
        "ETwoLowT": fmt(at(d["GARCH_t"], 0.5), 2),
        "ETwoHighT": fmt(at(d["GARCH_t"], 2.5), 2),
        "ETwoMidT": fmt(at(d["GARCH_t"], 1.0), 2),
        "ETwoIid": fmt(float(d["iid_t"].mean()), 2),
        "ETwoPaths": f"{N_PATHS:,}".replace(",", "{,}"),
    })


if __name__ == "__main__":
    run(compute, plot)
