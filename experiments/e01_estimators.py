"""E01. How well do standard estimators track conditional VaR and ES?

Daily returns follow a GARCH(1,1) with Student t innovations (nu = 5),
unconditional volatility 1.2% a day. The true one-day-ahead VaR_99 and ES_97.5
are known from the true conditional volatility. Five estimators use the last
500 days: historical simulation (HS), unconditional normal, filtered historical
simulation (FHS, GARCH-t fit then empirical standardized quantiles), GARCH-t
(parametric), and GARCH-EVT (generalized Pareto tail on standardized losses).

(a) One path: true ES_97.5 and the HS and FHS estimates over six years.
(b) 1,000 independent windows: estimate / true for ES_97.5, by method.

Output: figures/fig01_estimators.png, results/e01_estimators.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_int, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import evt, measures as ms, volatility as vol
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e01_estimators"
G = vol.Garch(omega=1.44e-4 * 0.02, alpha_g=0.08, beta_g=0.90, nu=5.0)
W = 500
R = 1000
METHODS = ("HS", "Normal", "FHS", "GARCH-t", "GARCH-EVT")


def estimates(r, alphas=(0.99, 0.975)):
    """VaR and ES (positive losses) of the next day from a return window r.

    Returns {method: {alpha: (VaR, ES)}}; the GARCH model is fitted once.
    """
    g = vol.fit_garch(r, "t")
    s2 = vol.garch_filter(r, g.omega, g.alpha_g, g.beta_g)
    s_next = np.sqrt(s2[-1])
    zl = -r / np.sqrt(s2[:-1])
    out = {m: {} for m in METHODS}
    for a in alphas:
        out["HS"][a] = ms.var_es_empirical(-r, a)
        out["Normal"][a] = ms.parametric_normal(-r, a)
        v, e = ms.var_es_empirical(zl, a)
        out["FHS"][a] = (s_next * v, s_next * e)
        v, e = ms.t_var_es(a, g.nu, ms.std_t_scale(g.nu))
        out["GARCH-t"][a] = (s_next * v, s_next * e)
        v, e, *_ = evt.pot_var_es(zl, a, 0.9)
        out["GARCH-EVT"][a] = (s_next * v, s_next * e)
    return out


def true_risk(s_next, alpha):
    v, e = ms.t_var_es(alpha, G.nu, ms.std_t_scale(G.nu))
    return s_next * v, s_next * e


def compute():
    rng = np.random.default_rng(101)
    # (a) one long path, refit every 25 days
    n = W + 1500
    r, sig = G.simulate(n, rng)
    days, true_es, hs_es, fhs_es = [], [], [], []
    for t in range(W, n, 25):
        win = r[t - W:t]
        est = estimates(win, (0.975,))
        s_next = np.sqrt(G.omega + G.alpha_g * r[t - 1] ** 2 + G.beta_g * sig[t - 1] ** 2)
        days.append(t - W)
        true_es.append(true_risk(s_next, 0.975)[1])
        hs_es.append(est["HS"][0.975][1])
        fhs_es.append(est["FHS"][0.975][1])
    # (b) independent windows
    ratio = np.empty((R, len(METHODS), 2))           # [:, :, 0] VaR_99, [:, :, 1] ES_97.5
    for k in range(R):
        r, sig = G.simulate(W, rng)
        s_next = np.sqrt(G.omega + G.alpha_g * r[-1] ** 2 + G.beta_g * sig[-1] ** 2)
        est = estimates(r)
        tv, _ = true_risk(s_next, 0.99)
        _, te = true_risk(s_next, 0.975)
        for j, m in enumerate(METHODS):
            ratio[k, j, 0] = est[m][0.99][0] / tv
            ratio[k, j, 1] = est[m][0.975][1] / te
    save_raw(NAME, days=days, true_es=true_es, hs_es=hs_es, fhs_es=fhs_es, ratio=ratio)


def plot():
    use_style()
    d = load_raw(NAME)
    ratio = d["ratio"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    a1.plot(d["days"], 100 * d["true_es"], "-", color="0.2", label=r"True ES$_{97.5}$")
    a1.plot(d["days"], 100 * d["hs_es"], "--", label="Historical Simulation")
    a1.plot(d["days"], 100 * d["fhs_es"], "o-", ms=2.5, label="Filtered HS (GARCH-t)")
    a1.set_xlabel(r"Trading Day $t$")
    a1.set_ylabel(r"One-Day ES$_{97.5}$ (% of value)")
    a1.set_ylim(0, 1.45 * 100 * max(d["true_es"].max(), d["hs_es"].max(), d["fhs_es"].max()))
    a1.legend(loc="upper left", fontsize=7.5)
    panel_label(a1, "(a)")

    lr = np.log(ratio[:, :, 1])
    pos = np.arange(len(METHODS))
    bp = a2.boxplot([lr[:, j] for j in pos], positions=pos, widths=0.55, whis=(5, 95),
                    showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor("#cfe0f0")
    a2.axhline(0, color="0.3", ls=":")
    a2.set_xticks(pos)
    a2.set_xticklabels(METHODS, fontsize=7.5)
    a2.tick_params(axis="x", which="minor", bottom=False, top=False)
    a2.set_xlabel("Estimator (500-Day Window)")
    a2.set_ylabel(r"$\ln$(Estimated / True ES$_{97.5}$)")
    a2.set_ylim(-1.2, 1.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig01_estimators", FIG)

    def rmse(j, m):
        return float(np.sqrt(np.mean(np.log(ratio[:, j, m]) ** 2)))

    nums = {"EOneR": fmt_int(R), "EOneW": str(W)}
    keys = {"HS": "HS", "Normal": "Norm", "FHS": "FHS", "GARCH-t": "GT", "GARCH-EVT": "EVT"}
    for j, m in enumerate(METHODS):
        nums[f"EOneRmseES{keys[m]}"] = fmt(rmse(j, 1), 2)
        nums[f"EOneRmseVaR{keys[m]}"] = fmt(rmse(j, 0), 2)
        nums[f"EOneUnderES{keys[m]}"] = pct(np.mean(ratio[:, j, 1] < 1), 0)
        nums[f"EOneUnderHalfES{keys[m]}"] = pct(np.mean(ratio[:, j, 1] < 0.67), 0)
    write_numbers(NAME, nums)


if __name__ == "__main__":
    run(compute, plot)
