"""E09. Tail dependence and rare-event simulation in a credit portfolio.

Portfolio: 1,000 obligors in five rating grades (PD 0.1%, 0.5%, 1%, 2%, 5%,
200 names each), exposures lognormal with total 1 (fixed seed), LGD 45%,
asset correlation rho = 0.15.

(a) Loss quantiles at 99% to 99.99% under the Gaussian copula and Student t
    copulas (nu = 10, 4) with the same marginal PDs and rho, from 2 million
    scenarios each.
(b) Importance sampling (Glasserman and Li 2005) against plain Monte Carlo for
    P(L > x): relative standard error per scenario, both from the same
    number of scenarios, and the variance reduction factor.

Output: figures/fig09_copula_is.png, results/e09_copula_is.{json,tex}
"""
import numpy as np

from common import FIG, fmt, fmt_int, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import credit as cr
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e09_copula_is"
RHO = 0.15
N_SIM = 2_000_000
N_IS = 20_000
QS = np.array([0.99, 0.995, 0.999, 0.9995, 0.9999])


def portfolio():
    rng = np.random.default_rng(900)
    p = np.repeat([0.001, 0.005, 0.01, 0.02, 0.05], 200)
    e = rng.lognormal(0.0, 1.0, 1000)
    return p, 0.45 * e / e.sum()


def compute():
    p, c = portfolio()
    quant = {}
    gauss = None
    for j, (name, cop, nu) in enumerate((("gauss", "gaussian", None), ("t10", "t", 10), ("t4", "t", 4))):
        L = cr.simulate_losses(p, c, RHO, N_SIM, rng=910 + j, copula=cop, nu=nu)
        quant[name] = np.quantile(L, QS)
        if name == "gauss":
            gauss = L
    xs = np.quantile(gauss, [0.99, 0.999, 0.9999])
    mc_p = np.array([np.mean(gauss > x) for x in xs])
    mc_rel = np.sqrt((1 - mc_p) / (mc_p * N_IS))            # plain MC with N_IS scenarios
    is_est, is_se = [], []
    for k, x in enumerate(xs):
        est, se, _ = cr.is_tail_probability(p, c, RHO, x, N_IS, rng=920 + k)
        is_est.append(est)
        is_se.append(se)
    save_raw(NAME, Q=QS, xs=xs, mc_p=mc_p, mc_rel=mc_rel, is_est=is_est, is_se=is_se, **quant)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    lab = 100 * (1 - d["Q"])
    for key, mk, name in (("gauss", "o-", "Gaussian Copula"), ("t10", "s--", r"$t$ Copula, $\nu = 10$"),
                          ("t4", "^:", r"$t$ Copula, $\nu = 4$")):
        a1.semilogx(lab, 100 * d[key], mk, ms=3, label=name)
    a1.invert_xaxis()
    a1.set_xlabel(r"Tail Probability $1 - q$ (%)")
    a1.set_ylabel(r"Loss Quantile (% of Exposure)")
    a1.set_ylim(0, 1.4 * 100 * d["t4"].max())
    a1.legend(loc="upper left", fontsize=7.5)
    panel_label(a1, "(a)")
    is_rel = np.array(d["is_se"]) / np.array(d["is_est"])
    x = np.arange(3)
    a2.bar(x - 0.18, 100 * d["mc_rel"], 0.36, label="Plain Monte Carlo", color="#cfe0f0", edgecolor="0.4")
    a2.bar(x + 0.18, 100 * is_rel, 0.36, label="Importance Sampling", color="#0C5DA5")
    a2.set_xticks(x)
    a2.set_xticklabels([r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$"])
    a2.tick_params(axis="x", which="minor", bottom=False, top=False)
    a2.set_yscale("log")
    a2.set_xlabel(r"Target Probability $P(L > x)$")
    a2.set_ylabel(fr"Relative Error, {N_IS // 1000}k Scenarios (%)")
    a2.set_ylim(0.1, 300)
    a2.legend(loc="upper left", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig09_copula_is", FIG)
    vrf = (d["mc_rel"] / is_rel) ** 2
    i999 = int(np.flatnonzero(np.isclose(d["Q"], 0.999))[0])
    write_numbers(NAME, {
        "ENineGaussQ": pct(float(d["gauss"][i999]), 2),
        "ENineTTenQ": pct(float(d["t10"][i999]), 2),
        "ENineTFourQ": pct(float(d["t4"][i999]), 2),
        "ENineRatioTFour": fmt(float(d["t4"][i999] / d["gauss"][i999]), 2),
        "ENineVrfLow": fmt(float(vrf[0]), 0),
        "ENineVrfHigh": fmt_int(float(vrf[-1])),
        "ENineIsRelHigh": pct(float(is_rel[-1]), 1),
        "ENineMcRelHigh": pct(float(d["mc_rel"][-1]), 0),
        "ENineNsim": fmt_int(N_SIM),
        "ENineNis": fmt_int(N_IS),
    })


if __name__ == "__main__":
    run(compute, plot)
