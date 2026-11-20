"""E04. Backtesting ES with the Acerbi-Szekely Z2 test against counting VaR exceptions.

The model forecasts a Student t loss distribution (nu = 5, unit scale) every
day for T = 250 days. Under H0 the losses follow it.

(a) Null distribution of Z2: simulated 5% and 0.01% critical values, compared
    with the -0.7 and -1.8 reported by Acerbi and Szekely (2014).
(b) Power against volatility underestimation (true scale = k > 1) at 5% size:
    Z2 against the one-sided VaR_99 exception count test (reject when the
    count is at least the smallest x with P(X >= x) <= 5%).
(c) Power against heavier tails with the same variance: the truth is t_nu
    rescaled to the model's variance, nu from 30 down to 3.

Output: figures/fig04_es_backtest.png, results/e04_es_backtest.{json,tex}
"""
import numpy as np
from scipy import stats

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import backtest as bt, measures as ms
from stochastic_risk.plotting import panel_label, plt, save, use_style

NAME = "e04_es_backtest"
NU = 5.0
T = 250
R_NULL = 200_000
R_POW = 20_000
K_GRID = np.linspace(1.0, 1.6, 13)
NU_GRID = np.array([30.0, 15.0, 10.0, 7.0, 5.0, 4.0, 3.0])


def z2_batch(losses, var, es, alpha=0.975):
    """Z2 for each row of a (reps x T) loss matrix."""
    x = -losses
    hit = (x + var) < 0
    return np.sum(x * hit, axis=1) / (T * (1 - alpha) * es) + 1.0


def var_count_crit(p=0.01, level=0.05):
    x = np.arange(T + 1)
    sf = stats.binom.sf(x - 1, T, p)                  # P(X >= x)
    return int(x[np.argmax(sf <= level)])


def compute():
    rng = np.random.default_rng(401)
    v975, e975 = ms.t_var_es(0.975, NU)
    v99, _ = ms.t_var_es(0.99, NU)
    z0 = np.concatenate([z2_batch(rng.standard_t(NU, (20_000, T)), v975, e975)
                         for _ in range(R_NULL // 20_000)])
    crit5, crit001 = np.quantile(z0, [0.05, 0.0001])
    xc = var_count_crit()
    pow_k = np.zeros((len(K_GRID), 2))
    for i, k in enumerate(K_GRID):
        L = k * rng.standard_t(NU, (R_POW, T))
        pow_k[i, 0] = np.mean(z2_batch(L, v975, e975) < crit5)
        pow_k[i, 1] = np.mean((L > v99).sum(axis=1) >= xc)
    pow_nu = np.zeros((len(NU_GRID), 2))
    model_sd = np.sqrt(NU / (NU - 2))
    for i, nu in enumerate(NU_GRID):
        L = rng.standard_t(nu, (R_POW, T)) * ms.std_t_scale(nu) * model_sd
        pow_nu[i, 0] = np.mean(z2_batch(L, v975, e975) < crit5)
        pow_nu[i, 1] = np.mean((L > v99).sum(axis=1) >= xc)
    size_var = float(stats.binom.sf(xc - 1, T, 0.01))
    save_raw(NAME, z0=z0[:50_000], crit=np.array([crit5, crit001]), xc=xc, size_var=size_var,
             K=K_GRID, pow_k=pow_k, NU=NU_GRID, pow_nu=pow_nu)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    a1, a2, a3 = axs
    a1.hist(d["z0"], bins=80, density=True, color="#cfe0f0", edgecolor="0.5", lw=0.3)
    a1.axvline(d["crit"][0], color="#c0392b", ls="--", label=f"5% Level, {d['crit'][0]:.2f}")
    a1.axvline(-0.7, color="0.2", ls=":", label="Acerbi-Szekely, $-0.7$")
    a1.set_xlabel(r"Test Statistic $Z_2$")
    a1.set_ylabel("Density Under $H_0$")
    a1.set_xlim(-2.5, 1.2)
    a1.set_ylim(0, 1.6 * a1.get_ylim()[1])
    a1.legend(loc="upper left", fontsize=6.5)
    panel_label(a1, "(a)")
    for ax, x, pw, lab in ((a2, d["K"], d["pow_k"], r"Volatility Ratio $k$"),
                           (a3, d["NU"], d["pow_nu"], r"True Tail Index $\nu$")):
        ax.plot(x, 100 * pw[:, 0], "o-", ms=3, label=r"ES Test $Z_2$")
        ax.plot(x, 100 * pw[:, 1], "s--", ms=3, label=r"VaR$_{99}$ Count")
        ax.axhline(5, color="0.4", ls=":")
        ax.set_xlabel(lab)
        ax.set_ylabel(r"Power at 5% Size (%)")
        ax.set_ylim(0, 140)
        ax.legend(loc="upper left", fontsize=6.5)
    a3.invert_xaxis()
    a3.set_xscale("log")
    a3.set_xticks([3, 5, 10, 30])
    a3.set_xticklabels(["3", "5", "10", "30"])
    panel_label(a2, "(b)")
    panel_label(a3, "(c)")
    fig.tight_layout()
    save(fig, "fig04_es_backtest", FIG, width_in=7.2)

    i12 = int(np.argmin(np.abs(d["K"] - 1.2)))
    i3 = int(np.argmin(np.abs(d["NU"] - 3.0)))
    write_numbers(NAME, {
        "EFourCritFive": fmt(float(d["crit"][0]), 2),
        "EFourCritTiny": fmt(float(d["crit"][1]), 2),
        "EFourXc": str(int(d["xc"])),
        "EFourSizeVaR": pct(float(d["size_var"])),
        "EFourPowZTwoK": pct(float(d["pow_k"][i12, 0]), 0),
        "EFourPowVaRK": pct(float(d["pow_k"][i12, 1]), 0),
        "EFourPowZTwoNu": pct(float(d["pow_nu"][i3, 0]), 0),
        "EFourPowVaRNu": pct(float(d["pow_nu"][i3, 1]), 0),
    })


if __name__ == "__main__":
    run(compute, plot)
