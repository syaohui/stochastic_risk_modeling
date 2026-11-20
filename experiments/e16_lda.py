"""E16. Operational risk capital: the single-loss approximation and estimation uncertainty.

Loss distribution approach: N ~ Poisson(25) losses a year, lognormal
severities with mu = 10 (median about 22,000) and sigma = 1.5, 2.0, 2.5.

(a) Annual-loss VaR at levels 99% to 99.99%: single-loss approximation (SLA)
    against the exact compound distribution (FFT of the discretized severity,
    2^21 grid points), as a ratio.
(b) Capital estimation risk: the bank observes n losses from the true
    lognormal (sigma = 2.0), fits it by maximum likelihood and computes the
    99.9% VaR by the SLA. Distribution of the estimate relative to the SLA at
    the true parameters, over 5,000 repetitions, for n = 100, 250 and 1,000.
    Using the SLA on both sides isolates parameter error from the
    approximation error shown in (a).

Output: figures/fig16_lda.png, results/e16_lda.{json,tex}
"""
import numpy as np
from scipy import stats

from common import FIG, fmt, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import oprisk as op
from stochastic_risk.plotting import WIDE, panel_label, plt, save, use_style

NAME = "e16_lda"
LAM, MU = 25.0, 10.0
SIGMAS = (1.5, 2.0, 2.5)
ALPHAS = 1 - np.geomspace(1e-2, 1e-4, 9)
N_OBS = (100, 250, 1000)
REPS = 5000


def exact_var(sigma, alphas):
    ppf = lambda u: stats.lognorm.ppf(u, sigma, scale=np.exp(MU))   # noqa: E731
    top = 4.0 * op.sla_var(alphas.max(), LAM, ppf)
    n = 2 ** 21
    h = top / n
    fx = op.discretize(lambda v: stats.lognorm.cdf(v, sigma, scale=np.exp(MU)), h, n)
    fs = op.compound_poisson_fft(LAM, fx)
    return np.array([op.quantile_from_pmf(fs, h, a) for a in alphas])


def compute():
    ratio = np.zeros((len(SIGMAS), len(ALPHAS)))
    for i, s in enumerate(SIGMAS):
        ex = exact_var(s, ALPHAS)
        sla = np.array([op.sla_var(a, LAM, lambda u: stats.lognorm.ppf(u, s, scale=np.exp(MU))) for a in ALPHAS])
        ratio[i] = sla / ex
    rng = np.random.default_rng(1601)
    s_true = 2.0
    true = op.sla_var(0.999, LAM, lambda u: stats.lognorm.ppf(u, s_true, scale=np.exp(MU)))
    est = np.zeros((len(N_OBS), REPS))
    for i, n in enumerate(N_OBS):
        x = np.log(rng.lognormal(MU, s_true, (REPS, n)))
        m, sd = x.mean(axis=1), x.std(axis=1)          # lognormal MLE
        z = stats.norm.ppf(1 - 0.001 / LAM)
        est[i] = np.exp(m + sd * z) / true
    save_raw(NAME, alphas=ALPHAS, ratio=ratio, est=est, true=true)


def plot():
    use_style()
    d = load_raw(NAME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=WIDE)
    tail = 100 * (1 - d["alphas"])
    for i, (s, mk) in enumerate(zip(SIGMAS, ("o-", "s--", "^:"))):
        a1.semilogx(tail, d["ratio"][i], mk, ms=3, label=rf"Lognormal $\sigma = {s}$")
    a1.invert_xaxis()
    a1.axhline(1, color="0.4", ls=":")
    a1.set_xlabel(r"Annual Tail Probability $1 - \alpha$ (%)")
    a1.set_ylabel("SLA / Exact Annual VaR")
    a1.set_ylim(0.6, 1.25)
    a1.legend(loc="lower right", fontsize=7.5)
    panel_label(a1, "(a)")
    bins = np.geomspace(0.1, 20, 60)
    for i, n in enumerate(N_OBS):
        a2.hist(d["est"][i], bins=bins, histtype="step", lw=1.2, density=False, label=rf"$n = {n}$ Losses")
    a2.set_xscale("log")
    a2.axvline(1, color="0.3", ls=":")
    a2.set_xlabel(r"Estimated / True 99.9% Capital")
    a2.set_ylabel("Number of Repetitions")
    a2.set_ylim(0, 1.45 * a2.get_ylim()[1])
    a2.legend(loc="upper right", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "fig16_lda", FIG)
    i999 = int(np.argmin(np.abs(d["alphas"] - 0.999)))
    q = np.quantile(d["est"], [0.05, 0.95], axis=1)
    write_numbers(NAME, {
        "ESixteenSlaTwoNine": fmt(float(d["ratio"][1][i999]), 2),
        "ESixteenSlaTwoTwo": fmt(float(d["ratio"][1][0]), 2),
        "ESixteenSlaOneFive": fmt(float(d["ratio"][0][i999]), 2),
        "ESixteenSlaTwoFive": fmt(float(d["ratio"][2][i999]), 2),
        "ESixteenLoHundred": fmt(float(q[0, 0]), 2),
        "ESixteenHiHundred": fmt(float(q[1, 0]), 2),
        "ESixteenLoThousand": fmt(float(q[0, 2]), 2),
        "ESixteenHiThousand": fmt(float(q[1, 2]), 2),
        "ESixteenTrue": fmt(float(d["true"]) / 1e6, 1),
    })


if __name__ == "__main__":
    run(compute, plot)
