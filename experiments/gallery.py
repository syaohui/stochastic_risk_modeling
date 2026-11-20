"""Gallery: one showcase figure per topic that shows the randomness itself.

The experiments e01 to e19 measure estimators and tests. These figures show the
simulated objects behind them: paths, conditional densities, scenario clouds
and sampling distributions. Models and parameters follow the matching
experiments unless a line below says otherwise.

g01 market risk       GARCH-t returns with the true 99% VaR band; conditional loss densities in 3D
g02 backtesting       exception-count laws and Basel zones; power surface of the exact count test
g03 model risk        sampling law of the plug-in VaR; worst-case exponential tilts
g04 credit            Vasicek tail-probability surface; Gaussian and t copula samples (PD 5%, rho 0.3)
g05 PD validation     sampled ROC curves; AUC sampling densities
g06 counterparty      Hull-White swap value paths; value densities over time with the exposure part
g07 Monte Carlo       nested simulation tree (horizon 3 months for display); random and Sobol points
g08 operational risk  one simulated year of losses; annual loss densities against severity spread
g09 stress testing    3D scenario cloud with the loss plane; calm and stressed ellipses

Units: returns and losses in percent of value, swap values in percent of
notional, operational losses in currency units, risk factors in standard
deviations.

Output: figures/gallery/gNN_*.png, results/gallery.{json,tex}
"""
import numpy as np
from matplotlib import cm, colormaps, colors
from matplotlib.collections import LineCollection, PolyCollection
from scipy import stats
from scipy.stats import qmc

from common import FIG, fmt, fmt_int, load_raw, pct, run, save_raw, write_numbers
from stochastic_risk import ccr, credit, measures, model_risk, stress
from stochastic_risk import volatility as vol
from stochastic_risk.plotting import panel_label, plt, save, style_3d, use_style

NAME = "gallery"
OUT = FIG / "gallery"
TALL = (7.2, 3.1)
RED, BLUE, ORANGE = "#FF2C00", "#0C5DA5", "#FF9500"

# g01: the GARCH-t model of e01 (unconditional volatility 1.2% a day)
GARCH = vol.Garch(omega=1.44e-4 * 0.02, alpha_g=0.08, beta_g=0.90, nu=5.0)
# g04: one-factor credit model; the copula panels use a larger PD and correlation for visibility
PD_V, RHO_COP, PD_COP, NU_COP = 0.01, 0.30, 0.05, 4.0
# g05: binormal scores with AUC 0.76, 50 non-defaulters per default (as e11)
MU_AUC = np.sqrt(2.0) * stats.norm.ppf(0.76)
# g06: the swap of e12
HW = ccr.HullWhite(0.05, 0.01, 0.03)
PAY = np.arange(1.0, 6.0)
TIMES = np.round(np.arange(0, 61) / 12.0, 12)
# g08: loss distribution approach of e16
LAM, MU_SEV = 25.0, 10.0
# g09: calm and stressed correlations of e18
W_STRESS = np.array([1.0, 0.6, -0.8])
C_CALM = np.array([[1.0, 0.3, -0.5], [0.3, 1.0, -0.2], [-0.5, -0.2, 1.0]])
C_STRESS = np.array([[1.0, 0.48, -0.80], [0.48, 1.0, -0.32], [-0.80, -0.32, 1.0]])
LOSS_STRESS = 4.0


def fast_auc(sd, sn):
    """Mann-Whitney AUC by ranks (continuous scores, no ties)."""
    nd, nn = len(sd), len(sn)
    ranks = stats.rankdata(np.concatenate([sd, sn]))
    return (ranks[:nd].sum() - nd * (nd + 1) / 2.0) / (nd * nn)


def roc_on_grid(sd, sn, fpr):
    """Empirical TPR at the given false positive rates."""
    thr = np.quantile(sn, 1.0 - fpr, method="higher")
    return (sd[None, :] > thr[:, None]).mean(axis=1)


def compute():
    rng = np.random.default_rng(9100)
    out = {}

    # g01: the most volatile of 50 seeded one-year paths (largest max/min volatility ratio)
    best = max(range(50), key=lambda s: (lambda sg: sg.max() / sg.min())(GARCH.simulate(250, rng=s)[1]))
    r, sig = GARCH.simulate(250, rng=best)
    out.update(g1_seed=best, g1_r=r, g1_sig=sig)

    # g02: exact one-sided binomial test of p0 = 1% with size at most 5%
    p_grid = np.linspace(0.005, 0.04, 36)
    n_grid = np.arange(100, 2001, 50)
    power = np.empty((len(p_grid), len(n_grid)))
    xc = np.empty(len(n_grid), int)
    for j, n in enumerate(n_grid):
        x = np.arange(n + 1)
        xc[j] = x[np.argmax(stats.binom.sf(x - 1, n, 0.01) <= 0.05)]
        power[:, j] = stats.binom.sf(xc[j] - 1, n, p_grid)
    out.update(g2_p=p_grid, g2_n=n_grid, g2_power=power, g2_xc=xc)

    # g03: sampling law of plug-in normal 99% VaR over true VaR
    ns = np.array([25, 50, 100, 250, 1000])
    z = stats.norm.ppf(0.99)
    edges = np.linspace(0.3, 1.7, 281)
    dens = np.empty((len(ns), len(edges) - 1))
    for i, n in enumerate(ns):
        lbar = rng.standard_normal(400_000) / np.sqrt(n)
        s = np.sqrt(rng.chisquare(n - 1, 400_000) / (n - 1))
        dens[i] = np.histogram((lbar + s * z) / z, edges, density=True)[0]
    exceed = np.array([model_risk.plugin_normal_exceedance(int(n), 0.99) for n in ns])
    out.update(g3_n=ns, g3_edges=edges, g3_dens=dens, g3_exceed=exceed)

    # g04: log10 P(default rate > x) of the infinitely granular portfolio, and copula samples
    xg = np.linspace(0.001, 0.15, 150)
    rg = np.linspace(0.03, 0.30, 55)
    X, R = np.meshgrid(xg, rg)
    arg = (np.sqrt(1 - R) * stats.norm.ppf(X) - stats.norm.ppf(PD_V)) / np.sqrt(R)
    logtail = stats.norm.logsf(arg) / np.log(10.0)
    n_cop = 20_000
    y = rng.standard_normal(n_cop)
    eps = rng.standard_normal((n_cop, 2))
    lat = np.sqrt(RHO_COP) * y[:, None] + np.sqrt(1 - RHO_COP) * eps
    w = rng.chisquare(NU_COP, n_cop) / NU_COP
    u_g = stats.norm.cdf(lat)
    u_t = stats.t.cdf(lat / np.sqrt(w)[:, None], NU_COP)
    out.update(g4_x=xg, g4_rho=rg, g4_logtail=logtail, g4_ug=u_g, g4_ut=u_t)

    # g05: ROC curves and AUC sampling densities
    fpr = np.linspace(0, 1, 201)
    rocs = {}
    for nd in (10, 100):
        rocs[nd] = np.array([roc_on_grid(rng.normal(MU_AUC, 1, nd), rng.standard_normal(50 * nd), fpr)
                             for _ in range(150)])
    nds = np.array([10, 30, 100])
    aucs = np.array([[fast_auc(rng.normal(MU_AUC, 1, nd), rng.standard_normal(50 * nd))
                      for _ in range(4000)] for nd in nds])
    out.update(g5_fpr=fpr, g5_roc10=rocs[10], g5_roc100=rocs[100], g5_nd=nds, g5_auc=aucs)

    # g06: Hull-White swap values
    x, _ = HW.simulate(TIMES, 20_000, rng=1206)
    K = ccr.par_rate(HW, PAY, 1.0)
    V = ccr.swap_values(HW, TIMES, x, PAY, 1.0, K)
    m = ccr.exposure_metrics(TIMES, V)
    vedges = np.linspace(-12, 12, 121)
    vdens = np.array([np.histogram(100 * V[:, j], vedges, density=True)[0] for j in range(len(TIMES))])
    out.update(g6_t=TIMES, g6_paths=100 * V[:120], g6_EE=100 * m["EE"], g6_PFE=100 * m["PFE"],
               g6_edges=vedges, g6_dens=vdens, g6_pos=(V > 0).mean(axis=0))

    # g07: nested tree (outer horizon 3 months for display) and point sets
    s0, rr, sg, tau, T = 100.0, 0.02, 0.25, 0.25, 1.0
    t_out = np.linspace(0, tau, 33)
    dW = rng.standard_normal((6, len(t_out) - 1)) * np.sqrt(np.diff(t_out))
    outer = s0 * np.exp(np.concatenate([np.zeros((6, 1)),
                                        np.cumsum((rr - 0.5 * sg**2) * np.diff(t_out) + sg * dW, 1)], 1))
    t_in = np.linspace(tau, T, 49)
    inner = np.empty((6, 30, len(t_in)))
    for k in range(6):
        dWi = rng.standard_normal((30, len(t_in) - 1)) * np.sqrt(np.diff(t_in))
        inc = np.cumsum((rr - 0.5 * sg**2) * np.diff(t_in) + sg * dWi, 1)
        inner[k] = outer[k, -1] * np.exp(np.concatenate([np.zeros((30, 1)), inc], 1))
    pts_mc = rng.random((512, 2))
    pts_qmc = qmc.Sobol(2, scramble=True, seed=1207).random(512)
    disc = np.array([qmc.discrepancy(pts_mc, method="L2-star"), qmc.discrepancy(pts_qmc, method="L2-star")])
    out.update(g7_tout=t_out, g7_outer=outer, g7_tin=t_in, g7_inner=inner, g7_mc=pts_mc, g7_qmc=pts_qmc,
               g7_disc=disc)

    # g08: one year of losses, 40 cumulative years, annual loss densities by severity spread
    n_years = 40
    days = np.arange(366)
    cum = np.zeros((n_years, len(days)))
    first = None
    for yidx in range(n_years):
        n_ev = rng.poisson(LAM)
        t_ev = np.sort(rng.uniform(0, 365, n_ev))
        size = np.exp(MU_SEV + 2.0 * rng.standard_normal(n_ev))
        cum[yidx] = np.array([size[t_ev <= dd].sum() for dd in days])
        if first is None:
            first = (t_ev, size)
    sig_grid = np.linspace(1.0, 2.6, 9)
    ledges = np.linspace(4.5, 10.0, 111)
    ldens = np.empty((len(sig_grid), len(ledges) - 1))
    lq = np.empty(len(sig_grid))
    for i, s in enumerate(sig_grid):
        logS = []
        for _ in range(4):                       # 4 x 250,000 simulated years
            Y = 250_000
            N = rng.poisson(LAM, Y)
            Xs = np.exp(MU_SEV + s * rng.standard_normal(N.sum()))
            S = np.bincount(np.repeat(np.arange(Y), N), weights=Xs, minlength=Y)
            logS.append(np.log10(np.maximum(S, 1.0)))
        logS = np.concatenate(logS)
        ldens[i] = np.histogram(logS, ledges)[0] / (len(logS) * np.diff(ledges))
        lq[i] = np.quantile(logS, 0.999)
    out.update(g8_tev=first[0], g8_size=first[1], g8_days=days, g8_cum=cum, g8_sig=sig_grid,
               g8_edges=ledges, g8_dens=ldens, g8_q999=lq)

    # g09: scenario cloud and most likely scenarios
    xs = rng.multivariate_normal(np.zeros(3), C_CALM, 3000)
    xc_calm, d_calm, _ = stress.reverse_stress_point(W_STRESS, C_CALM, LOSS_STRESS)
    xc_str, d_str, _ = stress.reverse_stress_point(W_STRESS, C_STRESS, LOSS_STRESS)
    # regime-switching one-year paths of the e19 portfolio (ten factors, equal weights)
    d10 = 10
    cov_c = 0.01**2 * stress.corr_matrix(d10, 0.2)
    cov_s = 0.025**2 * stress.corr_matrix(d10, 0.6)
    n_paths, n_days = 300, 250
    cum = np.empty((n_paths, n_days + 1))
    share = np.empty(n_paths)
    for k in range(n_paths):
        start = int(rng.random() < 0.004 / (0.004 + 0.02))      # stationary start
        rr_, ss_ = stress.simulate_regimes(n_days, 0.004, 0.02, cov_c, cov_s, rng=rng, start=start)
        cum[k] = np.concatenate([[0.0], np.cumsum(rr_ @ np.full(d10, 0.1))])
        share[k] = ss_.mean()
    out.update(g9_x=xs, g9_xcalm=xc_calm, g9_xstress=xc_str, g9_d=np.array([d_calm, d_str]),
               g9_cum=100 * cum, g9_share=share)

    save_raw(NAME, **out)


# ------------------------------------------------------------------ plotting helpers
def _surface(ax, X, Y, Z, cmap="viridis", vmin=None, vmax=None):
    norm = colors.Normalize(np.nanmin(Z) if vmin is None else vmin, np.nanmax(Z) if vmax is None else vmax)
    return ax.plot_surface(X, Y, Z, facecolors=colormaps[cmap](norm(Z)), rstride=1, cstride=1,
                           linewidth=0, antialiased=False, shade=False)


def _fences(ax, positions, grid, dens, cut, body_colors, tail_color=RED, alpha=0.88, base=0.0):
    """Density slices standing at `positions` along the x axis (grid along y, density along z).

    The part of slice k at or beyond cut[k] is filled with tail_color; the two
    parts do not overlap, so the 3D depth sorting cannot hide one behind the other.
    `base` is the floor of the z axis (for log densities).
    """
    for k, x0 in enumerate(positions):
        y, z = grid, np.maximum(dens[k], base)
        c = float(np.clip(cut[k], y[0], y[-1]))
        zc = float(np.interp(c, y, z))
        lo, hi = y < c, y > c
        core = [(y[0], base), *zip(y[lo], z[lo]), (c, zc), (c, base)]
        tail = [(c, base), (c, zc), *zip(y[hi], z[hi]), (y[-1], base)]
        pc = PolyCollection([core, tail], facecolors=[body_colors[k], tail_color], edgecolors="0.15",
                            linewidths=0.35, alpha=alpha)
        ax.add_collection3d(pc, zs=x0, zdir="x")


def _tag3d(ax, text, x=0.02, y=0.95):
    ax.text2D(x, y, text, transform=ax.transAxes, fontweight="bold", fontsize=9)


def g01(d):
    r, sig = d["g1_r"], d["g1_sig"]
    s5 = measures.std_t_scale(5.0)
    var99 = 100 * sig * s5 * stats.t.ppf(0.99, 5)
    t = np.arange(1, len(r) + 1)
    hit = -100 * r > var99
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    a1.fill_between(t, -var99, var99, color=BLUE, alpha=0.15, lw=0, label="True 99% VaR Band")
    a1.plot(t, 100 * r, color="0.25", lw=0.7, label="Simulated Daily Return")
    a1.plot(t[hit], 100 * r[hit], "o", color=RED, ms=4, label="VaR Exception")
    a1.set_xlim(0, 251)
    lim = 1.25 * np.abs(100 * r).max()
    a1.set_ylim(-lim, 1.55 * lim)
    a1.set_xlabel(r"Trading Day $t$")
    a1.set_ylabel(r"Return $r_t$ (%)")
    a1.legend(loc="upper left", ncol=2, fontsize=7)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    lg = np.linspace(-4, 10, 141)
    pos = t[4::10]
    sc = 100 * sig[4::10] * s5
    dens = np.log10(stats.t.pdf(lg[None, :] / sc[:, None], 5) / sc[:, None])
    cmap = colormaps["viridis"]
    norm = colors.Normalize(sig.min(), sig.max())
    _fences(a2, pos, lg, dens, var99[4::10], [cmap(norm(s)) for s in sig[4::10]], base=-4.0)
    style_3d(a2, elev=24, azim=-62)
    a2.set_xlim(0, 250)
    a2.set_ylim(-4, 10)
    a2.set_zlim(-4, 0)
    a2.set_box_aspect(None, zoom=0.82)
    a2.set_xlabel(r"Trading Day $t$", labelpad=0)
    a2.set_ylabel(r"Loss $\ell$ (%)", labelpad=0)
    a2.set_zlabel(r"Log$_{10}$ Density", labelpad=-1)
    a2.text2D(0.98, 0.88, "Red: Loss Beyond 99% VaR", transform=a2.transAxes, ha="right", fontsize=7.5)
    _tag3d(a2, "(b)")
    save(fig, "g01_market_risk", OUT)
    return {"GalOneExceptions": str(int(hit.sum())), "GalOneVolRatio": fmt(sig.max() / sig.min(), 1)}


def g02(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    k = np.arange(0, 16)
    for lo, hi, c in ((-0.5, 4.5, "#00B945"), (4.5, 9.5, ORANGE), (9.5, 15.5, RED)):
        a1.axvspan(lo, hi, color=c, alpha=0.12, lw=0)
    a1.bar(k - 0.2, stats.binom.pmf(k, 250, 0.01), 0.4, color=BLUE, label="Correct Model, $p$ = 1%")
    a1.bar(k + 0.2, stats.binom.pmf(k, 250, 0.02), 0.4, color="#845B97", label="Wrong Model, $p$ = 2%")
    for x0, txt in ((2.0, "Green"), (7.0, "Amber"), (12.5, "Red")):
        a1.text(x0, 0.335, txt, ha="center", fontsize=8)
    a1.set_xlim(-0.5, 15.5)
    a1.set_xticks([0, 5, 10, 15])
    a1.set_ylim(0, 0.37)
    a1.set_xlabel(r"Exceptions in 250 Days $x$")
    a1.set_ylabel("Probability")
    a1.legend(loc="center right", fontsize=7)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    P, N = np.meshgrid(100 * d["g2_p"], d["g2_n"], indexing="ij")
    Z = d["g2_power"]
    a2.computed_zorder = False
    srf = _surface(a2, P, N, Z, vmin=0, vmax=1)
    srf.set_zorder(1)
    cs = a2.contour(P, N, Z + 0.01, levels=[0.81], colors=RED, linewidths=1.8)
    cs.set_zorder(5)
    style_3d(a2, elev=27, azim=-122)
    a2.set_box_aspect(None, zoom=0.82)
    a2.set_xlabel(r"True Rate $p$ (%)", labelpad=0)
    a2.set_ylabel(r"Length $n$ (days)", labelpad=1)
    a2.set_zlabel("Rejection Probability", labelpad=-1)
    a2.set_zlim(0, 1)
    a2.text2D(0.98, 0.88, "Red: 80% Power", transform=a2.transAxes, ha="right", fontsize=7.5)
    _tag3d(a2, "(b)")
    save(fig, "g02_backtesting", OUT)
    j250 = int(np.argmin(np.abs(d["g2_n"] - 250)))
    i2 = int(np.argmin(np.abs(d["g2_p"] - 0.02)))
    pw = d["g2_power"][i2]
    n80 = d["g2_n"][np.argmax(pw >= 0.8)]
    return {"GalTwoPowTwoFifty": pct(float(pw[j250]), 0), "GalTwoNEighty": fmt_int(n80)}


def g03(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    mid = 0.5 * (d["g3_edges"][1:] + d["g3_edges"][:-1])
    cmap = colormaps["viridis"]
    ns = d["g3_n"]
    for j in range(len(ns) - 1, -1, -1):
        y0 = j * 0.85
        h = d["g3_dens"][j] / d["g3_dens"][j].max()
        c = cmap(0.15 + 0.7 * j / (len(ns) - 1))
        a1.fill_between(mid, y0, y0 + h, color=c, alpha=0.85, lw=0)
        a1.plot(mid, y0 + h, color="0.15", lw=0.6)
        a1.text(1.6, y0 + 0.1, f"$n$ = {ns[j]}\nP(Exceed) = {100 * d['g3_exceed'][j]:.2f}%", ha="left",
                va="bottom", fontsize=6.8, linespacing=1.0)
    a1.axvline(1.0, color=RED, lw=1.0, ls="--")
    a1.set_xlim(0.3, 2.25)
    a1.set_xticks([0.5, 1.0, 1.5, 2.0])
    a1.set_ylim(-0.1, 0.85 * len(ns) + 0.35)
    a1.set_yticks([])
    a1.set_xlabel(r"Estimated / True 99% VaR")
    a1.set_ylabel("Sampling Density (Scaled)")
    panel_label(a1, "(a)")

    x = np.linspace(-4, 6, 400)
    etas = np.linspace(0, 0.5, 11)
    norm = colors.Normalize(0, 0.5)
    es0 = measures.normal_var_es(0.975)[1]
    for eta in etas:
        th = np.sqrt(2 * eta)
        c = cmap(norm(eta))
        a2.plot(x, stats.norm.pdf(x, th), color=c, lw=1.0)
        a2.plot(es0 + th, stats.norm.pdf(es0 + th, th), "o", color=c, ms=3.5, mec="0.2", mew=0.4)
    a2.set_xlim(-4, 6)
    a2.set_ylim(0, 0.5)
    a2.set_xlabel(r"Loss $L$ (standard deviations)")
    a2.set_ylabel("Density")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=a2, pad=0.02)
    cb.set_label(r"Relative Entropy Budget $\eta$")
    cb.ax.tick_params(direction="in", labelsize=8)
    a2.text(5.8, 0.45, r"Dots: ES$_{97.5}$ $+\sqrt{2\eta}$", ha="right", fontsize=7.5)
    panel_label(a2, "(b)")
    fig.tight_layout()
    save(fig, "g03_model_risk", OUT)
    return {}


def g04(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([-0.01, 0.0, 0.5, 1.0], projection="3d")
    Xm, Rm = np.meshgrid(100 * d["g4_x"], d["g4_rho"])
    Z = np.maximum(d["g4_logtail"], -8.0)
    a1.computed_zorder = False
    srf = _surface(a1, Xm, Rm, Z, vmin=-8, vmax=0)
    srf.set_zorder(1)
    cs = a1.contour(Xm, Rm, Z + 0.03, levels=[-2.97], colors=RED, linewidths=1.8, linestyles="solid")
    cs.set_zorder(5)
    style_3d(a1, elev=26, azim=-130)
    a1.set_box_aspect(None, zoom=0.82)
    a1.set_xlabel("Default Rate $x$ (%)", labelpad=0)
    a1.set_ylabel(r"Correlation $\rho$", labelpad=0)
    a1.set_zlabel(r"Log$_{10}$ P(Rate $> x$)", labelpad=-1)
    a1.set_zlim(-8, 0)
    a1.text2D(0.97, 0.88, "Red: 99.9% Quantile", transform=a1.transAxes, ha="right", fontsize=7.5)
    _tag3d(a1, "(a)", x=0.04)

    res = {}
    zoom = 0.3
    for k, (key, title, x0) in enumerate((("g4_ug", "Gaussian", 0.555), ("g4_ut", r"Student $t_4$", 0.8))):
        ax = fig.add_axes([x0, 0.17, 0.185, 0.68])
        u = d[key]
        joint = (u[:, 0] < PD_COP) & (u[:, 1] < PD_COP)
        show = (u[:, 0] < zoom) & (u[:, 1] < zoom)
        ax.add_patch(plt.Rectangle((0, 0), PD_COP, PD_COP, color=RED, alpha=0.15, lw=0))
        ax.plot(u[show & ~joint, 0], u[show & ~joint, 1], ".", color=BLUE, ms=1.6, alpha=0.6)
        ax.plot(u[joint, 0], u[joint, 1], ".", color=RED, ms=2.6)
        ax.set_xlim(0, zoom)
        ax.set_ylim(0, zoom)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_xlabel(r"Score $U_1$")
        ax.set_ylabel(r"Score $U_2$", labelpad=1)
        ax.text(0.97, 0.97, f"{title}\n{joint.sum()} Joint Defaults", transform=ax.transAxes, ha="right",
                va="top", fontsize=7, bbox=dict(fc="white", ec="0.5", lw=0.5, pad=1.5))
        panel_label(ax, "(b)" if k == 0 else "(c)")
        res["GalFourJoint" + ("Gauss" if k == 0 else "T")] = fmt_int(joint.sum())
    save(fig, "g04_credit_portfolio", OUT)
    return res


def g05(d):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    fpr = d["g5_fpr"]
    true_tpr = stats.norm.sf(stats.norm.isf(np.clip(fpr, 1e-9, 1 - 1e-9)) - MU_AUC)
    for ax, key, nd, lab in ((axs[0], "g5_roc10", 10, "(a)"), (axs[1], "g5_roc100", 100, "(b)")):
        segs = [np.column_stack([fpr, r]) for r in d[key]]
        ax.add_collection(LineCollection(segs, colors=BLUE, alpha=0.08, lw=0.7))
        ax.plot(fpr, true_tpr, color=RED, lw=1.5, label="True ROC")
        ax.plot([0, 1], [0, 1], ":", color="0.4", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.text(0.97, 0.05, f"{nd} Defaults", ha="right", fontsize=8)
        ax.legend(loc="center right", fontsize=7, bbox_to_anchor=(1.0, 0.32))
        panel_label(ax, lab)
    ax = axs[2]
    cmap = colormaps["viridis"]
    res = {}
    for i, nd in enumerate(d["g5_nd"]):
        a = d["g5_auc"][i]
        h, e = np.histogram(a, np.linspace(0.4, 1.0, 61), density=True)
        mid = 0.5 * (e[1:] + e[:-1])
        ax.fill_between(mid, h, color=cmap(0.15 + 0.35 * i), alpha=0.55, lw=0, label=f"{nd} Defaults")
        lo, hi = np.quantile(a, [0.05, 0.95])
        res[f"GalFiveLo{('Ten', 'Thirty', 'Hundred')[i]}"] = fmt(lo, 2)
        res[f"GalFiveHi{('Ten', 'Thirty', 'Hundred')[i]}"] = fmt(hi, 2)
    ax.axvline(0.76, color=RED, lw=1.0, ls="--")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 1.35 * ax.get_ylim()[1])
    ax.set_xlabel("Estimated AUC")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=7)
    panel_label(ax, "(c)")
    fig.tight_layout()
    save(fig, "g05_pd_validation", OUT)
    return res


def g06(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.075, 0.17, 0.38, 0.72])
    t = d["g6_t"]
    paths = d["g6_paths"]
    segs = [np.column_stack([t, p]) for p in paths]
    lc = LineCollection(segs, cmap="RdBu_r", norm=colors.TwoSlopeNorm(0, -8, 8), lw=0.5, alpha=0.6)
    lc.set_array(paths[:, 24])
    a1.add_collection(lc)
    a1.axhline(0, color="0.3", lw=0.6)
    a1.plot(t, d["g6_EE"], color="0.1", lw=1.6, label="Expected Exposure")
    a1.plot(t, d["g6_PFE"], color=ORANGE, lw=1.6, ls="--", label=r"PFE$_{97.5}$")
    a1.set_xlim(0, 5)
    a1.set_ylim(-13, 19)
    a1.set_xlabel(r"Time $t$ (years)")
    a1.set_ylabel(r"Swap Value $V(t)$ (% of Notional)")
    a1.legend(loc="upper right", fontsize=7, ncol=2)
    panel_label(a1, "(a)")

    a2 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    mid = 0.5 * (d["g6_edges"][1:] + d["g6_edges"][:-1])
    idx = np.arange(3, 60, 3)
    pos = t[idx]
    dens = d["g6_dens"][idx]
    _fences(a2, pos, mid, dens, np.zeros(len(idx)), ["#9ecae1"] * len(idx), tail_color=ORANGE)
    style_3d(a2, elev=24, azim=-60)
    a2.set_xlim(0, 5)
    a2.set_ylim(-12, 12)
    a2.set_zlim(0, 1.02 * dens.max())
    a2.set_box_aspect(None, zoom=0.82)
    a2.set_xlabel(r"Time $t$ (years)", labelpad=0)
    a2.set_ylabel(r"Value $V$ (%)", labelpad=0)
    a2.set_zlabel("Density (1/%)", labelpad=-1)
    a2.text2D(0.98, 0.88, "Orange: Exposure $V^+$", transform=a2.transAxes, ha="right", fontsize=7.5)
    _tag3d(a2, "(b)")
    save(fig, "g06_counterparty", OUT)
    return {}


def g07(d):
    fig = plt.figure(figsize=(7.2, 2.8))
    a1 = fig.add_axes([0.07, 0.17, 0.42, 0.72])
    cmap = colormaps["viridis"]
    for k in range(6):
        c = cmap(0.1 + 0.15 * k)
        a1.plot(d["g7_tout"], d["g7_outer"][k], color=c, lw=1.6)
        for p in d["g7_inner"][k]:
            a1.plot(d["g7_tin"], p, color=c, lw=0.4, alpha=0.45)
        a1.plot(d["g7_tout"][-1], d["g7_outer"][k, -1], "o", color=c, ms=4, mec="0.15", mew=0.5)
    a1.axvline(0.25, color=RED, lw=0.9, ls="--")
    a1.text(0.12, 0.92, "Outer\nScenarios", transform=a1.transAxes, ha="center", va="top", fontsize=7.5)
    a1.text(0.62, 0.92, "Inner Repricing Paths", transform=a1.transAxes, ha="center", va="top", fontsize=7.5)
    a1.set_xlim(0, 1)
    lo, hi = d["g7_inner"].min(), d["g7_inner"].max()
    a1.set_ylim(0.95 * lo, 1.12 * hi)
    a1.set_xlabel(r"Time $t$ (years)")
    a1.set_ylabel(r"Underlying Price $S_t$ (currency)")
    panel_label(a1, "(a)")
    for k, (key, title, x0) in enumerate((("g7_mc", "Random", 0.565), ("g7_qmc", "Sobol", 0.815))):
        ax = fig.add_axes([x0, 0.17, 0.165, 0.72])
        p = d[key]
        ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), cmap="viridis", s=3, lw=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel(r"Coordinate $u_1$")
        ax.set_ylabel(r"Coordinate $u_2$", labelpad=1)
        ax.text(0.5, 1.03, f"{title}, 512 Points", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        ax.text(-0.02, 1.03, "(b)" if k == 0 else "(c)", transform=ax.transAxes, ha="right", va="bottom",
                fontweight="bold")
    save(fig, "g07_monte_carlo", OUT)
    disc = d["g7_disc"]
    return {"GalSevenDiscRatio": fmt(np.sqrt(disc[0] / disc[1]), 1)}


def g08(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([0.085, 0.62, 0.37, 0.3])
    t_ev, size = d["g8_tev"], d["g8_size"]
    a1.vlines(t_ev, 1e2, size, color=BLUE, lw=1.0)
    a1.plot(t_ev, size, "o", color=BLUE, ms=3)
    big = np.argmax(size)
    a1.plot(t_ev[big], size[big], "o", color=RED, ms=5)
    a1.set_yscale("log")
    a1.set_xlim(0, 365)
    a1.set_ylim(1e2, 30 * size.max())
    a1.set_xlabel("Day of Year", labelpad=1)
    a1.set_ylabel("Loss (currency)")
    a1.text(0.98, 0.86, "One Simulated Year", transform=a1.transAxes, ha="right", fontsize=7.5)
    panel_label(a1, "(a)")
    a2 = fig.add_axes([0.085, 0.15, 0.37, 0.3])
    cum = d["g8_cum"] / 1e6
    for c in cum[1:]:
        a2.step(d["g8_days"], c, where="post", color="0.6", lw=0.5, alpha=0.6)
    a2.step(d["g8_days"], cum[0], where="post", color=RED, lw=1.4)
    a2.set_xlim(0, 365)
    a2.set_ylim(0, 1.1 * cum[:, -1].max())
    a2.set_xlabel("Day of Year")
    a2.set_ylabel("Total (millions)")
    a3 = fig.add_axes([0.47, 0.0, 0.53, 1.0], projection="3d")
    mid = 0.5 * (d["g8_edges"][1:] + d["g8_edges"][:-1])
    cmap = colormaps["viridis"]
    sg = d["g8_sig"]
    with np.errstate(divide="ignore"):
        ld = np.log10(d["g8_dens"])
    _fences(a3, sg, mid, ld, d["g8_q999"], [cmap(0.15 + 0.7 * i / (len(sg) - 1)) for i in range(len(sg))],
            base=-4.0)
    style_3d(a3, elev=24, azim=-60)
    a3.set_xlim(sg[0] - 0.05, sg[-1] + 0.05)
    a3.set_ylim(4.5, 10.0)
    a3.set_zlim(-4, 1)
    a3.set_box_aspect(None, zoom=0.82)
    a3.set_xlabel(r"Severity Spread $\sigma$", labelpad=0)
    a3.set_ylabel(r"Log$_{10}$ Annual Loss", labelpad=0)
    a3.set_zlabel(r"Log$_{10}$ Density", labelpad=-1)
    a3.text2D(0.98, 0.88, "Red: Beyond 99.9% Quantile", transform=a3.transAxes, ha="right", fontsize=7.5)
    _tag3d(a3, "(b)")
    save(fig, "g08_operational_risk", OUT)
    share = size.max() / size.sum()
    return {"GalEightEvents": str(len(size)), "GalEightShare": pct(float(share), 0)}


def g09(d):
    fig = plt.figure(figsize=TALL)
    a1 = fig.add_axes([-0.02, 0.0, 0.56, 1.0], projection="3d")
    x = d["g9_x"]
    loss = -(x @ W_STRESS)
    norm = colors.TwoSlopeNorm(0, -5, 5)
    a1.scatter(x[:, 0], x[:, 1], x[:, 2], c=loss, cmap="RdBu_r", norm=norm, s=2, alpha=0.55, lw=0)
    g = np.linspace(-4, 4, 30)
    E, Rr = np.meshgrid(g, g)
    Sp = (LOSS_STRESS + W_STRESS[0] * E + W_STRESS[1] * Rr) / -W_STRESS[2]
    Sp = np.where(np.abs(Sp) <= 4, Sp, np.nan)
    a1.plot_surface(E, Rr, Sp, color=ORANGE, alpha=0.25, lw=0)
    p = d["g9_xcalm"]
    a1.plot([0, p[0]], [0, p[1]], [0, p[2]], color="0.1", lw=1.4)
    a1.scatter([p[0]], [p[1]], [p[2]], color="#FFD400", s=70, marker="*", edgecolor="0.1", lw=0.6,
               label=f"Most Likely Scenario, Loss {LOSS_STRESS:.0f}")
    style_3d(a1, elev=18, azim=-50)
    a1.set_xlim(-4, 4)
    a1.set_ylim(-4, 4)
    a1.set_zlim(-4, 4)
    a1.set_box_aspect(None, zoom=0.85)
    a1.set_xlabel("Equity (sd)", labelpad=0)
    a1.set_ylabel("Rate (sd)", labelpad=0)
    a1.set_zlabel("Spread (sd)", labelpad=-1)
    a1.legend(loc="upper right", fontsize=7, bbox_to_anchor=(0.98, 0.97))
    _tag3d(a1, "(a)", x=0.04)

    a2 = fig.add_axes([0.6, 0.17, 0.3, 0.72])
    cum, share = d["g9_cum"], d["g9_share"]
    order = np.argsort(share)
    days = np.arange(cum.shape[1])
    pnorm = colors.Normalize(0, 0.6)
    lc = LineCollection([np.column_stack([days, cum[k]]) for k in order], cmap="plasma", norm=pnorm,
                        lw=0.6, alpha=0.75)
    lc.set_array(share[order])
    a2.add_collection(lc)
    a2.set_xlim(0, days[-1])
    lim = np.abs(cum).max()
    a2.set_ylim(-1.05 * lim, 1.05 * lim)
    a2.set_xlabel("Trading Day")
    a2.set_ylabel("Cumulative Return (%)")
    cb = fig.colorbar(lc, ax=a2, pad=0.03, fraction=0.08)
    cb.set_label("Share of Days in Stress")
    cb.ax.tick_params(direction="in", labelsize=8)
    panel_label(a2, "(b)")
    save(fig, "g09_stress_testing", OUT)
    dd = d["g9_d"]
    return {"GalNineDCalm": fmt(float(dd[0]), 2), "GalNineDStress": fmt(float(dd[1]), 2)}


def plot():
    use_style()
    d = load_raw(NAME)
    numbers = {}
    for f in (g01, g02, g03, g04, g05, g06, g07, g08, g09):
        numbers.update(f(d))
        plt.close("all")
    write_numbers(NAME, numbers)


if __name__ == "__main__":
    run(compute, plot)
