"""Animated GIFs of the simulations behind the slides.

a1 Monte Carlo VaR      200 GARCH-t return paths grow day by day, then the number of
                        simulated paths grows to 6,000 while the histogram of 10-day
                        returns and the VaR and ES estimates settle
a2 credit defaults      400 obligors, 20 simulated years per frame: the worst year of
                        each 100-year block with its systematic factor and defaults,
                        and the default-count histogram converging to the exact
                        binomial mixture
a3 exposure paths       Hull-White swap values week by week: paths, EE, PFE and the
                        value distribution with its exposure part
a4 operational losses   40 simulated years per frame: one year's losses (refreshed
                        every 200 years), the annual-loss histogram and the running
                        99.9% quantile estimate
a5 stress scenarios     3D cloud of risk-factor scenarios with the loss plane and the
                        most likely scenario, turning 2.25 degrees per frame

Models and parameters follow e01 (a1), e08 (a2), e12 (a3), e16 (a4) and e18 (a5).

Frames are drawn at 150 dpi (1080 pixels wide for the 7.2-inch panels) and
encoded with one shared 255-colour palette and no dithering, so lines and text
stay sharp. Rendering takes several minutes of CPU time, so it runs on a
compute server:

    python experiments/animations.py              all five GIFs
    python experiments/animations.py a2 a4        selected GIFs
    python experiments/animations.py --poster DIR last frame of each as PNG (for the figure audit)
    python experiments/animations.py --sample DIR 12 evenly spaced frames of each, as a small test GIF

Output: figures/animations/aN_*.gif
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy import stats

from common import FIG
from stochastic_risk import ccr, credit, stress
from stochastic_risk import volatility as vol
from stochastic_risk.measures import var_es_empirical
from stochastic_risk.plotting import plt, save, style_3d, use_style

OUT = FIG / "animations"
RED, BLUE, ORANGE = "#FF2C00", "#0C5DA5", "#FF9500"
SIZE = (7.2, 3.0)
DPI = 150
A5_DPI = 125     # the rotating 3D cloud changes every pixel, so it is kept smaller
HOLD_MS = 2500     # the last frame stays this long before the loop restarts


def _title(fig, text):
    fig.text(0.5, 0.975, text, ha="center", va="top", fontsize=9.5, fontweight="bold")


# ------------------------------------------------------------------ a1
def a1():
    G = vol.Garch(omega=1.44e-4 * 0.02, alpha_g=0.08, beta_g=0.90, nu=5.0)
    rng = np.random.default_rng(1301)
    N, H, SHOW, SUB = 6000, 10, 200, 6
    s2 = np.full(N, G.uncond_var)
    cum = np.zeros((N, H + 1))
    for h in range(H):
        r = np.sqrt(s2) * G.innovations(N, rng)
        cum[:, h + 1] = cum[:, h] + r
        s2 = G.omega + G.alpha_g * r * r + G.beta_g * s2
    cum *= 100
    ref = 100 * G.simulate_paths(G.uncond_var, H, 200_000, rng=1302)
    edges = np.linspace(-20, 15, 71)
    mid = 0.5 * (edges[1:] + edges[:-1])
    fine = np.linspace(-20, 15, 141)
    ref_h = np.histogram(ref, fine, density=True)[0]
    ref_mid = 0.5 * (fine[1:] + fine[:-1])
    days = np.arange(H + 1)
    taus = np.linspace(0, H, H * SUB + 1)[1:]                    # phase A: time grows
    counts = np.round(np.geomspace(SHOW, N, 110)).astype(int)    # phase B: paths grow
    plan = [("A", t) for t in taus] + [("B", n) for n in counts]
    norm = colors.TwoSlopeNorm(0, -10, 10)
    shown = cum[:SHOW]

    fig = plt.figure(figsize=SIZE)
    _title(fig, "Monte Carlo 10-Day VaR and ES: GARCH-t Paths")
    a = fig.add_axes([0.085, 0.16, 0.5, 0.72])
    b = fig.add_axes([0.7, 0.16, 0.27, 0.72])

    def update(i):
        phase, v = plan[i]
        a.cla()
        b.cla()
        tau = float(v) if phase == "A" else float(H)
        lo = min(int(np.floor(tau)), H - 1)
        wgt = tau - lo
        now = shown[:, lo] * (1 - wgt) + shown[:, lo + 1] * wgt
        segs = [np.column_stack([np.append(days[:lo + 1], tau), np.append(p[:lo + 1], q)])
                for p, q in zip(shown, now)]
        lc = LineCollection(segs, cmap="RdBu", norm=norm, lw=0.6, alpha=0.6)
        lc.set_array(shown[:, -1])
        a.add_collection(lc)
        a.set_xlim(0, H)
        a.set_ylim(-20, 15)
        a.set_xlabel("Day")
        a.set_ylabel("Cumulative Return (%)")
        b.set_ylim(-20, 15)
        b.set_xlim(0, 1.25 * ref_h.max())
        b.set_xlabel("Density (1/%)")
        if phase == "A":
            allnow = cum[:, lo] * (1 - wgt) + cum[:, lo + 1] * wgt
            h = np.histogram(allnow, edges, density=True)[0]
            b.barh(mid, h, height=np.diff(edges), color=BLUE, alpha=0.75, lw=0)
            b.set_ylabel(f"Return by Day {tau:.1f} (%)")
            a.axvline(tau, color="0.4", lw=0.8)
            a.text(0.03, 0.96, f"Day {tau:.1f} of {H}\n{SHOW} of {N:,} Paths Shown", transform=a.transAxes,
                   va="top", fontsize=8, bbox=dict(fc="white", ec="0.5", lw=0.5))
            return
        n = int(v)
        v99 = var_es_empirical(-cum[:n, -1], 0.99)[0]
        es = var_es_empirical(-cum[:n, -1], 0.975)[1]
        for ax in (a, b):
            ax.axhline(-v99, color=RED, ls="--", lw=1.3, label="99% VaR")
            ax.axhline(-es, color="0.1", ls=":", lw=1.5, label="97.5% ES")
        a.legend(loc="lower left", fontsize=7.5)
        a.text(0.03, 0.96, f"Estimate from the First {n:,} Paths\nVaR$_{{99}}$ = {v99:.2f}%   "
               f"ES$_{{97.5}}$ = {es:.2f}%", transform=a.transAxes, va="top", fontsize=8,
               bbox=dict(fc="white", ec="0.5", lw=0.5))
        h = np.histogram(cum[:n, -1], edges, density=True)[0]
        b.barh(mid, h, height=np.diff(edges), color=np.where(mid < -v99, RED, BLUE), alpha=0.75, lw=0)
        b.plot(ref_h, ref_mid, color="0.1", lw=1.0, label="200,000 Paths")
        b.set_ylabel("10-Day Return (%)")
        b.legend(loc="upper right", fontsize=7)

    return fig, update, len(plan), 20, DPI


# ------------------------------------------------------------------ a2
def a2():
    n_ob, pd_, rho = 400, 0.01, 0.12
    per, n_frames, block = 20, 200, 5          # 20 years a frame; the grid changes every 5 frames
    rng = np.random.default_rng(1303)
    Y = rng.standard_normal(per * n_frames)
    D = rng.random((per * n_frames, n_ob)) < credit.cond_pd(pd_, rho, Y)[:, None]
    cnt = D.sum(axis=1)
    pmf = credit.homogeneous_loss_pmf(n_ob, pd_, rho)
    q999 = int(np.argmax(np.cumsum(pmf) >= 0.999))
    kmax = 60
    yg = np.linspace(-4, 4, 400)
    cmap = colors.ListedColormap(["#d6e4f0", RED])
    span = per * block

    fig = plt.figure(figsize=SIZE)
    _title(fig, "Credit Portfolio: 400 Obligors, PD 1%, Asset Correlation 0.12")
    g = fig.add_axes([0.05, 0.16, 0.22, 0.66])
    f = fig.add_axes([0.35, 0.16, 0.25, 0.66])
    h = fig.add_axes([0.71, 0.16, 0.27, 0.66])

    def update(k):
        last = per * (k + 1) - 1
        b0 = max(k // block - 1, 0) * span
        b1 = min(b0 + span, last + 1)
        worst = b0 + int(np.argmax(cnt[b0:b1]))
        for ax in (g, f, h):
            ax.cla()
        g.imshow(D[worst].reshape(20, 20), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        g.set_xticks([])
        g.set_yticks([])
        g.set_xlabel(f"Worst of Years {b0 + 1:,} to {b1:,}:\n{cnt[worst]} Defaults", fontsize=8.5)
        g.set_ylabel("400 Obligors")
        y = Y[worst]
        f.plot(yg, stats.norm.pdf(yg), color="0.2", lw=1.0)
        f.fill_between(yg[yg >= y], stats.norm.pdf(yg[yg >= y]), color=RED, alpha=0.3, lw=0)
        f.axvline(y, color=RED, lw=1.6)
        f.text(0.04, 0.95, f"$Y$ = {y:+.2f}\nPD($Y$) = {100 * credit.cond_pd(pd_, rho, y):.1f}%",
               transform=f.transAxes, ha="left", va="top", fontsize=8)
        f.set_xlim(-4, 4)
        f.set_ylim(0, 0.5)
        f.set_xlabel("Systematic Factor $Y$")
        f.set_ylabel("Density")
        c = np.bincount(np.minimum(cnt[:last + 1], kmax), minlength=kmax + 1) / (last + 1)
        h.bar(np.arange(kmax + 1), np.where(c > 0, c, np.nan), width=1.0, color=BLUE, alpha=0.6, lw=0,
              label="Simulated")
        h.plot(np.arange(kmax + 1), pmf[:kmax + 1], color="0.1", lw=1.0, label="Exact")
        h.axvline(q999, color=RED, ls="--", lw=1.2, label="99.9% Quantile")
        h.set_yscale("log")
        h.set_ylim(1e-5, 1.0)
        h.set_xlim(-0.5, kmax)
        h.set_xlabel("Defaults per Year")
        h.set_ylabel("Probability")
        h.legend(loc="upper right", fontsize=7)
        h.text(0.03, 0.05, f"{last + 1:,} Years", transform=h.transAxes, fontsize=8)

    return fig, update, n_frames, 20, DPI


# ------------------------------------------------------------------ a3
def a3():
    hw = ccr.HullWhite(0.05, 0.01, 0.03)
    pay = np.arange(1.0, 6.0)
    times = np.round(np.arange(0, 261) / 52.0, 12)          # weekly; contains every payment date
    x, _ = hw.simulate(times, 20_000, rng=1304)
    K = ccr.par_rate(hw, pay, 1.0)
    V = ccr.swap_values(hw, times, x, pay, 1.0, K)
    m = ccr.exposure_metrics(times, V)
    V, ee, pfe = 100 * V, 100 * m["EE"], 100 * m["PFE"]
    edges = np.linspace(-12, 16, 85)
    mid = 0.5 * (edges[1:] + edges[:-1])
    frames = np.arange(2, 260)

    fig = plt.figure(figsize=SIZE)
    _title(fig, "Counterparty Exposure: Five-Year Payer Swap under Hull-White")
    a = fig.add_axes([0.085, 0.16, 0.5, 0.72])
    b = fig.add_axes([0.7, 0.16, 0.27, 0.72])

    def update(i):
        j = frames[i]
        a.cla()
        b.cla()
        segs = [np.column_stack([times[:j + 1], p[:j + 1]]) for p in V[:150]]
        a.add_collection(LineCollection(segs, colors="0.55", lw=0.4, alpha=0.5))
        a.plot(times[:j + 1], ee[:j + 1], color="0.05", lw=1.8, label="Expected Exposure")
        a.plot(times[:j + 1], pfe[:j + 1], color=ORANGE, lw=1.8, ls="--", label=r"PFE$_{97.5}$")
        a.axvline(times[j], color=RED, lw=0.9)
        a.axhline(0, color="0.3", lw=0.5)
        a.set_xlim(0, 5)
        a.set_ylim(-12, 16)
        a.set_xlabel(r"Time $t$ (years)")
        a.set_ylabel("Swap Value (% of Notional)")
        a.legend(loc="upper right", fontsize=7.5, ncol=2)
        dens = np.histogram(V[:, j], edges, density=True)[0]
        b.barh(mid, dens, height=np.diff(edges), color=np.where(mid > 0, ORANGE, "#9ecae1"), lw=0)
        b.axhline(0, color="0.3", lw=0.5)
        b.set_ylim(-12, 16)
        b.set_xlim(0, 0.6)
        b.set_xlabel("Density (1/%)")
        b.set_ylabel(f"Value at $t$ = {times[j]:.2f} (%)")
        b.text(0.95, 0.96, f"EE = {ee[j]:.2f}%\nPFE = {pfe[j]:.2f}%", transform=b.transAxes, ha="right",
               va="top", fontsize=8)

    return fig, update, len(frames), 24, DPI


# ------------------------------------------------------------------ a4
def a4():
    lam, mu, s = 25.0, 10.0, 2.0
    per, n_frames, block = 40, 200, 5          # 40 years a frame; the shown year changes every 5 frames
    rng = np.random.default_rng(1305)
    years = {}
    totals = np.empty(per * n_frames)
    for yidx in range(per * n_frames):
        n = rng.poisson(lam)
        size = np.exp(mu + s * rng.standard_normal(n))
        totals[yidx] = size.sum()
        if yidx % (per * block) == per * block - 1 or yidx == per - 1:
            years[yidx] = (np.sort(rng.uniform(0, 365, n)), size)
    ref = []
    for _ in range(4):
        N = rng.poisson(lam, 250_000)
        X = np.exp(mu + s * rng.standard_normal(N.sum()))
        ref.append(np.bincount(np.repeat(np.arange(len(N)), N), weights=X, minlength=len(N)))
    q_ref = np.log10(np.quantile(np.concatenate(ref), 0.999))
    edges = np.linspace(5, 9.5, 91)
    mid = 0.5 * (edges[1:] + edges[:-1])

    fig = plt.figure(figsize=SIZE)
    _title(fig, "Operational Losses: 25 Events a Year, Lognormal Severities")
    a = fig.add_axes([0.085, 0.16, 0.45, 0.72])
    b = fig.add_axes([0.64, 0.16, 0.33, 0.72])

    def update(k):
        a.cla()
        b.cla()
        n = per * (k + 1)
        key = max(y for y in years if y < n)
        t_ev, size = years[key]
        big = np.argmax(size)
        a.vlines(t_ev, 1e2, size, color=BLUE, lw=1.0)
        a.plot(t_ev, size, "o", color=BLUE, ms=3)
        a.plot(t_ev[big], size[big], "o", color=RED, ms=6)
        a.set_yscale("log")
        a.set_xlim(0, 365)
        a.set_ylim(1e2, 1e9)
        a.set_xlabel("Day of Year")
        a.set_ylabel("Loss (currency)")
        a.text(0.03, 0.96, f"Year {key + 1:,}: {len(size)} Losses\nLargest = {100 * size.max() / size.sum():.0f}% of Total",
               transform=a.transAxes, va="top", fontsize=8, bbox=dict(fc="white", ec="0.5", lw=0.5))
        lt = np.log10(totals[:n])
        dens = np.histogram(lt, edges)[0] / (n * np.diff(edges))
        b.bar(mid, np.where(dens > 0, dens, np.nan), width=np.diff(edges), color=np.where(mid > q_ref, RED, BLUE),
              alpha=0.7, lw=0)
        q_hat = np.quantile(lt, 0.999)
        b.axvline(q_ref, color="0.1", ls="--", lw=1.2, label="99.9%, 10$^6$ Years")
        b.axvline(q_hat, color=RED, lw=1.4, label=f"99.9%, {n:,} Years")
        b.set_yscale("log")
        b.set_ylim(1e-4, 10)
        b.set_xlim(5, 9.5)
        b.set_xlabel(r"Log$_{10}$ Annual Loss")
        b.set_ylabel("Density")
        b.legend(loc="upper right", fontsize=7)

    return fig, update, n_frames, 20, DPI


# ------------------------------------------------------------------ a5
def _plane_in_cube(nvec, c, half):
    """Polygon where the plane nvec . x = c cuts the cube [-half, half]^3, vertices in order."""
    pts = []
    corners = np.array([[i, j, k] for i in (-half, half) for j in (-half, half) for k in (-half, half)], float)
    for a_ in range(8):
        for b_ in range(a_ + 1, 8):
            pa, pb = corners[a_], corners[b_]
            if np.count_nonzero(pa != pb) != 1:
                continue
            fa, fb = nvec @ pa - c, nvec @ pb - c
            if fa * fb < 0:
                pts.append(pa + fa / (fa - fb) * (pb - pa))
    pts = np.array(pts)
    ctr = pts.mean(axis=0)
    u = pts[0] - ctr
    u /= np.linalg.norm(u)
    v = np.cross(nvec / np.linalg.norm(nvec), u)
    ang = np.arctan2((pts - ctr) @ v, (pts - ctr) @ u)
    return pts[np.argsort(ang)]


def a5():
    w = np.array([1.0, 0.6, -0.8])
    C = np.array([[1.0, 0.3, -0.5], [0.3, 1.0, -0.2], [-0.5, -0.2, 1.0]])
    rng = np.random.default_rng(1306)
    x = rng.multivariate_normal(np.zeros(3), C, 2000)
    loss = -(x @ w)
    xs, d, _ = stress.reverse_stress_point(w, C, 4.0)
    plane = _plane_in_cube(-w, 4.0, 4.0)       # scenarios with loss -w'x = 4, clipped to the cube
    n = 160
    fig = plt.figure(figsize=(6.0, 4.6))
    _title(fig, "Reverse Stress Test: Scenarios, Loss Plane, Most Likely Scenario")
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.92], projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=loss, cmap="RdBu_r", norm=colors.TwoSlopeNorm(0, -5, 5), s=6,
               alpha=0.85, lw=0)
    ax.add_collection3d(Poly3DCollection([plane], facecolor=ORANGE, edgecolor="none", alpha=0.35))
    ax.plot([0, xs[0]], [0, xs[1]], [0, xs[2]], color="0.1", lw=1.5)
    ax.scatter([xs[0]], [xs[1]], [xs[2]], color="#FFD400", s=110, marker="*", edgecolor="0.1", lw=0.6,
               label=f"Most Likely Loss of 4 ($d$ = {d:.2f})")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    ax.set_xlabel("Equity (sd)", labelpad=0)
    ax.set_ylabel("Rate (sd)", labelpad=0)
    ax.set_zlabel("Spread (sd)", labelpad=0)
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.02, 0.98))
    ax.set_box_aspect(None, zoom=0.85)
    ax.grid(False)

    def update(i):
        style_3d(ax, elev=24 + 10 * np.sin(2 * np.pi * i / n), azim=-50 + 360 * i / n)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.96, 0.96, 0.96, 1.0))

    return fig, update, n, 20, A5_DPI


BUILDERS = {"a1": (a1, "a1_monte_carlo_var"), "a2": (a2, "a2_credit_defaults"), "a3": (a3, "a3_exposure_paths"),
            "a4": (a4, "a4_operational_losses"), "a5": (a5, "a5_reverse_stress_3d")}


# ------------------------------------------------------------------ encoder
def _grab(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def write_gif(fig, update, idx, fps, path, dpi=DPI):
    """Render frames `idx` and write one GIF with a shared palette and no dithering."""
    fig.set_dpi(dpi)
    frames = []
    for i in idx:
        update(i)
        frames.append(_grab(fig))
    pick = np.unique(np.linspace(0, len(frames) - 1, min(10, len(frames))).astype(int))
    w, h = frames[0].size
    strip = Image.new("RGB", (w, h * len(pick)), "white")
    for k, j in enumerate(pick):
        strip.paste(frames[j], (0, k * h))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    dur = [int(round(1000 / fps))] * len(out)
    dur[-1] = HOLD_MS
    out[0].save(path, save_all=True, append_images=out[1:], duration=dur, loop=0, disposal=1)
    return w, h


def main(argv):
    use_style()
    mode, target = None, None
    for flag in ("--poster", "--sample"):
        if flag in argv:
            mode, target = flag, Path(argv[argv.index(flag) + 1])
            argv = [a for a in argv if a not in (flag, str(target))]
    names = [a for a in argv if a in BUILDERS] or list(BUILDERS)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        build, stem = BUILDERS[name]
        fig, update, n, fps, dpi = build()
        if mode == "--poster":
            update(n - 1)
            save(fig, stem + "_poster", target)
        elif mode == "--sample":
            target.mkdir(parents=True, exist_ok=True)
            idx = np.unique(np.linspace(0, n - 1, 12).astype(int))
            path = target / f"{stem}_sample.gif"
            w, h = write_gif(fig, update, idx, fps, path, dpi)
            print(f"{stem}: {n} frames at {fps} fps, {w}x{h} px; sample {path.stat().st_size / 1e3 / len(idx):.0f} kB/frame",
                  flush=True)
        else:
            path = OUT / f"{stem}.gif"
            w, h = write_gif(fig, update, range(n), fps, path, dpi)
            print(f"{stem}.gif  {n} frames  {fps} fps  {w}x{h}  {path.stat().st_size / 1e6:.1f} MB", flush=True)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
