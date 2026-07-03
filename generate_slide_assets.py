"""
generate_slide_assets.py
=========================
Generates presentation-quality figures for the slide deck and saves them
to the current directory. Run once:  python3 generate_slide_assets.py
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import orbital_mechanics as om

plt.rcParams["mathtext.fontset"] = "cm"

BG       = "#05060a"
PANEL    = "#0d1018"
INNER_C  = "#4fc3f7"
OUTER_C  = "#ff7043"
TRANSFER = "#ffd54f"
TEXT_C   = "#e0e6f0"
MUTED    = "#8893a8"
BURN_C   = "#ff1744"
GREEN    = "#69f0ae"

FIGSIZE = (10.0, 5.625)   # 16:9
DPI = 160


def _starfield(ax, n=240, seed=3):
    rng = np.random.default_rng(seed)
    ax.scatter(rng.uniform(0, 1, n), rng.uniform(0, 1, n),
               s=rng.uniform(0.2, 2.0, n) ** 2, c="white",
               alpha=rng.uniform(0.2, 0.8, n), transform=ax.transAxes,
               zorder=0, linewidths=0)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_concept():
    """Annotated Hohmann transfer concept diagram."""
    r1, r2 = 1.0, 2.3
    a = (r1 + r2) / 2
    e = (r2 - r1) / (r2 + r1)
    p = a * (1 - e ** 2)

    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=BG)
    _clean(ax); _starfield(ax)
    lim = 2.9
    ax.set_xlim(-lim, lim * 1.15); ax.set_ylim(-lim * 0.9, lim * 0.9)
    ax.set_aspect("equal")

    # central body
    for k, gr in enumerate(np.linspace(2.6, 1.0, 6)):
        ax.add_patch(Circle((0, 0), 0.16 * gr, color="#0a84ff",
                             alpha=0.05 + 0.02 * k, zorder=1, lw=0))
    ax.add_patch(Circle((0, 0), 0.16, color="#1565c0", zorder=3))
    ax.text(0, -0.34, "Central body", color=TEXT_C, ha="center", va="top",
            fontsize=11, weight="bold")

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r1 * np.cos(th), r1 * np.sin(th), "--", color=INNER_C, lw=1.6)
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), "--", color=OUTER_C, lw=1.6)

    nu = np.linspace(0, 2 * np.pi, 400)
    rr = p / (1 + e * np.cos(nu))
    ax.plot(rr * np.cos(nu), rr * np.sin(nu), ":", color=TRANSFER, lw=1.0, alpha=0.4)
    nu2 = np.linspace(0, np.pi, 200)
    rr2 = p / (1 + e * np.cos(nu2))
    ax.plot(rr2 * np.cos(nu2), rr2 * np.sin(nu2), "-", color=TRANSFER, lw=2.6, zorder=4)

    # burn markers
    ax.add_patch(FancyArrowPatch((r1, 0), (r1, 0.7), color=BURN_C,
                                 arrowstyle="-|>", mutation_scale=18, lw=2, zorder=6))
    ax.add_patch(FancyArrowPatch((-r2, 0), (-r2, -0.7), color=BURN_C,
                                 arrowstyle="-|>", mutation_scale=18, lw=2, zorder=6))
    ax.plot([r1], [0], "o", color="white", ms=9, zorder=7, mec=BURN_C, mew=1.5)
    ax.plot([-r2], [0], "o", color="white", ms=9, zorder=7, mec=BURN_C, mew=1.5)

    ax.text(r1 + 0.08, 0.85, r"BURN 1  ($\Delta v_1$)", color=BURN_C, fontsize=11,
            weight="bold", ha="left")
    ax.text(-r2, -0.95, r"BURN 2  ($\Delta v_2$)", color=BURN_C, fontsize=11,
            weight="bold", ha="center", va="top")
    ax.text(0.62, 1.05, "Start orbit", color=INNER_C, fontsize=11, rotation=0)
    ax.text(1.55, 1.75, "Target orbit", color=OUTER_C, fontsize=11)
    ax.text(0.15, 2.05, "Transfer ellipse", color=TRANSFER, fontsize=11, weight="bold")

    ax.set_title("The Hohmann Transfer: two burns, one coast",
                 color=TEXT_C, fontsize=15, weight="bold", pad=2, y=0.96)
    fig.savefig("slide_concept.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved slide_concept.png")


def fig_deltav_bar():
    """Horizontal bar chart of total delta-v per scenario."""
    keys = ["geo-graveyard", "leo-geo", "leo-lunar", "earth-mars", "earth-jupiter"]
    labels, vals = [], []
    for k in keys:
        sc = om.PRESETS[k]
        res = om.hohmann_transfer(sc.mu, sc.r1, sc.r2)
        labels.append(sc.name.split(" (")[0])
        vals.append(res.dv_total)

    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax = fig.add_axes([0.30, 0.12, 0.64, 0.74], facecolor=BG)
    _clean(ax)
    colors = [MUTED, INNER_C, GREEN, OUTER_C, BURN_C]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.6, log=True)
    for yi, v in zip(y, vals):
        ax.text(v * 1.12, yi, f"{v:,.0f} m/s", color=TEXT_C, va="center",
                fontsize=11, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT_C, fontsize=11)
    ax.set_xlim(5, 40000)
    ax.tick_params(colors=MUTED)
    ax.set_title("Total $\\Delta v$ budget by mission (log scale)",
                 color=TEXT_C, fontsize=15, weight="bold", pad=14)
    fig.savefig("slide_deltav.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved slide_deltav.png")


def fig_tof_bar():
    """Horizontal bar chart of time of flight per scenario (log scale)."""
    keys = ["leo-geo", "geo-graveyard", "leo-lunar", "earth-mars", "earth-jupiter"]
    labels, hours = [], []
    for k in keys:
        sc = om.PRESETS[k]
        res = om.hohmann_transfer(sc.mu, sc.r1, sc.r2)
        labels.append(sc.name.split(" (")[0])
        hours.append(res.tof / 3600.0)

    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax = fig.add_axes([0.30, 0.12, 0.64, 0.74], facecolor=BG)
    _clean(ax)
    colors = [INNER_C, MUTED, GREEN, OUTER_C, BURN_C]
    y = np.arange(len(labels))
    ax.barh(y, hours, color=colors, height=0.6, log=True)
    for yi, hh in zip(y, hours):
        txt = om.format_duration(hh * 3600.0)
        ax.text(hh * 1.15, yi, txt, color=TEXT_C, va="center",
                fontsize=11, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, color=TEXT_C, fontsize=11)
    ax.set_xlim(1, 100000)
    ax.tick_params(colors=MUTED)
    ax.set_title("Time of flight by mission (log scale, hours)",
                 color=TEXT_C, fontsize=15, weight="bold", pad=14)
    fig.savefig("slide_tof.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved slide_tof.png")


def fig_bielliptic():
    """Hohmann vs bi-elliptic total dv (normalised) vs orbit ratio."""
    ratios = np.linspace(2, 30, 280)
    mu, r1 = 1.0, 1.0
    vc1 = om.circular_velocity(mu, r1)
    hoh, bie = [], []
    for R in ratios:
        r2 = R * r1
        h = om.hohmann_transfer(mu, r1, r2)
        b = om.bielliptic_transfer(mu, r1, r2, r_b=1e6 * r1)  # r_b -> infinity
        hoh.append(h.dv_total / vc1)
        bie.append(b.dv_total / vc1)

    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax = fig.add_axes([0.10, 0.13, 0.86, 0.74], facecolor=BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.plot(ratios, hoh, color=TRANSFER, lw=2.6, label="Hohmann (2 burns)")
    ax.plot(ratios, bie, color=GREEN, lw=2.6, ls="--",
            label=r"Bi-elliptic ($r_b\to\infty$, 3 burns)")
    ax.axvline(11.94, color=BURN_C, lw=1.5, ls=":")
    ax.text(11.94, 0.30, "  ratio = 11.94\n  (cross-over)", color=BURN_C,
            fontsize=10, va="bottom")
    ax.set_xlabel(r"orbit ratio  $r_2 / r_1$", color=TEXT_C, fontsize=12)
    ax.set_ylabel(r"total $\Delta v\ /\ v_{c1}$", color=TEXT_C, fontsize=12)
    ax.set_title("When Hohmann stops being optimal",
                 color=TEXT_C, fontsize=15, weight="bold", pad=12)
    leg = ax.legend(facecolor=PANEL, edgecolor=MUTED, fontsize=11, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TEXT_C)
    fig.savefig("slide_bielliptic.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved slide_bielliptic.png")


def fig_phase_angle():
    """Launch-window phase-angle diagram for Earth -> Mars."""
    sc = om.EARTH_TO_MARS
    lw = om.launch_window(sc.mu, sc.r1, sc.r2)
    r1, r2 = 1.0, 1.524
    a = (r1 + r2) / 2
    e = (r2 - r1) / (r2 + r1)
    p = a * (1 - e ** 2)
    phase = lw.phase_angle_rad

    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=BG)
    _clean(ax); _starfield(ax)
    lim = 1.95
    ax.set_xlim(-lim, lim * 1.2); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # Sun
    for k, gr in enumerate(np.linspace(3.0, 1.0, 6)):
        ax.add_patch(Circle((0, 0), 0.10 * gr, color="#ffb300",
                             alpha=0.05 + 0.03 * k, zorder=1, lw=0))
    ax.add_patch(Circle((0, 0), 0.10, color="#ffb300", zorder=3))
    ax.text(0, -0.22, "Sun", color=TEXT_C, ha="center", va="top", fontsize=11, weight="bold")

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r1 * np.cos(th), r1 * np.sin(th), "--", color=INNER_C, lw=1.4)
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), "--", color=OUTER_C, lw=1.4)

    nu2 = np.linspace(0, np.pi, 200)
    rr2 = p / (1 + e * np.cos(nu2))
    ax.plot(rr2 * np.cos(nu2), rr2 * np.sin(nu2), "-", color=TRANSFER, lw=2.4, zorder=4)

    # Earth at departure (angle 0)
    ax.add_patch(Circle((r1, 0), 0.07, color="#1565c0", zorder=6))
    ax.text(r1 + 0.05, -0.16, "Earth\n(departure)", color=INNER_C, fontsize=9.5, ha="left", va="top")
    # Mars at departure (angle phase)
    mx, my = r2 * math.cos(phase), r2 * math.sin(phase)
    ax.add_patch(Circle((mx, my), 0.07, color="#e64a19", zorder=6))
    ax.text(mx + 0.05, my + 0.12, "Mars\n(at departure)", color=OUTER_C, fontsize=9.5)
    # Mars at arrival (angle 180)
    ax.add_patch(Circle((-r2, 0), 0.07, color="#e64a19", alpha=0.35, zorder=5))
    ax.text(-r2, -0.14, "Mars\n(at arrival)", color=OUTER_C, fontsize=9, ha="center", va="top", alpha=0.8)

    # phase arc
    arc_r = 0.55
    aa = np.linspace(0, phase, 60)
    ax.plot(arc_r * np.cos(aa), arc_r * np.sin(aa), color=TEXT_C, lw=1.4)
    ax.plot([0, r1], [0, 0], "-", color=INNER_C, lw=0.9, alpha=0.5)
    ax.plot([0, mx], [0, my], "-", color=OUTER_C, lw=0.9, alpha=0.5)
    ax.text(arc_r * 1.25 * math.cos(phase / 2), arc_r * 1.25 * math.sin(phase / 2),
            rf"$\phi = {lw.phase_angle_deg:+.1f}^\circ$", color=TEXT_C, fontsize=13, weight="bold")

    ax.set_title("Launch window: where Mars must be at departure",
                 color=TEXT_C, fontsize=15, weight="bold", pad=2, y=0.96)
    fig.savefig("slide_phase_angle.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved slide_phase_angle.png")


def fig_visviva():
    """Velocity-vs-radius profile illustrating the two burns (LEO -> GEO)."""
    sc = om.LEO_TO_GEO
    res = om.hohmann_transfer(sc.mu, sc.r1, sc.r2)
    r1, r2 = sc.r1, sc.r2
    a_t = res.a_transfer
    rr = np.linspace(r1 * 0.9, r2 * 1.05, 300)
    v_circ = np.sqrt(sc.mu / rr)
    v_tr = np.sqrt(sc.mu * (2 / rr - 1 / a_t))

    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    ax = fig.add_axes([0.11, 0.14, 0.85, 0.72], facecolor=BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    km = 1e3
    ax.plot(rr / km, v_circ, color=INNER_C, lw=2.4, label="circular-orbit speed")
    ax.plot(rr / km, v_tr, color=TRANSFER, lw=2.4, ls="--", label="transfer-ellipse speed")

    ax.plot([r1 / km], [res.v1_circular], "o", color=INNER_C, ms=9)
    ax.plot([r1 / km], [res.v_transfer_peri], "o", color=TRANSFER, ms=9)
    ax.plot([r2 / km], [res.v2_circular], "o", color=INNER_C, ms=9)
    ax.plot([r2 / km], [res.v_transfer_apo], "o", color=TRANSFER, ms=9)
    ax.annotate("", xy=(r1 / km, res.v_transfer_peri), xytext=(r1 / km, res.v1_circular),
                arrowprops=dict(arrowstyle="-|>", color=BURN_C, lw=2))
    ax.annotate("", xy=(r2 / km, res.v2_circular), xytext=(r2 / km, res.v_transfer_apo),
                arrowprops=dict(arrowstyle="-|>", color=BURN_C, lw=2))
    ax.text(r1 / km * 1.02, (res.v1_circular + res.v_transfer_peri) / 2,
            f" $\\Delta v_1$ = {res.dv1:,.0f} m/s", color=BURN_C, fontsize=11, weight="bold")
    ax.text(r2 / km * 0.62, (res.v2_circular + res.v_transfer_apo) / 2,
            f"$\\Delta v_2$ = {res.dv2:,.0f} m/s", color=BURN_C, fontsize=11, weight="bold")

    ax.set_xlabel("orbital radius (km)", color=TEXT_C, fontsize=12)
    ax.set_ylabel("speed (m/s)", color=TEXT_C, fontsize=12)
    ax.set_title("Vis-viva: speed up to climb, slow down to circularise",
                 color=TEXT_C, fontsize=15, weight="bold", pad=12)
    leg = ax.legend(facecolor=PANEL, edgecolor=MUTED, fontsize=11)
    for t in leg.get_texts():
        t.set_color(TEXT_C)
    fig.savefig("slide_visviva.png", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved slide_visviva.png")


if __name__ == "__main__":
    fig_concept()
    fig_deltav_bar()
    fig_tof_bar()
    fig_bielliptic()
    fig_phase_angle()
    fig_visviva()
    print("All concept figures generated.")
