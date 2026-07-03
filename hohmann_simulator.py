"""
hohmann_simulator.py
====================

Animated Hohmann transfer simulator built on orbital_mechanics.py.

Usage:
    python3 hohmann_simulator.py --scenario earth-mars
    python3 hohmann_simulator.py --scenario leo-lunar
    python3 hohmann_simulator.py --scenario earth-jupiter
    python3 hohmann_simulator.py --scenario geo-graveyard
    python3 hohmann_simulator.py --scenario leo-geo --save transfer.gif
    python3 hohmann_simulator.py --scenario earth-mars --save transfer.mp4
    python3 hohmann_simulator.py --scenario earth-mars --save-frame poster.png
    python3 hohmann_simulator.py --list-scenarios

GIF vs MP4:
    GIFs are portable but large. For a ~10x smaller file at the same quality,
    save as .mp4 (requires ffmpeg: https://ffmpeg.org/download.html).
    Install on macOS:  brew install ffmpeg
    Install on Linux:  sudo apt install ffmpeg
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge

import orbital_mechanics as om


BG = "#05060a"
PANEL = "#0d1018"
INNER_C = "#4fc3f7"
OUTER_C = "#ff7043"
TRANSFER_C = "#ffd54f"
CRAFT_C = "#ffffff"
BURN_C = "#ff1744"
TEXT_C = "#e0e6f0"
MUTED = "#8893a8"

# Visual properties for departure / arrival body discs, keyed by scenario name.
# Each entry: r1_color, r1_label, r1_radius_frac,
#             r2_color, r2_label, r2_radius_frac
# radius_frac is a fraction of r_outer (the larger orbit radius).
_BODY_STYLES: dict[str, dict] = {
    "LEO -> GEO (Earth)": {
        "r1_color": "#4fc3f7",   # pale blue  – LEO / spacecraft
        "r1_label": "LEO",
        "r1_radius_frac": 0.018,
        "r2_color": "#ffd54f",   # gold       – GEO belt
        "r2_label": "GEO",
        "r2_radius_frac": 0.018,
    },
    "Earth -> Mars (Sun)": {
        "r1_color": "#1565c0",   # deep blue  – Earth
        "r1_label": "Earth",
        "r1_radius_frac": 0.022,
        "r2_color": "#e64a19",   # burnt orange – Mars
        "r2_label": "Mars",
        "r2_radius_frac": 0.018,
    },
    "LEO -> Lunar (Earth)": {
        "r1_color": "#4fc3f7",   # pale blue  – LEO
        "r1_label": "LEO",
        "r1_radius_frac": 0.018,
        "r2_color": "#b0bec5",   # cool grey  – Moon
        "r2_label": "Moon",
        "r2_radius_frac": 0.022,
    },
    "Earth -> Jupiter (Sun)": {
        "r1_color": "#1565c0",   # deep blue  – Earth
        "r1_label": "Earth",
        "r1_radius_frac": 0.016,
        "r2_color": "#ff8f00",   # amber      – Jupiter
        "r2_label": "Jupiter",
        "r2_radius_frac": 0.030,  # Jupiter is big — give it a larger disc
    },
    "GEO -> Graveyard (Earth)": {
        "r1_color": "#ffd54f",   # gold       – GEO
        "r1_label": "GEO",
        "r1_radius_frac": 0.018,
        "r2_color": "#78909c",   # blue-grey  – Graveyard band
        "r2_label": "Graveyard",
        "r2_radius_frac": 0.018,
    },
}


def _starfield(ax, n=320, seed=7):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, n)
    ys = rng.uniform(0, 1, n)
    sizes = rng.uniform(0.2, 2.2, n) ** 2
    alphas = rng.uniform(0.25, 0.9, n)
    ax.scatter(xs, ys, s=sizes, c="white", alpha=alphas,
               transform=ax.transAxes, zorder=0, linewidths=0)


def _ellipse_points(a, e, num=400):
    theta = np.linspace(0, 2 * np.pi, num)
    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(theta))
    return r * np.cos(theta), r * np.sin(theta)


def build_figure(scenario: om.Scenario, result: om.HohmannResult):
    # Use Computer Modern math fonts so equations render with a clean,
    # publication-style (LaTeX-like) look. mathtext needs no LaTeX install.
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.default"] = "regular"

    r1, r2 = result.r1, result.r2
    a_t = result.a_transfer
    e_t = abs(r2 - r1) / (r1 + r2)
    r_outer = max(r1, r2)

    fig = plt.figure(figsize=(13, 7.3), facecolor=BG)
    ax = fig.add_axes([0.02, 0.04, 0.62, 0.92], facecolor=BG)
    panel = fig.add_axes([0.66, 0.04, 0.32, 0.92], facecolor=PANEL)

    for a in (ax, panel):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    _starfield(ax)

    lim = 1.25 * r_outer
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # Central body glow + disc
    _CENTRAL_BODY_COLORS = {
        "Earth": "#1565c0",
        "Sun":   "#ffb300",
    }
    body_r = 0.045 * r_outer
    cb_color = _CENTRAL_BODY_COLORS.get(scenario.central_body, "#ffb300")
    for k, gr in enumerate(np.linspace(2.6, 1.0, 6)):
        ax.add_patch(Circle((0, 0), body_r * gr, color=cb_color,
                             alpha=0.05 + 0.02 * k, zorder=1, linewidth=0))
    ax.add_patch(Circle((0, 0), body_r, color=cb_color, zorder=3))
    ax.text(0, -body_r * 1.9, scenario.central_body, color=TEXT_C,
            ha="center", va="top", fontsize=11, weight="bold")

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r1 * np.cos(th), r1 * np.sin(th), "--", color=INNER_C,
            lw=1.4, alpha=0.9, zorder=2)
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), "--", color=OUTER_C,
            lw=1.4, alpha=0.9, zorder=2)

    ex, ey = _ellipse_points(a_t, e_t)
    if r2 < r1:
        ex = -ex
    ax.plot(ex, ey, ":", color=TRANSFER_C, lw=1.0, alpha=0.35, zorder=2)

    # --- departure / arrival planet discs (keyed by scenario name) ---
    bstyle = _BODY_STYLES.get(scenario.name, {})
    disc_r1 = bstyle.get("r1_radius_frac", 0.02) * r_outer
    disc_r2 = bstyle.get("r2_radius_frac", 0.02) * r_outer
    r1_col = bstyle.get("r1_color", INNER_C)
    r2_col = bstyle.get("r2_color", OUTER_C)
    r1_planet_label = bstyle.get("r1_label", scenario.r1_label)
    r2_planet_label = bstyle.get("r2_label", scenario.r2_label)

    # departure planet sits at (r1, 0) — burn-1 position
    for k, gr in enumerate(np.linspace(2.0, 1.0, 4)):
        ax.add_patch(Circle((r1, 0), disc_r1 * gr,
                             color=r1_col, alpha=0.08 + 0.04 * k,
                             zorder=3, linewidth=0))
    ax.add_patch(Circle((r1, 0), disc_r1, color=r1_col, zorder=4, linewidth=0))
    ax.text(r1, disc_r1 * 2.4, r1_planet_label, color=r1_col,
            fontsize=8.5, ha="center", va="bottom")

    # arrival planet sits at (-r2, 0) — burn-2 position
    for k, gr in enumerate(np.linspace(2.0, 1.0, 4)):
        ax.add_patch(Circle((-r2, 0), disc_r2 * gr,
                             color=r2_col, alpha=0.08 + 0.04 * k,
                             zorder=3, linewidth=0))
    ax.add_patch(Circle((-r2, 0), disc_r2, color=r2_col, zorder=4, linewidth=0))
    ax.text(-r2, disc_r2 * 2.4, r2_planet_label, color=r2_col,
            fontsize=8.5, ha="center", va="bottom")

    # --- launch-window / phase-angle analysis --------------------------------
    lw = om.launch_window(scenario.mu, r1, r2)
    optimal = om.hohmann_is_optimal(r1, r2)
    ratio = max(r1, r2) / min(r1, r2)

    # For heliocentric (planet-to-planet) transfers, visualise where the
    # target must be at DEPARTURE so both arrive at the apsis simultaneously.
    if scenario.central_body == "Sun":
        phase = lw.phase_angle_rad
        tx, ty = r2 * math.cos(phase), r2 * math.sin(phase)
        # radial guides: departure planet (angle 0) and target-at-departure
        ax.plot([0, r1], [0, 0], "-", color=r1_col, lw=0.8, alpha=0.4, zorder=2)
        ax.plot([0, tx], [0, ty], "-", color=r2_col, lw=0.8, alpha=0.4, zorder=2)
        # faint "ghost" of the target at departure
        ax.add_patch(Circle((tx, ty), disc_r2, color=r2_col, alpha=0.30,
                             zorder=3, linewidth=0))
        ax.text(tx, ty + disc_r2 * 2.4, "target @ departure", color=r2_col,
                fontsize=7.5, ha="center", va="bottom", alpha=0.85)
        # phase-angle arc near the centre
        arc_r = 0.30 * r_outer
        aa = np.linspace(0, phase, 60)
        ax.plot(arc_r * np.cos(aa), arc_r * np.sin(aa), color=TEXT_C,
                lw=1.0, alpha=0.6, zorder=2)
        mid = phase / 2.0
        ax.text(arc_r * 1.15 * math.cos(mid), arc_r * 1.15 * math.sin(mid),
                rf"$\phi = {lw.phase_angle_deg:+.1f}^\circ$", color=TEXT_C,
                fontsize=9.5, ha="center", va="center")

    # compact analysis box (lower-left, applies to every scenario)
    analysis = "\n".join([
        f"phase angle  phi = {lw.phase_angle_deg:+6.1f} deg",
        f"synodic period   = {om.format_duration(lw.synodic_period)}",
        f"orbit ratio r2/r1= {ratio:6.2f}",
        f"Hohmann optimal  : {'yes' if optimal else 'no (bi-elliptic)'}",
    ])
    ax.text(0.015, 0.02, analysis, transform=ax.transAxes, color=MUTED,
            fontsize=8.5, family="monospace", va="bottom", ha="left", zorder=8,
            bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=MUTED, alpha=0.55))

    ax.set_title(f"Hohmann Transfer  |  {scenario.name}",
                 color=TEXT_C, fontsize=14, weight="bold", pad=10)

    (craft,) = ax.plot([], [], "o", color=CRAFT_C, ms=9, zorder=6,
                       markeredgecolor=BURN_C, markeredgewidth=1.2)
    (trail,) = ax.plot([], [], "-", color=CRAFT_C, lw=1.3, alpha=0.6, zorder=5)
    (burn_flash,) = ax.plot([], [], "*", color=BURN_C, ms=26, zorder=7)
    phase_txt = ax.text(0.5, 0.97, "", transform=ax.transAxes, color=TEXT_C,
                        ha="center", va="top", fontsize=12, weight="bold")

    _fill_panel(panel, scenario, result)

    handles = dict(fig=fig, ax=ax, craft=craft, trail=trail,
                   burn_flash=burn_flash, phase_txt=phase_txt,
                   r1=r1, r2=r2, a_t=a_t, e_t=e_t)
    return handles


def _fill_panel(panel, scenario: om.Scenario, res: om.HohmannResult):
    panel.set_xlim(0, 1); panel.set_ylim(0, 1)

    def header(y, txt, color):
        panel.text(0.06, y, txt, color=color, fontsize=12.5, weight="bold",
                   transform=panel.transAxes, va="top", family="monospace")

    def row(y, label, value, color=TEXT_C, size=11, weight="normal"):
        """Two-column readout: math label on the left, value on the right."""
        panel.text(0.07, y, label, color=color, fontsize=size, weight=weight,
                   transform=panel.transAxes, va="top", ha="left")
        panel.text(0.95, y, value, color=color, fontsize=size, weight=weight,
                   transform=panel.transAxes, va="top", ha="right",
                   family="monospace")

    # ---- Mission parameters -------------------------------------------------
    header(0.985, "MISSION PARAMETERS", INNER_C)
    row(0.945, "Central body", scenario.central_body, size=10.5)
    row(0.917, r"$r_1$  (start)",  scenario.r1_label, INNER_C, size=10.5)
    row(0.889, r"$r_2$  (target)", scenario.r2_label, OUTER_C, size=10.5)

    # ---- Velocities ---------------------------------------------------------
    header(0.840, "VELOCITIES", OUTER_C)
    row(0.802, r"$v_{c}(r_1)$",      f"{res.v1_circular:,.0f} m/s")
    row(0.774, r"$v_{c}(r_2)$",      f"{res.v2_circular:,.0f} m/s")
    row(0.746, r"$v_{t}(r_1)$",      f"{res.v_transfer_peri:,.0f} m/s")
    row(0.718, r"$v_{t}(r_2)$",      f"{res.v_transfer_apo:,.0f} m/s")

    # ---- Delta-v budget -----------------------------------------------------
    header(0.668, r"$\Delta v$  BUDGET", TRANSFER_C)
    row(0.630, r"$\Delta v_1$", f"{res.dv1:,.1f} m/s")
    row(0.602, r"$\Delta v_2$", f"{res.dv2:,.1f} m/s")
    row(0.570, r"$\Delta v_{\mathrm{tot}}$", f"{res.dv_total:,.1f} m/s",
        "#ffffff", 11.5, "bold")
    row(0.535, r"$t_f$  (ToF)", om.format_duration(res.tof),
        "#ffffff", 11, "bold")

    # ---- Governing equations (proper mathtext, two-column layout) -----------
    header(0.500, "GOVERNING EQUATIONS", MUTED)

    def eqn_at(x, y, txt, color=MUTED, size=11):
        panel.text(x, y, txt, color=color, fontsize=size,
                   transform=panel.transAxes, va="top", ha="left")

    # Left column: the tall radical equations (need generous vertical spacing).
    left_eqs = [
        r"$v_{c} = \sqrt{\dfrac{\mu}{r}}$",
        r"$v = \sqrt{\mu\!\left(\dfrac{2}{r} - \dfrac{1}{a}\right)}$",
        r"$T = 2\pi\sqrt{\dfrac{a^{3}}{\mu}}$",
    ]
    # Right column: compact single-line relations (subscript notation keeps
    # them narrow enough to sit beside the left column without clipping).
    right_eqs = [
        r"$a_t = \dfrac{r_1 + r_2}{2}$",
        r"$\Delta v_1 = \left| v_{t,1} - v_{c,1} \right|$",
        r"$\Delta v_2 = \left| v_{c,2} - v_{t,2} \right|$",
        r"$t_f = \dfrac{1}{2}\, T_{\mathrm{tr}}$",
    ]

    for i, eq in enumerate(left_eqs):
        eqn_at(0.08, 0.455 - i * 0.098, eq, MUTED, 11)
    for i, eq in enumerate(right_eqs):
        eqn_at(0.54, 0.455 - i * 0.078, eq, MUTED, 10)

    # ---- Phases -------------------------------------------------------------
    header(0.150, "PHASES", MUTED)
    row(0.116, "1) coast on start orbit", "", INNER_C, 9.5)
    row(0.089, r"2) burn $\Delta v_1 \rightarrow$ transfer", "", BURN_C, 9.5)
    row(0.062, "3) coast half-ellipse", "", TRANSFER_C, 9.5)
    row(0.035, r"4) burn $\Delta v_2 \rightarrow$ circularise", "", OUTER_C, 9.5)


PH_START = 0.12
PH_BURN1 = 0.06
PH_TRANSFER = 0.50
PH_BURN2 = 0.06


def _position(frac, h):
    r1, r2, a_t, e_t = h["r1"], h["r2"], h["a_t"], h["e_t"]

    c0 = PH_START
    c1 = c0 + PH_BURN1
    c2 = c1 + PH_TRANSFER
    c3 = c2 + PH_BURN2

    if frac < c0:
        ang = -math.pi / 2 + (frac / c0) * (math.pi / 2)
        return r1 * math.cos(ang), r1 * math.sin(ang), "Coasting on start orbit", False
    if frac < c1:
        return r1, 0.0, r"BURN 1   $\Delta v_1 \rightarrow$  inject into transfer ellipse", True
    if frac < c2:
        s = (frac - c1) / PH_TRANSFER
        nu = math.pi * s
        p = a_t * (1 - e_t ** 2)
        r = p / (1 + e_t * math.cos(nu))
        x, y = r * math.cos(nu), r * math.sin(nu)
        if r2 < r1:
            x = -x
        return x, y, "Coasting on transfer ellipse", False
    if frac < c3:
        return -r2, 0.0, r"BURN 2   $\Delta v_2 \rightarrow$  circularise at target", True
    s = (frac - c3) / max(1e-9, 1 - c3)
    ang = math.pi + s * (math.pi / 2)
    return r2 * math.cos(ang), r2 * math.sin(ang), "Arrived: coasting on target orbit", False


def animate(scenario_key="earth-mars", save=None, save_frame=None, frames=420):
    scenario = om.PRESETS[scenario_key]
    result = om.hohmann_transfer(scenario.mu, scenario.r1, scenario.r2)
    print(om.summary_report(scenario, result))

    h = build_figure(scenario, result)
    trail_x, trail_y = [], []

    if save_frame:
        x, y, phase, _ = _position(PH_START + PH_BURN1 + PH_TRANSFER * 0.5, h)
        h["craft"].set_data([x], [y])
        h["phase_txt"].set_text("Coasting on transfer ellipse")
        ex, ey = [], []
        for f in np.linspace(0, PH_START + PH_BURN1 + PH_TRANSFER * 0.5, 120):
            px, py, _, _ = _position(f, h)
            ex.append(px); ey.append(py)
        h["trail"].set_data(ex, ey)
        h["fig"].savefig(save_frame, dpi=140, facecolor=BG)
        print(f"\nSaved poster frame -> {save_frame}")
        return

    def init():
        h["craft"].set_data([], [])
        h["trail"].set_data([], [])
        h["burn_flash"].set_data([], [])
        h["phase_txt"].set_text("")
        return h["craft"], h["trail"], h["burn_flash"], h["phase_txt"]

    def update(i):
        frac = i / (frames - 1)
        x, y, phase, is_burn = _position(frac, h)
        h["craft"].set_data([x], [y])
        trail_x.append(x); trail_y.append(y)
        h["trail"].set_data(trail_x, trail_y)
        if is_burn:
            h["burn_flash"].set_data([x], [y])
        else:
            h["burn_flash"].set_data([], [])
        h["phase_txt"].set_text(phase)
        return h["craft"], h["trail"], h["burn_flash"], h["phase_txt"]

    anim = FuncAnimation(h["fig"], update, frames=frames, init_func=init,
                         interval=25, blit=True, repeat=True)

    if save:
        if save.lower().endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(save, writer=PillowWriter(fps=30))
        elif save.lower().endswith(".mp4"):
            try:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=30, bitrate=1800,
                                      extra_args=["-vcodec", "libx264",
                                                  "-pix_fmt", "yuv420p"])
                anim.save(save, writer=writer)
            except Exception as exc:
                print(f"\nffmpeg not available ({exc}).")
                print("Install ffmpeg to enable MP4 export:")
                print("  macOS :  brew install ffmpeg")
                print("  Linux :  sudo apt install ffmpeg")
                print("Falling back to GIF …")
                gif_path = save[:-4] + ".gif"
                from matplotlib.animation import PillowWriter
                anim.save(gif_path, writer=PillowWriter(fps=30))
                print(f"Saved animation -> {gif_path}")
                return
        else:
            anim.save(save, fps=30)
        print(f"\nSaved animation -> {save}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Animated Hohmann transfer simulator")
    p.add_argument("--scenario", default="earth-mars",
                   choices=list(om.PRESETS.keys()),
                   help="which transfer to simulate (default: earth-mars)")
    p.add_argument("--list-scenarios", action="store_true",
                   help="print all available scenarios and exit")
    p.add_argument("--save", help="save animation to file (.gif or .mp4 — mp4 requires ffmpeg)")
    p.add_argument("--save-frame", help="save a single poster PNG and exit")
    p.add_argument("--frames", type=int, default=420)
    args = p.parse_args()

    if args.list_scenarios:
        print("\nAvailable scenarios:")
        print(f"  {'KEY':<18}  DESCRIPTION")
        print(f"  {'-'*18}  {'-'*38}")
        for key, sc in om.PRESETS.items():
            print(f"  {key:<18}  {sc.name}")
        print()
        return

    if args.save or args.save_frame:
        matplotlib.use("Agg")

    animate(args.scenario, save=args.save, save_frame=args.save_frame,
            frames=args.frames)


if __name__ == "__main__":
    main()
