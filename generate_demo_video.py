"""
generate_demo_video.py
=======================
Builds a narrated-style demo MP4 that walks through every feature of the
Hohmann Transfer Simulator, one segment at a time.

It reuses the real simulator animation for each scenario and stitches in
title / feature / chart cards. Output: demo_hohmann.mp4

Run:  python3 generate_demo_video.py
Requires: imageio + imageio-ffmpeg (bundled ffmpeg, no system install needed).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imageio

import orbital_mechanics as om
import hohmann_simulator as hs

# Computer Modern math font for professional (LaTeX-style) equation rendering
plt.rcParams["mathtext.fontset"] = "cm"

# ---------------------------------------------------------------------------
# Video constants
# ---------------------------------------------------------------------------
W, H = 1280, 720           # divisible by 16 -> clean H.264 / yuv420p
FIGSIZE = (12.8, 7.2)      # * dpi 100 = 1280 x 720
DPI = 100
FPS = 30
OUT = "demo_hohmann.mp4"

BG       = hs.BG
PANEL    = hs.PANEL
INNER_C  = hs.INNER_C
OUTER_C  = hs.OUTER_C
TRANSFER = hs.TRANSFER_C
TEXT_C   = hs.TEXT_C
MUTED    = hs.MUTED
BURN_C   = hs.BURN_C
GREEN    = "#69f0ae"


def _grab(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def _starfield(ax, n=240, seed=5):
    rng = np.random.default_rng(seed)
    ax.scatter(rng.uniform(0, 1, n), rng.uniform(0, 1, n),
               s=rng.uniform(0.2, 2.0, n) ** 2, c="white",
               alpha=rng.uniform(0.2, 0.8, n), transform=ax.transAxes,
               zorder=0, linewidths=0)


def _blank_fig():
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax


# ---------------------------------------------------------------------------
# Card builders (each returns one RGB frame)
# ---------------------------------------------------------------------------

def title_frame():
    fig, ax = _blank_fig()
    _starfield(ax)
    ax.text(0.5, 0.62, "Hohmann Transfer Simulator", color=TEXT_C,
            fontsize=34, weight="bold", ha="center", va="center")
    ax.text(0.5, 0.50, "Reaching other worlds on a fuel budget", color=TRANSFER,
            fontsize=17, ha="center", va="center", style="italic")
    ax.text(0.5, 0.30, "An interactive orbital-mechanics project", color=MUTED,
            fontsize=14, ha="center", va="center")
    frame = _grab(fig); plt.close(fig)
    return frame


def text_frame(title, lines, accent=INNER_C):
    fig, ax = _blank_fig()
    _starfield(ax, n=120, seed=9)
    ax.text(0.08, 0.84, title, color=accent, fontsize=27, weight="bold",
            ha="left", va="center")
    ax.plot([0.08, 0.92], [0.76, 0.76], color=accent, lw=1.5, alpha=0.6)
    y = 0.65
    for ln in lines:
        ax.text(0.10, y, "•", color=accent, fontsize=16, ha="left", va="center")
        ax.text(0.14, y, ln, color=TEXT_C, fontsize=16, ha="left", va="center")
        y -= 0.115
    frame = _grab(fig); plt.close(fig)
    return frame


def divider_frame(text, accent=TRANSFER):
    fig, ax = _blank_fig()
    _starfield(ax, n=180, seed=2)
    ax.text(0.5, 0.54, text, color=accent, fontsize=30, weight="bold",
            ha="center", va="center")
    ax.plot([0.30, 0.70], [0.44, 0.44], color=accent, lw=2, alpha=0.7)
    frame = _grab(fig); plt.close(fig)
    return frame


def physics_frame():
    """Physics card with professionally typeset (mathtext) equations."""
    fig, ax = _blank_fig()
    _starfield(ax, n=120, seed=9)
    ax.text(0.08, 0.86, "The physics behind it", color=INNER_C, fontsize=27,
            weight="bold", ha="left", va="center")
    ax.plot([0.08, 0.92], [0.78, 0.78], color=INNER_C, lw=1.5, alpha=0.6)

    rows = [
        (r"$v = \sqrt{\mu\left(\dfrac{2}{r} - \dfrac{1}{a}\right)}$",
         "vis-viva: speed anywhere on an orbit"),
        (r"$v = \sqrt{\dfrac{\mu}{r}}$",
         "circular-orbit speed"),
        (r"$T = 2\pi\sqrt{\dfrac{a^{3}}{\mu}}$",
         "Kepler's third law: sets the travel time"),
    ]
    y = 0.62
    for eq, caption in rows:
        ax.text(0.12, y, eq, color=TEXT_C, fontsize=26, ha="left", va="center")
        ax.text(0.48, y, caption, color=MUTED, fontsize=15, ha="left", va="center")
        y -= 0.165
    ax.text(0.12, 0.10, "Burn 1 speeds you up to climb; Burn 2 circularises at the top",
            color=TRANSFER, fontsize=15, ha="left", va="center", style="italic")
    frame = _grab(fig); plt.close(fig)
    return frame


def scenario_title_frame(key):
    sc = om.PRESETS[key]
    res = om.hohmann_transfer(sc.mu, sc.r1, sc.r2)
    lw = om.launch_window(sc.mu, sc.r1, sc.r2)
    optimal = om.hohmann_is_optimal(sc.r1, sc.r2)
    fig, ax = _blank_fig()
    _starfield(ax, n=140, seed=hash(key) % 1000)
    ax.text(0.5, 0.74, sc.name, color=TEXT_C, fontsize=26, weight="bold",
            ha="center", va="center")
    rows = [
        (f"Total \u0394v", f"{res.dv_total:,.0f} m/s", TRANSFER),
        ("Time of flight", om.format_duration(res.tof), INNER_C),
        ("Departure phase angle", f"{lw.phase_angle_deg:+.1f}\u00b0", OUTER_C),
        ("Hohmann optimal?", "yes" if optimal else "no (bi-elliptic better)", GREEN),
    ]
    y = 0.55
    for label, val, col in rows:
        ax.text(0.30, y, label, color=MUTED, fontsize=16, ha="left", va="center")
        ax.text(0.70, y, val, color=col, fontsize=16, weight="bold",
                ha="right", va="center")
        y -= 0.10
    frame = _grab(fig); plt.close(fig)
    return frame


def image_frames(path):
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Scenario animation frames (reuse the real simulator visualization)
# ---------------------------------------------------------------------------

def scenario_animation_frames(key, n_frames=150):
    sc = om.PRESETS[key]
    res = om.hohmann_transfer(sc.mu, sc.r1, sc.r2)
    h = hs.build_figure(sc, res)
    fig = h["fig"]
    fig.set_size_inches(*FIGSIZE)
    fig.set_dpi(DPI)

    trail_x, trail_y = [], []
    frames = []
    for i in range(n_frames):
        frac = i / (n_frames - 1)
        x, y, phase, is_burn = hs._position(frac, h)
        h["craft"].set_data([x], [y])
        trail_x.append(x); trail_y.append(y)
        h["trail"].set_data(trail_x, trail_y)
        if is_burn:
            h["burn_flash"].set_data([x], [y])
        else:
            h["burn_flash"].set_data([], [])
        h["phase_txt"].set_text(phase)
        frames.append(_grab(fig))
    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------

def hold(writer, frame, seconds):
    for _ in range(int(round(seconds * FPS))):
        writer.append_data(frame)


def main():
    writer = imageio.get_writer(
        OUT, fps=FPS, codec="libx264", quality=8,
        macro_block_size=16, pixelformat="yuv420p",
    )

    print("1/8  intro cards")
    hold(writer, title_frame(), 3.5)
    hold(writer, text_frame(
        "What is a Hohmann transfer?",
        ["The most fuel-efficient way to move between two circular orbits",
         "Two engine burns connected by one long, engine-off coast",
         "Proposed by Walter Hohmann in 1925, still used today",
         "Powers satellites, Moon missions, and interplanetary probes"],
        accent=TRANSFER), 6.0)

    print("2/8  concept + vis-viva cards")
    hold(writer, image_frames("slide_concept.png"), 5.5)
    hold(writer, physics_frame(), 6.0)
    hold(writer, image_frames("slide_visviva.png"), 5.5)

    print("3/8  scenarios divider")
    hold(writer, divider_frame("Five Real Mission Scenarios"), 2.8)

    scenarios = ["leo-geo", "earth-mars", "leo-lunar", "earth-jupiter", "geo-graveyard"]
    for idx, key in enumerate(scenarios, 1):
        print(f"4/8  scenario {idx}/5: {key}")
        hold(writer, scenario_title_frame(key), 2.6)
        for fr in scenario_animation_frames(key, n_frames=140):
            writer.append_data(fr)

    print("5/8  launch window feature")
    hold(writer, text_frame(
        "Feature: Launch windows",
        ["Planets move, so you must depart when the target will meet you",
         "The tool computes the required departure phase angle",
         "Earth -> Mars: Mars must lead by about 44 degrees",
         "...and the synodic period: a Mars window only every ~26 months"],
        accent=OUTER_C), 6.0)
    hold(writer, image_frames("slide_phase_angle.png"), 5.5)

    print("6/8  bi-elliptic feature")
    hold(writer, text_frame(
        "Feature: Knowing the limits",
        ["Hohmann is optimal only for orbit ratios below ~11.94",
         "Beyond that, a 3-burn bi-elliptic transfer uses less fuel",
         "The tool detects and flags this automatically",
         "Example: LEO -> Lunar (ratio ~57) favours bi-elliptic"],
        accent=GREEN), 6.0)
    hold(writer, image_frames("slide_bielliptic.png"), 5.5)

    print("7/8  results charts")
    hold(writer, divider_frame("Results & Validation", accent=INNER_C), 2.5)
    hold(writer, image_frames("slide_deltav.png"), 4.5)
    hold(writer, image_frames("slide_tof.png"), 4.5)

    print("8/8  closing card")
    hold(writer, text_frame(
        "Validated against real data",
        ["Geostationary period = 86,164 s = one sidereal day (exact)",
         "Earth-Mars synodic period = 779.7 days  (accepted: 779.9)",
         "Earth-Mars phase angle = +44.4 deg  (textbook: ~44)",
         "31 automated tests confirm every result"],
        accent=TRANSFER), 6.0)
    hold(writer, divider_frame("Thank you"), 3.0)

    writer.close()
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
