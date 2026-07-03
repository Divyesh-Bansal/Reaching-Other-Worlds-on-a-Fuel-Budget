# Hohmann Transfer & Orbital Mechanics Simulator

An animated, physics-accurate simulator of **Hohmann transfer orbits** — the
two-burn maneuver that moves a spacecraft between two circular orbits using the
least energy. The tool computes the full delta-v budget, time of flight, launch
phase angle, and synodic period from first principles, and renders the transfer
as an animation with a live data panel.

A full feature walkthrough is available in **`demo_hohmann.mp4`** (in this repository).

## What is a Hohmann transfer?

To move from a low circular orbit to a higher one, a spacecraft fires its engine
twice. The **first burn** raises the far side of the orbit, placing the craft on
an elliptical "transfer" path whose low point touches the start orbit and whose
high point touches the target orbit. The craft coasts halfway around this ellipse,
then a **second burn** circularizes it at the target altitude. It is the most
fuel-efficient two-impulse transfer between coplanar circular orbits — which is
exactly why real missions, from communications satellites to Mars probes, use it.

## Features

- Physics engine derived from the vis-viva equation and Kepler's third law
- Five built-in scenarios spanning Earth orbits, cislunar space, and interplanetary transfers
- Live data panel with delta-v budget, velocities, and LaTeX-rendered governing equations
- **Launch-window analysis**: required departure phase angle and synodic period
- **Bi-elliptic comparison**: flags when a three-burn transfer would beat Hohmann
- Export to animated GIF, MP4 (via ffmpeg), or a single poster PNG
- 31 validation tests against textbook and real-mission values

## Installation

```bash
git clone <your-repo-url>
cd hohmann-simulator
pip install -r requirements.txt
```

Python 3.9+ is required. MP4 export additionally needs
[ffmpeg](https://ffmpeg.org/download.html) (`brew install ffmpeg` on macOS,
`sudo apt install ffmpeg` on Linux).

## Usage

```bash
# Live animated window
python3 hohmann_simulator.py --scenario earth-mars

# List every available scenario
python3 hohmann_simulator.py --list-scenarios

# Save an animation (GIF or, ~10x smaller, MP4)
python3 hohmann_simulator.py --scenario leo-geo --save transfer.gif
python3 hohmann_simulator.py --scenario earth-mars --save transfer.mp4

# Save a single poster frame
python3 hohmann_simulator.py --scenario earth-mars --save-frame poster.png
```

Each run also prints a full text report to the console:

```
Hohmann Transfer Report : Earth -> Mars (Sun)
====================================================
TOTAL delta-v           : 5,596.0 m/s
Transfer time of flight : 258.92 days
Required phase angle    : +44.4 deg
Synodic period          : 779.67 days
Hohmann optimal?        : yes
====================================================
```

## Results & validation

All figures below are produced by the simulator. The "Reference" column lists the
independently published value the result is checked against.

| Scenario | Total Δv | Time of flight | Reference (published) |
|---|---|---|---|
| LEO → GEO | 3,857 m/s | 5.29 hours | ~3.9 km/s, ~5.3 h transfer [1] |
| Earth → Mars (heliocentric) | 5,596 m/s | 258.9 days | ~5.6 km/s, ~8.5 months [2] |
| LEO → Lunar distance | 3,913 m/s | 4.98 days | ~3.9 km/s, ~3–5 day coast [3] |
| Earth → Jupiter (heliocentric) | 14,437 m/s | 997.9 days (~2.7 yr) | ~14.4 km/s, ~2.7 yr [2] |
| GEO → graveyard (+300 km) | 10.9 m/s | 12.0 hours | ~11 m/s disposal burn [4] |

Additional cross-checks built into the test suite:

- **Geostationary period** computes to 86,164 s = one sidereal day (exact).
- **Earth–Mars synodic period** computes to 779.7 days vs. the accepted 779.9 days.
- **Earth–Mars departure phase angle** computes to +44.4° vs. the textbook ~44°.
- **Vis-viva** reduces to the circular-velocity formula when `a = r`.
- **Bi-elliptic** total Δv falls below Hohmann for orbit ratios above ~11.94.

## Model assumptions & limitations

This simulator uses the standard impulsive, two-body, coplanar idealization taught
in introductory astrodynamics. Being explicit about what it does *not* model:

1. **Impulsive burns.** Each maneuver is treated as an instantaneous change in
   velocity. Real engines burn over finite time, incurring gravity losses.
2. **Coplanar, circular orbits.** No inclination change and no starting
   eccentricity. Real transfers (e.g. to Mars) require an additional plane-change
   component.
3. **Two-body (patched-conic not applied).** For the interplanetary scenarios,
   the reported delta-v is the **heliocentric transfer Δv** — the change required
   in the Sun-centered frame. It is **not** the full mission Δv from a parking
   orbit, which would also include Earth-escape injection (~3.6 km/s for Mars) and
   target-arrival capture. A complete mission budget would chain these via the
   patched-conic approximation.
4. **No perturbations.** J2 oblateness, third-body gravity, solar radiation
   pressure, and atmospheric drag are all neglected.
5. **Hohmann is optimal only below an orbit ratio of ~11.94.** Above that, a
   bi-elliptic transfer needs less total delta-v (at the cost of far longer flight
   time). The simulator detects and reports this — for example, the LEO → Lunar
   scenario (ratio ≈ 56.8) is flagged as a case where bi-elliptic wins.

These simplifications keep the physics transparent and the results verifiable
against closed-form textbook solutions.

## Project structure

```
hohmann-simulator/
├── orbital_mechanics.py        # physics engine (no plotting dependencies)
├── hohmann_simulator.py        # animation, rendering, CLI
├── test_orbital_mechanics.py   # 31 pytest validation tests
├── pyproject.toml              # packaging + pytest config
├── requirements.txt
└── README.md
```

The physics engine is deliberately separated from the visualization so it can be
imported and reused on its own:

```python
import orbital_mechanics as om

result = om.hohmann_transfer(om.MU_EARTH, om.EARTH_RADIUS + 400e3,
                             om.EARTH_RADIUS + 35_786e3)
print(result.dv_total)        # 3856.6 m/s

window = om.launch_window(om.MU_SUN, om.EARTH_TO_MARS.r1, om.EARTH_TO_MARS.r2)
print(window.phase_angle_deg) # +44.4 deg
```

## Testing

```bash
pytest                 # run all 31 tests
pytest -v              # verbose
pytest --cov           # with coverage (needs pytest-cov)
```

## References

1. H. D. Curtis, *Orbital Mechanics for Engineering Students*, Ch. 6 (Hohmann transfers).
2. D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, interplanetary transfer chapter.
3. NASA, Apollo translunar trajectory documentation (~3-day coast to lunar distance).
4. Inter-Agency Space Debris Coordination Committee (IADC), GEO disposal (graveyard orbit) guidelines.

## License

MIT
