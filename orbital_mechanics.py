"""
orbital_mechanics.py
====================

Core physics engine for the Hohmann Transfer / Orbital Mechanics Simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


G = 6.67430e-11  # Universal gravitational constant [m^3 kg^-1 s^-2]
MU_EARTH = 3.986004418e14
MU_SUN   = 1.32712440018e20
EARTH_RADIUS  = 6.371e6
MOON_DISTANCE = 3.844e8        # mean Earth–Moon distance [m]
AU = 1.495978707e11            # 1 Astronomical Unit [m]

# Circular orbit radii of planets around the Sun (mean values)
R_JUPITER = 5.2044 * AU        # Jupiter mean orbital radius


@dataclass(frozen=True)
class Scenario:
    name: str
    central_body: str
    mu: float
    r1: float
    r2: float
    r1_label: str
    r2_label: str


LEO_TO_GEO = Scenario(
    name="LEO -> GEO (Earth)",
    central_body="Earth",
    mu=MU_EARTH,
    r1=EARTH_RADIUS + 400e3,
    r2=EARTH_RADIUS + 35_786e3,
    r1_label="LEO (400 km)",
    r2_label="GEO (35,786 km)",
)

EARTH_TO_MARS = Scenario(
    name="Earth -> Mars (Sun)",
    central_body="Sun",
    mu=MU_SUN,
    r1=1.000 * AU,
    r2=1.524 * AU,
    r1_label="Earth orbit (1.00 AU)",
    r2_label="Mars orbit (1.52 AU)",
)

LEO_TO_LUNAR = Scenario(
    name="LEO -> Lunar (Earth)",
    central_body="Earth",
    mu=MU_EARTH,
    r1=EARTH_RADIUS + 400e3,        # ISS-like LEO
    r2=MOON_DISTANCE,               # mean lunar distance
    r1_label="LEO (400 km alt)",
    r2_label="Lunar orbit (384,400 km)",
)

EARTH_TO_JUPITER = Scenario(
    name="Earth -> Jupiter (Sun)",
    central_body="Sun",
    mu=MU_SUN,
    r1=1.000 * AU,
    r2=R_JUPITER,
    r1_label="Earth orbit (1.00 AU)",
    r2_label="Jupiter orbit (5.20 AU)",
)

# GEO graveyard band sits ~300 km above GEO to avoid crowding the GEO belt.
GEO_TO_GRAVEYARD = Scenario(
    name="GEO -> Graveyard (Earth)",
    central_body="Earth",
    mu=MU_EARTH,
    r1=EARTH_RADIUS + 35_786e3,     # GEO
    r2=EARTH_RADIUS + 35_786e3 + 300e3,   # graveyard (+300 km)
    r1_label="GEO (35,786 km)",
    r2_label="Graveyard (36,086 km)",
)

PRESETS = {
    "leo-geo":          LEO_TO_GEO,
    "earth-mars":       EARTH_TO_MARS,
    "leo-lunar":        LEO_TO_LUNAR,
    "earth-jupiter":    EARTH_TO_JUPITER,
    "geo-graveyard":    GEO_TO_GRAVEYARD,
}


def circular_velocity(mu: float, r: float) -> float:
    """v = sqrt(mu / r)"""
    return math.sqrt(mu / r)


def vis_viva_velocity(mu: float, r: float, a: float) -> float:
    """v = sqrt( mu * (2/r - 1/a) )"""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def orbital_period(mu: float, a: float) -> float:
    """T = 2*pi*sqrt(a^3 / mu)"""
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def specific_orbital_energy(mu: float, a: float) -> float:
    """Specific mechanical (orbital) energy:  epsilon = -mu / (2a)  [J/kg]."""
    return -mu / (2.0 * a)


# Orbit-ratio threshold above which a bi-elliptic transfer always beats a
# Hohmann transfer on total delta-v.  (Classic result; ~11.93876 exactly.)
BIELLIPTIC_THRESHOLD = 11.93876


@dataclass(frozen=True)
class HohmannResult:
    r1: float
    r2: float
    a_transfer: float
    v1_circular: float
    v2_circular: float
    v_transfer_peri: float
    v_transfer_apo: float
    dv1: float
    dv2: float
    dv_total: float
    tof: float
    period1: float
    period2: float


def hohmann_transfer(mu: float, r1: float, r2: float) -> HohmannResult:
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbital radii must be positive.")

    a_t = (r1 + r2) / 2.0
    v1_circular = circular_velocity(mu, r1)
    v2_circular = circular_velocity(mu, r2)
    v_transfer_peri = vis_viva_velocity(mu, r1, a_t)
    v_transfer_apo = vis_viva_velocity(mu, r2, a_t)
    dv1 = abs(v_transfer_peri - v1_circular)
    dv2 = abs(v2_circular - v_transfer_apo)
    dv_total = dv1 + dv2
    tof = 0.5 * orbital_period(mu, a_t)

    return HohmannResult(
        r1=r1, r2=r2, a_transfer=a_t,
        v1_circular=v1_circular, v2_circular=v2_circular,
        v_transfer_peri=v_transfer_peri, v_transfer_apo=v_transfer_apo,
        dv1=dv1, dv2=dv2, dv_total=dv_total, tof=tof,
        period1=orbital_period(mu, r1), period2=orbital_period(mu, r2),
    )


@dataclass(frozen=True)
class BiEllipticResult:
    r1: float
    r2: float
    r_b: float          # apoapsis radius of the intermediate ellipses
    dv1: float          # raise apoapsis to r_b
    dv2: float          # at r_b, raise periapsis to r2
    dv3: float          # at r2, circularise (this is a retro-burn -> lowers)
    dv_total: float
    tof: float


def bielliptic_transfer(mu: float, r1: float, r2: float, r_b: float) -> BiEllipticResult:
    """Three-burn bi-elliptic transfer via an intermediate apoapsis r_b.

    For large orbit ratios (r2/r1 > ~11.94) this can require less total
    delta-v than a Hohmann transfer, at the cost of much longer flight time.
    r_b must be >= r2 (the intermediate apoapsis reaches beyond the target).
    """
    if r1 <= 0 or r2 <= 0 or r_b <= 0:
        raise ValueError("Orbital radii must be positive.")
    if r_b < r2:
        raise ValueError("Intermediate apoapsis r_b must be >= r2.")

    a1 = (r1 + r_b) / 2.0      # first transfer ellipse: r1 -> r_b
    a2 = (r2 + r_b) / 2.0      # second transfer ellipse: r_b -> r2

    v_c1 = circular_velocity(mu, r1)
    v_c2 = circular_velocity(mu, r2)

    # Burn 1: at r1, accelerate onto ellipse 1 (periapsis r1, apoapsis r_b)
    dv1 = vis_viva_velocity(mu, r1, a1) - v_c1
    # Burn 2: at r_b, accelerate from ellipse 1 apoapsis onto ellipse 2 apoapsis
    dv2 = vis_viva_velocity(mu, r_b, a2) - vis_viva_velocity(mu, r_b, a1)
    # Burn 3: at r2, decelerate from ellipse 2 periapsis to circular at r2
    dv3 = vis_viva_velocity(mu, r2, a2) - v_c2

    dv_total = abs(dv1) + abs(dv2) + abs(dv3)
    tof = 0.5 * orbital_period(mu, a1) + 0.5 * orbital_period(mu, a2)

    return BiEllipticResult(
        r1=r1, r2=r2, r_b=r_b,
        dv1=dv1, dv2=dv2, dv3=dv3,
        dv_total=dv_total, tof=tof,
    )


def hohmann_is_optimal(r1: float, r2: float) -> bool:
    """Hohmann is the optimal two-impulse transfer for orbit ratios below
    ~11.94.  Beyond that, a bi-elliptic transfer can achieve lower total
    delta-v.  Returns True when Hohmann is (or ties as) optimal."""
    ratio = max(r1, r2) / min(r1, r2)
    return ratio <= BIELLIPTIC_THRESHOLD


@dataclass(frozen=True)
class LaunchWindow:
    phase_angle_rad: float      # required lead/lag of target at departure
    phase_angle_deg: float
    synodic_period: float       # time between successive identical geometries
    tof: float                  # transfer time of flight


def launch_window(mu: float, r1: float, r2: float) -> LaunchWindow:
    """Phase-angle and synodic-period analysis for a Hohmann transfer.

    For an interplanetary transfer the target body must lead the spacecraft
    by a specific angle at departure so that both arrive at the apsis point
    simultaneously.  The required phase angle is

        phi = pi - omega_target * t_flight

    where omega_target is the target's mean angular rate.  The synodic period
    is the time between successive launch opportunities.
    """
    a_t = (r1 + r2) / 2.0
    tof = 0.5 * orbital_period(mu, a_t)

    T1 = orbital_period(mu, r1)
    T2 = orbital_period(mu, r2)
    omega2 = 2.0 * math.pi / T2          # target angular rate [rad/s]

    phase = math.pi - omega2 * tof       # wrap into (-pi, pi]
    phase = (phase + math.pi) % (2.0 * math.pi) - math.pi

    # Synodic period: 1 / |1/T1 - 1/T2|  (guard the equal-period case)
    denom = abs(1.0 / T1 - 1.0 / T2)
    synodic = math.inf if denom == 0 else 1.0 / denom

    return LaunchWindow(
        phase_angle_rad=phase,
        phase_angle_deg=math.degrees(phase),
        synodic_period=synodic,
        tof=tof,
    )


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "infinite"
    days = seconds / 86400.0
    if days >= 1.0:
        return f"{days:.2f} days"
    hours = seconds / 3600.0
    if hours >= 1.0:
        return f"{hours:.2f} hours"
    return f"{seconds / 60.0:.2f} minutes"


def summary_report(scenario: Scenario, result: HohmannResult) -> str:
    lw = launch_window(scenario.mu, scenario.r1, scenario.r2)
    ratio = max(scenario.r1, scenario.r2) / min(scenario.r1, scenario.r2)
    optimal = hohmann_is_optimal(scenario.r1, scenario.r2)

    lines = [
        f"Hohmann Transfer Report : {scenario.name}",
        "=" * 52,
        f"Central body            : {scenario.central_body}",
        f"Start orbit  (r1)       : {scenario.r1_label}",
        f"Target orbit (r2)       : {scenario.r2_label}",
        "-" * 52,
        f"Circular speed @ r1     : {result.v1_circular:,.1f} m/s",
        f"Circular speed @ r2     : {result.v2_circular:,.1f} m/s",
        f"Transfer speed @ r1     : {result.v_transfer_peri:,.1f} m/s",
        f"Transfer speed @ r2     : {result.v_transfer_apo:,.1f} m/s",
        "-" * 52,
        f"Burn 1 (delta-v1)       : {result.dv1:,.1f} m/s",
        f"Burn 2 (delta-v2)       : {result.dv2:,.1f} m/s",
        f"TOTAL delta-v           : {result.dv_total:,.1f} m/s",
        f"Transfer time of flight : {format_duration(result.tof)}",
        "-" * 52,
        f"Orbit ratio r2/r1       : {ratio:,.2f}",
        f"Required phase angle    : {lw.phase_angle_deg:+.1f} deg",
        f"Synodic period          : {format_duration(lw.synodic_period)}",
        f"Hohmann optimal?        : {'yes' if optimal else 'no (bi-elliptic better)'}",
        "=" * 52,
    ]
    return "\n".join(lines)
