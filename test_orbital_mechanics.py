"""
test_orbital_mechanics.py
=========================
Validation tests against known textbook values.

Run with:
    pytest                     # all tests, verbose by default via pytest.ini
    pytest -v                  # explicit verbose
    pytest -k "leo"            # filter by name
"""

import math
import pytest
import orbital_mechanics as om


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def approx(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def test_circular_velocity_leo():
    r = om.EARTH_RADIUS + 400e3
    v = om.circular_velocity(om.MU_EARTH, r)
    assert approx(v, 7670.0, tol=0.01), v


def test_geostationary_period_is_one_sidereal_day():
    r = om.EARTH_RADIUS + 35_786e3
    T = om.orbital_period(om.MU_EARTH, r)
    assert approx(T, 86164.0, tol=0.01), T


def test_vis_viva_reduces_to_circular():
    r = 7.0e6
    v_circ = om.circular_velocity(om.MU_EARTH, r)
    v_vv = om.vis_viva_velocity(om.MU_EARTH, r, a=r)
    assert approx(v_circ, v_vv, tol=1e-9), (v_circ, v_vv)


# ---------------------------------------------------------------------------
# LEO → GEO
# ---------------------------------------------------------------------------

def test_leo_to_geo_total_delta_v():
    res = om.hohmann_transfer(om.LEO_TO_GEO.mu, om.LEO_TO_GEO.r1, om.LEO_TO_GEO.r2)
    assert approx(res.dv_total, 3900.0, tol=0.05), res.dv_total
    assert res.dv1 > res.dv2


def test_leo_to_geo_transfer_time():
    res = om.hohmann_transfer(om.LEO_TO_GEO.mu, om.LEO_TO_GEO.r1, om.LEO_TO_GEO.r2)
    hours = res.tof / 3600.0
    assert approx(hours, 5.27, tol=0.05), hours


# ---------------------------------------------------------------------------
# Earth → Mars
# ---------------------------------------------------------------------------

def test_earth_to_mars_delta_v_and_time():
    res = om.hohmann_transfer(om.EARTH_TO_MARS.mu, om.EARTH_TO_MARS.r1, om.EARTH_TO_MARS.r2)
    assert approx(res.dv_total, 5600.0, tol=0.07), res.dv_total
    days = res.tof / 86400.0
    assert approx(days, 259.0, tol=0.05), days


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def test_lowering_transfer_is_symmetric_in_magnitude():
    up = om.hohmann_transfer(om.MU_EARTH, om.LEO_TO_GEO.r1, om.LEO_TO_GEO.r2)
    down = om.hohmann_transfer(om.MU_EARTH, om.LEO_TO_GEO.r2, om.LEO_TO_GEO.r1)
    assert approx(up.dv_total, down.dv_total, tol=1e-9), (up.dv_total, down.dv_total)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_same_orbit_zero_delta_v():
    """r1 == r2 → no transfer needed, both burns are zero."""
    r = om.EARTH_RADIUS + 500e3
    res = om.hohmann_transfer(om.MU_EARTH, r, r)
    assert res.dv1 == pytest.approx(0.0, abs=1e-6), res.dv1
    assert res.dv2 == pytest.approx(0.0, abs=1e-6), res.dv2
    assert res.dv_total == pytest.approx(0.0, abs=1e-6), res.dv_total


def test_same_orbit_tof_equals_full_period():
    """When r1 == r2 the 'transfer ellipse' is the same circle; half its
    period equals half the circular orbital period."""
    r = om.EARTH_RADIUS + 500e3
    res = om.hohmann_transfer(om.MU_EARTH, r, r)
    T_circle = om.orbital_period(om.MU_EARTH, r)
    assert res.tof == pytest.approx(T_circle / 2.0, rel=1e-9)


def test_very_large_radii_sun_centric():
    """Pluto-like orbit (~39 AU). Should not raise and dv_total must be positive."""
    r_earth = 1.0 * om.AU
    r_pluto = 39.5 * om.AU
    res = om.hohmann_transfer(om.MU_SUN, r_earth, r_pluto)
    assert res.dv_total > 0
    assert math.isfinite(res.dv_total)
    assert math.isfinite(res.tof)


def test_very_small_radii_low_earth():
    """Just above Earth's surface — extreme low orbit."""
    r_surface = om.EARTH_RADIUS + 10e3   # 10 km altitude
    r_leo = om.EARTH_RADIUS + 400e3
    res = om.hohmann_transfer(om.MU_EARTH, r_surface, r_leo)
    assert res.dv_total > 0
    assert math.isfinite(res.dv_total)


def test_negative_radius_raises():
    with pytest.raises(ValueError):
        om.hohmann_transfer(om.MU_EARTH, -1e6, 7e6)


def test_zero_radius_raises():
    with pytest.raises(ValueError):
        om.hohmann_transfer(om.MU_EARTH, 0.0, 7e6)


def test_result_fields_are_all_positive():
    """All computed quantities must be strictly positive for a valid transfer."""
    res = om.hohmann_transfer(om.MU_EARTH, om.LEO_TO_GEO.r1, om.LEO_TO_GEO.r2)
    for field in ("a_transfer", "v1_circular", "v2_circular",
                  "v_transfer_peri", "v_transfer_apo", "tof", "period1", "period2"):
        assert getattr(res, field) > 0, field


def test_transfer_semi_major_axis():
    """a_transfer must equal (r1 + r2) / 2 exactly."""
    r1, r2 = 7e6, 42e6
    res = om.hohmann_transfer(om.MU_EARTH, r1, r2)
    assert res.a_transfer == pytest.approx((r1 + r2) / 2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# New scenario smoke tests
# ---------------------------------------------------------------------------

def test_leo_to_lunar_tof_approx_4_days():
    """LEO → Lunar transfer should take roughly 3–5 days."""
    res = om.hohmann_transfer(om.LEO_TO_LUNAR.mu, om.LEO_TO_LUNAR.r1, om.LEO_TO_LUNAR.r2)
    days = res.tof / 86400.0
    assert 3.0 <= days <= 5.5, f"Expected ~3.9 days, got {days:.2f}"


def test_leo_to_lunar_delta_v_range():
    """Total Δv for LEO→Lunar should be in the ~3800–4000 m/s ballpark."""
    res = om.hohmann_transfer(om.LEO_TO_LUNAR.mu, om.LEO_TO_LUNAR.r1, om.LEO_TO_LUNAR.r2)
    assert 3700 <= res.dv_total <= 4100, res.dv_total


def test_earth_to_jupiter_tof_approx_2_years():
    """Earth → Jupiter transfer takes ~2.7 years (Hohmann approximation)."""
    res = om.hohmann_transfer(om.EARTH_TO_JUPITER.mu, om.EARTH_TO_JUPITER.r1, om.EARTH_TO_JUPITER.r2)
    years = res.tof / (365.25 * 86400)
    assert 2.0 <= years <= 3.5, f"Expected ~2.7 years, got {years:.2f}"


def test_earth_to_jupiter_delta_v_range():
    """Δv for Earth→Jupiter is ~14,400 m/s (large orbit ratio means big burns)."""
    res = om.hohmann_transfer(om.EARTH_TO_JUPITER.mu, om.EARTH_TO_JUPITER.r1, om.EARTH_TO_JUPITER.r2)
    assert 13_500 <= res.dv_total <= 15_500, res.dv_total


def test_geo_to_graveyard_very_small_delta_v():
    """GEO → graveyard is a tiny nudge — Δv should be well under 20 m/s."""
    res = om.hohmann_transfer(om.GEO_TO_GRAVEYARD.mu, om.GEO_TO_GRAVEYARD.r1, om.GEO_TO_GRAVEYARD.r2)
    assert 0 < res.dv_total < 20, res.dv_total


def test_all_presets_compute_without_error():
    """Every preset must produce a finite, positive result — no crashes."""
    for key, sc in om.PRESETS.items():
        res = om.hohmann_transfer(sc.mu, sc.r1, sc.r2)
        assert math.isfinite(res.dv_total), key
        assert math.isfinite(res.tof), key
        assert res.tof > 0, key


# ---------------------------------------------------------------------------
# Specific orbital energy
# ---------------------------------------------------------------------------

def test_specific_energy_is_negative_for_bound_orbit():
    eps = om.specific_orbital_energy(om.MU_EARTH, om.EARTH_RADIUS + 400e3)
    assert eps < 0


def test_specific_energy_matches_definition():
    a = 1.5e7
    eps = om.specific_orbital_energy(om.MU_EARTH, a)
    assert eps == pytest.approx(-om.MU_EARTH / (2 * a), rel=1e-12)


# ---------------------------------------------------------------------------
# Launch window / phase angle
# ---------------------------------------------------------------------------

def test_earth_mars_phase_angle_is_about_44_degrees():
    """Textbook Earth->Mars departure phase angle is ~44 degrees."""
    lw = om.launch_window(om.MU_SUN, om.EARTH_TO_MARS.r1, om.EARTH_TO_MARS.r2)
    assert lw.phase_angle_deg == pytest.approx(44.0, abs=2.0), lw.phase_angle_deg


def test_earth_mars_synodic_period_about_780_days():
    """Earth-Mars synodic period is ~779.9 days."""
    lw = om.launch_window(om.MU_SUN, om.EARTH_TO_MARS.r1, om.EARTH_TO_MARS.r2)
    days = lw.synodic_period / 86400.0
    assert days == pytest.approx(780.0, abs=10.0), days


def test_phase_angle_in_valid_range():
    """Phase angle must always wrap into (-180, 180] degrees."""
    for sc in om.PRESETS.values():
        lw = om.launch_window(sc.mu, sc.r1, sc.r2)
        assert -180.0 < lw.phase_angle_deg <= 180.0, sc.name


# ---------------------------------------------------------------------------
# Bi-elliptic transfer
# ---------------------------------------------------------------------------

def test_hohmann_optimal_below_threshold():
    """For small ratios (e.g. LEO->GEO ~6.6) Hohmann is optimal."""
    assert om.hohmann_is_optimal(om.LEO_TO_GEO.r1, om.LEO_TO_GEO.r2)


def test_hohmann_not_optimal_above_threshold():
    """For ratios above ~11.94 Hohmann is no longer optimal."""
    r1 = 7000e3
    r2 = 15.0 * r1
    assert not om.hohmann_is_optimal(r1, r2)


def test_bielliptic_beats_hohmann_at_large_ratio():
    """At ratio 16 with a far intermediate apoapsis, bi-elliptic wins on dv."""
    mu, r1 = om.MU_EARTH, 7000e3
    r2 = 16.0 * r1
    h = om.hohmann_transfer(mu, r1, r2)
    b = om.bielliptic_transfer(mu, r1, r2, r_b=50.0 * r1)
    assert b.dv_total < h.dv_total, (b.dv_total, h.dv_total)


def test_bielliptic_takes_longer_than_hohmann():
    """Bi-elliptic always trades time for fuel — ToF must be larger."""
    mu, r1 = om.MU_EARTH, 7000e3
    r2 = 16.0 * r1
    h = om.hohmann_transfer(mu, r1, r2)
    b = om.bielliptic_transfer(mu, r1, r2, r_b=50.0 * r1)
    assert b.tof > h.tof


def test_bielliptic_rejects_rb_below_r2():
    with pytest.raises(ValueError):
        om.bielliptic_transfer(om.MU_EARTH, 7e6, 4e7, r_b=3e7)
