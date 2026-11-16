"""Retroreflective (retrodirective) materials: closed-form checks and the metalens tag's measured values."""

import numpy as np
import pytest
import pyvista as pv

from radar_detector import geometry
from radar_detector.materials import MaterialLibrary
from radar_detector.rcs import RadarSettings, Target, _patches, compute_rcs, retro_taper

LAM = RadarSettings().wavelength
TAG = 0.053  # side of the tag in Ghasemi et al. 2026


def _plate(size=TAG, res=6, center=(0, 0, 0)) -> pv.PolyData:
    # Normal -X faces a radar behind the rider (azimuth 0)
    return pv.Plane(center=center, direction=(-1, 0, 0), i_size=size, j_size=size, i_resolution=res, j_resolution=res).triangulate()


def _target(mesh, R=1.0, A=0.0, s=0.0, q=0.0, half=40.0) -> Target:
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    m = len(faces)
    return Target(np.asarray(mesh.points, float), faces, np.full(m, R), np.full(m, A), np.full(m, s),
                  retro=np.full(m, q), retro_half_angle_deg=np.full(m, half))


def _solve(t, az=0.0, el=0.0):
    r = RadarSettings(distance_m=2000.0, azimuth_deg=az, elevation_deg=el, multibounce=False)
    return compute_rcs(t, r, np.zeros(3))


def _db(x):
    return 10 * np.log10(x)


def test_ideal_retro_matches_plate_at_normal_incidence():
    expected = 4 * np.pi * TAG**4 / LAM**2
    res = _solve(_target(_plate(), q=1.0))
    assert res.rcs_m2 == pytest.approx(expected, rel=0.01)
    assert res.retro_m2 == pytest.approx(expected, rel=0.01) and res.specular_m2 == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("az", [15.0, 30.0, 45.0])
def test_ideal_retro_holds_up_off_axis(az):
    """sigma = 4 pi (A cos theta)^2 / lambda^2, where a metal plate of the same size has collapsed."""
    expected = 4 * np.pi * (TAG**2 * np.cos(np.radians(az))) ** 2 / LAM**2
    retro = _solve(_target(_plate(res=12), q=1.0, half=60.0), az=az)
    metal = _solve(_target(_plate(res=12)), az=az)
    assert _db(retro.rcs_m2) == pytest.approx(_db(expected), abs=0.1)
    assert _db(metal.rcs_m2) < _db(expected) - 20


def test_coherent_across_a_tilted_patch_in_elevation():
    """The patch's faces sit at different ranges when tilted; phase conjugation keeps them in step."""
    expected = 4 * np.pi * (TAG**2 * np.cos(np.radians(25))) ** 2 / LAM**2
    assert _db(_solve(_target(_plate(res=12), q=1.0), el=25.0).rcs_m2) == pytest.approx(_db(expected), abs=0.1)


def test_coverage_and_one_sidedness():
    t = _target(_plate(), q=1.0, half=40.0)
    assert _solve(t, az=55.0).retro_m2 == 0.0  # beyond the half-angle plus the 10 degree roll-off
    assert 0 < _solve(t, az=45.0).retro_m2 < _solve(t, az=35.0).retro_m2  # inside the roll-off
    assert _solve(t, az=180.0).retro_m2 == 0.0  # from behind the tag
    np.testing.assert_allclose(retro_taper(np.array([0, 40, 45, 50, 60]), 40.0), [1, 1, 0.5, 0, 0], atol=1e-12)


def test_metalens_preset_matches_measured_tag():
    """Fitted to the 53 mm tag: 3.54 dBsm at broadside, -16.76 dBsm median over +/-40 degrees."""
    lib = MaterialLibrary()
    m = lib[lib.index("Metalens retro tag (measured)")]
    t = _target(_plate(res=12), R=m.reflectance, A=m.absorptance, s=m.roughness, q=m.retro, half=m.retro_half_angle_deg)
    assert _solve(t).rcs_dbsm == pytest.approx(3.54, abs=0.3)
    # Like the paper: 1 degree steps across the tag's 80 degree coverage
    off_axis = [_solve(t, az=float(a)).rcs_dbsm for a in range(-40, 41)]
    assert np.median(off_axis) == pytest.approx(-16.76, abs=1.0)


def test_separate_tags_are_separate_apertures():
    a, b = _plate(center=(0, -0.2, 0)), _plate(center=(0, 0.2, 0))
    faces = np.vstack([a.faces.reshape(-1, 4)[:, 1:], b.faces.reshape(-1, 4)[:, 1:] + a.n_points])
    labels = _patches(faces, a.n_points + b.n_points)
    na = a.n_cells
    assert len(set(labels[:na])) == 1 and len(set(labels[na:])) == 1 and labels[0] != labels[-1]


def test_retro_tag_on_the_biker():
    """A metalens tag behind the saddle beats the biker alone from the rear, and is invisible from the front."""
    lib = MaterialLibrary()
    biker = geometry.build_biker(lib)
    tag = geometry.make_attachment("Tag", geometry.flat_plate(TAG), lib.index("Retroreflector (ideal)"), True)
    base, _ = geometry.build_target(biker, lib)
    with_tag, _ = geometry.build_target(biker + [tag], lib)
    r = RadarSettings(azimuth_deg=20.0, multibounce=False)
    res = compute_rcs(with_tag, r)
    assert res.retro_m2 > 0 and res.rcs_m2 > 3 * compute_rcs(base, r).rcs_m2
    assert compute_rcs(with_tag, RadarSettings(azimuth_deg=180.0, multibounce=False)).retro_m2 == 0.0
