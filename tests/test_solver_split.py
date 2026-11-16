"""The illuminate / sum_fields split, band averaging and the shadow fast path.

`legacy_rcs.py` is a frozen copy of the solver from commit 69677f2, before it was split
into two stages. The split solver has to reproduce it exactly.
"""

import numpy as np
import pytest
import pyvista as pv

from radar_detector import geometry
from radar_detector.materials import MaterialLibrary
from radar_detector.rcs import (
    RadarSettings, Target, _shadow_transmission, _shadow_transmission_multi, compute_rcs, compute_rcs_band,
    illuminate, ray_engine, sum_fields,
)
from tests import legacy_rcs

FIELDS = ("rcs_m2", "specular_m2", "diffuse_m2", "multibounce_m2", "returned_fraction", "intercepted_fraction",
          "reflected_pct", "absorbed_pct", "passed_through_pct", "gain_dbi", "lit_faces", "rays_launched",
          "multibounce_rays")


def _target(mesh: pv.PolyData, R=1.0, A=0.0, s=0.0) -> Target:
    faces = mesh.triangulate().faces.reshape(-1, 4)[:, 1:]
    m = len(faces)
    return Target(np.asarray(mesh.points, float), faces, np.full(m, R), np.full(m, A), np.full(m, s))


def _legacy(t: Target) -> legacy_rcs.Target:
    return legacy_rcs.Target(t.vertices, t.faces, t.reflectance, t.absorptance, t.roughness, t.object_id,
                             t.object_names)


def _assert_same(new, old):
    for f in FIELDS:
        assert getattr(new, f) == pytest.approx(getattr(old, f), rel=1e-12, abs=1e-300), f
    np.testing.assert_allclose(new.face_rcs_m2, old.face_rcs_m2, rtol=1e-12, atol=0)
    # Per-object values changed on purpose (multi-bounce is now added coherently per object; see
    # test_single_object_value_equals_total), so only the keys are compared here.
    assert new.object_rcs_m2.keys() == old.object_rcs_m2.keys()
    assert new.notes == old.notes


def _far(az=0.0, mb=False):
    return dict(distance_m=2000.0, azimuth_deg=az, beam_width_deg=20.0, multibounce=mb)


@pytest.mark.parametrize("case", ["plate", "sphere", "trihedral"])
def test_split_matches_legacy_solver(case):
    if case == "plate":
        t, kw = _target(pv.Plane(direction=(1, 0, 0), i_size=0.1, j_size=0.1, i_resolution=3, j_resolution=3)), _far()
    elif case == "sphere":
        t, kw = _target(geometry.sphere((0, 0, 0), 0.1)), _far()
    else:
        t, kw = _target(geometry.corner_reflector(0.1, opening=(-1, 0, 0))), _far(mb=True)
    new = sum_fields(illuminate(t, RadarSettings(**kw), np.zeros(3)), RadarSettings().frequency_hz)
    old = legacy_rcs.compute_rcs(_legacy(t), legacy_rcs.RadarSettings(**kw), np.zeros(3))
    _assert_same(new, old)


@pytest.mark.parametrize("az", [0.0, 60.0, 180.0])
def test_split_matches_legacy_on_baseline_biker(az):
    """Shadowing with see-through materials, multi-bounce and per-object sums, all at once."""
    lib = MaterialLibrary()
    t, _ = geometry.build_target(geometry.build_biker(lib), lib)
    kw = dict(distance_m=10.0, azimuth_deg=az)
    _assert_same(compute_rcs(t, RadarSettings(**kw)), legacy_rcs.compute_rcs(_legacy(t), legacy_rcs.RadarSettings(**kw)))


def test_shadow_fast_path_matches_multi_hit():
    lib = MaterialLibrary()
    t, _ = geometry.build_target(geometry.build_biker(lib), lib)
    for az in (0.0, 90.0, 200.0):
        illum = illuminate(t, RadarSettings(azimuth_deg=az))
        p_r = illum.radar_position
        V, F = t.vertices, t.faces
        cent = V[F].mean(axis=1)
        ids = np.arange(len(F))
        dist = np.linalg.norm(cent - p_r, axis=1)
        fast = _shadow_transmission(t.bvh(), p_r, cent, dist, ids, t.transmittance)
        slow = _shadow_transmission_multi(t.bvh(), p_r, cent, dist, ids, t.transmittance)
        lit_fast, lit_slow = fast > 1e-6, slow > 1e-6
        np.testing.assert_array_equal(lit_fast, lit_slow)
        np.testing.assert_allclose(fast[lit_fast], slow[lit_slow], rtol=1e-12)


def test_band_average_of_plate_is_flat():
    """A plate's RCS barely changes over 1 GHz, so the band average sits on the single-frequency value."""
    t = _target(pv.Plane(direction=(1, 0, 0), i_size=0.1, j_size=0.1, i_resolution=3, j_resolution=3))
    r = RadarSettings(**_far())
    single = compute_rcs(t, r, np.zeros(3)).rcs_dbsm
    band = compute_rcs_band(t, r, np.zeros(3))
    assert band.rcs_dbsm == pytest.approx(single, abs=0.1)
    assert len(band.frequencies_hz) == r.n_freq


def test_band_average_smooths_two_plate_interference():
    """Two plates 20 cm apart in depth beat against each other across the band; the average sits in between."""
    a = _target(pv.Plane(center=(0, 0, 0), direction=(1, 0, 0), i_size=0.05, j_size=0.05))
    b = _target(pv.Plane(center=(0.2, 0, 0.1), direction=(1, 0, 0), i_size=0.05, j_size=0.05))
    t = Target(np.vstack([a.vertices, b.vertices]), np.vstack([a.faces, b.faces + len(a.vertices)]),
               np.ones(len(a.faces) * 2), np.zeros(len(a.faces) * 2), np.zeros(len(a.faces) * 2))
    r = RadarSettings(**_far())
    illum = illuminate(t, r, np.array([0.1, 0, 0.05]))
    singles = np.array([sum_fields(illum, f).rcs_m2 for f in r.band_frequencies()])
    assert singles.max() / singles.min() > 3  # the interference really does swing across the band
    band = compute_rcs_band(t, r, np.array([0.1, 0, 0.05]))
    assert singles.min() < band.rcs_m2 < singles.max()
    assert band.spread_db > 5


def test_zero_bandwidth_is_single_frequency():
    t = _target(geometry.sphere((0, 0, 0), 0.05))
    r = RadarSettings(**_far(), bandwidth_hz=0.0)
    assert compute_rcs_band(t, r, np.zeros(3)).rcs_m2 == compute_rcs(t, r, np.zeros(3)).rcs_m2


def test_ray_engine_reports_without_raising():
    name, embree = ray_engine()
    assert isinstance(name, str) and name
    assert isinstance(embree, bool)


def test_single_object_value_equals_total():
    """A lone corner reflector's per-object RCS is the scene total, multi-bounce included."""
    t = _target(geometry.corner_reflector(0.1, opening=(-1, 0, 0)))
    res = compute_rcs(t, RadarSettings(**_far(mb=True)), np.zeros(3))
    (val,) = res.object_rcs_m2.values()
    assert val == pytest.approx(res.rcs_m2, rel=1e-9)


def test_corner_reflector_dominates_object_table():
    """With a corner reflector behind the saddle, the table has to credit the corner, not the wheels."""
    lib = MaterialLibrary()
    biker = geometry.build_biker(lib)
    part = geometry.make_attachment("Corner", geometry.corner_reflector(0.1), lib.index("Aluminum"), True)
    t, _ = geometry.build_target(biker + [part], lib)
    res = compute_rcs(t, RadarSettings())
    assert max(res.object_rcs_m2, key=res.object_rcs_m2.get) == "Corner"
