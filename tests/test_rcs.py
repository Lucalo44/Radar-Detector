"""Check the solver against closed-form RCS results at 77 GHz."""

import numpy as np
import pyvista as pv
import pytest

from radar_detector.rcs import RadarSettings, Target, compute_rcs, triangle_phase_integral
from radar_detector import geometry

LAM = RadarSettings().wavelength


def _target(mesh: pv.PolyData, R=1.0, A=0.0, s=0.0) -> Target:
    faces = mesh.triangulate().faces.reshape(-1, 4)[:, 1:]
    m = len(faces)
    return Target(np.asarray(mesh.points, float), faces, np.full(m, R), np.full(m, A), np.full(m, s))


def _far(az=0.0, el=0.0, mb=False):
    return RadarSettings(distance_m=2000.0, azimuth_deg=az, elevation_deg=el, beam_width_deg=20.0, multibounce=mb)


def test_phase_integral_matches_quadrature():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=(3, 3)) * 0.01
        w = rng.normal(size=3) * 3000
        n = 600
        uu, vv = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
        m = uu + vv < 1
        pts = v[0] + uu[m][:, None] * (v[1] - v[0]) + vv[m][:, None] * (v[2] - v[0])
        c = v.mean(0)
        area = 0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))
        ref = np.exp(1j * (pts - c) @ w).mean() * area
        got = triangle_phase_integral(v[0:1], v[1:2], v[2:3], w[None])[0]
        assert abs(got - ref) < 2e-3 * area


def test_flat_plate_normal_incidence():
    a = 0.1
    plate = pv.Plane(center=(0, 0, 0), direction=(1, 0, 0), i_size=a, j_size=a, i_resolution=3, j_resolution=3)
    res = compute_rcs(_target(plate), _far(), aim_point=np.zeros(3))
    expected = 4 * np.pi * a**4 / LAM**2
    assert res.rcs_m2 == pytest.approx(expected, rel=0.01)


def test_flat_plate_reflectance_scales_power():
    a = 0.1
    plate = pv.Plane(direction=(1, 0, 0), i_size=a, j_size=a)
    res = compute_rcs(_target(plate, R=0.25, A=0.75), _far(), aim_point=np.zeros(3))
    assert res.rcs_m2 == pytest.approx(0.25 * 4 * np.pi * a**4 / LAM**2, rel=0.01)


def test_sphere_optical_region():
    r = 0.1
    sph = geometry.sphere((0, 0, 0), r)
    res = compute_rcs(_target(sph), _far(), aim_point=np.zeros(3))
    assert res.rcs_m2 == pytest.approx(np.pi * r**2, rel=0.35)


def test_trihedral_corner_reflector():
    a = 0.1
    cr = geometry.corner_reflector(a, opening=(-1, 0, 0))
    res = compute_rcs(_target(cr), _far(mb=True), aim_point=np.zeros(3))
    expected = 4 * np.pi * a**4 / (3 * LAM**2)
    # SBR with a finite ray grid: within ~1.5 dB of theory
    assert 10 * np.log10(res.rcs_m2 / expected) == pytest.approx(0.0, abs=1.5)
    assert res.multibounce_m2 > 10 * res.specular_m2


def test_absorber_hides_target():
    plate = pv.Plane(direction=(1, 0, 0), i_size=0.1, j_size=0.1)
    res = compute_rcs(_target(plate, R=0.0, A=1.0), _far(), aim_point=np.zeros(3))
    assert res.rcs_m2 < 1e-12


def test_shadowing_blocks_rear_plate():
    front = pv.Plane(center=(-0.05, 0, 0), direction=(1, 0, 0), i_size=0.2, j_size=0.2)
    back = pv.Plane(center=(0.05, 0, 0), direction=(1, 0, 0), i_size=0.1, j_size=0.1)
    absorber = _target(front, R=0.0, A=1.0)
    metal = _target(back)
    both = Target(
        np.vstack([absorber.vertices, metal.vertices]),
        np.vstack([absorber.faces, metal.faces + len(absorber.vertices)]),
        np.concatenate([absorber.reflectance, metal.reflectance]),
        np.concatenate([absorber.absorptance, metal.absorptance]),
        np.concatenate([absorber.roughness, metal.roughness]),
    )
    res = compute_rcs(both, _far(), aim_point=np.zeros(3))
    assert res.rcs_m2 < 1e-9
