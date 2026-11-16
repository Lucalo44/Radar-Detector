"""Δ vs. baseline, equivalent sphere, wording, table rows, heat scaling and the baseline cache."""

import numpy as np
import pytest

from radar_detector import compare as cmp
from radar_detector import geometry as geo
from radar_detector.materials import MaterialLibrary
from radar_detector.rcs import RadarSettings, RcsResult


def _res(total, objects, face=None):
    face = np.zeros(4) if face is None else np.asarray(face, float)
    return RcsResult(total, 0, 0, 0, 0, 0, 0, 0, 0, 0, face, objects, np.zeros(3), np.zeros(3), 0, 0, 0)


def test_delta_and_ratio():
    assert cmp.delta_db(2.0, 1.0) == pytest.approx(3.0103, abs=1e-4)
    assert cmp.delta_db(1.0, 10.0) == pytest.approx(-10.0)
    assert cmp.visibility(3.4) == "more" and cmp.visibility(0.9) == "same" and cmp.visibility(-1.5) == "less"
    assert cmp.ratio_text(3.4) == "about 2.2× the return"
    assert cmp.ratio_text(0.5) == "about the same return"
    assert cmp.ratio_text(-3.0) == "about 50% less return"


def test_equivalent_sphere():
    # A sphere of radius r has RCS pi r^2, so it has to come back as its own diameter
    for r in (0.05, 0.5, 2.0):
        assert cmp.equivalent_sphere_m(np.pi * r**2) == pytest.approx(2 * r)
    assert cmp.equivalent_sphere_m(1.62) == pytest.approx(1.436, abs=1e-3)
    assert cmp.fmt_length(1.436) == "1.4 m" and cmp.fmt_length(0.35) == "35 cm"


def test_formatting():
    assert cmp.fmt_rcs(1.62) == "+2.1 dBsm"
    assert cmp.fmt_rcs(1.62, "m²") == "1.62 m²"
    assert cmp.band_text(RadarSettings()) == "averaged over 76.5–77.5 GHz (9 frequencies)"
    assert cmp.sentence_case("averaged over 76.5–77.5 GHz") == "Averaged over 76.5–77.5 GHz"
    assert "single frequency" in cmp.band_text(RadarSettings(bandwidth_hz=0))
    assert cmp.reference_label(5.0) == "in the typical cyclist range"
    assert cmp.reference_label(0.0) == "in the typical pedestrian / cyclist range"


def test_object_rows():
    res = _res(10.0, {"Rider": 1.0, "Corner": 8.0, "Frame": 1.0})
    rows = cmp.object_rows(res, _res(2.0, {}), {"Corner"})
    assert [r.name for r in rows] == ["Corner", "Rider", "Frame"]
    assert rows[0].share_pct == pytest.approx(80.0)
    assert rows[0].vs_base_db == pytest.approx(10 * np.log10(10.0 / 2.0))
    assert rows[1].vs_base_db is None


def test_heat_values_match_object_totals():
    res = _res(10.0, {"A": 9.0, "B": 1.0}, face=[1.0, 2.0, 0.5, 0.5])
    h = cmp.heat_values(res, [("A", 0, 2), ("B", 2, 4)])
    assert h[:2].sum() == pytest.approx(9.0) and h[2:].sum() == pytest.approx(1.0)
    assert h[1] / h[0] == pytest.approx(2.0)  # the spread within an object is kept


def test_baseline_key_invalidates_on_edits():
    lib = MaterialLibrary()
    biker = geo.build_biker(lib)
    radar, aim = RadarSettings(), np.array([0.5, 0, 0.85])
    key = lambda: cmp.baseline_key(radar, [o.face_mat for o in biker], cmp.library_values(lib), aim)  # noqa: E731
    k0 = key()
    assert key() == k0  # stable

    cache = cmp.BaselineCache()
    cache.put(k0, _res(1.0, {}))
    lib[lib.index("Aluminum")].reflectance = 0.5  # material value edit
    assert key() != k0 and cache.get(key()) is None
    lib[lib.index("Aluminum")].reflectance = 0.99
    assert key() == k0 and cache.get(key()) is not None

    biker[1].face_mat[0] = lib.index("Steel")  # face material edit on the biker
    assert key() != k0
    biker[1].face_mat[0] = lib.index("Aluminum")
    assert key() == k0

    radar.distance_m = 12.0  # radar setting
    assert key() != k0


def test_cache_keeps_recent_only():
    c = cmp.BaselineCache(size=2)
    for k in "abc":
        c.put(k, _res(1.0, {}))
    assert c.get("a") is None and c.get("c") is not None and len(c) == 2
