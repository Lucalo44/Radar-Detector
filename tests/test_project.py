"""Project save/load round trip."""

import numpy as np
import pytest

from radar_detector import geometry as geo
from radar_detector.materials import MaterialLibrary
from radar_detector.project import load_project, save_project


def test_round_trip(tmp_path):
    lib = MaterialLibrary()
    custom = lib.add("Test foil", 0.8, 0.1, 0.2)
    biker = geo.build_biker(lib)
    biker[0].face_mat[:50] = custom  # a face-level edit on the baseline rider

    part = geo.make_attachment("Corner", geo.corner_reflector(0.1), lib.index("Aluminum"), True)
    part.position = np.array([0.1, 0.2, 1.3])
    part.rotation_deg = np.array([0.0, 15.0, 30.0])
    part.scale = 1.5
    part.visible = False
    part.face_mat[0] = custom
    settings = {"advanced": True, "distance_m": 12.5, "camera": [[1, 2, 3], [0, 0, 1], [0, 0, 1]]}

    path = save_project(tmp_path / "demo", lib, biker, [part], settings)
    assert path.suffix == ".rdproj" and path.exists()

    proj = load_project(path)
    assert proj.lib.names() == lib.names()
    assert proj.lib[custom].reflectance == pytest.approx(0.8)
    assert proj.settings == settings
    np.testing.assert_array_equal(proj.biker_face_mats[biker[0].name], biker[0].face_mat)

    (got,) = proj.attachments
    assert got.name == "Corner" and got.visible is False and got.scale == 1.5
    np.testing.assert_allclose(got.position, part.position)
    np.testing.assert_allclose(got.rotation_deg, part.rotation_deg)
    np.testing.assert_array_equal(got.face_mat, part.face_mat)
    np.testing.assert_allclose(got.world_mesh().points, part.world_mesh().points)
    np.testing.assert_array_equal(got.base_mesh.faces, part.base_mesh.faces)


def test_rejects_other_files(tmp_path):
    p = tmp_path / "x.rdproj"
    with open(p, "wb") as f:
        np.savez(f, a=np.zeros(3))
    with pytest.raises(ValueError):
        load_project(p)
