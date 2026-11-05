"""Save and load a Radar Detector project (.rdproj).

A project is a single compressed NumPy archive: a JSON header (settings, materials, object
transforms) plus the arrays for each attachment's mesh and per-face materials. Attachment meshes
are stored in full, so a project opens even if the original STL has moved or been deleted.
The baseline biker is rebuilt from code, so only its per-face material assignments are stored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

from . import geometry as geo
from .materials import Material, MaterialLibrary

FORMAT = "radar-detector-project"
VERSION = 1
EXTENSION = ".rdproj"


@dataclass
class Project:
    lib: MaterialLibrary
    attachments: list[geo.SceneObject]
    biker_face_mats: dict[str, np.ndarray] = field(default_factory=dict)  # builtin object name -> face_mat
    settings: dict = field(default_factory=dict)  # GUI state: mode, radar settings, toggles


def save_project(path: str | Path, lib: MaterialLibrary, biker: list[geo.SceneObject],
                 attachments: list[geo.SceneObject], settings: dict) -> Path:
    path = Path(path)
    if path.suffix != EXTENSION:
        path = path.with_suffix(EXTENSION)
    header = {
        "format": FORMAT,
        "version": VERSION,
        "materials": [asdict(m) for m in lib.materials],
        "settings": settings,
        "biker": [o.name for o in biker],
        "attachments": [
            {"name": o.name, "position": [float(v) for v in o.position],
             "rotation_deg": [float(v) for v in o.rotation_deg], "scale": float(o.scale), "visible": bool(o.visible)}
            for o in attachments
        ],
    }
    arrays = {"header": np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)}
    for i, o in enumerate(biker):
        arrays[f"biker{i}_face_mat"] = o.face_mat.astype(np.int32)
    for i, o in enumerate(attachments):
        m = o.base_mesh
        arrays[f"att{i}_points"] = np.asarray(m.points, dtype=np.float64)
        arrays[f"att{i}_faces"] = m.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        arrays[f"att{i}_face_mat"] = o.face_mat.astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:  # a file object stops NumPy from appending .npz
        np.savez_compressed(f, **arrays)
    tmp.replace(path)  # write-then-rename, so a failed save never clobbers the previous file
    return path


def load_project(path: str | Path) -> Project:
    with np.load(Path(path), allow_pickle=False) as data:
        if "header" not in data:
            raise ValueError("Not a Radar Detector project (no header).")
        header = json.loads(bytes(data["header"]).decode())
        if header.get("format") != FORMAT:
            raise ValueError("Not a Radar Detector project.")
        if header.get("version", 0) > VERSION:
            raise ValueError(f"This project was saved by a newer version (format {header['version']}).")

        lib = MaterialLibrary(materials=[Material(**m) for m in header["materials"]])
        n_mat = len(lib)

        biker_face_mats = {}
        for i, name in enumerate(header.get("biker", [])):
            key = f"biker{i}_face_mat"
            if key in data:
                biker_face_mats[name] = np.clip(data[key].astype(int), 0, n_mat - 1)

        attachments = []
        for i, a in enumerate(header["attachments"]):
            pts, tri = data[f"att{i}_points"], data[f"att{i}_faces"]
            faces = np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri]).ravel()
            mesh = pv.PolyData(pts, faces)
            obj = geo.SceneObject(a["name"], mesh, np.clip(data[f"att{i}_face_mat"].astype(int), 0, n_mat - 1))
            obj.position = np.array(a["position"], float)
            obj.rotation_deg = np.array(a["rotation_deg"], float)
            obj.scale = float(a["scale"])
            obj.visible = bool(a.get("visible", True))
            attachments.append(obj)
    return Project(lib, attachments, biker_face_mats, header.get("settings", {}))
