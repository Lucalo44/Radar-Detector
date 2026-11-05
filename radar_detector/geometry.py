"""Scene geometry: primitives, the baseline biker, imported attachments.

Coordinate frame: metres, +X is the direction of travel, +Y is the rider's
left, +Z is up, and the road surface is z = 0.

Curved primitives are tessellated so that each flat facet is narrower than
the first Fresnel zone of its curvature (w ~ sqrt(r lambda / 2)). Coarser
facets turn smooth tubes into a row of flat mirrors and give spiky,
overestimated glints at 77 GHz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

from .materials import MaterialLibrary
from .rcs import RadarSettings, Target

LAMBDA = RadarSettings().wavelength
HEAD_CENTER = np.array([0.75, 0.0, 1.61])  # baseline rider: shoulder (0.62, 0, 1.36) + neck offset
HEAD_RADIUS = 0.105
UNIT_SCALES = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _segments(radius: float, lo: int = 12, hi: int = 96) -> int:
    fresnel = np.sqrt(radius * LAMBDA / 2.0)
    return int(np.clip(np.ceil(2 * np.pi * radius / fresnel), lo, hi))


def _clean(mesh: pv.PolyData) -> pv.PolyData:
    return mesh.triangulate().clean(tolerance=1e-9)


def tube(p0, p1, radius: float, capping: bool = True) -> pv.PolyData:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    cyl = pv.Cylinder(
        center=(p0 + p1) / 2, direction=axis / length, radius=radius, height=length,
        resolution=_segments(radius), capping=capping,
    )
    return _clean(cyl)


def sphere(center, radius: float) -> pv.PolyData:
    n = _segments(radius, 16, 120)
    return _clean(pv.Sphere(radius=radius, center=center, theta_resolution=n, phi_resolution=max(n // 2, 8)))


def capsule(p0, p1, radius: float) -> pv.PolyData:
    return pv.merge([tube(p0, p1, radius, capping=False), sphere(p0, radius), sphere(p1, radius)])


def torus(center, ring_radius: float, cross_radius: float, axis=(0, 1, 0)) -> pv.PolyData:
    t = pv.ParametricTorus(
        ringradius=ring_radius, crosssectionradius=cross_radius,
        u_res=_segments(ring_radius, 24, 160), v_res=_segments(cross_radius, 10, 48), w_res=2,
    )
    t = _clean(t.extract_surface(algorithm="dataset_surface"))
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(z, axis)
    if np.linalg.norm(rot_axis) > 1e-9:
        ang = np.degrees(np.arccos(np.clip(z @ axis, -1, 1)))
        t = t.rotate_vector(rot_axis, ang, point=(0, 0, 0))
    return t.translate(center)


def _rotation_to(src, dst) -> np.ndarray:
    a = np.asarray(src, float) / np.linalg.norm(src)
    b = np.asarray(dst, float) / np.linalg.norm(dst)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def corner_reflector(edge: float = 0.1, opening=(-1, 0, 0)) -> pv.PolyData:
    """Triangular trihedral corner reflector with its mouth facing `opening`."""
    Rm = _rotation_to((1, 1, 1), opening)
    e = (np.eye(3) * edge) @ Rm.T
    pts = np.vstack([np.zeros(3), e])
    pts -= pts.mean(0)
    faces = np.array([[3, 0, 1, 2], [3, 0, 2, 3], [3, 0, 3, 1]]).ravel()
    m = pv.PolyData(pts, faces)
    # Subdivide so the shadowing test (done per face centroid) resolves partial shadowing
    return m.subdivide(3, subfilter="linear")


def flat_plate(size: float = 0.1, normal=(-1, 0, 0)) -> pv.PolyData:
    return _clean(pv.Plane(direction=normal, i_size=size, j_size=size, i_resolution=4, j_resolution=4))


# --------------------------------------------------------------------------- #
# Scene objects
# --------------------------------------------------------------------------- #
@dataclass
class SceneObject:
    name: str
    base_mesh: pv.PolyData  # triangulated, local coordinates
    face_mat: np.ndarray  # (M,) int material index per face
    builtin: bool = False
    visible: bool = True
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    _cache_key: tuple | None = field(default=None, repr=False)
    _cache: pv.PolyData | None = field(default=None, repr=False)

    @property
    def n_faces(self) -> int:
        return self.base_mesh.n_cells

    def world_mesh(self) -> pv.PolyData:
        key = (tuple(np.round(self.position, 9)), tuple(np.round(self.rotation_deg, 9)), round(self.scale, 12))
        if self._cache is None or self._cache_key != key:
            m = self.base_mesh.copy()
            m.points = m.points * self.scale
            rx, ry, rz = self.rotation_deg
            if rx:
                m = m.rotate_x(rx, point=(0, 0, 0))
            if ry:
                m = m.rotate_y(ry, point=(0, 0, 0))
            if rz:
                m = m.rotate_z(rz, point=(0, 0, 0))
            m = m.translate(self.position)
            self._cache, self._cache_key = m, key
        return self._cache

    def set_material(self, mat_index: int) -> None:
        self.face_mat[:] = mat_index


def _parts_object(name: str, parts: list[tuple[pv.PolyData, int]]) -> SceneObject:
    meshes, mats = [], []
    for mesh, mat in parts:
        meshes.append(mesh)
        mats.append(np.full(mesh.n_cells, mat, dtype=int))
    merged = pv.merge(meshes, merge_points=False)
    return SceneObject(name, merged, np.concatenate(mats), builtin=True)


def build_biker(lib: MaterialLibrary) -> list[SceneObject]:
    """A barebones rider on a road bike, built from tubes, spheres and tori."""
    al, steel, rubber = lib.index("Aluminum"), lib.index("Steel"), lib.index("Tire rubber")
    body, skin, saddle = lib.index("Clothed body"), lib.index("Skin"), lib.index("Saddle foam / leather")

    wheel_r, axle_z = 0.335, 0.335
    RA, FA = np.array([0.0, 0, axle_z]), np.array([1.0, 0, axle_z])
    BB = np.array([0.40, 0, 0.27])
    ST = np.array([0.26, 0, 0.78])  # seat tube top
    HT_top, HT_bot = np.array([0.88, 0, 0.76]), np.array([0.91, 0, 0.62])
    SADDLE = np.array([0.22, 0, 0.93])
    BAR = np.array([0.98, 0, 0.88])

    # ---- wheels ----
    wheels = []
    for c in (RA, FA):
        wheels.append((torus(c, wheel_r - 0.014, 0.014), rubber))
        wheels.append((torus(c, wheel_r - 0.040, 0.012), al))
        wheels.append((tube(c + [0, -0.05, 0], c + [0, 0.05, 0], 0.02), al))
        for i in range(16):
            ang = 2 * np.pi * i / 16
            rim = c + (wheel_r - 0.045) * np.array([np.cos(ang), 0, np.sin(ang)])
            hub = c + np.array([0, 0.035 * (1 if i % 2 else -1), 0])
            wheels.append((tube(hub, rim, 0.0012, capping=False), steel))

    # ---- frame ----
    frame = [
        (tube(BB, ST, 0.016), al),
        (tube(ST, HT_top, 0.016), al),
        (tube(BB, HT_bot, 0.020), al),
        (tube(HT_bot, HT_top, 0.022), al),
        (tube(ST, SADDLE + [0.0, 0, -0.03], 0.0135), al),
        (tube(HT_top, BAR + [-0.02, 0, 0], 0.014), al),
        (tube(BAR + [0, -0.21, 0], BAR + [0, 0.21, 0], 0.012), al),
        (tube(BB + [0, -0.075, 0], BB + [0, 0.075, 0], 0.018), al),
    ]
    for side in (-1, 1):
        off = np.array([0, 0.055 * side, 0])
        frame += [
            (tube(BB + off, RA + off, 0.010), al),
            (tube(ST + off * 0.5, RA + off, 0.009), al),
            (tube(HT_bot + off * 0.4, FA + off * 0.9, 0.012), al),
            (tube(BAR + [0, 0.21 * side, 0], BAR + [0.06, 0.21 * side, -0.12], 0.012), al),  # drops
        ]
    crank_a = BB + np.array([0.12, -0.09, -0.08])
    crank_b = BB + np.array([-0.12, 0.09, 0.08])
    frame += [
        (tube(BB + [0, -0.09, 0], crank_a, 0.008), al),
        (tube(BB + [0, 0.09, 0], crank_b, 0.008), al),
        (torus(BB + [0, 0.07, 0], 0.10, 0.004), steel),  # chainring
    ]
    # Rounded saddle (a flat-backed box would act as a mirror aimed at a rear radar)
    saddle_mesh = _clean(pv.Sphere(radius=1.0, theta_resolution=72, phi_resolution=36))
    saddle_mesh.points = saddle_mesh.points * np.array([0.13, 0.07, 0.025]) + SADDLE
    frame.append((saddle_mesh, saddle))

    # ---- rider ----
    hip = np.array([0.24, 0, 1.00])
    shoulder = np.array([0.62, 0, 1.36])
    rider = [
        (capsule(hip, shoulder, 0.15), body),
        (tube(shoulder + [0.03, 0, 0.05], shoulder + [0.07, 0, 0.13], 0.055), skin),  # neck
        (sphere(HEAD_CENTER, HEAD_RADIUS), skin),  # head
    ]
    for side in (-1, 1):
        sh = shoulder + [0, 0.18 * side, 0.0]
        elbow = np.array([0.80, 0.22 * side, 1.10])
        hand = np.array([1.00, 0.21 * side, 0.88])
        h = hip + [0, 0.10 * side, -0.02]
        pedal = crank_a if side < 0 else crank_b
        knee = np.array([0.62, 0.12 * side, 0.68]) if side < 0 else np.array([0.50, 0.12 * side, 0.80])
        foot = pedal + [0, 0.03 * side, 0.04]
        rider += [
            (capsule(sh, elbow, 0.05), body),
            (capsule(elbow, hand, 0.04), skin),
            (capsule(h, knee, 0.075), body),
            (capsule(knee, foot, 0.055), body),
        ]
    return [
        _parts_object("Rider (baseline)", rider),
        _parts_object("Bike frame (baseline)", frame),
        _parts_object("Wheels (baseline)", wheels),
    ]


# --------------------------------------------------------------------------- #
# Imports / attachments
# --------------------------------------------------------------------------- #
DEFAULT_ATTACH_POINT = np.array([0.10, 0.0, 0.82])  # behind the saddle, above the rear wheel


def load_mesh(path: str | Path, units: str = "mm") -> pv.PolyData:
    """Read STL / OBJ / PLY / VTK / GLTF and return a triangulated, centred surface."""
    data = pv.read(str(path))
    if isinstance(data, pv.MultiBlock):
        data = data.combine()
    if not isinstance(data, pv.PolyData):
        data = data.extract_surface()
    mesh = _clean(data)
    if mesh.n_cells == 0:
        raise ValueError("The file contains no surface triangles.")
    mesh.points = mesh.points * UNIT_SCALES[units]
    mesh.points = mesh.points - np.array(mesh.center)
    return mesh


def guess_units(raw_extent: float) -> str:
    """Pick the unit that makes a part's largest dimension a plausible bike accessory (2cm-2m)."""
    for units in ("mm", "cm", "in", "m"):
        if 0.02 <= raw_extent * UNIT_SCALES[units] <= 2.0:
            return units
    return "mm"


def make_attachment(name: str, mesh: pv.PolyData, mat_index: int, biker_present: bool,
                    mount: str = "saddle") -> SceneObject:
    """Wrap a centred mesh as a movable attachment.

    mount="saddle" puts it behind the saddle; mount="head" sits it flush against the back of the
    rider's head, facing a rear radar.
    """
    obj = SceneObject(name, mesh, np.full(mesh.n_cells, mat_index, dtype=int))
    if not biker_present:
        obj.position = np.array([0.5, 0.0, 0.8])
    elif mount == "head":
        half_depth = mesh.bounds[1]  # mesh is centred, so its +X extent is half its depth
        obj.position = HEAD_CENTER - np.array([HEAD_RADIUS + half_depth, 0.0, 0.0])
    else:
        obj.position = DEFAULT_ATTACH_POINT.copy()
    return obj


# --------------------------------------------------------------------------- #
# Scene -> solver target
# --------------------------------------------------------------------------- #
def build_target(objects: list[SceneObject], lib: MaterialLibrary):
    """Merge visible objects into a solver Target.

    Returns (target, spans) where spans[i] = (object, start_face, end_face).
    """
    R_lib, A_lib, s_lib, _ = lib.arrays()
    verts, faces, mats, oids, names, spans = [], [], [], [], [], []
    v_off = f_off = 0
    for obj in objects:
        if not obj.visible or obj.n_faces == 0:
            continue
        m = obj.world_mesh()
        f = m.faces.reshape(-1, 4)[:, 1:]
        verts.append(np.asarray(m.points, float))
        faces.append(f + v_off)
        mats.append(obj.face_mat)
        oids.append(np.full(len(f), len(names), dtype=int))
        names.append(obj.name)
        spans.append((obj, f_off, f_off + len(f)))
        v_off += m.n_points
        f_off += len(f)
    if not faces:
        return None, []
    mat = np.concatenate(mats)
    target = Target(
        vertices=np.vstack(verts), faces=np.vstack(faces),
        reflectance=R_lib[mat], absorptance=A_lib[mat], roughness=s_lib[mat],
        object_id=np.concatenate(oids), object_names=names,
    )
    return target, spans
