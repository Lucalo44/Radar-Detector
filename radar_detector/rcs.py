"""Monostatic radar cross section (RCS) solver.

The target is a triangle mesh in which every face carries a material
(reflectance R, absorptance A, roughness s). Three scattering mechanisms are
summed:

1. Specular single bounce, by physical optics (PO). Every illuminated,
   unshadowed face contributes the exact closed-form integral of the
   round-trip phase over its triangle, so faces larger than a wavelength are
   handled correctly. Contributions add coherently.

       sigma_spec = (4 pi / lambda^2) |sum_i Gamma_i cos(theta_i) w_i I_i e^{-j 2k R_i}|^2

2. Diffuse single bounce. The rough fraction s of each face's reflected
   power is scattered as a Lambertian surface (monostatic sigma = 4 rho A
   cos^2 theta) and adds incoherently.

3. Retrodirective return, for faces with a retroreflective material (tags such
   as metalens reflectors or Van Atta arrays). Each connected patch of such
   faces acts as one phase-conjugating aperture: its lit faces re-radiate
   straight back at the radar with a common phase, so the patch returns
   sigma = (4 pi / lambda^2) |sum_i sqrt(R_i q_i) A_i cos(theta_i) w_i g_i|^2,
   with g a taper to zero outside the material's coverage half-angle.

4. Multi-bounce, by shooting and bouncing rays (SBR). A grid of ray tubes is
   launched from the radar; tubes that reflect two or more times are summed
   coherently with the PO field using a ray-tube aperture integral. This is
   what captures corner reflectors and dihedrals, which single-bounce PO
   misses entirely.

Spherical-wave illumination (true per-face direction and range), a Gaussian
beam pattern with a user-set -3 dB cone, and shadowing (with partial
transmission through low-loss dielectrics) are included. Not included:
ground-plane multipath, angle-dependent Fresnel coefficients, polarization
and creeping/edge-diffraction waves.

The solve runs in two stages. `illuminate` does everything that doesn't
depend on frequency (geometry, beam weights, shadow rays and the full SBR ray
paths). `sum_fields` turns that into an RCS at one frequency, which is only
NumPy sums. `compute_rcs_band` reuses one illumination for every frequency in
the radar's sweep, so band averaging costs little more than a single solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import trimesh

C0 = 299_792_458.0
DEFAULT_FREQ_HZ = 77e9
LN2 = np.log(2.0)
NEAR_FIELD = "Target is in the radar's near field"
OPAQUE_T = 1e-6
RETRO_ROLLOFF_DEG = 10.0  # raised-cosine taper from full return at the half-angle to none this far past it  # below this transmittance a blocker leaves a face unlit (same threshold as the lit test)


# --------------------------------------------------------------------------- #
# Inputs / outputs
# --------------------------------------------------------------------------- #
@dataclass
class RadarSettings:
    distance_m: float = 10.0
    azimuth_deg: float = 0.0  # 0 = radar directly behind the rider, 90 = rider's left, 180 = head-on
    elevation_deg: float = 0.0  # + = radar above the aim point, looking down
    beam_width_deg: float = 20.0  # full -3 dB cone angle
    frequency_hz: float = DEFAULT_FREQ_HZ
    multibounce: bool = True
    max_bounces: int = 3
    ray_spacing_m: float = 0.005
    max_rays: int = 400_000
    bandwidth_hz: float = 1e9  # FMCW sweep the result is averaged over (0 = single frequency)
    n_freq: int = 9
    aspect_avg_deg: float = 0.0  # > 0 also averages az - d, az, az + d

    @property
    def wavelength(self) -> float:
        return C0 / self.frequency_hz

    @property
    def k(self) -> float:
        return 2.0 * np.pi / self.wavelength

    @property
    def gain(self) -> float:
        """Antenna gain of an ideal Gaussian beam with this -3 dB cone (linear)."""
        theta = np.radians(max(self.beam_width_deg, 0.1))
        return float(max(16.0 * LN2 / theta**2, 1.0))

    def band_frequencies(self) -> np.ndarray:
        if self.bandwidth_hz <= 0 or self.n_freq <= 1:
            return np.array([self.frequency_hz])
        half = 0.5 * self.bandwidth_hz
        return np.linspace(self.frequency_hz - half, self.frequency_hz + half, int(self.n_freq))

    def look_direction(self) -> np.ndarray:
        """Unit vector from the target toward the radar."""
        a, e = np.radians(self.azimuth_deg), np.radians(self.elevation_deg)
        # Rider travels along +X, +Y is the rider's left, +Z is up.
        return np.array([-np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])

    def position(self, aim: np.ndarray) -> np.ndarray:
        return np.asarray(aim, float) + self.distance_m * self.look_direction()


@dataclass(eq=False)
class Target:
    vertices: np.ndarray  # (N, 3) float, metres
    faces: np.ndarray  # (M, 3) int
    reflectance: np.ndarray  # (M,)
    absorptance: np.ndarray  # (M,)
    roughness: np.ndarray  # (M,)
    object_id: np.ndarray | None = None  # (M,) int
    object_names: list[str] = field(default_factory=list)
    retro: np.ndarray | None = None  # (M,) retroreflective share of reflected power (None = all zero)
    retro_half_angle_deg: np.ndarray | None = None  # (M,) retro coverage half-angle
    # Ray-casting acceleration structure, built on first use and reused by every solve on this
    # Target (sweeps and charts call the solver hundreds of times on the same mesh). Treat a
    # Target's arrays as read-only once it has been solved.
    _bvh: object = field(default=None, init=False, repr=False)

    @property
    def transmittance(self) -> np.ndarray:
        return np.clip(1.0 - self.reflectance - self.absorptance, 0.0, 1.0)

    def bvh(self) -> trimesh.Trimesh:
        if self._bvh is None:
            self._bvh = _build_tm(self)
        return self._bvh

    def __getstate__(self):  # the Embree scene can't be pickled; worker processes rebuild it
        state = self.__dict__.copy()
        state["_bvh"] = None
        return state


@dataclass
class RcsResult:
    rcs_m2: float
    specular_m2: float
    diffuse_m2: float
    multibounce_m2: float
    returned_fraction: float  # P_received / P_transmitted
    intercepted_fraction: float  # P_on_target / P_transmitted
    reflected_pct: float  # of intercepted power
    absorbed_pct: float
    passed_through_pct: float
    gain_dbi: float
    face_rcs_m2: np.ndarray  # (M,) incoherent per-face contribution, for heat maps
    object_rcs_m2: dict[str, float]
    radar_position: np.ndarray
    aim_point: np.ndarray
    lit_faces: int
    rays_launched: int
    multibounce_rays: int
    notes: list[str] = field(default_factory=list)
    frequencies_hz: np.ndarray | None = None  # frequencies averaged (None = the single frequency)
    aspects_deg: np.ndarray | None = None  # azimuths averaged (None = the single azimuth)
    samples_rcs_m2: np.ndarray | None = None  # total RCS at each averaged sample, before averaging
    retro_m2: float = 0.0  # retrodirective part on its own (adds coherently with the rest)

    @property
    def rcs_dbsm(self) -> float:
        return 10.0 * np.log10(max(self.rcs_m2, 1e-30))

    @property
    def returned_pct(self) -> float:
        return 100.0 * self.returned_fraction

    @property
    def returned_db(self) -> float:
        return 10.0 * np.log10(max(self.returned_fraction, 1e-300))

    @property
    def spread_db(self) -> float:
        """Max-min of the averaged samples in dB: how much the value moves within the band."""
        if self.samples_rcs_m2 is None or len(self.samples_rcs_m2) < 2:
            return 0.0
        db = 10 * np.log10(np.maximum(self.samples_rcs_m2, 1e-30))
        return float(db.max() - db.min())


# --------------------------------------------------------------------------- #
# Maths helpers
# --------------------------------------------------------------------------- #
def _norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=-1)


def triangle_phase_integral(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Exact integral of exp(j w . (r - c)) over each triangle (c = centroid).

    Uses the divided-difference form of the simplex integral, which is
    numerically stable when two or three vertex phases coincide:
        I = 2 A g[p0, p1, p2],  g(x) = -exp(jx)
    with g[a, b] = -j exp(j(a+b)/2) sinc((b-a)/2) evaluated without cancellation.
    """
    c = (v0 + v1 + v2) / 3.0
    p = np.stack([np.einsum("ij,ij->i", w, v - c) for v in (v0, v1, v2)], axis=1)
    p.sort(axis=1)
    a, b, cc = p[:, 0], p[:, 1], p[:, 2]
    area = 0.5 * _norm(np.cross(v1 - v0, v2 - v0))

    def first_dd(x, y):
        return -1j * np.exp(0.5j * (x + y)) * np.sinc((y - x) / (2.0 * np.pi))

    d1, d2 = first_dd(a, b), first_dd(b, cc)
    span = cc - a
    out = 0.5 * np.exp(1j * (a + b + cc) / 3.0)
    big = span > 1e-9
    out[big] = (d2[big] - d1[big]) / span[big]
    return 2.0 * area * out


def _orthonormal_basis(d: np.ndarray):
    d = d / np.linalg.norm(d)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def _beam_power(psi: np.ndarray, beam_width_rad: float) -> np.ndarray:
    """One-way Gaussian power pattern, -3 dB at psi = beam_width / 2."""
    return np.exp(-4.0 * LN2 * (psi / beam_width_rad) ** 2)


def retro_taper(theta_deg: np.ndarray, half_angle_deg: np.ndarray) -> np.ndarray:
    """Field weight of a retroreflective face vs. incidence angle: 1 inside the coverage half-angle,
    a raised-cosine roll-off over RETRO_ROLLOFF_DEG past it, then 0."""
    x = np.clip((np.asarray(theta_deg) - half_angle_deg) / RETRO_ROLLOFF_DEG, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def _patches(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Connected-component label for each face, joining faces that share a vertex."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    m = len(faces)
    rows = np.repeat(np.arange(m), 3)
    cols = m + faces.ravel()
    n = m + n_vertices
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels[:m]


def _build_tm(target: Target) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=target.vertices, faces=target.faces, process=False, validate=False)


def ray_engine() -> tuple[str, bool]:
    """Name of trimesh's active ray intersector and whether it is the Embree one.

    Without Embree, trimesh silently falls back to a pure-Python intersector that is
    roughly 100x slower, which turns a sub-second solve into 20 s or more.
    """
    tm = trimesh.creation.box()
    ray = tm.ray
    name = f"{type(ray).__module__}.{type(ray).__name__}"
    return name, "pyembree" in type(ray).__module__ or "embree" in type(ray).__module__


# --------------------------------------------------------------------------- #
# Shadowing
# --------------------------------------------------------------------------- #
def _shadow_transmission_multi(tm, origin, centroids, dist, face_ids, T_face):
    """One-way power transmission from the radar to each face centroid (all hits).

    Every surface crossed before the face multiplies the path by its
    transmittance, so an opaque surface (T = 0) fully shadows the face and a
    thin plastic shell (T ~ 0.9) only dims it.
    """
    n = len(face_ids)
    if n == 0:
        return np.ones(0)
    dirs = (centroids - origin) / dist[:, None]
    origins = np.repeat(origin[None, :], n, axis=0)
    tri, ray, loc = tm.ray.intersects_id(origins, dirs, multiple_hits=True, max_hits=32, return_locations=True)
    if len(tri) == 0:
        return np.ones(n)
    hit_d = _norm(loc - origin)
    tol = 1e-4 + 1e-6 * dist[ray]
    blocking = (hit_d < dist[ray] - tol) & (tri != face_ids[ray])
    log_t = np.zeros(n)
    np.add.at(log_t, ray[blocking], np.log(np.clip(T_face[tri[blocking]], 1e-12, 1.0)))
    return np.exp(log_t)


def _shadow_transmission(tm, origin, centroids, dist, face_ids, T_face):
    """Same answer as `_shadow_transmission_multi` for every face that ends up lit, but cheaper.

    A single first-hit cast settles most rays: either nothing sits in front of the face
    (transmission 1), or the first thing in front is opaque (the face is unlit, whatever lies
    behind). Only rays whose first blocker is see-through need the full multi-hit cast.
    """
    n = len(face_ids)
    if n == 0:
        return np.ones(0)
    dirs = (centroids - origin) / dist[:, None]
    origins = np.repeat(origin[None, :], n, axis=0)
    tri, ray, loc = tm.ray.intersects_id(origins, dirs, multiple_hits=False, return_locations=True)
    out = np.ones(n)
    if len(tri) == 0:
        return out
    hit_d = _norm(loc - origin)
    tol = 1e-4 + 1e-6 * dist[ray]
    blocking = (hit_d < dist[ray] - tol) & (tri != face_ids[ray])
    b_ray, b_tri = ray[blocking], tri[blocking]
    opaque = T_face[b_tri] < OPAQUE_T
    out[b_ray[opaque]] = 0.0
    redo = b_ray[~opaque]
    if len(redo):
        out[redo] = _shadow_transmission_multi(tm, origin, centroids[redo], dist[redo], face_ids[redo], T_face)
    return out


# --------------------------------------------------------------------------- #
# Shooting and bouncing rays (multi-bounce only)
# --------------------------------------------------------------------------- #
@dataclass
class _SbrExits:
    """Rays leaving a 2nd+ reflection toward the radar, from one bounce step.

    Everything here is frequency-independent. The aperture factor F depends on k, so it
    is formed in `_sbr_field` from `dout`, `s_dir` and `tube`, not stored.
    """

    amp: np.ndarray  # field amplitude product (beam x reflections x transmissions)
    tube: np.ndarray  # ray-tube width at the exit face
    first_r: np.ndarray  # range to the first hit
    back: np.ndarray  # range from the exit point back to the radar
    path: np.ndarray  # path length from the radar to the exit point
    dout: np.ndarray  # reflected direction
    s_dir: np.ndarray  # direction from the exit point to the radar
    face: np.ndarray  # exit face index
    clear: np.ndarray  # exit path back to the radar is unobstructed


def _sbr_trace(tm, target, radar: RadarSettings, p_r, aim, normals, gamma_spec, T_face, notes):
    """Trace the ray-tube grid. Returns (exits, n_rays)."""
    bore = aim - p_r
    R0 = np.linalg.norm(bore)
    bore /= R0
    u, v = _orthonormal_basis(bore)
    theta3 = np.radians(radar.beam_width_deg)

    # Project every vertex onto the aim plane (perspective from the radar)
    rel = target.vertices - p_r
    depth = rel @ bore
    front = depth > 1e-6
    if not np.any(front):
        return [], 0
    proj = rel[front] * (R0 / depth[front])[:, None]
    pu, pv = proj @ u, proj @ v
    beam_r = R0 * np.tan(min(theta3, np.radians(170)))  # rays beyond ~2x the -3 dB half-width carry < 6% field
    umin, umax = max(pu.min(), -beam_r), min(pu.max(), beam_r)
    vmin, vmax = max(pv.min(), -beam_r), min(pv.max(), beam_r)
    if umax <= umin or vmax <= vmin:
        return [], 0

    d = radar.ray_spacing_m
    n_est = ((umax - umin) / d + 1) * ((vmax - vmin) / d + 1)
    if n_est > radar.max_rays:
        d = d * np.sqrt(n_est / radar.max_rays)
        notes.append(f"Multi-bounce ray spacing coarsened to {d * 1000:.1f} mm to stay under {radar.max_rays:,} rays.")
    gu = np.arange(umin, umax + d * 0.5, d)
    gv = np.arange(vmin, vmax + d * 0.5, d)
    UU, VV = np.meshgrid(gu, gv)
    pts = p_r + R0 * bore + UU.ravel()[:, None] * u + VV.ravel()[:, None] * v
    dirs = pts - p_r
    dirs /= _norm(dirs)[:, None]
    n_rays = len(dirs)

    psi = np.arccos(np.clip(dirs @ bore, -1, 1))
    amp = _beam_power(psi, theta3)  # two-way field = one-way power
    origins = np.repeat(p_r[None, :], n_rays, axis=0)
    path = np.zeros(n_rays)
    first_r = np.full(n_rays, np.nan)
    n_ref = np.zeros(n_rays, dtype=int)
    alive = np.arange(n_rays)

    exits: list[_SbrExits] = []
    eps = 1e-5
    for _ in range(radar.max_bounces + 4):  # extra events allow transmissions
        if len(alive) == 0:
            break
        loc, idx_ray, idx_tri = tm.ray.intersects_location(origins[alive], dirs[alive], multiple_hits=False)
        if len(idx_ray) == 0:
            break
        rays = alive[idx_ray]
        seg = _norm(loc - origins[rays])
        path[rays] += seg
        first = np.isnan(first_r[rays])
        first_r[rays[first]] = path[rays[first]]

        n = normals[idx_tri].copy()
        flip = np.einsum("ij,ij->i", n, dirs[rays]) > 0
        n[flip] *= -1
        T = T_face[idx_tri]
        transmit = T >= 0.5
        reflect = ~transmit

        # Transmission: carry on straight, dimmed by sqrt(T) in field
        rt = rays[transmit]
        amp[rt] *= np.sqrt(T[transmit])
        origins[rt] = loc[transmit] + dirs[rt] * eps

        # Reflection: mirror the direction
        rr = rays[reflect]
        nr = n[reflect]
        amp[rr] *= gamma_spec[idx_tri[reflect]]
        dnew = dirs[rr] - 2.0 * np.einsum("ij,ij->i", dirs[rr], nr)[:, None] * nr
        dirs[rr] = dnew
        origins[rr] = loc[reflect] + nr * eps
        n_ref[rr] += 1

        # Exit event for rays on their 2nd+ reflection
        sel = n_ref[rr] >= 2
        if np.any(sel):
            r2 = rr[sel]
            hit = loc[reflect][sel]
            to_radar = p_r - hit
            back = _norm(to_radar)
            s_dir = to_radar / back[:, None]
            # Exit path back to the radar must be clear
            clear = ~tm.ray.intersects_any(hit + s_dir * eps, s_dir)
            exits.append(_SbrExits(
                amp=amp[r2], tube=d * first_r[r2] / R0, first_r=first_r[r2], back=back, path=path[r2],
                dout=dirs[r2], s_dir=s_dir, face=idx_tri[reflect][sel], clear=clear,
            ))

        keep = (n_ref[rays] < radar.max_bounces) & (amp[rays] > 1e-4)
        alive = np.unique(rays[keep])
    return exits, n_rays


def _sbr_field(exits: list[_SbrExits], k: float, lam: float, R0: float, n_faces: int, obj_id: np.ndarray):
    """Coherent multi-bounce field at wavenumber k.

    Returns (S, face_power, contributing, S_obj), where S_obj is the coherent field per object,
    attributed by exit face.
    """
    S = 0j
    face_power = np.zeros(n_faces)
    S_obj = np.zeros(int(obj_id.max()) + 1 if len(obj_id) else 1, dtype=complex)
    contributing = 0
    for e in exits:
        dout, s_dir, tube = e.dout, e.s_dir, e.tube
        delta = k * (dout - s_dir)
        # Aperture factor of a square ray tube: sinc in both transverse axes
        helper = np.where(np.abs(dout[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
        tu = np.cross(dout, helper)
        tu /= _norm(tu)[:, None]
        tv = np.cross(dout, tu)
        F = np.sinc(np.einsum("ij,ij->i", delta, tu) * tube / (2 * np.pi)) * np.sinc(
            np.einsum("ij,ij->i", delta, tv) * tube / (2 * np.pi)
        )
        F *= np.clip(np.einsum("ij,ij->i", dout, s_dir), 0.0, 1.0)  # obliquity
        contrib = e.amp * F * tube**2 * (R0 / e.first_r) * (R0 / e.back) * np.exp(-1j * k * (e.path + e.back))
        contrib *= e.clear
        S += contrib.sum()
        np.add.at(face_power, e.face, (4 * np.pi / lam**2) * np.abs(contrib) ** 2)
        np.add.at(S_obj, obj_id[e.face], contrib)
        contributing += int(np.count_nonzero(np.abs(contrib) > 0))
    return S, face_power, contributing, S_obj


# --------------------------------------------------------------------------- #
# Stage 1: illumination (frequency-independent)
# --------------------------------------------------------------------------- #
@dataclass
class Illumination:
    """Everything about a solve that doesn't depend on frequency."""

    n_faces: int
    obj_id: np.ndarray  # (M,)
    names: list[str]
    radar_position: np.ndarray
    aim_point: np.ndarray
    distance_m: float
    gain: float
    lit: np.ndarray  # indices of lit faces
    v0: np.ndarray  # lit triangle vertices
    v1: np.ndarray
    v2: np.ndarray
    khat: np.ndarray  # unit vector radar -> lit face centroid
    Ri: np.ndarray  # range to each lit face centroid
    po_amp: np.ndarray  # PO amplitude per lit face: gamma cos(theta) P tau spread
    diffuse: np.ndarray  # diffuse RCS per lit face (m^2; incoherent, so k-free)
    sbr_exits: list[_SbrExits]
    rays_launched: int
    intercepted_fraction: float
    reflected_pct: float
    absorbed_pct: float
    passed_through_pct: float
    extent_m: float  # largest bounding-box side, for the far-field note
    notes: list[str] = field(default_factory=list)
    retro_faces: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # face indices returning retro
    retro_amp: np.ndarray = field(default_factory=lambda: np.zeros(0))  # k-free field amplitude per retro face
    retro_range: np.ndarray = field(default_factory=lambda: np.zeros(0))  # phase reference range of its patch


def illuminate(target: Target, radar: RadarSettings, aim_point: np.ndarray | None = None) -> Illumination:
    V = np.asarray(target.vertices, float)
    F = np.asarray(target.faces, int)
    M = len(F)
    notes: list[str] = []
    if aim_point is None:
        aim_point = 0.5 * (V.min(0) + V.max(0))
    aim = np.asarray(aim_point, float)
    p_r = radar.position(aim)
    R0 = radar.distance_m
    bore = (aim - p_r) / R0
    theta3 = np.radians(radar.beam_width_deg)
    obj_id = target.object_id if target.object_id is not None else np.zeros(M, int)
    names = target.object_names or ["Target"]

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cent = (v0 + v1 + v2) / 3.0
    cr = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * _norm(cr)
    good = area > 1e-14
    normals = np.zeros_like(cr)
    normals[good] = cr[good] / (2 * area[good])[:, None]

    dvec = cent - p_r
    Ri = _norm(dvec)
    khat = dvec / Ri[:, None]
    cos_i = np.abs(np.einsum("ij,ij->i", normals, khat))  # two-sided surfaces
    psi = np.arccos(np.clip(khat @ bore, -1, 1))
    P = _beam_power(psi, theta3)

    R = np.asarray(target.reflectance, float)
    A = np.asarray(target.absorptance, float)
    s = np.asarray(target.roughness, float)
    T = target.transmittance

    cand = np.nonzero(good & (cos_i > 1e-6) & (P > 1e-5))[0]
    tm = target.bvh()
    tau = np.zeros(M)
    tau[cand] = _shadow_transmission(tm, p_r, cent[cand], Ri[cand], cand, T)
    lit = cand[tau[cand] > 1e-6]

    q = np.zeros(M) if target.retro is None else np.asarray(target.retro, float)
    spread = (R0 / Ri[lit]) ** 2
    gamma = np.sqrt(R[lit] * (1.0 - s[lit] - q[lit]))  # specular share; the retro share is handled below
    po_amp = gamma * cos_i[lit] * P[lit] * tau[lit] * spread
    diff = 4.0 * R[lit] * s[lit] * area[lit] * cos_i[lit] ** 2 * (P[lit] * tau[lit] * spread) ** 2

    exits, n_rays = [], 0
    if radar.multibounce and radar.max_bounces >= 2:
        gamma_all = np.sqrt(R * (1.0 - s - q))
        exits, n_rays = _sbr_trace(tm, target, radar, p_r, aim, normals, gamma_all, T, notes)

    # ---- retrodirective faces ----
    retro_faces, retro_amp, retro_range = np.zeros(0, int), np.zeros(0), np.zeros(0)
    rl = lit[q[lit] > 0]
    if len(rl):
        half = (np.full(M, 40.0) if target.retro_half_angle_deg is None
                else np.asarray(target.retro_half_angle_deg, float))
        # One-sided: a tag only works from the side its face normal points to
        facing = np.einsum("ij,ij->i", normals[rl], -khat[rl]) > 0
        rl = rl[facing]
        theta = np.degrees(np.arccos(np.clip(cos_i[rl], 0.0, 1.0)))
        g = retro_taper(theta, half[rl])
        rl, theta, g = rl[g > 0], theta[g > 0], g[g > 0]
        if len(rl):
            sp = (R0 / Ri[rl]) ** 2
            retro_amp = np.sqrt(R[rl] * q[rl]) * area[rl] * cos_i[rl] * P[rl] * tau[rl] * sp * g
            # Each connected patch is one phase-conjugating aperture: its faces share the patch's
            # amplitude-weighted mean range as their phase reference
            lab = _patches(F[q > 0], len(V))
            patch = np.full(M, -1)
            patch[np.nonzero(q > 0)[0]] = lab
            pl = patch[rl]
            uniq, inv = np.unique(pl, return_inverse=True)
            w_sum = np.bincount(inv, weights=retro_amp)
            r_ref = np.bincount(inv, weights=retro_amp * Ri[rl]) / np.maximum(w_sum, 1e-300)
            retro_faces, retro_range = rl, r_ref[inv]

    # ---- energy budget ----
    G = radar.gain
    incident = area[lit] * cos_i[lit] * P[lit] * tau[lit] * spread  # m^2 of boresight power density
    first_surface = tau[lit] > 0.999
    intercepted = incident[first_surface].sum()
    reflected = (R[lit] * incident).sum()
    absorbed = (A[lit] * incident).sum()
    if intercepted > 0:
        refl_pct = 100 * min(reflected / intercepted, 1.0)
        abs_pct = 100 * min(absorbed / intercepted, 1.0 - refl_pct / 100)
        thru_pct = max(0.0, 100.0 - refl_pct - abs_pct)
    else:
        refl_pct = abs_pct = thru_pct = 0.0
    intercepted_frac = min(G * intercepted / (4 * np.pi * R0**2), 1.0)

    return Illumination(
        n_faces=M, obj_id=obj_id, names=names, radar_position=p_r, aim_point=aim, distance_m=R0, gain=G,
        lit=lit, v0=v0[lit], v1=v1[lit], v2=v2[lit], khat=khat[lit], Ri=Ri[lit], po_amp=po_amp, diffuse=diff,
        sbr_exits=exits, rays_launched=int(n_rays), intercepted_fraction=float(intercepted_frac),
        reflected_pct=float(refl_pct), absorbed_pct=float(abs_pct), passed_through_pct=float(thru_pct),
        extent_m=float(np.ptp(V, axis=0).max()), notes=notes,
        retro_faces=retro_faces, retro_amp=retro_amp, retro_range=retro_range,
    )


# --------------------------------------------------------------------------- #
# Stage 2: fields at one frequency
# --------------------------------------------------------------------------- #
def sum_fields(illum: Illumination, frequency_hz: float = DEFAULT_FREQ_HZ) -> RcsResult:
    lam = C0 / frequency_hz
    k = 2.0 * np.pi / lam
    M, lit, R0 = illum.n_faces, illum.lit, illum.distance_m
    notes = list(illum.notes)

    # ---- specular PO ----
    I = triangle_phase_integral(illum.v0, illum.v1, illum.v2, -2.0 * k * illum.khat)
    po = illum.po_amp * I * np.exp(-2j * k * illum.Ri)
    S_po = po.sum()
    # ---- diffuse ----
    diff = illum.diffuse
    sigma_diff = float(diff.sum())
    # ---- multi-bounce ----
    S_mb, face_mb, n_mb, S_mb_obj = _sbr_field(illum.sbr_exits, k, lam, R0, M, illum.obj_id)

    # ---- retrodirective ----
    rf = illum.retro_faces
    retro = illum.retro_amp * np.exp(-2j * k * illum.retro_range)
    S_retro = retro.sum() if len(rf) else 0j

    scale = 4.0 * np.pi / lam**2
    sigma_spec = scale * abs(S_po) ** 2
    sigma_mb = scale * abs(S_mb) ** 2
    sigma_retro = scale * abs(S_retro) ** 2
    if len(rf):
        sigma = scale * abs(S_po + S_mb + S_retro) ** 2 + sigma_diff
    else:
        sigma = scale * abs(S_po + S_mb) ** 2 + sigma_diff

    face = np.zeros(M)
    face[lit] = scale * np.abs(po) ** 2 + diff
    face += face_mb
    if len(rf):
        np.add.at(face, rf, scale * np.abs(retro) ** 2)
        np.add.at(S_mb_obj, illum.obj_id[rf], retro)  # per-object field: retro joins PO + multi-bounce

    # Per-object RCS: each object's PO and multi-bounce fields (multi-bounce attributed by exit face) added
    # coherently, plus its diffuse power. For a one-object scene this equals the total exactly. (Up to commit
    # 69677f2 the multi-bounce part was added as incoherent ray power, which undercounted corner reflectors.)
    obj_id, names = illum.obj_id, illum.names
    obj_rcs: dict[str, float] = {}
    for oid in np.unique(obj_id):
        m = obj_id[lit] == oid
        val = scale * abs(po[m].sum() + S_mb_obj[oid]) ** 2 + diff[m].sum()
        obj_rcs[names[oid] if oid < len(names) else f"Object {oid}"] = float(val)

    G = illum.gain
    returned = G**2 * lam**2 * sigma / ((4 * np.pi) ** 3 * R0**4)

    far_field = 2 * illum.extent_m**2 / lam
    if R0 < far_field:
        notes.append(
            f"{NEAR_FIELD} (far field starts at ~{far_field:.0f} m); "
            "RCS here is distance-dependent, which is expected for close-range automotive radar."
        )

    return RcsResult(
        rcs_m2=float(sigma),
        specular_m2=float(sigma_spec),
        diffuse_m2=sigma_diff,
        multibounce_m2=float(sigma_mb),
        returned_fraction=float(returned),
        intercepted_fraction=illum.intercepted_fraction,
        reflected_pct=illum.reflected_pct,
        absorbed_pct=illum.absorbed_pct,
        passed_through_pct=illum.passed_through_pct,
        gain_dbi=float(10 * np.log10(G)),
        face_rcs_m2=face,
        object_rcs_m2=obj_rcs,
        radar_position=illum.radar_position,
        aim_point=illum.aim_point,
        lit_faces=int(len(lit)),
        rays_launched=illum.rays_launched,
        multibounce_rays=int(n_mb),
        notes=notes,
        retro_m2=float(sigma_retro),
    )


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def compute_rcs(target: Target, radar: RadarSettings, aim_point: np.ndarray | None = None) -> RcsResult:
    """Single-frequency solve at radar.frequency_hz."""
    return sum_fields(illuminate(target, radar, aim_point), radar.frequency_hz)


def average_results(results: list[RcsResult]) -> RcsResult:
    """Power average of several solves of the same target (frequencies or aspects)."""
    if len(results) == 1:
        return results[0]
    mean = lambda attr: float(np.mean([getattr(r, attr) for r in results]))  # noqa: E731
    first = results[len(results) // 2]  # the centre sample carries the geometry
    objects = {name: float(np.mean([r.object_rcs_m2.get(name, 0.0) for r in results])) for name in first.object_rcs_m2}
    # The near-field note quotes a frequency-dependent distance, so keep only the centre sample's copy
    notes = list(dict.fromkeys(first.notes + [n for r in results for n in r.notes if not n.startswith(NEAR_FIELD)]))
    return RcsResult(
        rcs_m2=mean("rcs_m2"), specular_m2=mean("specular_m2"), diffuse_m2=mean("diffuse_m2"),
        multibounce_m2=mean("multibounce_m2"), returned_fraction=mean("returned_fraction"),
        intercepted_fraction=mean("intercepted_fraction"), reflected_pct=mean("reflected_pct"),
        absorbed_pct=mean("absorbed_pct"), passed_through_pct=mean("passed_through_pct"), gain_dbi=first.gain_dbi,
        face_rcs_m2=np.mean([r.face_rcs_m2 for r in results], axis=0), object_rcs_m2=objects,
        radar_position=first.radar_position, aim_point=first.aim_point, lit_faces=first.lit_faces,
        rays_launched=first.rays_launched, multibounce_rays=max(r.multibounce_rays for r in results), notes=notes,
        retro_m2=mean("retro_m2"),
        samples_rcs_m2=np.array([r.rcs_m2 for r in results]),
    )


def compute_rcs_band(target: Target, radar: RadarSettings, aim_point: np.ndarray | None = None) -> RcsResult:
    """RCS averaged over the radar's FMCW sweep: one ray-casting pass, n_freq field sums."""
    illum = illuminate(target, radar, aim_point)
    freqs = radar.band_frequencies()
    res = average_results([sum_fields(illum, f) for f in freqs])
    if len(freqs) > 1:
        res.frequencies_hz = freqs
    return res


def solve(target: Target, radar: RadarSettings, aim_point: np.ndarray | None = None) -> RcsResult:
    """What the app shows: band averaged and, if radar.aspect_avg_deg > 0, averaged over three bearings."""
    if radar.aspect_avg_deg <= 0:
        return compute_rcs_band(target, radar, aim_point)
    d = radar.aspect_avg_deg
    azs = np.array([radar.azimuth_deg - d, radar.azimuth_deg, radar.azimuth_deg + d])
    parts = [compute_rcs_band(target, replace(radar, azimuth_deg=float(a)), aim_point) for a in azs]
    samples = np.concatenate([p.samples_rcs_m2 if p.samples_rcs_m2 is not None else [p.rcs_m2] for p in parts])
    res = average_results(parts)
    res.frequencies_hz = parts[1].frequencies_hz
    res.aspects_deg = azs
    res.samples_rcs_m2 = samples
    return res
