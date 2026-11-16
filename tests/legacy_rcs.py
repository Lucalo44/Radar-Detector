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

3. Multi-bounce, by shooting and bouncing rays (SBR). A grid of ray tubes is
   launched from the radar; tubes that reflect two or more times are summed
   coherently with the PO field using a ray-tube aperture integral. This is
   what captures corner reflectors and dihedrals, which single-bounce PO
   misses entirely.

Spherical-wave illumination (true per-face direction and range), a Gaussian
beam pattern with a user-set -3 dB cone, and shadowing (with partial
transmission through low-loss dielectrics) are included. Not included:
ground-plane multipath, angle-dependent Fresnel coefficients, polarization
and creeping/edge-diffraction waves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

C0 = 299_792_458.0
DEFAULT_FREQ_HZ = 77e9
LN2 = np.log(2.0)


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

    def look_direction(self) -> np.ndarray:
        """Unit vector from the target toward the radar."""
        a, e = np.radians(self.azimuth_deg), np.radians(self.elevation_deg)
        # Rider travels along +X, +Y is the rider's left, +Z is up.
        return np.array([-np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])

    def position(self, aim: np.ndarray) -> np.ndarray:
        return np.asarray(aim, float) + self.distance_m * self.look_direction()


@dataclass
class Target:
    vertices: np.ndarray  # (N, 3) float, metres
    faces: np.ndarray  # (M, 3) int
    reflectance: np.ndarray  # (M,)
    absorptance: np.ndarray  # (M,)
    roughness: np.ndarray  # (M,)
    object_id: np.ndarray | None = None  # (M,) int
    object_names: list[str] = field(default_factory=list)

    @property
    def transmittance(self) -> np.ndarray:
        return np.clip(1.0 - self.reflectance - self.absorptance, 0.0, 1.0)


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

    @property
    def rcs_dbsm(self) -> float:
        return 10.0 * np.log10(max(self.rcs_m2, 1e-30))

    @property
    def returned_pct(self) -> float:
        return 100.0 * self.returned_fraction

    @property
    def returned_db(self) -> float:
        return 10.0 * np.log10(max(self.returned_fraction, 1e-300))


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


def _build_tm(target: Target) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=target.vertices, faces=target.faces, process=False, validate=False)


# --------------------------------------------------------------------------- #
# Shadowing
# --------------------------------------------------------------------------- #
def _shadow_transmission(tm, origin, centroids, dist, face_ids, T_face):
    """One-way power transmission from the radar to each face centroid.

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


# --------------------------------------------------------------------------- #
# Shooting and bouncing rays (multi-bounce only)
# --------------------------------------------------------------------------- #
def _sbr(tm, target, radar: RadarSettings, p_r, aim, normals, gamma_spec, T_face, notes):
    k = radar.k
    lam = radar.wavelength
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
        return 0j, np.zeros(len(target.faces)), 0, 0
    proj = rel[front] * (R0 / depth[front])[:, None]
    pu, pv = proj @ u, proj @ v
    beam_r = R0 * np.tan(min(theta3, np.radians(170)))  # rays beyond ~2x the -3 dB half-width carry < 6% field
    umin, umax = max(pu.min(), -beam_r), min(pu.max(), beam_r)
    vmin, vmax = max(pv.min(), -beam_r), min(pv.max(), beam_r)
    if umax <= umin or vmax <= vmin:
        return 0j, np.zeros(len(target.faces)), 0, 0

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

    S = 0j
    face_power = np.zeros(len(target.faces))
    contributing = 0
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

        # Exit contribution for rays on their 2nd+ reflection
        sel = n_ref[rr] >= 2
        if np.any(sel):
            r2 = rr[sel]
            hit = loc[reflect][sel]
            to_radar = p_r - hit
            back = _norm(to_radar)
            s_dir = to_radar / back[:, None]
            dout = dirs[r2]
            # Exit path back to the radar must be clear
            clear = ~tm.ray.intersects_any(hit + s_dir * eps, s_dir)
            tube = d * first_r[r2] / R0
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
            contrib = amp[r2] * F * tube**2 * (R0 / first_r[r2]) * (R0 / back) * np.exp(-1j * k * (path[r2] + back))
            contrib *= clear
            S += contrib.sum()
            np.add.at(face_power, idx_tri[reflect][sel], (4 * np.pi / lam**2) * np.abs(contrib) ** 2)
            contributing += int(np.count_nonzero(np.abs(contrib) > 0))

        keep = (n_ref[rays] < radar.max_bounces) & (amp[rays] > 1e-4)
        alive = np.unique(rays[keep])
    return S, face_power, n_rays, contributing


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def compute_rcs(target: Target, radar: RadarSettings, aim_point: np.ndarray | None = None) -> RcsResult:
    V = np.asarray(target.vertices, float)
    F = np.asarray(target.faces, int)
    M = len(F)
    notes: list[str] = []
    lam, k = radar.wavelength, radar.k
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
    tm = _build_tm(target)
    tau = np.zeros(M)
    tau[cand] = _shadow_transmission(tm, p_r, cent[cand], Ri[cand], cand, T)
    lit = cand[tau[cand] > 1e-6]

    spread = (R0 / Ri[lit]) ** 2
    # ---- specular PO ----
    gamma = np.sqrt(R[lit] * (1.0 - s[lit]))
    amp = gamma * cos_i[lit] * P[lit] * tau[lit] * spread
    I = triangle_phase_integral(v0[lit], v1[lit], v2[lit], -2.0 * k * khat[lit])
    po = amp * I * np.exp(-2j * k * Ri[lit])
    S_po = po.sum()
    # ---- diffuse ----
    diff = 4.0 * R[lit] * s[lit] * area[lit] * cos_i[lit] ** 2 * (P[lit] * tau[lit] * spread) ** 2
    sigma_diff = float(diff.sum())
    # ---- multi-bounce ----
    S_mb, face_mb, n_rays, n_mb = 0j, np.zeros(M), 0, 0
    if radar.multibounce and radar.max_bounces >= 2:
        gamma_all = np.sqrt(R * (1.0 - s))
        S_mb, face_mb, n_rays, n_mb = _sbr(tm, target, radar, p_r, aim, normals, gamma_all, T, notes)

    scale = 4.0 * np.pi / lam**2
    sigma_spec = scale * abs(S_po) ** 2
    sigma_mb = scale * abs(S_mb) ** 2
    sigma = scale * abs(S_po + S_mb) ** 2 + sigma_diff

    face = np.zeros(M)
    face[lit] = scale * np.abs(po) ** 2 + diff
    face += face_mb

    # Per-object coherent RCS (PO + diffuse; multi-bounce attributed by last face)
    obj_rcs: dict[str, float] = {}
    for oid in np.unique(obj_id):
        m = obj_id[lit] == oid
        mb_power = face_mb[obj_id == oid].sum()
        val = scale * abs(po[m].sum()) ** 2 + diff[m].sum() + mb_power
        obj_rcs[names[oid] if oid < len(names) else f"Object {oid}"] = float(val)

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
    returned = G**2 * lam**2 * sigma / ((4 * np.pi) ** 3 * R0**4)

    far_field = 2 * (np.ptp(V, axis=0).max()) ** 2 / lam
    if R0 < far_field:
        notes.append(
            f"Target is in the radar's near field (far field starts at ~{far_field:.0f} m); "
            "RCS here is distance-dependent, which is expected for close-range automotive radar."
        )

    return RcsResult(
        rcs_m2=float(sigma),
        specular_m2=float(sigma_spec),
        diffuse_m2=sigma_diff,
        multibounce_m2=float(sigma_mb),
        returned_fraction=float(returned),
        intercepted_fraction=float(intercepted_frac),
        reflected_pct=float(refl_pct),
        absorbed_pct=float(abs_pct),
        passed_through_pct=float(thru_pct),
        gain_dbi=float(10 * np.log10(G)),
        face_rcs_m2=face,
        object_rcs_m2=obj_rcs,
        radar_position=p_r,
        aim_point=aim,
        lit_faces=int(len(lit)),
        rays_launched=int(n_rays),
        multibounce_rays=int(n_mb),
        notes=notes,
    )
