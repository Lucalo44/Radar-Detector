"""Turning solver output into numbers a designer can act on. No Qt in here, so it can be tested.

The headline question is: compared with the plain cyclist, does my part make the rider easier or
harder for a radar to see, and by how much? Everything here is in service of answering that.
"""

from __future__ import annotations

import hashlib
from dataclasses import astuple, dataclass

import numpy as np

from .rcs import RadarSettings, RcsResult

SAME_BAND_DB = 1.0  # changes inside +/- this are within the solver's own speckle

# Rough monostatic RCS ranges at 76-81 GHz, for orientation only. These still need checking
# against the literature (see docs/numbers.md), and the UI labels them as typical ranges.
REFERENCE_RANGES: list[tuple[str, float, float]] = [
    ("Pedestrian", -10.0, 0.0),
    ("Cyclist", 0.0, 10.0),
    ("Car", 10.0, 20.0),
]


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def to_db(x: float) -> float:
    return float(10.0 * np.log10(max(x, 1e-30)))


def delta_db(with_m2: float, base_m2: float) -> float:
    return to_db(with_m2) - to_db(base_m2)


def visibility(delta: float) -> str:
    """'more', 'same' or 'less' visible than the biker alone."""
    if delta > SAME_BAND_DB:
        return "more"
    if delta < -SAME_BAND_DB:
        return "less"
    return "same"


def ratio_text(delta: float) -> str:
    """The change as a plain-words ratio of returned power."""
    r = 10 ** (delta / 10)
    if abs(delta) <= SAME_BAND_DB:
        return "about the same return"
    if r >= 1:
        return f"about {r:.1f}× the return" if r < 10 else f"about {r:.0f}× the return"
    return f"about {100 * (1 - r):.0f}% less return"


def equivalent_sphere_m(sigma_m2: float) -> float:
    """Diameter of a metal sphere with this RCS (optical region: sigma = pi r^2)."""
    return float(2.0 * np.sqrt(max(sigma_m2, 0.0) / np.pi))


def fmt_length(m: float) -> str:
    if m >= 1.0:
        return f"{m:.1f} m"
    if m >= 0.01:
        return f"{m * 100:.0f} cm"
    return f"{m * 1000:.1f} mm"


def fmt_rcs(m2: float, unit: str = "dBsm") -> str:
    if unit == "m²":
        return f"{m2:.3g} m²"
    return f"{to_db(m2):+.1f} dBsm"


def fmt_rcs_pair(m2: float, unit: str = "dBsm") -> str:
    """Primary unit first, the other in brackets."""
    other = "m²" if unit == "dBsm" else "dBsm"
    return f"{fmt_rcs(m2, unit)} ({fmt_rcs(m2, other)})"


def sentence_case(text: str) -> str:
    """Capitalise the first letter only (str.capitalize would turn GHz into ghz)."""
    return text[:1].upper() + text[1:]


def band_text(radar: RadarSettings) -> str:
    f = radar.band_frequencies()
    if len(f) > 1:
        txt = f"averaged over {f[0] / 1e9:.1f}–{f[-1] / 1e9:.1f} GHz ({len(f)} frequencies)"
    else:
        txt = f"single frequency, {f[0] / 1e9:.1f} GHz"
    if radar.aspect_avg_deg > 0:
        txt += f" and ±{radar.aspect_avg_deg:g}° in bearing"
    return txt


def reference_label(dbsm: float) -> str:
    """Which typical range(s) a value falls in, for the line under the scale."""
    hits = [name.lower() for name, lo, hi in REFERENCE_RANGES if lo <= dbsm <= hi]
    if hits:
        return "in the typical " + " / ".join(hits) + " range"
    lo = REFERENCE_RANGES[0][1]
    return "below the typical pedestrian range" if dbsm < lo else "above the typical car range"


# --------------------------------------------------------------------------- #
# Per-object table
# --------------------------------------------------------------------------- #
@dataclass
class ObjectRow:
    name: str
    rcs_m2: float
    share_pct: float  # of the sum of per-object values
    vs_base_db: float | None  # attachments only: how much this part alone lifts the biker's total
    attachment: bool

    @property
    def dbsm(self) -> float:
        return to_db(self.rcs_m2)


def object_rows(result: RcsResult, base: RcsResult | None, attachment_names: set[str]) -> list[ObjectRow]:
    """Rows sorted by contribution, largest first.

    Per-object values add coherently inside each object but not across objects, so the shares are
    of the sum of the per-object values, not of the coherent total.
    For an attachment, "vs. biker alone" is 10 log10(1 + sigma_part / sigma_biker): what that part
    would add to the biker's return if nothing else changed.
    """
    total = sum(result.object_rcs_m2.values()) or 1e-30
    rows = []
    for name, val in result.object_rcs_m2.items():
        att = name in attachment_names
        vs = None
        if att and base is not None and base.rcs_m2 > 0:
            vs = to_db(base.rcs_m2 + val) - to_db(base.rcs_m2)
        rows.append(ObjectRow(name, val, 100.0 * val / total, vs, att))
    rows.sort(key=lambda r: -r.rcs_m2)
    return rows


# --------------------------------------------------------------------------- #
# Baseline cache
# --------------------------------------------------------------------------- #
def baseline_key(radar: RadarSettings, biker_face_mats: list[np.ndarray], lib_values: list[tuple],
                 aim: np.ndarray) -> str:
    """Everything the biker-alone result depends on: radar settings, the biker's per-face
    materials, the library's material values and the aim point."""
    h = hashlib.sha1()
    h.update(repr(astuple(radar)).encode())
    h.update(repr([tuple(round(float(x), 12) for x in v) for v in lib_values]).encode())
    h.update(np.round(np.asarray(aim, float), 9).tobytes())
    for fm in biker_face_mats:
        h.update(np.ascontiguousarray(fm, dtype=np.int64).tobytes())
    return h.hexdigest()


def library_values(lib) -> list[tuple]:
    return [(m.reflectance, m.absorptance, m.roughness, m.retro, m.retro_half_angle_deg) for m in lib.materials]


class BaselineCache:
    """The biker-alone result for the current settings. Holds a few recent keys, so flipping a
    setting back and forth doesn't re-solve."""

    def __init__(self, size: int = 8):
        self._items: dict[str, RcsResult] = {}
        self._size = size

    def get(self, key: str) -> RcsResult | None:
        return self._items.get(key)

    def put(self, key: str, res: RcsResult) -> None:
        self._items.pop(key, None)
        self._items[key] = res
        while len(self._items) > self._size:
            self._items.pop(next(iter(self._items)))

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# Heat map
# --------------------------------------------------------------------------- #
def heat_values(result: RcsResult, spans: list[tuple[str, int, int]]) -> np.ndarray:
    """Per-face values for the heat map, scaled so each object's faces add up to its table value.

    The solver's per-face values are incoherent (each face's own power). For a corner reflector,
    whose rays add in phase, that hides the part doing most of the work. Scaling per object keeps
    the spread across an object's faces but makes the objects compare the way the table does.
    spans: (object name, first face, end face) for each object in the merged target.
    """
    face = np.asarray(result.face_rcs_m2, float)
    out = face.copy()
    for name, a, b in spans:
        total = face[a:b].sum()
        target = result.object_rcs_m2.get(name)
        if target is not None and total > 0:
            out[a:b] = face[a:b] * (target / total)
    return out
