"""Material library for 77 GHz radar scattering.

Each material is described by three power fractions measured at normal
incidence, plus a roughness term:

    reflectance  R : fraction of incident power that is reflected
    absorptance  A : fraction of incident power that is absorbed
    transmittance T = 1 - R - A : fraction that passes through the surface
    roughness    s : fraction of the *reflected* power that is scattered
                     diffusely (Lambertian) instead of specularly

The default values are engineering estimates at 77 GHz (lambda ~= 3.9 mm)
derived from typical complex permittivities (e.g. skin eps ~ 6.5 - j8 gives
|Gamma|^2 ~ 0.32). They are a reasonable starting point, not lab data.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field


@dataclass
class Material:
    name: str
    reflectance: float
    absorptance: float
    roughness: float = 0.0
    color: str = "#999999"
    note: str = ""

    def __post_init__(self) -> None:
        self.reflectance = float(min(max(self.reflectance, 0.0), 1.0))
        self.absorptance = float(min(max(self.absorptance, 0.0), 1.0 - self.reflectance))
        self.roughness = float(min(max(self.roughness, 0.0), 1.0))

    @property
    def transmittance(self) -> float:
        return max(0.0, 1.0 - self.reflectance - self.absorptance)

    @property
    def rgb(self) -> tuple[int, int, int]:
        h = self.color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


DEFAULT_MATERIALS: list[Material] = [
    Material("Aluminum", 0.99, 0.01, 0.02, "#b8c2cc", "Bike frames, rims, brackets. Near-perfect conductor."),
    Material("Steel", 0.98, 0.02, 0.03, "#7d8790", "Spokes, chain, hardware."),
    Material("Carbon fiber composite", 0.85, 0.15, 0.05, "#3a3a3a", "Conductive fibres reflect most of the energy."),
    Material("Clothed body", 0.33, 0.67, 0.50, "#3a6ea5", "Skin under thin, mostly transparent jersey fabric."),
    Material("Skin", 0.35, 0.65, 0.30, "#e0ac69", "High water content; lossy dielectric."),
    Material("Tire rubber", 0.25, 0.70, 0.30, "#1e1e1e", "Carbon-black loaded rubber."),
    Material("Saddle foam / leather", 0.08, 0.30, 0.30, "#5a3a22", ""),
    Material("ABS / hard plastic", 0.05, 0.03, 0.05, "#9a9a94", "Low-loss dielectric; mostly transparent."),
    Material("Polycarbonate", 0.06, 0.02, 0.02, "#cfe8ff", "Lens and light housings."),
    Material("Nylon fabric", 0.02, 0.05, 0.30, "#c46b3a", "Bags and panniers (dry)."),
    Material("Metallized fabric", 0.90, 0.08, 0.30, "#d4d48a", "Conductive textile."),
    Material("Wood (dry)", 0.04, 0.30, 0.10, "#a07040", ""),
    Material("Glass", 0.18, 0.20, 0.02, "#a8d8d8", ""),
    Material("Water / wet surface", 0.60, 0.40, 0.10, "#2f7fbf", ""),
    Material("Radar absorber (RAM)", 0.01, 0.99, 0.00, "#4a2a4a", "Carbon-loaded absorber foam."),
]


def _auto_color(index: int) -> str:
    h = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.85)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


@dataclass
class MaterialLibrary:
    """Append-only list of materials. Faces reference materials by index."""

    materials: list[Material] = field(default_factory=lambda: [Material(**m.__dict__) for m in DEFAULT_MATERIALS])

    def __len__(self) -> int:
        return len(self.materials)

    def __getitem__(self, i: int) -> Material:
        return self.materials[i]

    def names(self) -> list[str]:
        return [m.name for m in self.materials]

    def index(self, name: str) -> int:
        for i, m in enumerate(self.materials):
            if m.name == name:
                return i
        raise KeyError(name)

    def add(self, name: str, reflectance: float, absorptance: float, roughness: float, color: str | None = None) -> int:
        name = name.strip()
        base, n = re.sub(r" \(\d+\)$", "", name) or "Custom", 1  # "Foil (2)" -> "Foil (3)", not "Foil (2) (2)"
        existing = set(self.names())
        while name in existing or not name.strip():
            n += 1
            name = f"{base} ({n})"
        self.materials.append(Material(name, reflectance, absorptance, roughness, color or _auto_color(len(self.materials))))
        return len(self.materials) - 1

    def arrays(self):
        """Per-material numpy arrays (R, A, s) for fast lookup by face index."""
        import numpy as np

        R = np.array([m.reflectance for m in self.materials])
        A = np.array([m.absorptance for m in self.materials])
        s = np.array([m.roughness for m in self.materials])
        rgb = np.array([m.rgb for m in self.materials], dtype=np.uint8)
        return R, A, s, rgb
