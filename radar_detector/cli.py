"""Command-line entry point.

    python run.py            launch the app
    python run.py --check    print versions and which ray engine is active, then time one solve
"""

from __future__ import annotations

import platform
import sys
import time
from importlib import metadata

PACKAGES = ("numpy", "scipy", "trimesh", "embreex", "pyvista", "pyvistaqt", "vtk", "PySide6", "matplotlib")


def check() -> int:
    print(f"Python      {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    for pkg in PACKAGES:
        try:
            ver = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            ver = "not installed"
        print(f"{pkg:<11} {ver}")

    from .rcs import RadarSettings, ray_engine, solve

    name, embree = ray_engine()
    print(f"\nRay engine  {name}")
    if not embree:
        print("            Embree NOT active: solves will take 20 s or more instead of under a second.\n"
              "            Try: pip install embreex   (see the README's Setup section)")
        return 1
    from . import geometry as geo
    from .materials import MaterialLibrary

    lib = MaterialLibrary()
    target, _ = geo.build_target(geo.build_biker(lib), lib)
    t0 = time.perf_counter()
    res = solve(target, RadarSettings())
    print(f"Test solve  baseline biker, {len(target.faces):,} faces, 10 m: "
          f"{res.rcs_dbsm:+.1f} dBsm in {time.perf_counter() - t0:.2f} s")
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        sys.exit(check())
    from .gui import main as gui_main

    gui_main()
