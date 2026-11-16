"""Every Docs page loads, starts with a heading, and its links and images point at real files."""

import re

import pytest

from radar_detector.docs_tab import DOCS_DIR, PAGES
from radar_detector.materials import DEFAULT_MATERIALS


@pytest.mark.parametrize("slug,title,fname", PAGES)
def test_page_loads_with_heading(slug, title, fname):
    text = (DOCS_DIR / fname).read_text(encoding="utf-8")
    assert text.lstrip().startswith("# ")
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://")):
            continue
        assert (DOCS_DIR / target).exists(), f"{fname} links to missing {target}"


def test_materials_page_lists_every_preset():
    text = (DOCS_DIR / "materials.md").read_text(encoding="utf-8")
    for m in DEFAULT_MATERIALS:
        q = f"{m.retro:.3g} (±{m.retro_half_angle_deg:.0f}°)" if m.retro > 0 else "0"
        row = f"| {m.name} | {m.reflectance:.2f} | {m.absorptance:.2f} | {m.transmittance:.2f} | {m.roughness:.2f} | {q} |"
        assert row in text, f"materials.md is out of date for {m.name}"
