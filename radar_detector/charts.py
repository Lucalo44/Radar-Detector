"""Returned-power comparison charts: baseline biker vs. biker with attachments, against distance."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Aspects in the order they're plotted. Azimuth follows the app: 0 = radar behind, 90 = rider's left.
ASPECTS = [("Rear", 0.0), ("Left side", 90.0), ("Right side", 270.0), ("Front", 180.0)]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
WITH_COLOR = "#2a78d6"  # series 1
BASE_COLOR = "#8c8b85"  # neutral reference
ASPECT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8)


def save_comparison(folder: str | Path, distances: np.ndarray, base_db: dict[str, np.ndarray],
                    with_db: dict[str, np.ndarray], attachment_names: list[str], ref_distance: float = 10.0) -> list[Path]:
    """Write the two charts and a CSV of every value. Returns the paths written.

    base_db / with_db map aspect name -> returned power (dB re transmitted) at each distance.
    """
    from matplotlib.figure import Figure  # no pyplot, so this never touches the Qt app's backend

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    parts = ", ".join(attachment_names) if attachment_names else "attachments"
    names = [n for n, _ in ASPECTS]
    i_ref = int(np.argmin(np.abs(distances - ref_distance)))

    # ---- 1. returned power vs distance, one panel per aspect, each with a difference strip ----
    # Returned power spans ~50 dB over 2-50 m (1/R^4), so an attachment's effect is invisible on that
    # axis alone. The strip under each panel plots the difference on its own scale.
    deltas = {n: with_db[n] - base_db[n] for n in names}
    # At least +/-1 dB so solver speckle on a part that adds nothing doesn't read as a big swing
    d_lim = max(1.0, 1.15 * max(float(np.abs(d).max()) for d in deltas.values()))
    fig = Figure(figsize=(10, 9), facecolor=SURFACE)
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 1.3, 0.5, 3, 1.3], hspace=0.1, wspace=0.05,
                          left=0.1, right=0.98, top=0.9, bottom=0.07)  # row 2 is a spacer
    p_lo = min(float(min(base_db[n].min(), with_db[n].min())) for n in names)
    p_hi = max(float(max(base_db[n].max(), with_db[n].max())) for n in names)
    pad = 0.05 * (p_hi - p_lo)
    first = None
    for k, name in enumerate(names):
        r, c = 3 * (k // 2), k % 2
        ax = fig.add_subplot(gs[r, c], sharex=first)
        first = first or ax
        dx = fig.add_subplot(gs[r + 1, c], sharex=first)
        for a in (ax, dx):
            _style(a)
        ax.plot(distances, with_db[name], color=WITH_COLOR, lw=2, label=f"Biker + {parts}")
        # Baseline on top, so it stays visible where the two lines coincide
        ax.plot(distances, base_db[name], color=BASE_COLOR, lw=1.5, ls=(0, (4, 3)), label="Biker alone", zorder=3)
        ax.set_ylim(p_lo - pad, p_hi + pad)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.text(0.98, 0.95, f"{deltas[name][i_ref]:+.1f} dB at {distances[i_ref]:.0f} m", transform=ax.transAxes,
                ha="right", va="top", color=INK, fontsize=9)
        ax.tick_params(labelbottom=False)
        dx.axhline(0, color=INK2, lw=1)
        dx.fill_between(distances, 0, deltas[name], color=WITH_COLOR, alpha=0.18, lw=0)
        dx.plot(distances, deltas[name], color=WITH_COLOR, lw=1.5)
        dx.set_ylim(-d_lim, d_lim)
        if c == 0:
            ax.set_ylabel("Returned power\n(dB re transmitted)", color=INK2, fontsize=9)
            dx.set_ylabel("Change (dB)", color=INK2, fontsize=9)
        else:
            ax.tick_params(labelleft=False)
            dx.tick_params(labelleft=False)
        if r == 3:
            dx.set_xlabel("Distance (m)", color=INK2, fontsize=9)
        else:
            dx.tick_params(labelbottom=False)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    fig.suptitle("Returned radar power vs. distance", x=0.06, ha="left", color=INK, fontsize=12)
    p_power = folder / "returned_power_vs_distance.png"
    fig.savefig(p_power, dpi=200)

    # ---- 2. what the attachments add, all aspects on one axis ----
    fig = Figure(figsize=(9, 5), facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    ax.axhline(0, color=INK2, lw=1)
    ends = {}
    for (name, _), color in zip(ASPECTS, ASPECT_COLORS):
        d = with_db[name] - base_db[name]
        ax.plot(distances, d, color=color, lw=2, label=name)
        ends[name] = d[-1]
    # Direct labels at the line ends, nudged apart so lines that finish together don't overprint
    ax.set_ylim(-d_lim, d_lim)
    gap = 0.06 * 2 * d_lim
    placed = []
    for name, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.annotate(name, (distances[-1], y), xytext=(distances[-1] + 0.012 * distances[-1], y_lab),
                    textcoords="data", va="center", color=INK, fontsize=9)
    ax.set_xlim(distances[0], distances[-1] * 1.12)
    ax.set_xlabel("Distance (m)", color=INK2, fontsize=9)
    ax.set_ylabel("Change in returned power (dB)", color=INK2, fontsize=9)
    ax.set_title(f"Effect of {parts} on returned radar power", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncol=4)
    fig.tight_layout()
    p_delta = folder / "attachment_effect_vs_distance.png"
    fig.savefig(p_delta, dpi=200)

    # ---- 3. the numbers ----
    p_csv = folder / "returned_power_vs_distance.csv"
    with open(p_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance_m"] + [f"{n} {k}" for n in names for k in ("baseline_db", "with_attachments_db", "delta_db")])
        for i, dist in enumerate(distances):
            row = [f"{dist:.2f}"]
            for n in names:
                row += [f"{base_db[n][i]:.3f}", f"{with_db[n][i]:.3f}", f"{with_db[n][i] - base_db[n][i]:.3f}"]
            w.writerow(row)
    return [p_power, p_delta, p_csv]
