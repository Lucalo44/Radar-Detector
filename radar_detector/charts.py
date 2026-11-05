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

    # ---- 1. returned power vs distance, one panel per aspect ----
    fig = Figure(figsize=(10, 7.2), facecolor=SURFACE)
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        _style(ax)
        ax.plot(distances, base_db[name], color=BASE_COLOR, lw=2, ls=(0, (5, 3)), label="Biker alone")
        ax.plot(distances, with_db[name], color=WITH_COLOR, lw=2, label=f"Biker + {parts}")
        delta = with_db[name][i_ref] - base_db[name][i_ref]
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.text(0.98, 0.95, f"{delta:+.1f} dB at {distances[i_ref]:.0f} m", transform=ax.transAxes,
                ha="right", va="top", color=INK, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Distance (m)", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Returned power (dB re transmitted)", color=INK2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    fig.suptitle("Returned radar power vs. distance", x=0.06, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
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
