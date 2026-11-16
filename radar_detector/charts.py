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


def comparison_figures(distances: np.ndarray, base_db: dict[str, np.ndarray], with_db: dict[str, np.ndarray],
                       attachment_names: list[str], ref_distance: float = 10.0):
    """The two comparison charts as matplotlib Figures: (power vs. distance, attachment effect).

    base_db / with_db map aspect name -> returned power (dB re transmitted) at each distance.
    """
    from matplotlib.figure import Figure  # no pyplot, so this never touches the Qt app's backend

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
    fig_power = fig

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
    return fig_power, fig


def save_comparison(folder: str | Path, distances: np.ndarray, base_db: dict[str, np.ndarray],
                    with_db: dict[str, np.ndarray], attachment_names: list[str], ref_distance: float = 10.0) -> list[Path]:
    """Write the two charts and a CSV of every value. Returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in ASPECTS]
    fig_power, fig_delta = comparison_figures(distances, base_db, with_db, attachment_names, ref_distance)
    p_power = folder / "returned_power_vs_distance.png"
    fig_power.savefig(p_power, dpi=200)
    p_delta = folder / "attachment_effect_vs_distance.png"
    fig_delta.savefig(p_delta, dpi=200)

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


# --------------------------------------------------------------------------- #
# Aspect sweep
# --------------------------------------------------------------------------- #
SWEEP_MARK = "#eb6834"


def sweep_figure(az: np.ndarray, dbsm: np.ndarray, current_az: float, distance: float, base_dbsm: np.ndarray | None = None):
    """Polar plot of total RCS against radar bearing. Returns (figure, axes)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.5, 5.5), facecolor=SURFACE)
    ax = fig.add_subplot(projection="polar", facecolor=SURFACE)
    ax.set_theta_zero_location("S")  # 0 deg (radar behind rider) at the bottom, rider faces up
    ax.set_theta_direction(-1)  # 90 deg = rider's left
    th = np.radians(np.append(az, az[0] + 360))
    if base_dbsm is not None:
        ax.plot(th, np.append(base_dbsm, base_dbsm[0]), color=BASE_COLOR, lw=1.5, ls=(0, (4, 3)), label="Biker alone")
    ax.plot(th, np.append(dbsm, dbsm[0]), color=WITH_COLOR, lw=2, label="Current scene")
    cur = float(np.interp(current_az % 360, np.append(az, 360), np.append(dbsm, dbsm[0])))
    ax.plot([np.radians(current_az)], [cur], "o", ms=9, color=SWEEP_MARK, mec=SURFACE, mew=2)
    ax.annotate(f"current {current_az:.0f}°: {cur:+.1f} dBsm", (np.radians(current_az), cur),
                xytext=(12, 8), textcoords="offset points", color=INK, fontsize=9)
    vals = dbsm if base_dbsm is None else np.concatenate([dbsm, base_dbsm])
    ax.set_ylim(np.floor(vals.min() / 5) * 5, np.ceil(vals.max() / 5) * 5 + 2)
    ax.grid(color="#d8d7d2", lw=0.6)
    ax.spines["polar"].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0° rear", "45°", "90° left", "135°", "180° front", "225°", "270° right", "315°"])
    ax.set_rlabel_position(22.5)
    ax.set_title(f"Total RCS (dBsm) vs. radar bearing, {distance:.1f} m", color=INK, fontsize=11, pad=18)
    if base_dbsm is not None:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", bbox_to_anchor=(-0.12, -0.12))
    fig.tight_layout()
    return fig, ax


def save_sweep_csv(path: str | Path, az: np.ndarray, dbsm: np.ndarray, base_dbsm: np.ndarray | None = None) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["azimuth_deg", "rcs_dbsm", "rcs_m2"] + (["biker_alone_dbsm", "delta_db"] if base_dbsm is not None else []))
        for i, (a, d) in enumerate(zip(az, dbsm)):
            row = [f"{a:.1f}", f"{d:.3f}", f"{10 ** (d / 10):.6g}"]
            if base_dbsm is not None:
                row += [f"{base_dbsm[i]:.3f}", f"{d - base_dbsm[i]:.3f}"]
            w.writerow(row)
    return path
