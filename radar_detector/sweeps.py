"""Aspect sweeps and distance charts, run in parallel across processes. No Qt in here.

Each solve in a sweep is independent, so they spread across worker processes. Solves are sent
in chunks that share one Target, so the mesh is pickled once per chunk instead of once per
solve, and each worker builds its Embree BVH once per chunk.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .rcs import RadarSettings, Target, solve

ASPECT_STEP_DEG = 5.0
CHART_DISTANCES = np.arange(2.0, 50.0 + 1e-9, 2.0)


@dataclass
class Task:
    target: Target
    aim: np.ndarray
    radar: RadarSettings


def _solve_chunk(target: Target, aim: np.ndarray, radars: list[RadarSettings]) -> list[tuple[float, float]]:
    out = []
    for r in radars:
        res = solve(target, r, aim)
        out.append((res.rcs_m2, res.returned_fraction))
    return out


# --------------------------------------------------------------------------- #
# Process pool
# --------------------------------------------------------------------------- #
_pool: ProcessPoolExecutor | None = None


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _get_pool() -> ProcessPoolExecutor | None:
    """One pool for the life of the app; spawning workers costs about a second each."""
    global _pool
    if _pool is None:
        try:
            # spawn, never fork: forking a process that holds Qt and VTK threads is unsafe
            _pool = ProcessPoolExecutor(max_workers=default_workers(), mp_context=mp.get_context("spawn"))
        except (OSError, ValueError, NotImplementedError):
            return None
    return _pool


@atexit.register
def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def run_tasks(tasks: list[Task], progress: Callable[[int, int], None] | None = None,
              cancelled: Callable[[], bool] | None = None, parallel: bool = True) -> list[tuple[float, float]] | None:
    """Solve every task. Returns [(rcs_m2, returned_fraction)] in task order, or None if cancelled."""
    n = len(tasks)
    results: list[tuple[float, float] | None] = [None] * n
    progress = progress or (lambda done, total: None)
    cancelled = cancelled or (lambda: False)

    # Group by Target so each chunk carries one mesh
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault(id(t.target), []).append(i)
    pool = _get_pool() if parallel and n > 1 else None
    workers = default_workers() if pool else 1
    size = max(1, int(np.ceil(n / (workers * 4))))
    chunks = [idx[j:j + size] for idx in groups.values() for j in range(0, len(idx), size)]

    done = 0
    progress(0, n)
    if pool is None:
        for chunk in chunks:
            if cancelled():
                return None
            t0 = tasks[chunk[0]]
            for i, r in zip(chunk, _solve_chunk(t0.target, t0.aim, [tasks[i].radar for i in chunk])):
                results[i] = r
            done += len(chunk)
            progress(done, n)
        return results  # type: ignore[return-value]

    futures: dict[Future, list[int]] = {}
    for chunk in chunks:
        t0 = tasks[chunk[0]]
        futures[pool.submit(_solve_chunk, t0.target, t0.aim, [tasks[i].radar for i in chunk])] = chunk
    pending = set(futures)
    while pending:
        finished, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
        if cancelled():
            for f in pending:
                f.cancel()
            return None
        for f in finished:
            chunk = futures[f]
            for i, r in zip(chunk, f.result()):
                results[i] = r
            done += len(chunk)
        if finished:
            progress(done, n)
    return results  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# The two jobs the app runs
# --------------------------------------------------------------------------- #
def aspect_tasks(target: Target, aim: np.ndarray, radar: RadarSettings, step: float = ASPECT_STEP_DEG):
    az = np.arange(0.0, 360.0, step)
    return az, [Task(target, aim, replace(radar, azimuth_deg=float(a))) for a in az]


def chart_tasks(base: Target, with_parts: Target, aim: np.ndarray, radar: RadarSettings, aspects,
                distances: np.ndarray = CHART_DISTANCES):
    """Tasks in (aspect, distance, [base, with]) order, matching `unpack_chart`."""
    tasks = []
    for _, az in aspects:
        for d in distances:
            r = replace(radar, azimuth_deg=float(az), distance_m=float(d))
            tasks += [Task(base, aim, r), Task(with_parts, aim, r)]
    return tasks


def unpack_chart(results: list[tuple[float, float]], aspects, distances: np.ndarray = CHART_DISTANCES):
    """Returned power in dB re transmitted, as {aspect: array over distance}, for base and with."""
    base_db, with_db = {}, {}
    k = 0
    for name, _ in aspects:
        b, w = np.zeros(len(distances)), np.zeros(len(distances))
        for i in range(len(distances)):
            b[i] = 10 * np.log10(max(results[k][1], 1e-300))
            w[i] = 10 * np.log10(max(results[k + 1][1], 1e-300))
            k += 2
        base_db[name], with_db[name] = b, w
    return base_db, with_db
