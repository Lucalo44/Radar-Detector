"""Background solving, so the window stays responsive while the solver runs.

Two kinds of work:

1. `SolveService` runs the solve behind the headline numbers on one worker thread. A new request
   while a solve is running replaces any queued one, so at most one stale solve finishes, and its
   result is dropped by generation number.
2. `BatchJob` runs an aspect sweep or a chart set through `sweeps.run_tasks`, which spreads the
   solves across processes, and reports progress. It can be cancelled.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from PySide6 import QtCore

from .rcs import RadarSettings, RcsResult, Target, solve
from .sweeps import Task, run_tasks


@dataclass
class SolveRequest:
    gen: int
    target: Target
    radar: RadarSettings
    aim: np.ndarray
    base_target: Target | None = None  # set when the biker-alone baseline isn't cached yet
    base_key: str | None = None
    extra: dict = field(default_factory=dict)  # passed through untouched (spans etc.)


@dataclass
class SolveReply:
    request: SolveRequest
    result: RcsResult | None
    base_result: RcsResult | None
    seconds: float
    error: str | None = None


class _SolveWorker(QtCore.QObject):
    done = QtCore.Signal(object)

    @QtCore.Slot(object)
    def run(self, req: SolveRequest):
        t0 = time.perf_counter()
        try:
            res = solve(req.target, req.radar, req.aim)
            base = solve(req.base_target, req.radar, req.aim) if req.base_target is not None else None
            self.done.emit(SolveReply(req, res, base, time.perf_counter() - t0))
        except Exception:  # noqa: BLE001  (report, don't kill the thread)
            self.done.emit(SolveReply(req, None, None, time.perf_counter() - t0, traceback.format_exc()))


class SolveService(QtCore.QObject):
    finished = QtCore.Signal(object)  # SolveReply, on the GUI thread
    busy_changed = QtCore.Signal(bool)
    _submit = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QtCore.QThread(self)
        self._worker = _SolveWorker()
        self._worker.moveToThread(self._thread)
        self._submit.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._thread.start()
        self._busy = False
        self._pending: SolveRequest | None = None

    def request(self, req: SolveRequest):
        if self._busy:
            self._pending = req  # latest wins
            return
        self._start(req)

    def _start(self, req: SolveRequest):
        self._busy = True
        self.busy_changed.emit(True)
        self._submit.emit(req)

    def _on_done(self, reply: SolveReply):
        self.finished.emit(reply)
        if self._pending is not None:
            req, self._pending = self._pending, None
            self._start(req)
        else:
            self._busy = False
            self.busy_changed.emit(False)

    def shutdown(self):
        self._thread.quit()
        self._thread.wait(10_000)


class BatchJob(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    succeeded = QtCore.Signal(object)  # list[(rcs_m2, returned_fraction)]
    failed = QtCore.Signal(str)
    was_cancelled = QtCore.Signal()

    def __init__(self, tasks: list[Task], parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            out = run_tasks(self.tasks, progress=self.progress.emit, cancelled=lambda: self._cancel)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
            return
        if out is None:
            self.was_cancelled.emit()
        else:
            self.succeeded.emit(out)
