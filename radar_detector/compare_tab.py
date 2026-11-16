"""Compare tab: returned power vs. distance against the biker alone, and the aspect sweep.

Both run in the background through `jobs.BatchJob`, with a progress bar and a cancel button
in the tab, and show their charts inline. "Save…" writes the same charts and a CSV to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PySide6 import QtCore, QtWidgets

from . import charts, sweeps
from .jobs import BatchJob

MUTED = "color: palette(placeholder-text)"


class _RunBar(QtWidgets.QWidget):
    run = QtCore.Signal()
    cancel = QtCore.Signal()
    save = QtCore.Signal()

    def __init__(self, run_text: str, tip: str, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.b_run = QtWidgets.QPushButton(run_text)
        self.b_run.setToolTip(tip)
        self.b_run.setStyleSheet("font-weight:600")
        self.b_run.clicked.connect(self.run)
        lay.addWidget(self.b_run)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setVisible(False)
        lay.addWidget(self.bar, 1)
        self.b_cancel = QtWidgets.QPushButton("Cancel")
        self.b_cancel.setVisible(False)
        self.b_cancel.clicked.connect(self.cancel)
        lay.addWidget(self.b_cancel)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet(MUTED)
        lay.addWidget(self.status, 1)
        self.b_save = QtWidgets.QPushButton("Save…")
        self.b_save.setEnabled(False)
        self.b_save.clicked.connect(self.save)
        lay.addWidget(self.b_save)

    def running(self, on: bool):
        self.b_run.setEnabled(not on)
        self.bar.setVisible(on)
        self.b_cancel.setVisible(on)
        self.status.setVisible(not on)
        if on:
            self.bar.setValue(0)

    def set_progress(self, done: int, total: int):
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(done)
        self.bar.setFormat(f"{done} / {total} solves")


class CompareTab(QtWidgets.QWidget):
    def __init__(self, chart_inputs: Callable, sweep_inputs: Callable, parent=None):
        """chart_inputs() -> (base_target, with_target, aim, radar, part_names); raises ValueError with a
        user-facing message if there's nothing to compare. sweep_inputs() -> (target, base_target | None, aim, radar)."""
        super().__init__(parent)
        self._chart_inputs, self._sweep_inputs = chart_inputs, sweep_inputs
        self._job: BatchJob | None = None
        self._chart_data = None
        self._sweep_data = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        lay.addWidget(self.tabs)
        self._build_charts()
        self._build_sweep()

    # ------------------------------------------------------------------ #
    def _canvas(self, fig):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        return FigureCanvasQTAgg(fig)

    def _build_charts(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        intro = QtWidgets.QLabel(
            "Returned power from 2 m to 50 m, seen from the rear, both sides and the front, for the biker alone and "
            "with the ticked attachments. Uses the current radar settings (cone, elevation, band averaging).")
        intro.setWordWrap(True)
        intro.setStyleSheet(MUTED)
        lay.addWidget(intro)
        self.chart_bar = _RunBar("Run comparison", "4 aspects × 25 distances × 2 scenes = 200 solves, in parallel.")
        self.chart_bar.run.connect(self.run_charts)
        self.chart_bar.cancel.connect(self._cancel)
        self.chart_bar.save.connect(self.save_charts)
        lay.addWidget(self.chart_bar)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.chart_body = QtWidgets.QWidget()
        self.chart_lay = QtWidgets.QVBoxLayout(self.chart_body)
        placeholder = QtWidgets.QLabel("Press Run comparison to compute the charts.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet(MUTED)
        self.chart_lay.addWidget(placeholder)
        scroll.setWidget(self.chart_body)
        lay.addWidget(scroll, 1)
        self.tabs.addTab(w, "Vs. distance")

    def _build_sweep(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        intro = QtWidgets.QLabel(
            "Total RCS as the radar circles the rider in 5° steps, at the current distance. With attachments and the "
            "biker shown, the biker alone is swept too and drawn dashed.")
        intro.setWordWrap(True)
        intro.setStyleSheet(MUTED)
        lay.addWidget(intro)
        self.sweep_bar = _RunBar("Run aspect sweep 0–360°", "72 bearings (144 with the baseline), in parallel.")
        self.sweep_bar.run.connect(self.run_sweep)
        self.sweep_bar.cancel.connect(self._cancel)
        self.sweep_bar.save.connect(self.save_sweep)
        lay.addWidget(self.sweep_bar)
        body = QtWidgets.QHBoxLayout()
        self.sweep_plot = QtWidgets.QVBoxLayout()
        placeholder = QtWidgets.QLabel("Press Run aspect sweep to compute the polar plot.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet(MUTED)
        self.sweep_plot.addWidget(placeholder)
        body.addLayout(self.sweep_plot, 3)
        self.sweep_table = QtWidgets.QTableWidget(0, 2)
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sweep_table.setHorizontalHeaderLabels(["Bearing (°)", "RCS (dBsm)"])
        self.sweep_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        body.addWidget(self.sweep_table, 1)
        lay.addLayout(body, 1)
        self.tabs.addTab(w, "Aspect sweep")

    @staticmethod
    def _clear(layout: QtWidgets.QLayout):
        while layout.count():
            it = layout.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()

    # ------------------------------------------------------------------ #
    def mark_stale(self):
        for bar, data in ((self.chart_bar, self._chart_data), (self.sweep_bar, self._sweep_data)):
            if data is not None:
                bar.status.setText("Scene or radar changed since this run. Run again to update.")

    def _start(self, bar: _RunBar, tasks, on_done):
        if self._job is not None:
            return
        self._job = job = BatchJob(tasks, self)
        self._bar = bar
        bar.running(True)
        job.progress.connect(bar.set_progress)

        def finish():
            bar.running(False)
            self._job = None

        def ok(results):
            finish()
            on_done(results)

        def cancelled():
            finish()
            bar.status.setText("Cancelled.")

        def failed(tb):
            finish()
            bar.status.setText("Failed, see the message.")
            QtWidgets.QMessageBox.critical(self, "Run failed", tb[-2000:])

        job.succeeded.connect(ok)
        job.was_cancelled.connect(cancelled)
        job.failed.connect(failed)
        job.finished.connect(job.deleteLater)
        job.start()

    def _cancel(self):
        if self._job is not None:
            self._job.cancel()

    def busy(self) -> bool:
        return self._job is not None

    # ------------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------------ #
    def run_charts(self):
        try:
            base_t, with_t, aim, radar, names = self._chart_inputs()
        except ValueError as e:
            QtWidgets.QMessageBox.information(self, "Nothing to compare", str(e))
            return
        tasks = sweeps.chart_tasks(base_t, with_t, aim, radar, charts.ASPECTS)

        def done(results):
            base_db, with_db = sweeps.unpack_chart(results, charts.ASPECTS)
            self._chart_data = (sweeps.CHART_DISTANCES, base_db, with_db, names)
            self._show_charts()
            self.chart_bar.status.setText(f"{len(tasks)} solves at {radar.beam_width_deg:.0f}° cone, "
                                          f"{radar.elevation_deg:.0f}° elevation.")
            self.chart_bar.b_save.setEnabled(True)

        self._start(self.chart_bar, tasks, done)

    def _show_charts(self):
        self._clear(self.chart_lay)
        fig_power, fig_delta = charts.comparison_figures(*self._chart_data)
        for fig, h in ((fig_delta, 420), (fig_power, 720)):
            c = self._canvas(fig)
            c.setMinimumHeight(h)
            self.chart_lay.addWidget(c)

    def save_charts(self):
        if self._chart_data is None:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save charts to folder", str(Path.home()))
        if folder:
            paths = charts.save_comparison(folder, *self._chart_data)
            QtWidgets.QMessageBox.information(self, "Charts saved", "Saved:\n" + "\n".join(str(p) for p in paths))

    # ------------------------------------------------------------------ #
    # Aspect sweep
    # ------------------------------------------------------------------ #
    def run_sweep(self):
        try:
            target, base_t, aim, radar = self._sweep_inputs()
        except ValueError as e:
            QtWidgets.QMessageBox.information(self, "Nothing to sweep", str(e))
            return
        az, tasks = sweeps.aspect_tasks(target, aim, radar)
        n = len(tasks)
        if base_t is not None:
            tasks += sweeps.aspect_tasks(base_t, aim, radar)[1]

        def done(results):
            db = np.array([10 * np.log10(max(r[0], 1e-30)) for r in results])
            base_db = db[n:] if base_t is not None else None
            self._sweep_data = (az, db[:n], radar.azimuth_deg, radar.distance_m, base_db)
            self._show_sweep()
            self.sweep_bar.status.setText(f"{len(tasks)} solves at {radar.distance_m:.1f} m."
                                          + ("  Δ = change vs. biker alone." if base_t is not None else ""))
            self.sweep_bar.b_save.setEnabled(True)

        self._start(self.sweep_bar, tasks, done)

    def _show_sweep(self):
        az, db, cur, dist, base_db = self._sweep_data
        self._clear(self.sweep_plot)
        fig, ax = charts.sweep_figure(az, db, cur, dist, base_db)
        canvas = self._canvas(fig)
        self.sweep_plot.addWidget(canvas)
        hover = ax.annotate("", (0, 0), xytext=(10, -14), textcoords="offset points", fontsize=9, color=charts.INK,
                            bbox=dict(boxstyle="round", fc="white", ec="#d8d7d2"))
        hover.set_visible(False)

        def on_move(ev):
            if ev.inaxes is not ax or ev.xdata is None:
                if hover.get_visible():
                    hover.set_visible(False)
                    canvas.draw_idle()
                return
            deg = np.degrees(ev.xdata) % 360
            i = int(np.argmin(np.abs(((az - deg) + 180) % 360 - 180)))
            hover.xy = (np.radians(az[i]), db[i])
            extra = f"  ({db[i] - base_db[i]:+.1f} dB vs. alone)" if base_db is not None else ""
            hover.set_text(f"{az[i]:.0f}°  {db[i]:+.2f} dBsm{extra}")
            hover.set_visible(True)
            canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_move)

        cols = ["Bearing (°)", "RCS (dBsm)"] + (["Δ (dB)"] if base_db is not None else [])
        self.sweep_table.setColumnCount(len(cols))
        self.sweep_table.setHorizontalHeaderLabels(cols)
        self.sweep_table.setRowCount(len(az))
        for i, (a, d) in enumerate(zip(az, db)):
            vals = [f"{a:.0f}", f"{d:+.2f}"] + ([f"{d - base_db[i]:+.1f}"] if base_db is not None else [])
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.sweep_table.setItem(i, j, it)

    def save_sweep(self):
        if self._sweep_data is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save sweep", "aspect_sweep.csv", "CSV (*.csv)")
        if path:
            az, db, _, _, base_db = self._sweep_data
            charts.save_sweep_csv(path, az, db, base_db)
