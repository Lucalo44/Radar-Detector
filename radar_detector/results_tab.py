"""Results tab: the headline numbers, the per-object table and the solver's notes."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import compare as cmp
from .rcs import RadarSettings, RcsResult

MUTED = "color: palette(placeholder-text)"
VIS_COLORS = {"more": "#178a5c", "same": "#8c8b85", "less": "#c77700"}
VIS_WORDS = {"more": "more visible to the radar", "same": "within the solver's ±1 dB noise",
             "less": "less visible to the radar"}


def help_button(tip: str) -> QtWidgets.QToolButton:
    b = QtWidgets.QToolButton()
    b.setText("?")
    b.setToolTip(tip)
    b.setAutoRaise(True)
    b.setStyleSheet("QToolButton{font-weight:600; border:1px solid palette(mid); border-radius:9px;"
                    "min-width:16px; max-width:16px; min-height:16px; max-height:16px; padding:0}")
    return b


class ReferenceScale(QtWidgets.QWidget):
    """A dBsm axis with the typical pedestrian / cyclist / car ranges and a marker for the result."""

    LO, HI = -20.0, 30.0
    FILLS = ["#d9e6f5", "#d8efe4", "#f6e3cf"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value: float | None = None
        self.base: float | None = None
        self.setMinimumHeight(54)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_values(self, value: float | None, base: float | None):
        self.value, self.base = value, base
        self.update()

    def _x(self, db: float, w: float) -> float:
        return 20 + (np.clip(db, self.LO, self.HI) - self.LO) / (self.HI - self.LO) * (w - 40)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        ink = self.palette().color(QtGui.QPalette.WindowText)
        bar_top, bar_h = 14, 16
        font = p.font()
        font.setPointSizeF(font.pointSizeF() * 0.85)
        p.setFont(font)
        for (name, lo, hi), fill in zip(cmp.REFERENCE_RANGES, self.FILLS):
            x0, x1 = self._x(lo, w), self._x(hi, w)
            p.fillRect(QtCore.QRectF(x0, bar_top, x1 - x0, bar_h), QtGui.QColor(fill))
            p.setPen(QtGui.QColor("#3d3c39"))
            p.drawText(QtCore.QRectF(x0, bar_top, x1 - x0, bar_h), QtCore.Qt.AlignCenter, name)
        p.setPen(ink)
        for db in range(int(self.LO), int(self.HI) + 1, 10):
            x = self._x(db, w)
            p.drawLine(QtCore.QPointF(x, bar_top + bar_h), QtCore.QPointF(x, bar_top + bar_h + 4))
            p.drawText(QtCore.QRectF(x - 20, bar_top + bar_h + 4, 40, 14), QtCore.Qt.AlignCenter, f"{db:+d}" if db else "0")
        if self.base is not None:
            x = self._x(self.base, w)
            p.setPen(QtGui.QPen(QtGui.QColor("#8c8b85"), 2, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(x, bar_top - 2), QtCore.QPointF(x, bar_top + bar_h + 2))
        if self.value is not None:
            x = self._x(self.value, w)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor("#2a78d6"))
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x - 6, 2), QtCore.QPointF(x + 6, 2), QtCore.QPointF(x, bar_top + 2)]))
            p.setPen(QtGui.QPen(QtGui.QColor("#2a78d6"), 2))
            p.drawLine(QtCore.QPointF(x, bar_top), QtCore.QPointF(x, bar_top + bar_h))
        p.end()


class ResultsTab(QtWidgets.QWidget):
    help_requested = QtCore.Signal(str, str)  # page, heading
    object_clicked = QtCore.Signal(str)
    unit_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(4)

        # ---- 1. change vs. biker alone ----
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>Change vs. biker alone</b>"))
        b = help_button("What do these numbers mean?")
        b.clicked.connect(lambda: self.help_requested.emit("numbers", ""))
        row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.lbl_delta = QtWidgets.QLabel("—")
        self.lbl_delta.setStyleSheet("font-size:34px; font-weight:700")
        lay.addWidget(self.lbl_delta)
        self.lbl_delta_sub = QtWidgets.QLabel("")
        self.lbl_delta_sub.setWordWrap(True)
        lay.addWidget(self.lbl_delta_sub)
        self.lbl_band = QtWidgets.QLabel("")
        self.lbl_band.setStyleSheet(MUTED)
        self.lbl_band.setWordWrap(True)
        lay.addWidget(self.lbl_band)
        lay.addSpacing(14)

        # ---- 2. total RCS ----
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("<b>Total radar cross section</b>"))
        row.addStretch()
        row.addWidget(QtWidgets.QLabel("Units:"))
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["dBsm", "m²"])
        self.unit_combo.currentTextChanged.connect(self._on_unit)
        row.addWidget(self.unit_combo)
        lay.addLayout(row)
        self.lbl_rcs = QtWidgets.QLabel("—")
        self.lbl_rcs.setStyleSheet("font-size:22px; font-weight:600")
        lay.addWidget(self.lbl_rcs)
        self.scale = ReferenceScale()
        lay.addWidget(self.scale)
        self.lbl_scale = QtWidgets.QLabel("")
        self.lbl_scale.setStyleSheet(MUTED)
        self.lbl_scale.setWordWrap(True)
        lay.addWidget(self.lbl_scale)
        self.lbl_sphere = QtWidgets.QLabel("")
        self.lbl_sphere.setWordWrap(True)
        lay.addWidget(self.lbl_sphere)
        lay.addSpacing(8)
        self.lbl_power = QtWidgets.QLabel("")
        self.lbl_power.setStyleSheet(MUTED)
        self.lbl_power.setWordWrap(True)
        lay.addWidget(self.lbl_power)
        lay.addSpacing(14)

        # ---- 3. per-object table ----
        lay.addWidget(QtWidgets.QLabel("<b>Where the return comes from</b>"))
        hint = QtWidgets.QLabel("Click a row to show that object in the 3D view.")
        hint.setStyleSheet(MUTED)
        lay.addWidget(hint)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setMinimumHeight(150)
        self.table.cellClicked.connect(lambda r, _c: self._on_row(r))
        lay.addWidget(self.table)

        # ---- 4. breakdown (Advanced) and notes ----
        self.details = QtWidgets.QLabel("")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addSpacing(8)
        lay.addWidget(self.details)
        self.notes = QtWidgets.QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color:#c77700")
        lay.addWidget(self.notes)
        lay.addStretch()
        self.set_advanced(False)

    # ------------------------------------------------------------------ #
    @property
    def unit(self) -> str:
        return self.unit_combo.currentText()

    def set_unit(self, unit: str):
        if unit in ("dBsm", "m²"):
            self.unit_combo.setCurrentText(unit)

    def set_advanced(self, adv: bool):
        self.details.setVisible(adv)

    def _on_unit(self, unit: str):
        if self._last is not None:
            self.show_result(*self._last)
        self.unit_changed.emit(unit)

    def _on_row(self, r: int):
        it = self.table.item(r, 0)
        if it is not None:
            self.object_clicked.emit(it.data(QtCore.Qt.UserRole))

    # ------------------------------------------------------------------ #
    def show_empty(self, message: str):
        self._last = None
        self.lbl_delta.setText("—")
        self.lbl_delta.setStyleSheet("font-size:34px; font-weight:700")
        self.lbl_delta_sub.setText(message)
        for w in (self.lbl_band, self.lbl_rcs, self.lbl_scale, self.lbl_sphere, self.lbl_power, self.details, self.notes):
            w.setText("")
        self.lbl_rcs.setText("—")
        self.scale.set_values(None, None)
        self.table.setRowCount(0)

    def show_result(self, res: RcsResult, base: RcsResult | None, radar: RadarSettings, attachments: set[str],
                    no_base_reason: str = ""):
        self._last = (res, base, radar, attachments, no_base_reason)
        unit = self.unit
        # ---- 1. change vs. biker alone ----
        if base is not None:
            d = cmp.delta_db(res.rcs_m2, base.rcs_m2)
            vis = cmp.visibility(d)
            self.lbl_delta.setText(f"{d:+.1f} dB")
            self.lbl_delta.setStyleSheet(f"font-size:34px; font-weight:700; color:{VIS_COLORS[vis]}")
            self.lbl_delta_sub.setText(f"{cmp.sentence_case(cmp.ratio_text(d))}: {VIS_WORDS[vis]}. "
                                       f"Biker alone: {cmp.fmt_rcs(base.rcs_m2, unit)}.")
        else:
            self.lbl_delta.setText("—")
            self.lbl_delta.setStyleSheet("font-size:34px; font-weight:700")
            self.lbl_delta_sub.setText(no_base_reason)
        band = cmp.sentence_case(cmp.band_text(radar))
        if res.spread_db > 0:
            band += f". Across those samples the total moves by {res.spread_db:.1f} dB, which is why one frequency alone isn't trustworthy."
        self.lbl_band.setText(band)

        # ---- 2. total RCS ----
        self.lbl_rcs.setText(cmp.fmt_rcs_pair(res.rcs_m2, unit))
        self.scale.set_values(res.rcs_dbsm, base.rcs_dbsm if base is not None else None)
        self.lbl_scale.setText(f"{res.rcs_dbsm:+.1f} dBsm is {cmp.reference_label(res.rcs_dbsm)}. "
                               "Typical ranges from literature, rough and for orientation only (sources in Docs)."
                               + ("  Dashed line: biker alone." if base is not None else ""))
        dia = cmp.equivalent_sphere_m(res.rcs_m2)
        self.lbl_sphere.setText(f"Equivalent to a <b>{cmp.fmt_length(dia)}</b> metal sphere. "
                                "That's a size comparison for the same RCS, not a claim about the part's size.")
        self.lbl_power.setText(f"Returned power: {res.returned_db:.1f} dB re transmitted at {radar.distance_m:.1f} m "
                               f"(falls as 1/R⁴, so it mostly tracks distance; the change above is what your design controls).")

        # ---- 3. table ----
        rows = cmp.object_rows(res, base, attachments)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Object", f"RCS ({unit})", "Share", "vs. biker alone"])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            name = QtWidgets.QTableWidgetItem(r.name)
            name.setData(QtCore.Qt.UserRole, r.name)
            val = QtWidgets.QTableWidgetItem(f"{r.dbsm:+.1f}" if unit == "dBsm" else f"{r.rcs_m2:.3g}")
            share = QtWidgets.QTableWidgetItem(f"{r.share_pct:.0f} %")
            vs = QtWidgets.QTableWidgetItem("" if r.vs_base_db is None else f"{r.vs_base_db:+.1f} dB")
            if r.attachment:
                vs.setToolTip("How much this part alone would lift the biker's total return.")
            for j, it in enumerate((name, val, share, vs)):
                if j:
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        # ---- 4. breakdown ----
        f = lambda m2: cmp.fmt_rcs(m2, unit)  # noqa: E731
        self.details.setText(
            f"<b>Breakdown</b>: specular {f(res.specular_m2)} · diffuse {f(res.diffuse_m2)} · "
            f"multi-bounce {f(res.multibounce_m2)}"
            + (f" · retroreflective {f(res.retro_m2)}" if res.retro_m2 > 0 else "")
            + " (these add coherently, so they need not sum to the total).<br>"
            f"<b>Beam on target</b>: {100 * res.intercepted_fraction:.2f}% of transmitted power hits the scene. "
            f"Of that, {res.reflected_pct:.0f}% is reflected, {res.absorbed_pct:.0f}% absorbed and "
            f"{res.passed_through_pct:.0f}% passes through.<br>"
            f"<b>Radar</b>: gain {res.gain_dbi:.1f} dBi · {res.lit_faces:,} lit faces · "
            f"{res.rays_launched:,} rays ({res.multibounce_rays:,} multi-bounce returns)")
        self.notes.setText("\n".join(res.notes))
