"""Small custom widgets and helpers used by the main window."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

DIST_MIN, DIST_MAX, DIST_STEPS = 1.0, 200.0, 1000  # log-scaled distance slider
DIST_TICKS = (2.0, 10.0, 50.0)  # the range the comparison charts use
HEAT_TOP_DB = 50.0



def _dist_to_slider(d: float) -> int:
    return int(round(DIST_STEPS * np.log(d / DIST_MIN) / np.log(DIST_MAX / DIST_MIN)))


def _slider_to_dist(v: int) -> float:
    d = DIST_MIN * (DIST_MAX / DIST_MIN) ** (v / DIST_STEPS)
    return round(d * 2) / 2 if d >= 5 else round(d, 1)  # 0.5 m steps past 5 m, 0.1 m below


def _heat_rgb(face_db: np.ndarray, top: float) -> np.ndarray:
    from matplotlib import colormaps

    x = np.clip((face_db - (top - HEAT_TOP_DB)) / HEAT_TOP_DB, 0, 1)
    return (colormaps["inferno"](x)[:, :3] * 255).astype(np.uint8)


class TickLabels(QtWidgets.QWidget):
    """Labels under the log distance slider at 2 m / 10 m / 50 m."""

    def __init__(self, slider: QtWidgets.QSlider, parent=None):
        super().__init__(parent)
        self.slider = slider
        self.setFixedHeight(14)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        f = p.font()
        f.setPointSizeF(f.pointSizeF() * 0.8)
        p.setFont(f)
        p.setPen(self.palette().color(QtGui.QPalette.PlaceholderText))
        opt = QtWidgets.QStyleOptionSlider()
        self.slider.initStyleOption(opt)
        groove = self.slider.style().subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, self.slider)
        handle = self.slider.style().subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self.slider)
        x0 = groove.left() + handle.width() / 2
        span = groove.width() - handle.width()
        for d in DIST_TICKS:
            x = x0 + span * _dist_to_slider(d) / DIST_STEPS
            p.drawText(QtCore.QRectF(x - 20, 0, 40, 14), QtCore.Qt.AlignCenter, f"{d:g} m")
        p.end()


class HeatLegend(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.top: float | None = None
        self.setFixedSize(300, 30)

    def set_top(self, top: float | None):
        self.top = top
        self.update()

    def paintEvent(self, _):
        if self.top is None:
            return
        p = QtGui.QPainter(self)
        from matplotlib import colormaps

        cm = colormaps["inferno"]
        grad = QtGui.QLinearGradient(0, 0, 160, 0)
        for t in np.linspace(0, 1, 9):
            r, g, b, _ = cm(t)
            grad.setColorAt(t, QtGui.QColor.fromRgbF(r, g, b))
        p.fillRect(QtCore.QRectF(0, 2, 160, 10), grad)
        f = p.font()
        f.setPointSizeF(f.pointSizeF() * 0.8)
        p.setFont(f)
        p.setPen(self.palette().color(QtGui.QPalette.WindowText))
        p.drawText(QtCore.QRectF(0, 13, 160, 16), QtCore.Qt.AlignLeft, f"{self.top - HEAT_TOP_DB:+.0f}")
        p.drawText(QtCore.QRectF(0, 13, 160, 16), QtCore.Qt.AlignRight, f"{self.top:+.0f}")
        p.drawText(QtCore.QRectF(166, 0, 134, 28), QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap,
                   "dBsm per face,\ntop 50 dB")
        p.end()


def _swatch(hex_color: str) -> QtGui.QIcon:
    pm = QtGui.QPixmap(14, 14)
    pm.fill(QtGui.QColor(hex_color))
    return QtGui.QIcon(pm)


def _dspin(lo, hi, val, step, dec=3, suffix="") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s
