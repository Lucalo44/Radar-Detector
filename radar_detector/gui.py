"""Radar Detector desktop GUI (PySide6 + PyVista)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from vtkmodules.vtkRenderingCore import vtkCellPicker

from . import geometry as geo
from .materials import MaterialLibrary
from .rcs import RadarSettings, RcsResult, compute_rcs

MUTED = "color: palette(placeholder-text)"
HIGHLIGHT = np.array([255, 214, 0], dtype=np.uint8)
SIMPLE_RADAR = dict(azimuth_deg=0.0, elevation_deg=0.0, beam_width_deg=20.0, multibounce=True, max_bounces=3, ray_spacing_m=0.005)
MESH_FILTER = "3D geometry (*.stl *.obj *.ply *.vtk *.vtp *.glb *.gltf *.off);;All files (*)"


def _fmt_pct(frac: float) -> str:
    pct = 100.0 * frac
    if pct == 0:
        return "0 %"
    if pct >= 0.01:
        return f"{pct:.3f} %"
    return f"{pct:.3e} %"


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


# =========================================================================== #
# Aspect sweep dialog
# =========================================================================== #
class SweepDialog(QtWidgets.QDialog):
    INK, INK2, SURFACE, SERIES, MARK = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834"

    def __init__(self, parent, az: np.ndarray, dbsm: np.ndarray, current_az: float, distance: float):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.az, self.dbsm = az, dbsm
        self.setWindowTitle("Aspect sweep")
        self.resize(900, 560)
        lay = QtWidgets.QHBoxLayout(self)

        fig = Figure(figsize=(5.5, 5.5), facecolor=self.SURFACE)
        self.canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(projection="polar", facecolor=self.SURFACE)
        ax.set_theta_zero_location("S")  # 0 deg (radar behind rider) at the bottom, rider faces up
        ax.set_theta_direction(-1)  # 90 deg = rider's left
        th = np.radians(np.append(az, az[0] + 360))
        r = np.append(dbsm, dbsm[0])
        ax.plot(th, r, color=self.SERIES, lw=2)
        cur = float(np.interp(current_az % 360, np.append(az, 360), np.append(dbsm, dbsm[0])))
        ax.plot([np.radians(current_az)], [cur], "o", ms=9, color=self.MARK, mec=self.SURFACE, mew=2)
        ax.annotate(f"current {current_az:.0f}°: {cur:.1f} dBsm", (np.radians(current_az), cur),
                    xytext=(12, 8), textcoords="offset points", color=self.INK, fontsize=9)
        lo = np.floor(dbsm.min() / 5) * 5
        hi = np.ceil(dbsm.max() / 5) * 5 + 2
        ax.set_ylim(lo, hi)
        ax.grid(color="#d8d7d2", lw=0.6)
        ax.spines["polar"].set_color("#d8d7d2")
        ax.tick_params(colors=self.INK2, labelsize=8)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0° rear", "45°", "90° left", "135°", "180° front", "225°", "270° right", "315°"])
        ax.set_rlabel_position(22.5)
        ax.set_title(f"Total RCS (dBsm) vs. angle of incidence, {distance:.1f} m", color=self.INK, fontsize=11, pad=18)
        self._hover = ax.annotate("", (0, 0), xytext=(10, -14), textcoords="offset points", fontsize=9, color=self.INK,
                                  bbox=dict(boxstyle="round", fc="white", ec="#d8d7d2"))
        self._hover.set_visible(False)
        self._ax = ax
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        fig.tight_layout()
        lay.addWidget(self.canvas, 3)

        side = QtWidgets.QVBoxLayout()
        table = QtWidgets.QTableWidget(len(az), 2)
        table.setHorizontalHeaderLabels(["Angle (°)", "RCS (dBsm)"])
        for i, (a, d) in enumerate(zip(az, dbsm)):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{a:.0f}"))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{d:.2f}"))
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        side.addWidget(table)
        btn = QtWidgets.QPushButton("Save CSV…")
        btn.clicked.connect(self._save)
        side.addWidget(btn)
        lay.addLayout(side, 1)

    def _on_move(self, ev):
        if ev.inaxes is not self._ax or ev.xdata is None:
            if self._hover.get_visible():
                self._hover.set_visible(False)
                self.canvas.draw_idle()
            return
        deg = np.degrees(ev.xdata) % 360
        i = int(np.argmin(np.abs(((self.az - deg) + 180) % 360 - 180)))
        self._hover.xy = (np.radians(self.az[i]), self.dbsm[i])
        self._hover.set_text(f"{self.az[i]:.0f}°  {self.dbsm[i]:.2f} dBsm")
        self._hover.set_visible(True)
        self.canvas.draw_idle()

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save sweep", "aspect_sweep.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["azimuth_deg", "rcs_dbsm", "rcs_m2"])
                for a, d in zip(self.az, self.dbsm):
                    w.writerow([f"{a:.1f}", f"{d:.3f}", f"{10 ** (d / 10):.6g}"])


# =========================================================================== #
# Main window
# =========================================================================== #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Detector — 77 GHz bicycle RCS")
        self.resize(1500, 950)

        self.lib = MaterialLibrary()
        self.biker = geo.build_biker(self.lib)
        self.attachments: list[geo.SceneObject] = []
        self.selection: dict[int, set[int]] = {}  # id(obj) -> face ids
        self.result: RcsResult | None = None
        self.spans: list = []
        self.actor_map: dict[str, geo.SceneObject] = {}
        self._adj_cache: dict[int, tuple] = {}
        self._press_pos = None
        self._first_render = True

        self._debounce = QtCore.QTimer(self, singleShot=True, interval=250)
        self._debounce.timeout.connect(self.compute)

        # ---- layout ----
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(500)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        self.panel_layout = QtWidgets.QVBoxLayout(panel)
        scroll.setWidget(panel)
        split.addWidget(scroll)

        view = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(view)
        vlay.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        for text, fn in (("Fit bike", lambda: self.fit_view(False)), ("Fit whole scene", lambda: self.fit_view(True)),
                         ("Side view", lambda: self.set_view("side")), ("Rear view", lambda: self.set_view("rear")),
                         ("Top view", lambda: self.set_view("top"))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        self.view_hint = QtWidgets.QLabel("")
        self.view_hint.setStyleSheet(MUTED)
        bar.addWidget(self.view_hint)
        vlay.addLayout(bar)
        self.plotter = QtInteractor(view)
        self.plotter.set_background("#f4f4f2", top="#ffffff")
        vlay.addWidget(self.plotter.interactor, 1)
        split.addWidget(view)
        split.setStretchFactor(1, 1)

        self._build_mode_box()
        self._build_geometry_box()
        self._build_radar_box()
        self._build_faces_box()
        self._build_results_box()
        self.panel_layout.addStretch()

        iren = self.plotter.iren.interactor
        iren.AddObserver("LeftButtonPressEvent", self._on_lpress, 10.0)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_lrelease, 10.0)

        self.refresh_object_list()
        self.apply_mode()
        self.compute()

    # ------------------------------------------------------------------ #
    # Panel construction
    # ------------------------------------------------------------------ #
    def _group(self, title: str) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox(title)
        self.panel_layout.addWidget(g)
        return g

    def _build_mode_box(self):
        g = self._group("Mode")
        lay = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.rb_simple = QtWidgets.QRadioButton("Simple")
        self.rb_adv = QtWidgets.QRadioButton("Advanced")
        self.rb_simple.setChecked(True)
        row.addWidget(self.rb_simple)
        row.addWidget(self.rb_adv)
        row.addStretch()
        lay.addLayout(row)
        self.rb_simple.toggled.connect(self.apply_mode)
        self.cb_biker = QtWidgets.QCheckBox("Show baseline biker (uncheck to analyse geometry alone)")
        self.cb_biker.setChecked(True)
        self.cb_biker.toggled.connect(self._on_biker_toggle)
        lay.addWidget(self.cb_biker)

    def _build_geometry_box(self):
        g = self._group("Geometry")
        lay = QtWidgets.QVBoxLayout(g)
        self.obj_list = QtWidgets.QListWidget()
        self.obj_list.setMaximumHeight(130)
        self.obj_list.currentRowChanged.connect(self._on_object_selected)
        lay.addWidget(self.obj_list)

        row = QtWidgets.QHBoxLayout()
        b_imp = QtWidgets.QPushButton("Import geometry…")
        b_imp.clicked.connect(self.import_geometry)
        row.addWidget(b_imp)
        b_sample = QtWidgets.QToolButton()
        b_sample.setText("Add sample")
        b_sample.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(b_sample)
        menu.addAction("Corner reflector (10 cm, aluminum)", lambda: self.add_sample("corner"))
        menu.addAction("Flat plate (10 cm, aluminum)", lambda: self.add_sample("plate"))
        menu.addAction("Sphere (10 cm dia., aluminum)", lambda: self.add_sample("sphere"))
        b_sample.setMenu(menu)
        row.addWidget(b_sample)
        self.b_remove = QtWidgets.QPushButton("Remove")
        self.b_remove.clicked.connect(self.remove_object)
        row.addWidget(self.b_remove)
        lay.addLayout(row)

        form = QtWidgets.QFormLayout()
        mrow = QtWidgets.QHBoxLayout()
        self.obj_mat_combo = QtWidgets.QComboBox()
        self.obj_mat_combo.activated.connect(self._on_object_material)
        mrow.addWidget(self.obj_mat_combo, 1)
        self.obj_mat_mixed = QtWidgets.QLabel("")
        self.obj_mat_mixed.setStyleSheet("color:#d08a00")
        mrow.addWidget(self.obj_mat_mixed)
        form.addRow("Material:", mrow)
        self.obj_mat_info = QtWidgets.QLabel("")
        self.obj_mat_info.setStyleSheet(MUTED)
        self.obj_mat_info.setWordWrap(True)
        form.addRow("", self.obj_mat_info)

        self.pos = [_dspin(-10, 10, 0, 0.01, 3, " m") for _ in range(3)]
        self.rot = [_dspin(-360, 360, 0, 5, 1, "°") for _ in range(3)]
        self.scale = _dspin(0.001, 1000, 1, 0.1, 3, "×")
        for label, spins in (("Position X/Y/Z:", self.pos), ("Rotation X/Y/Z:", self.rot)):
            r = QtWidgets.QHBoxLayout()
            for s in spins:
                s.valueChanged.connect(self._on_transform)
                r.addWidget(s)
            form.addRow(label, r)
        self.scale.valueChanged.connect(self._on_transform)
        form.addRow("Scale:", self.scale)
        self.transform_widgets = self.pos + self.rot + [self.scale]
        lay.addLayout(form)
        hint = QtWidgets.QLabel("+X = direction of travel, +Y = rider's left, +Z = up, road at Z = 0.")
        hint.setStyleSheet(MUTED)
        lay.addWidget(hint)

    def _build_radar_box(self):
        g = self._group("Radar (77 GHz, λ = 3.89 mm)")
        form = QtWidgets.QFormLayout(g)
        drow = QtWidgets.QHBoxLayout()
        self.dist_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dist_slider.setRange(10, 2000)  # 0.1 m steps -> 1..200 m
        self.dist_slider.setValue(100)
        self.dist_spin = _dspin(1, 200, 10, 0.5, 1, " m")
        self.dist_slider.valueChanged.connect(lambda v: self.dist_spin.setValue(v / 10))
        self.dist_spin.valueChanged.connect(self._on_distance)
        drow.addWidget(self.dist_slider, 1)
        drow.addWidget(self.dist_spin)
        form.addRow("Distance:", drow)

        self.simple_radar_note = QtWidgets.QLabel("Fixed in Simple mode: radar directly behind the rider, 20° cone, multi-bounce on.")
        self.simple_radar_note.setWordWrap(True)
        self.simple_radar_note.setStyleSheet(MUTED)
        form.addRow(self.simple_radar_note)

        self.beam_spin = _dspin(1, 120, 20, 1, 1, "°")
        self.az_spin = _dspin(-180, 360, 0, 5, 1, "°")
        self.el_spin = _dspin(-45, 80, 0, 1, 1, "°")
        self.mb_check = QtWidgets.QCheckBox("Multi-bounce (shooting & bouncing rays)")
        self.mb_check.setChecked(True)
        self.bounce_spin = QtWidgets.QSpinBox()
        self.bounce_spin.setRange(2, 8)
        self.bounce_spin.setValue(3)
        self.spacing_spin = _dspin(0.5, 50, 5, 0.5, 1, " mm")
        self.adv_radar_rows = []
        for label, w in (("Cone width (-3 dB):", self.beam_spin), ("Angle of incidence (azimuth):", self.az_spin),
                         ("Elevation:", self.el_spin), ("", self.mb_check), ("Max bounces:", self.bounce_spin),
                         ("Ray spacing:", self.spacing_spin)):
            lab = QtWidgets.QLabel(label)
            form.addRow(lab, w)
            self.adv_radar_rows.append((lab, w))
        self.az_hint = QtWidgets.QLabel("Azimuth: 0° = radar behind rider, 90° = rider's left, 180° = head-on.")
        self.az_hint.setStyleSheet(MUTED)
        self.az_hint.setWordWrap(True)
        form.addRow(self.az_hint)
        for w in (self.beam_spin, self.az_spin, self.el_spin, self.spacing_spin):
            w.valueChanged.connect(self._schedule)
        self.bounce_spin.valueChanged.connect(self._schedule)
        self.mb_check.toggled.connect(self._schedule)

    def _build_faces_box(self):
        self.faces_box = g = self._group("Face materials (Advanced)")
        lay = QtWidgets.QVBoxLayout(g)
        self.pick_check = QtWidgets.QCheckBox("Pick faces with the mouse")
        self.pick_check.toggled.connect(lambda _: self.refresh_scene())
        lay.addWidget(self.pick_check)
        hint = QtWidgets.QLabel("Click toggles one face. Shift+click selects the smooth region around it "
                                "(faces within the grow angle). Drag still rotates the view.")
        hint.setWordWrap(True)
        hint.setStyleSheet(MUTED)
        lay.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        self.grow_angle = _dspin(0.5, 90, 20, 1, 1, "°")
        b_grow = QtWidgets.QPushButton("Grow selection")
        b_grow.clicked.connect(self.grow_selection)
        row.addWidget(b_grow)
        row.addWidget(QtWidgets.QLabel("angle"))
        row.addWidget(self.grow_angle)
        lay.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Select object", self.select_object_faces), ("Select by material", self.select_by_material),
                         ("Clear", self.clear_selection)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.sel_label = QtWidgets.QLabel("0 faces selected")
        lay.addWidget(self.sel_label)

        form = QtWidgets.QFormLayout()
        self.face_mat_combo = QtWidgets.QComboBox()
        self.face_mat_combo.currentIndexChanged.connect(self._load_material_values)
        form.addRow("Material:", self.face_mat_combo)
        self.R_spin = _dspin(0, 1, 0.5, 0.01, 3)
        self.A_spin = _dspin(0, 1, 0.5, 0.01, 3)
        self.s_spin = _dspin(0, 1, 0.0, 0.05, 2)
        self.T_label = QtWidgets.QLabel("")
        form.addRow("Reflectance R:", self.R_spin)
        form.addRow("Absorptance A:", self.A_spin)
        form.addRow("Transmittance T = 1−R−A:", self.T_label)
        form.addRow("Roughness (diffuse share):", self.s_spin)
        for s in (self.R_spin, self.A_spin):
            s.valueChanged.connect(self._update_T)
        lay.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        b_assign = QtWidgets.QPushButton("Assign to selected faces")
        b_assign.setStyleSheet("font-weight:600")
        b_assign.clicked.connect(self.assign_material)
        row.addWidget(b_assign)
        b_new = QtWidgets.QPushButton("Save as new material…")
        b_new.clicked.connect(self.new_material)
        row.addWidget(b_new)
        b_upd = QtWidgets.QPushButton("Update material")
        b_upd.clicked.connect(self.update_material)
        row.addWidget(b_upd)
        lay.addLayout(row)

    def _build_results_box(self):
        g = self._group("Results")
        lay = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.b_compute = QtWidgets.QPushButton("Compute")
        self.b_compute.setStyleSheet("font-weight:600; padding:6px 14px")
        self.b_compute.clicked.connect(self.compute)
        row.addWidget(self.b_compute)
        self.auto_check = QtWidgets.QCheckBox("Auto-update")
        self.auto_check.setChecked(True)
        row.addWidget(self.auto_check)
        row.addStretch()
        lay.addLayout(row)

        grid = QtWidgets.QGridLayout()
        big = "font-size:20px; font-weight:600"
        small = MUTED
        grid.addWidget(QtWidgets.QLabel("Returning radar power"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("Total radar cross section"), 0, 1)
        self.lbl_power = QtWidgets.QLabel("—")
        self.lbl_rcs = QtWidgets.QLabel("—")
        self.lbl_power.setStyleSheet(big)
        self.lbl_rcs.setStyleSheet(big)
        grid.addWidget(self.lbl_power, 1, 0)
        grid.addWidget(self.lbl_rcs, 1, 1)
        self.lbl_power_sub = QtWidgets.QLabel("")
        self.lbl_rcs_sub = QtWidgets.QLabel("")
        self.lbl_power_sub.setStyleSheet(small)
        self.lbl_rcs_sub.setStyleSheet(small)
        grid.addWidget(self.lbl_power_sub, 2, 0)
        grid.addWidget(self.lbl_rcs_sub, 2, 1)
        lay.addLayout(grid)

        self.adv_results = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(self.adv_results)
        al.setContentsMargins(0, 6, 0, 0)
        self.details = QtWidgets.QLabel("")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        al.addWidget(self.details)
        self.obj_table = QtWidgets.QTableWidget(0, 3)
        self.obj_table.setHorizontalHeaderLabels(["Object", "RCS (m²)", "dBsm"])
        self.obj_table.verticalHeader().setVisible(False)
        self.obj_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.obj_table.setMaximumHeight(150)
        al.addWidget(self.obj_table)
        row = QtWidgets.QHBoxLayout()
        self.heat_check = QtWidgets.QCheckBox("Show per-face RCS heat map")
        self.heat_check.toggled.connect(lambda _: self.refresh_scene())
        row.addWidget(self.heat_check)
        b_sweep = QtWidgets.QPushButton("Aspect sweep 0–360°")
        b_sweep.clicked.connect(self.sweep)
        row.addWidget(b_sweep)
        al.addLayout(row)
        lay.addWidget(self.adv_results)
        self.notes = QtWidgets.QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color:#d08a00")
        lay.addWidget(self.notes)

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    @property
    def advanced(self) -> bool:
        return self.rb_adv.isChecked()

    def objects(self) -> list[geo.SceneObject]:
        for o in self.biker:
            o.visible = self.cb_biker.isChecked()
        return self.biker + self.attachments

    def visible_objects(self):
        return [o for o in self.objects() if o.visible]

    def current_object(self) -> geo.SceneObject | None:
        item = self.obj_list.currentItem()
        return item.data(QtCore.Qt.UserRole) if item else None

    def radar_settings(self) -> RadarSettings:
        d = self.dist_spin.value()
        if not self.advanced:
            return RadarSettings(distance_m=d, **SIMPLE_RADAR)
        return RadarSettings(
            distance_m=d, azimuth_deg=self.az_spin.value(), elevation_deg=self.el_spin.value(),
            beam_width_deg=self.beam_spin.value(), multibounce=self.mb_check.isChecked(),
            max_bounces=self.bounce_spin.value(), ray_spacing_m=self.spacing_spin.value() / 1000,
        )

    def aim_point(self) -> np.ndarray:
        objs = self.visible_objects()
        if not objs:
            return np.array([0.5, 0, 0.8])
        b = np.array([o.world_mesh().bounds for o in objs])
        lo = np.array([b[:, 0].min(), b[:, 2].min(), b[:, 4].min()])
        hi = np.array([b[:, 1].max(), b[:, 3].max(), b[:, 5].max()])
        return 0.5 * (lo + hi)

    def _schedule(self, *_):
        self.result = None
        if self.auto_check.isChecked():
            self._debounce.start()
        else:
            self.refresh_scene()

    # ------------------------------------------------------------------ #
    # Mode / list management
    # ------------------------------------------------------------------ #
    def apply_mode(self):
        adv = self.advanced
        self.simple_radar_note.setVisible(not adv)
        for lab, w in self.adv_radar_rows:
            lab.setVisible(adv)
            w.setVisible(adv)
        self.az_hint.setVisible(adv)
        self.faces_box.setVisible(adv)
        self.adv_results.setVisible(adv)
        if not adv:
            self.pick_check.setChecked(False)
            self.heat_check.setChecked(False)
            self.clear_selection(refresh=False)
        self.view_hint.setText("Advanced: tick 'Pick faces' to select faces by clicking." if adv else "Simple mode")
        self.refresh_object_list()
        self._schedule()

    def _on_biker_toggle(self, _):
        self.refresh_object_list()
        self._schedule()

    def refresh_material_combos(self):
        for combo in (self.obj_mat_combo, self.face_mat_combo):
            cur = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            for m in self.lib.materials:
                combo.addItem(_swatch(m.color), m.name)
            combo.setCurrentIndex(max(cur, 0))
            combo.blockSignals(False)
        self._load_material_values()

    def refresh_object_list(self, select: geo.SceneObject | None = None):
        if self.obj_mat_combo.count() != len(self.lib):
            self.refresh_material_combos()
        prev = select or self.current_object()
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        for o in self.objects():
            if not o.visible:
                continue
            it = QtWidgets.QListWidgetItem(f"{o.name}  ·  {o.n_faces:,} faces")
            it.setData(QtCore.Qt.UserRole, o)
            if o.builtin:
                it.setText(it.text() + "  (baseline)" if "(baseline)" not in o.name else it.text())
            self.obj_list.addItem(it)
        self.obj_list.blockSignals(False)
        rows = [self.obj_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.obj_list.count())]
        if prev in rows:
            self.obj_list.setCurrentRow(rows.index(prev))
        elif self.attachments and self.attachments[-1] in rows:
            self.obj_list.setCurrentRow(rows.index(self.attachments[-1]))
        elif rows:
            self.obj_list.setCurrentRow(0)
        self._on_object_selected()

    def _on_object_selected(self, *_):
        o = self.current_object()
        enable = o is not None and not o.builtin
        for w in self.transform_widgets:
            w.setEnabled(enable)
            w.blockSignals(True)
        self.b_remove.setEnabled(enable)
        self.obj_mat_combo.setEnabled(o is not None)
        if o is not None:
            for s, v in zip(self.pos, o.position):
                s.setValue(float(v))
            for s, v in zip(self.rot, o.rotation_deg):
                s.setValue(float(v))
            self.scale.setValue(o.scale)
            mats, counts = np.unique(o.face_mat, return_counts=True)
            dominant = int(mats[np.argmax(counts)])
            self.obj_mat_combo.setCurrentIndex(dominant)
            self.obj_mat_mixed.setText("(mixed)" if len(mats) > 1 else "")
            m = self.lib[dominant]
            self.obj_mat_info.setText(f"R {m.reflectance:.2f} · A {m.absorptance:.2f} · T {m.transmittance:.2f} · rough {m.roughness:.2f}"
                                      + (f" — {m.note}" if m.note else ""))
        for w in self.transform_widgets:
            w.blockSignals(False)

    def _on_object_material(self, idx: int):
        o = self.current_object()
        if o is None:
            return
        o.set_material(idx)
        self._on_object_selected()
        self._schedule()

    def _on_transform(self, *_):
        o = self.current_object()
        if o is None or o.builtin:
            return
        o.position = np.array([s.value() for s in self.pos])
        o.rotation_deg = np.array([s.value() for s in self.rot])
        o.scale = self.scale.value()
        self._schedule()

    def _on_distance(self, v: float):
        self.dist_slider.blockSignals(True)
        self.dist_slider.setValue(int(round(v * 10)))
        self.dist_slider.blockSignals(False)
        self._schedule()

    # ------------------------------------------------------------------ #
    # Geometry actions
    # ------------------------------------------------------------------ #
    def import_geometry(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import geometry", str(Path.home()), MESH_FILTER)
        if not path:
            return
        units, ok = QtWidgets.QInputDialog.getItem(
            self, "Units", "What units is this file in?", list(geo.UNIT_SCALES), 2, False)
        if not ok:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            mesh = geo.load_mesh(path, units)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Import failed", f"Could not read {path}:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        size = np.ptp(np.asarray(mesh.points), axis=0)
        if size.max() > 3:
            QtWidgets.QMessageBox.warning(self, "Large part",
                f"This part is {size.max():.2f} m across. If that's wrong, re-import with different units or use Scale.")
        if mesh.n_cells > 300_000:
            QtWidgets.QMessageBox.information(self, "Dense mesh",
                f"{mesh.n_cells:,} triangles. This will work but computes slowly; decimating in your CAD tool helps.")
        obj = geo.make_attachment(Path(path).stem, mesh, self.lib.index("ABS / hard plastic"), self.cb_biker.isChecked())
        self.attachments.append(obj)
        self.refresh_object_list(select=obj)
        self.fit_view(False)
        self._schedule()

    def add_sample(self, kind: str):
        al = self.lib.index("Aluminum")
        if kind == "corner":
            mesh, name = geo.corner_reflector(0.10, opening=(-1, 0, 0)), "Corner reflector"
        elif kind == "plate":
            mesh, name = geo.flat_plate(0.10, normal=(-1, 0, 0)), "Flat plate"
        else:
            mesh, name = geo.sphere((0, 0, 0), 0.05), "Sphere"
        obj = geo.make_attachment(name, mesh, al, self.cb_biker.isChecked())
        self.attachments.append(obj)
        self.refresh_object_list(select=obj)
        self._schedule()

    def remove_object(self):
        o = self.current_object()
        if o is None or o.builtin:
            return
        self.attachments.remove(o)
        self.selection.pop(id(o), None)
        self.refresh_object_list()
        self._schedule()

    # ------------------------------------------------------------------ #
    # Face selection
    # ------------------------------------------------------------------ #
    def _sel_count(self) -> int:
        return sum(len(v) for v in self.selection.values())

    def _update_sel_label(self):
        self.sel_label.setText(f"{self._sel_count():,} faces selected")

    def _obj_by_id(self, oid: int):
        for o in self.objects():
            if id(o) == oid:
                return o
        return None

    def _adjacency_labels(self, obj: geo.SceneObject, angle_deg: float) -> np.ndarray:
        key = (id(obj.base_mesh), round(angle_deg, 3))
        if key in self._adj_cache:
            return self._adj_cache[key]
        m = obj.base_mesh
        tm = trimesh.Trimesh(np.asarray(m.points), m.faces.reshape(-1, 4)[:, 1:], process=False)
        adj, ang = tm.face_adjacency, tm.face_adjacency_angles
        ok = adj[ang < np.radians(angle_deg)]
        n = len(tm.faces)
        graph = coo_matrix((np.ones(len(ok)), (ok[:, 0], ok[:, 1])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        self._adj_cache[key] = labels
        return labels

    def _grow(self, obj, seeds) -> set[int]:
        labels = self._adjacency_labels(obj, self.grow_angle.value())
        return set(np.nonzero(np.isin(labels, labels[list(seeds)]))[0].tolist())

    def grow_selection(self):
        for oid, faces in list(self.selection.items()):
            obj = self._obj_by_id(oid)
            if obj is not None and faces:
                self.selection[oid] = self._grow(obj, faces)
        self._update_sel_label()
        self.refresh_scene()

    def select_object_faces(self):
        o = self.current_object()
        if o is not None:
            self.selection[id(o)] = set(range(o.n_faces))
            self._update_sel_label()
            self.refresh_scene()

    def select_by_material(self):
        idx = self.face_mat_combo.currentIndex()
        for o in self.visible_objects():
            ids = set(np.nonzero(o.face_mat == idx)[0].tolist())
            if ids:
                self.selection[id(o)] = ids
        self._update_sel_label()
        self.refresh_scene()

    def clear_selection(self, refresh: bool = True):
        self.selection.clear()
        self._update_sel_label()
        if refresh:
            self.refresh_scene()

    def _on_lpress(self, iren, _evt):
        self._press_pos = iren.GetEventPosition()

    def _on_lrelease(self, iren, _evt):
        if not (self.advanced and self.pick_check.isChecked()) or self._press_pos is None:
            return
        x, y = iren.GetEventPosition()
        if abs(x - self._press_pos[0]) + abs(y - self._press_pos[1]) > 4:
            return  # it was a drag (rotate), not a click
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return
        actor, cell = picker.GetActor(), picker.GetCellId()
        if actor is None or cell < 0:
            return
        obj = self.actor_map.get(actor.GetAddressAsString("vtkObject"))
        if obj is None:
            return
        sel = self.selection.setdefault(id(obj), set())
        if iren.GetShiftKey():
            sel |= self._grow(obj, [cell])
        elif cell in sel:
            sel.discard(cell)
        else:
            sel.add(cell)
        rows = [self.obj_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.obj_list.count())]
        if obj in rows and self.current_object() is not obj:
            self.obj_list.setCurrentRow(rows.index(obj))
        self._update_sel_label()
        self.refresh_scene()

    # ------------------------------------------------------------------ #
    # Materials (advanced)
    # ------------------------------------------------------------------ #
    def _load_material_values(self, *_):
        i = self.face_mat_combo.currentIndex()
        if i < 0:
            return
        m = self.lib[i]
        for s, v in ((self.R_spin, m.reflectance), (self.A_spin, m.absorptance), (self.s_spin, m.roughness)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self._update_T()

    def _update_T(self, *_):
        R, A = self.R_spin.value(), self.A_spin.value()
        if R + A > 1:
            self.A_spin.blockSignals(True)
            self.A_spin.setValue(1 - R)
            self.A_spin.blockSignals(False)
            A = 1 - R
        self.T_label.setText(f"{max(0.0, 1 - R - A):.3f}")

    def assign_material(self):
        if not self._sel_count():
            QtWidgets.QMessageBox.information(self, "No faces selected", "Tick 'Pick faces' and click faces first.")
            return
        idx = self.face_mat_combo.currentIndex()
        for oid, faces in self.selection.items():
            obj = self._obj_by_id(oid)
            if obj is not None and faces:
                obj.face_mat[list(faces)] = idx
        self.clear_selection(refresh=False)
        self._on_object_selected()
        self._schedule()

    def new_material(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New material", "Name:", text="Custom material")
        if not ok:
            return
        idx = self.lib.add(name, self.R_spin.value(), self.A_spin.value(), self.s_spin.value())
        self.refresh_material_combos()
        self.face_mat_combo.setCurrentIndex(idx)

    def update_material(self):
        i = self.face_mat_combo.currentIndex()
        m = self.lib[i]
        m.reflectance, m.absorptance, m.roughness = self.R_spin.value(), self.A_spin.value(), self.s_spin.value()
        m.__post_init__()
        self._on_object_selected()
        self._schedule()

    # ------------------------------------------------------------------ #
    # Compute & display
    # ------------------------------------------------------------------ #
    def compute(self):
        from .geometry import build_target

        objs = self.visible_objects()
        target, spans = build_target(objs, self.lib)
        if target is None:
            self.result, self.spans = None, []
            self.lbl_power.setText("—")
            self.lbl_rcs.setText("—")
            self.lbl_power_sub.setText("No geometry in the scene")
            self.lbl_rcs_sub.setText("")
            self.refresh_scene()
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            radar = self.radar_settings()
            res = compute_rcs(target, radar, self.aim_point())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.result, self.spans = res, spans
        self.lbl_power.setText(_fmt_pct(res.returned_fraction))
        self.lbl_power_sub.setText(f"of transmitted power ({res.returned_db:.1f} dB)")
        self.lbl_rcs.setText(f"{res.rcs_m2:.3g} m²")
        self.lbl_rcs_sub.setText(f"{res.rcs_dbsm:+.1f} dBsm")
        self.details.setText(
            f"<b>Breakdown</b> — specular {res.specular_m2:.3g} m² · diffuse {res.diffuse_m2:.3g} m² · "
            f"multi-bounce {res.multibounce_m2:.3g} m² (these add coherently, so they need not sum to the total).<br>"
            f"<b>Beam on target</b> — {100 * res.intercepted_fraction:.2f}% of transmitted power hits the object. "
            f"Of that, {res.reflected_pct:.0f}% is reflected, {res.absorbed_pct:.0f}% absorbed, "
            f"{res.passed_through_pct:.0f}% passes through.<br>"
            f"<b>Radar</b> — gain {res.gain_dbi:.1f} dBi · {res.lit_faces:,} lit faces · "
            f"{res.rays_launched:,} rays ({res.multibounce_rays:,} multi-bounce returns)"
        )
        self.obj_table.setRowCount(len(res.object_rcs_m2))
        for i, (name, val) in enumerate(sorted(res.object_rcs_m2.items(), key=lambda kv: -kv[1])):
            self.obj_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.obj_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{val:.4g}"))
            self.obj_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{10 * np.log10(max(val, 1e-30)):.1f}"))
        self.notes.setText("\n".join(res.notes))
        self.refresh_scene()

    def refresh_scene(self):
        p = self.plotter
        cam = None if self._first_render else p.camera_position
        p.clear()
        self.actor_map = {}
        _, _, _, rgb_lib = self.lib.arrays()
        adv = self.advanced
        heat = adv and self.heat_check.isChecked() and self.result is not None
        face_db = None
        if heat:
            face_db = 10 * np.log10(np.maximum(self.result.face_rcs_m2, 1e-12))
            top = face_db.max()
            clim = (top - 50, top)
        span_map = {id(o): (a, b) for o, a, b in self.spans}
        bar_shown = False
        for obj in self.visible_objects():
            m = obj.world_mesh().copy()
            if heat and id(obj) in span_map:
                a, b = span_map[id(obj)]
                m.cell_data["RCS per face (dBsm)"] = face_db[a:b]
                actor = p.add_mesh(m, scalars="RCS per face (dBsm)", cmap="inferno", clim=clim, show_scalar_bar=not bar_shown,
                                   scalar_bar_args=dict(title="RCS per face (dBsm)", color="#0b0b0b", vertical=True))
                bar_shown = True
            else:
                colors = rgb_lib[obj.face_mat].copy()
                sel = self.selection.get(id(obj))
                if sel:
                    colors[list(sel)] = HIGHLIGHT
                m.cell_data["colors"] = colors
                show_edges = adv and self.pick_check.isChecked()
                actor = p.add_mesh(m, scalars="colors", rgb=True, show_edges=show_edges, edge_color="#6b6b6b",
                                   line_width=0.3, ambient=0.25, diffuse=0.8, specular=0.15)
            self.actor_map[actor.GetAddressAsString("vtkObject")] = obj

        # ---- radar, beam cone, distance ----
        radar = self.radar_settings()
        aim = self.aim_point()
        p_r = radar.position(aim)
        bore = (aim - p_r) / radar.distance_m
        L = radar.distance_m * 1.08
        half = np.radians(radar.beam_width_deg / 2)
        cone = pv.Cone(center=p_r + bore * L / 2, direction=-bore, height=L, radius=L * np.tan(half), resolution=64, capping=False)
        p.add_mesh(cone, color="#eb6834", opacity=0.10, pickable=False)
        p.add_mesh(pv.Cube(center=p_r, x_length=0.18, y_length=0.18, z_length=0.10), color="#c0392b", pickable=False)
        p.add_mesh(pv.Line(p_r, aim), color="#c0392b", line_width=2, pickable=False)
        p.add_mesh(pv.Sphere(radius=0.025, center=aim), color="#c0392b", pickable=False)
        p.add_point_labels([0.5 * (p_r + aim)], [f"{radar.distance_m:.1f} m"], font_size=16, point_size=0,
                           shape_color="white", shape_opacity=0.8, text_color="#0b0b0b", always_visible=True, pickable=False)
        p.add_point_labels([p_r + np.array([0, 0, 0.25])], ["Radar 77 GHz"], font_size=12, point_size=0,
                           shape_opacity=0.0, text_color="#c0392b", always_visible=True, pickable=False)
        # ---- road ----
        c = 0.5 * (p_r + aim)
        size = max(4.0, 1.3 * np.linalg.norm(p_r[:2] - aim[:2]) + 3)
        ground = pv.Plane(center=(c[0], c[1], 0), direction=(0, 0, 1), i_size=size, j_size=size, i_resolution=20, j_resolution=20)
        p.add_mesh(ground, color="#d9d9d4", opacity=0.5, show_edges=True, edge_color="#c4c4be", pickable=False)
        p.add_axes()
        p.add_text(f"Radar distance: {radar.distance_m:.1f} m   ·   azimuth {radar.azimuth_deg:.0f}°   ·   cone {radar.beam_width_deg:.0f}°",
                   position="upper_left", font_size=11, color="#0b0b0b")

        if self._first_render:
            self._first_render = False
            self.set_view("iso")
        elif cam is not None:
            p.camera_position = cam
        p.render()

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #
    def _bounds(self, whole: bool):
        objs = self.visible_objects()
        pts = [np.array(o.world_mesh().bounds).reshape(3, 2).T for o in objs]
        if whole or not pts:
            radar = self.radar_settings()
            p_r = radar.position(self.aim_point())
            pts.append(np.vstack([p_r - 0.2, p_r + 0.2]))
        allp = np.vstack(pts)
        lo, hi = allp.min(0), allp.max(0)
        return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    def fit_view(self, whole: bool):
        self.plotter.reset_camera(bounds=self._bounds(whole))
        self.plotter.render()

    def set_view(self, which: str):
        p = self.plotter
        aim = self.aim_point()
        dirs = {"side": (0, 1, 0), "rear": (-1, 0, 0), "top": (0, 0, 1), "iso": (-1.0, 1.3, 0.8)}
        d = np.array(dirs[which], float)
        d /= np.linalg.norm(d)
        up = (1, 0, 0) if which == "top" else (0, 0, 1)
        p.camera_position = [tuple(aim + 5 * d), tuple(aim), up]
        p.reset_camera(bounds=self._bounds(which == "iso" and self.dist_spin.value() < 15))
        p.render()

    # ------------------------------------------------------------------ #
    # Aspect sweep
    # ------------------------------------------------------------------ #
    def sweep(self):
        target, _ = geo.build_target(self.visible_objects(), self.lib)
        if target is None:
            return
        base = self.radar_settings()
        aim = self.aim_point()
        az = np.arange(0, 360, 5.0)
        out = np.zeros_like(az)
        dlg = QtWidgets.QProgressDialog("Sweeping angle of incidence…", "Cancel", 0, len(az), self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(0)
        for i, a in enumerate(az):
            if dlg.wasCanceled():
                return
            r = RadarSettings(**{**base.__dict__, "azimuth_deg": float(a)})
            out[i] = compute_rcs(target, r, aim).rcs_dbsm
            dlg.setValue(i + 1)
            QtWidgets.QApplication.processEvents()
        SweepDialog(self, az, out, base.azimuth_deg, base.distance_m).exec()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Radar Detector")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
