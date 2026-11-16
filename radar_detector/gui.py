"""Radar Detector desktop GUI (PySide6 + PyVista)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import compare as cmp
from . import geometry as geo
from .compare_tab import CompareTab
from .docs_tab import DocsTab
from .jobs import SolveReply, SolveRequest, SolveService
from .materials import MaterialLibrary
from .rcs import RadarSettings, RcsResult, ray_engine
from .project_ui import ProjectMixin
from .results_tab import VIS_COLORS, ResultsTab, help_button
from .scene_view import SceneMixin
from .widgets import DIST_MAX, DIST_MIN, DIST_STEPS, HeatLegend, TickLabels, _dist_to_slider, _dspin, _slider_to_dist, _swatch

MUTED = "color: palette(placeholder-text)"
SIMPLE_RADAR = dict(azimuth_deg=0.0, elevation_deg=0.0, beam_width_deg=20.0, multibounce=True, max_bounces=3,
                    ray_spacing_m=0.005, bandwidth_hz=1e9, n_freq=9, aspect_avg_deg=0.0)
MESH_FILTER = "3D geometry (*.stl *.obj *.ply *.vtk *.vtp *.glb *.gltf *.off);;All files (*)"
DECIMATE_OFFER = 50_000  # offer decimation on import above this many faces




# =========================================================================== #
# Main window
# =========================================================================== #
class MainWindow(SceneMixin, ProjectMixin, QtWidgets.QMainWindow):
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
        self.base_result: RcsResult | None = None
        self.baseline = cmp.BaselineCache()
        self._gen = 0  # solve generation; replies from older generations are dropped
        self._scene_actors: dict[int, tuple] = {}  # id(obj) -> (actor, rendered mesh, source world mesh)
        self._radar_actors: list = []
        self._radar_key = None
        self._flash_obj: geo.SceneObject | None = None
        self._flash_timer = QtCore.QTimer(self, singleShot=True, interval=1500)
        self._flash_timer.timeout.connect(self._end_flash)

        self._debounce = QtCore.QTimer(self, singleShot=True, interval=250)
        self._debounce.timeout.connect(self.compute)
        self.solver = SolveService(self)
        self.solver.finished.connect(self._on_solved)

        # ---- layout ----
        central = QtWidgets.QWidget()
        clay = QtWidgets.QVBoxLayout(central)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(0)
        self.setCentralWidget(central)
        self.engine_banner = QtWidgets.QLabel("")
        self.engine_banner.setWordWrap(True)
        self.engine_banner.setStyleSheet("background:#fff1d6; color:#5c3d00; padding:8px 12px; border-bottom:1px solid #e8c98a")
        self.engine_banner.setVisible(False)
        clay.addWidget(self.engine_banner)
        split = QtWidgets.QSplitter()
        clay.addWidget(split, 1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(500)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        self.panel_layout = QtWidgets.QVBoxLayout(panel)
        scroll.setWidget(panel)
        # Left column: the geometry list stays pinned at the top; everything else scrolls beneath it
        left = QtWidgets.QWidget()
        left.setMinimumWidth(500)
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)
        self.pinned_layout = QtWidgets.QVBoxLayout()
        self.pinned_layout.setContentsMargins(9, 9, 9, 4)
        left_lay.addLayout(self.pinned_layout)
        left_lay.addWidget(scroll, 1)
        split.addWidget(left)

        # Right side: tabs. Scene (3D view), Results, Compare, Docs
        self.tabs = QtWidgets.QTabWidget()
        view = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(view)
        vlay.setContentsMargins(0, 4, 0, 0)
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
        hbar = QtWidgets.QHBoxLayout()
        self.heat_check = QtWidgets.QCheckBox("Show where the return comes from (per-face heat map)")
        self.heat_check.toggled.connect(lambda _: self.refresh_scene())
        hbar.addWidget(self.heat_check)
        self.heat_legend = HeatLegend()
        hbar.addWidget(self.heat_legend)
        hbar.addStretch()
        vlay.addLayout(hbar)
        self.plotter = QtInteractor(view)
        self.plotter.set_background("#f4f4f2", top="#ffffff")
        self.plotter.add_axes()
        vlay.addWidget(self.plotter.interactor, 1)
        self.tabs.addTab(view, "Scene")
        self.results_tab = ResultsTab()
        self.results_tab.help_requested.connect(self.show_docs)
        self.results_tab.object_clicked.connect(self._show_object_by_name)
        self.results_tab.unit_changed.connect(lambda _: self._update_summary())
        self.tabs.addTab(self.results_tab, "Results")
        self.compare_tab = CompareTab(self._chart_inputs, self._sweep_inputs)
        self.tabs.addTab(self.compare_tab, "Compare")
        self.docs_tab = DocsTab()
        self.tabs.addTab(self.docs_tab, "Docs")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)

        self._build_mode_box()
        self._build_geometry_box()
        self._build_radar_box()
        self._build_faces_box()
        self.panel_layout.addStretch()
        self.project_path: Path | None = None
        self._default_settings = self._gather_settings()
        self._build_menu()
        self._check_engine()

        # Mouse observers go on the interactor *style*: pyvista's style grabs focus on button-down, after which
        # the interactor delivers the release only to the style, so an interactor observer never sees it.
        iren = self.plotter.iren.interactor
        style = self.plotter.iren.style
        style.AddObserver("LeftButtonPressEvent", lambda *_: self._on_lpress(iren, None))
        style.AddObserver("LeftButtonReleaseEvent", lambda *_: self._on_lrelease(iren, None))
        # An observer on the style replaces its default wheel handler (which dollies toward the focal point)
        style.AddObserver("MouseWheelForwardEvent", lambda *_: self._zoom_at_cursor(1.2))
        style.AddObserver("MouseWheelBackwardEvent", lambda *_: self._zoom_at_cursor(1 / 1.2))

        self.refresh_object_list()
        self.apply_mode()
        self.compute()

    # ------------------------------------------------------------------ #
    # Panel construction
    # ------------------------------------------------------------------ #
    def _group(self, title: str, pinned: bool = False) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox(title)
        (self.pinned_layout if pinned else self.panel_layout).addWidget(g)
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
        g = self._group("Geometry", pinned=True)
        lay = QtWidgets.QVBoxLayout(g)
        self.obj_list = QtWidgets.QListWidget()
        self.obj_list.setMaximumHeight(120)
        self.obj_list.currentRowChanged.connect(self._on_object_selected)
        self.obj_list.itemChanged.connect(self._on_item_checked)
        lay.addWidget(self.obj_list)
        hint = QtWidgets.QLabel("Tick or untick an imported part to include or exclude it.")
        hint.setStyleSheet(MUTED)
        lay.addWidget(hint)

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
        menu.addAction("Metalens retro tag (53 mm, measured)", lambda: self.add_sample("metalens"))
        menu.addAction("Ideal retroreflector (53 mm)", lambda: self.add_sample("retro"))
        b_sample.setMenu(menu)
        row.addWidget(b_sample)
        self.b_remove = QtWidgets.QPushButton("Delete")
        self.b_remove.clicked.connect(self.remove_object)
        row.addWidget(self.b_remove)
        lay.addLayout(row)

        # One-line result, pinned under the list so it stays visible while editing
        g = self._group("Result", pinned=True)
        lay = QtWidgets.QVBoxLayout(g)
        self.summary = QtWidgets.QLabel("—")
        self.summary.setStyleSheet("font-size:15px; font-weight:600")
        self.summary.setWordWrap(True)
        self.summary.setCursor(QtCore.Qt.PointingHandCursor)
        self.summary.setToolTip("Click for the full results")
        self.summary.mousePressEvent = lambda _e: self.tabs.setCurrentWidget(self.results_tab)
        lay.addWidget(self.summary)
        row = QtWidgets.QHBoxLayout()
        self.summary_sub = QtWidgets.QLabel("")
        self.summary_sub.setStyleSheet(MUTED)
        self.summary_sub.setWordWrap(True)
        row.addWidget(self.summary_sub, 1)
        self.b_compute = QtWidgets.QPushButton("Compute")
        self.b_compute.clicked.connect(self.compute)
        row.addWidget(self.b_compute)
        self.auto_check = QtWidgets.QCheckBox("Auto-update")
        self.auto_check.setChecked(True)
        row.addWidget(self.auto_check)
        lay.addLayout(row)

        # Placement and material for the selected object scroll with the rest of the panel
        g = self._group("Selected object")
        lay = QtWidgets.QVBoxLayout(g)
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
        self.obj_mat_estimate = QtWidgets.QLabel("Preset values are engineering <b>estimates</b> at 77 GHz, not lab data.")
        self.obj_mat_estimate.setStyleSheet(MUTED)
        self.obj_mat_estimate.setWordWrap(True)
        form.addRow("", self.obj_mat_estimate)

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
        outer = QtWidgets.QVBoxLayout(g)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)
        drow = QtWidgets.QGridLayout()
        drow.setVerticalSpacing(0)
        self.dist_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.dist_slider.setRange(0, DIST_STEPS)  # log scale, 1..200 m
        self.dist_slider.setValue(_dist_to_slider(10.0))
        self.dist_spin = _dspin(DIST_MIN, DIST_MAX, 10, 0.5, 1, " m")
        self.dist_slider.valueChanged.connect(lambda v: self.dist_spin.setValue(_slider_to_dist(v)))
        self.dist_spin.valueChanged.connect(self._on_distance)
        drow.addWidget(self.dist_slider, 0, 0)
        drow.addWidget(self.dist_spin, 0, 1)
        drow.addWidget(TickLabels(self.dist_slider), 1, 0)
        drow.setColumnStretch(0, 1)
        form.addRow("Distance:", drow)

        self.simple_radar_note = QtWidgets.QLabel("Fixed in Simple mode: radar directly behind the rider, 20° cone, "
                                                  "multi-bounce on, averaged over 76.5–77.5 GHz.")
        self.simple_radar_note.setWordWrap(True)
        self.simple_radar_note.setStyleSheet(MUTED)
        outer.addWidget(self.simple_radar_note)

        # Everything below is Advanced only, in one container so Simple mode keeps no hidden-row spacing
        self.adv_radar = QtWidgets.QWidget()
        adv = QtWidgets.QVBoxLayout(self.adv_radar)
        adv.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.adv_radar)
        f2 = QtWidgets.QFormLayout()
        adv.addLayout(f2)
        self.az_spin = _dspin(-180, 360, 0, 5, 1, "°")
        self.el_spin = _dspin(-45, 80, 0, 1, 1, "°")
        self.beam_spin = _dspin(1, 120, 20, 1, 1, "°")
        f2.addRow("Radar bearing:", self.az_spin)
        self.az_hint = QtWidgets.QLabel("0° = radar behind the rider, 90° = rider's left, 180° = head-on.")
        self.az_hint.setStyleSheet(MUTED)
        self.az_hint.setWordWrap(True)
        f2.addRow("", self.az_hint)
        f2.addRow("Elevation:", self.el_spin)
        f2.addRow("Cone width (−3 dB):", self.beam_spin)

        self.solver_toggle = QtWidgets.QToolButton()
        self.solver_toggle.setText("Solver settings")
        self.solver_toggle.setCheckable(True)
        self.solver_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.solver_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.solver_toggle.setAutoRaise(True)
        self.solver_toggle.setStyleSheet("QToolButton{font-weight:600; border:none}")
        adv.addWidget(self.solver_toggle)
        self.solver_box = QtWidgets.QWidget()
        f3 = QtWidgets.QFormLayout(self.solver_box)
        f3.setContentsMargins(16, 0, 0, 0)
        adv.addWidget(self.solver_box)
        self.solver_box.setVisible(False)
        self.solver_toggle.toggled.connect(self._on_solver_toggle)

        mrow = QtWidgets.QHBoxLayout()
        self.mb_check = QtWidgets.QCheckBox("Multi-bounce (shooting & bouncing rays)")
        self.mb_check.setChecked(True)
        mrow.addWidget(self.mb_check)
        b = help_button("How multi-bounce works")
        b.clicked.connect(lambda: self.show_docs("solver", "Step 4: shooting"))
        mrow.addWidget(b)
        mrow.addStretch()
        f3.addRow(mrow)
        self.bounce_spin = QtWidgets.QSpinBox()
        self.bounce_spin.setRange(2, 8)
        self.bounce_spin.setValue(3)
        f3.addRow("Max bounces:", self.bounce_spin)
        self.spacing_spin = _dspin(0.5, 50, 5, 0.5, 1, " mm")
        f3.addRow("Ray spacing:", self.spacing_spin)
        self.bw_spin = _dspin(0, 5, 1, 0.25, 2, " GHz")
        self.bw_spin.setToolTip("FMCW sweep the result is averaged over. 0 = one frequency (77 GHz).")
        f3.addRow("Band averaging:", self.bw_spin)
        self.nf_spin = QtWidgets.QSpinBox()
        self.nf_spin.setRange(1, 33)
        self.nf_spin.setValue(9)
        self.nf_spin.setToolTip("Frequencies sampled across the band. Ray casting is shared, so each extra one is cheap.")
        f3.addRow("Frequencies:", self.nf_spin)
        self.aspect_check = QtWidgets.QCheckBox("Also average over ±1° of bearing (3 samples, triples solve time)")
        f3.addRow(self.aspect_check)
        self.solve_time = QtWidgets.QLabel("")
        self.solve_time.setStyleSheet(MUTED)
        f3.addRow(self.solve_time)

        for w in (self.beam_spin, self.az_spin, self.el_spin, self.spacing_spin, self.bw_spin):
            w.valueChanged.connect(self._schedule)
        for w in (self.bounce_spin, self.nf_spin):
            w.valueChanged.connect(self._schedule)
        self.mb_check.toggled.connect(self._schedule)
        self.aspect_check.toggled.connect(self._schedule)

    def _on_solver_toggle(self, on: bool):
        self.solver_toggle.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        self.solver_box.setVisible(on)

    def _build_faces_box(self):
        self.faces_box = g = self._group("Face materials (Advanced)")
        lay = QtWidgets.QVBoxLayout(g)
        hint = QtWidgets.QLabel("Click a surface in the 3D view to select it. Shift+click adds or removes a surface, "
                                "Option/Alt+click toggles a single triangle, and clicking empty space clears. "
                                "Drag still rotates the view.")
        hint.setWordWrap(True)
        hint.setStyleSheet(MUTED)
        lay.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        self.grow_angle = _dspin(0.5, 90, 20, 1, 1, "°")
        self.grow_angle.setToolTip("A surface ends where neighbouring faces bend by more than this angle.")
        row.addWidget(QtWidgets.QLabel("Surface edge angle:"))
        row.addWidget(self.grow_angle)
        row.addStretch()
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

        src = QtWidgets.QHBoxLayout()
        self.rb_preset = QtWidgets.QRadioButton("Preset")
        self.rb_custom = QtWidgets.QRadioButton("Custom")
        self.rb_preset.setChecked(True)
        src.addWidget(QtWidgets.QLabel("Material:"))
        src.addWidget(self.rb_preset)
        src.addWidget(self.rb_custom)
        src.addStretch()
        lay.addLayout(src)
        self.rb_custom.toggled.connect(self._on_material_source)

        form = QtWidgets.QFormLayout()
        self.face_mat_combo = QtWidgets.QComboBox()
        self.face_mat_combo.currentIndexChanged.connect(self._load_material_values)
        self.preset_label = QtWidgets.QLabel("Preset:")
        form.addRow(self.preset_label, self.face_mat_combo)
        self.custom_name = QtWidgets.QLineEdit()
        self.custom_name.setPlaceholderText("Custom material")
        self.custom_color_btn = QtWidgets.QPushButton()
        self.custom_color_btn.setFixedWidth(60)
        self.custom_color_btn.clicked.connect(self._pick_custom_color)
        self._custom_color = "#b04fc4"
        nrow = QtWidgets.QHBoxLayout()
        nrow.addWidget(self.custom_name, 1)
        nrow.addWidget(QtWidgets.QLabel("Colour:"))
        nrow.addWidget(self.custom_color_btn)
        self.custom_name_label = QtWidgets.QLabel("Name:")
        self.custom_row = QtWidgets.QWidget()
        self.custom_row.setLayout(nrow)
        nrow.setContentsMargins(0, 0, 0, 0)
        form.addRow(self.custom_name_label, self.custom_row)
        self.R_spin = _dspin(0, 1, 0.5, 0.01, 3)
        self.A_spin = _dspin(0, 1, 0.5, 0.01, 3)
        self.s_spin = _dspin(0, 1, 0.0, 0.05, 2)
        self.T_label = QtWidgets.QLabel("")
        form.addRow("Reflectance R:", self.R_spin)
        form.addRow("Absorptance A:", self.A_spin)
        form.addRow("Transmittance T = 1−R−A:", self.T_label)
        form.addRow("Roughness (diffuse share):", self.s_spin)
        self.q_spin = _dspin(0, 1, 0.0, 0.05, 3)
        self.q_spin.setToolTip("Share of the reflected power sent straight back to the radar, as by a metalens tag, "
                               "Van Atta array or Luneburg lens. Roughness + retro share can't exceed 1.")
        self.half_spin = _dspin(0, 89, 40, 5, 0, "°")
        self.half_spin.setToolTip("Retro return holds up to this angle from the face normal, then rolls off over 10°.")
        qrow = QtWidgets.QHBoxLayout()
        qrow.addWidget(self.q_spin, 1)
        qrow.addWidget(QtWidgets.QLabel("within ±"))
        qrow.addWidget(self.half_spin)
        b = help_button("How retroreflective materials work")
        b.clicked.connect(lambda: self.show_docs("solver", "Step 3b"))
        qrow.addWidget(b)
        form.addRow("Retroreflective share:", qrow)
        for sp in (self.R_spin, self.A_spin):
            sp.valueChanged.connect(self._update_T)
        for sp in (self.s_spin, self.q_spin):
            sp.valueChanged.connect(self._update_q)
        lay.addLayout(form)
        self.mat_hint = QtWidgets.QLabel("")
        self.mat_hint.setWordWrap(True)
        self.mat_hint.setStyleSheet(MUTED)
        lay.addWidget(self.mat_hint)
        row = QtWidgets.QHBoxLayout()
        b_assign = QtWidgets.QPushButton("Assign to selected faces")
        b_assign.setStyleSheet("font-weight:600")
        b_assign.clicked.connect(self.assign_material)
        row.addWidget(b_assign)
        b_obj = QtWidgets.QPushButton("Assign to whole object")
        b_obj.setToolTip("Applies this material to every face of the object selected in the geometry list.")
        b_obj.clicked.connect(self.assign_material_to_object)
        row.addWidget(b_obj)
        lay.addLayout(row)
        self._on_material_source()

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
            bandwidth_hz=self.bw_spin.value() * 1e9, n_freq=self.nf_spin.value(),
            aspect_avg_deg=1.0 if self.aspect_check.isChecked() else 0.0,
        )

    @staticmethod
    def _centre(objs) -> np.ndarray:
        b = np.array([o.world_mesh().bounds for o in objs])
        lo = np.array([b[:, 0].min(), b[:, 2].min(), b[:, 4].min()])
        hi = np.array([b[:, 1].max(), b[:, 3].max(), b[:, 5].max()])
        return 0.5 * (lo + hi)

    def aim_point(self) -> np.ndarray:
        """The radar aims at the biker's centre whenever the biker is shown, so adding or moving a part never
        moves the aim. That keeps the comparison with the biker alone like-for-like."""
        if self.cb_biker.isChecked():
            return self._centre(self.biker)
        objs = self.visible_objects()
        if not objs:
            return np.array([0.5, 0, 0.8])
        return self._centre(objs)

    def _schedule(self, *_):
        self.result = None
        self.compare_tab.mark_stale()
        if self.auto_check.isChecked():
            self._debounce.start()
        else:
            self.summary_sub.setText("Out of date. Press Compute.")
        self.refresh_scene()

    # ------------------------------------------------------------------ #
    # Mode / list management
    # ------------------------------------------------------------------ #
    def apply_mode(self):
        adv = self.advanced
        self.simple_radar_note.setVisible(not adv)
        self.adv_radar.setVisible(adv)
        self.faces_box.setVisible(adv)
        self.results_tab.set_advanced(adv)
        if not adv:
            self.clear_selection(refresh=False)
        self.view_hint.setText("Advanced: click a surface to select it" if adv else "Simple mode: click an object to select it")
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
        if not self.rb_custom.isChecked():  # never overwrite values the user is typing in Custom mode
            self._load_material_values()

    def refresh_object_list(self, select: geo.SceneObject | None = None):
        if self.obj_mat_combo.count() != len(self.lib):
            self.refresh_material_combos()
        prev = select or self.current_object()
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        for o in self.objects():
            if o.builtin and not o.visible:
                continue
            it = QtWidgets.QListWidgetItem(f"{o.name}  ·  {o.n_faces:,} faces")
            it.setData(QtCore.Qt.UserRole, o)
            if o.builtin:
                it.setText(it.text() + "  (baseline)" if "(baseline)" not in o.name else it.text())
            else:  # attachments get an include checkbox, so their effect can be toggled on and off
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if o.visible else QtCore.Qt.Unchecked)
                it.setToolTip("Untick to leave this part out of the scene and the result without deleting it.")
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

    def _on_item_checked(self, item: QtWidgets.QListWidgetItem):
        o = item.data(QtCore.Qt.UserRole)
        if o is None or o.builtin:
            return
        visible = item.checkState() == QtCore.Qt.Checked
        if visible != o.visible:
            o.visible = visible
            if not visible:
                self.selection.pop(id(o), None)
                self._update_sel_label()
            self._schedule()

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
            self.obj_mat_estimate.setVisible(m.note != "Custom")
            self.obj_mat_info.setText(f"R {m.reflectance:.2f} · A {m.absorptance:.2f} · T {m.transmittance:.2f} · rough {m.roughness:.2f}"
                                      + (f" · retro {m.retro:.3g} within ±{m.retro_half_angle_deg:.0f}°" if m.retro > 0 else "")
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
        self.dist_slider.setValue(_dist_to_slider(v))
        self.dist_slider.blockSignals(False)
        self._schedule()

    # ------------------------------------------------------------------ #
    # Geometry actions
    # ------------------------------------------------------------------ #
    def import_geometry(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import geometry", str(Path.home()), MESH_FILTER)
        if not path:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            mesh = geo.load_mesh(path, "m")  # raw file units; scaled below once the user confirms units
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Import failed", f"Could not read {path}:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        raw = np.ptp(np.asarray(mesh.points), axis=0)
        unit_names = list(geo.UNIT_SCALES)
        guess = geo.guess_units(float(raw.max()))
        units, ok = QtWidgets.QInputDialog.getItem(
            self, "Units",
            f"Part spans {raw[0]:.3g} × {raw[1]:.3g} × {raw[2]:.3g} file units.\nWhat units is this file in?",
            unit_names, unit_names.index(guess), False)
        if not ok:
            return
        mesh.points = mesh.points * geo.UNIT_SCALES[units]
        size = raw * geo.UNIT_SCALES[units]
        if size.max() > 3:
            QtWidgets.QMessageBox.warning(self, "Large part",
                f"This part is {size.max():.2f} m across. If that's wrong, re-import with different units or use Scale.")
        elif size.max() < 0.02:
            QtWidgets.QMessageBox.warning(self, "Tiny part",
                f"This part is only {size.max() * 1000:.1f} mm across. If that's wrong, re-import with different units or use Scale.")
        if mesh.n_cells > DECIMATE_OFFER:
            mesh = self._offer_decimate(mesh)
        obj = geo.make_attachment(Path(path).stem, mesh, self.lib.index("ABS / hard plastic"), self.cb_biker.isChecked(),
                                  mount="head")
        self.attachments.append(obj)
        self.refresh_object_list(select=obj)
        self.fit_view(False)
        self._schedule()

    def _offer_decimate(self, mesh: pv.PolyData) -> pv.PolyData:
        """Dense CAD exports slow every solve down. Offer to thin them, showing the face count before and after."""
        n = mesh.n_cells
        target, ok = QtWidgets.QInputDialog.getInt(
            self, "Dense mesh",
            f"This part has {n:,} triangles, which slows every solve.\n\n"
            "Decimate to how many triangles? Keep the original count to skip.\n"
            "Of note, flat faces survive decimation well; tight curves can turn faceted, "
            "which makes their glints spikier at 77 GHz.",
            min(n, 40_000), 1_000, n, 1_000)
        if not ok or target >= n:
            return mesh
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            out = mesh.decimate(1.0 - target / n).triangulate().clean(tolerance=1e-9)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if out.n_cells == 0:
            QtWidgets.QMessageBox.warning(self, "Decimation failed", "Decimation removed every face; keeping the original.")
            return mesh
        self.statusBar().showMessage(f"Decimated from {n:,} to {out.n_cells:,} triangles.", 8000)
        QtWidgets.QMessageBox.information(self, "Decimated", f"Decimated from {n:,} to {out.n_cells:,} triangles.")
        return out

    def add_sample(self, kind: str):
        al = self.lib.index("Aluminum")
        if kind == "corner":
            mesh, name = geo.corner_reflector(0.10, opening=(-1, 0, 0)), "Corner reflector"
        elif kind == "plate":
            mesh, name = geo.flat_plate(0.10, normal=(-1, 0, 0)), "Flat plate"
        elif kind in ("metalens", "retro"):
            # A 53 mm tag, the size in Ghasemi et al. 2026, facing a radar behind the rider
            mesh = geo.flat_plate(0.053, normal=(-1, 0, 0))
            name, al = (("Metalens retro tag", self.lib.index("Metalens retro tag (measured)")) if kind == "metalens"
                        else ("Ideal retroreflector", self.lib.index("Retroreflector (ideal)")))
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
        if self._press_pos is None:
            return
        x, y = iren.GetEventPosition()
        px, py = self._press_pos
        self._press_pos = None
        if abs(x - px) + abs(y - py) > 4:
            return  # it was a drag (rotate), not a click
        obj, cell = self._pick(x, y)
        if obj is not None:
            rows = [self.obj_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.obj_list.count())]
            if obj in rows and self.current_object() is not obj:
                self.obj_list.setCurrentRow(rows.index(obj))
        if not self.advanced:
            return
        # pyvistaqt doesn't forward Alt/Option to VTK, so read modifiers from Qt directly
        mods = QtWidgets.QApplication.keyboardModifiers()
        shift = bool(mods & QtCore.Qt.ShiftModifier)
        alt = bool(mods & QtCore.Qt.AltModifier)
        if obj is None:
            if not (shift or alt) and self.selection:
                self.clear_selection(refresh=False)
                self._recolor()
            return
        surface = {cell} if alt else self._grow(obj, [cell])
        if shift or alt:  # toggle this surface in or out of the selection
            sel = self.selection.setdefault(id(obj), set())
            if surface <= sel:
                sel -= surface
            else:
                sel |= surface
        else:
            self.selection = {id(obj): surface}
        self._update_sel_label()
        self._recolor()

    def _recolor(self):
        """Repaint material colours and the selection highlight without rebuilding the scene."""
        self._paint()

    # ------------------------------------------------------------------ #
    # Materials (advanced)
    # ------------------------------------------------------------------ #
    def _on_material_source(self, *_):
        custom = self.rb_custom.isChecked()
        for w in (self.R_spin, self.A_spin, self.s_spin, self.q_spin, self.half_spin):
            w.setEnabled(custom)  # preset values are shown read-only, so a preset can't be edited by accident
        for w in (self.custom_name_label, self.custom_row):
            w.setVisible(custom)
        self.preset_label.setText("Start from:" if custom else "Preset:")
        self.mat_hint.setText(
            "Set your own values. Assigning saves this as a new material in the library (it never changes a preset) "
            "and it then appears in every material list." if custom else
            "Presets are applied exactly as listed. Switch to Custom to set your own values.")
        self._set_custom_color(self._custom_color)
        if not custom:
            self._load_material_values()

    def _set_custom_color(self, hex_color: str):
        self._custom_color = hex_color
        self.custom_color_btn.setStyleSheet(f"background:{hex_color}; border:1px solid #888; border-radius:3px; min-height:18px")

    def _pick_custom_color(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._custom_color), self, "Material colour")
        if c.isValid():
            self._set_custom_color(c.name())

    def _load_material_values(self, *_):
        i = self.face_mat_combo.currentIndex()
        if i < 0:
            return
        m = self.lib[i]
        for sp, v in ((self.R_spin, m.reflectance), (self.A_spin, m.absorptance), (self.s_spin, m.roughness),
                      (self.q_spin, m.retro), (self.half_spin, m.retro_half_angle_deg)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self._update_T()

    def _update_T(self, *_):
        R, A = self.R_spin.value(), self.A_spin.value()
        if R + A > 1:
            self.A_spin.blockSignals(True)
            self.A_spin.setValue(1 - R)
            self.A_spin.blockSignals(False)
            A = 1 - R
        self.T_label.setText(f"{max(0.0, 1 - R - A):.3f}")

    def _update_q(self, *_):
        if self.s_spin.value() + self.q_spin.value() > 1:  # diffuse + retro shares of the reflected power
            self.q_spin.blockSignals(True)
            self.q_spin.setValue(1 - self.s_spin.value())
            self.q_spin.blockSignals(False)

    def _material_to_assign(self) -> int:
        """Library index for the panel's current material, adding a custom material if needed."""
        if not self.rb_custom.isChecked():
            return self.face_mat_combo.currentIndex()
        R, A, s = self.R_spin.value(), self.A_spin.value(), self.s_spin.value()
        q, half = self.q_spin.value(), self.half_spin.value()
        name = self.custom_name.text().strip() or "Custom material"
        # Re-assigning the same custom material reuses its entry instead of piling up copies
        for i, m in enumerate(self.lib.materials):
            if (m.name == name and m.color == self._custom_color and abs(m.reflectance - R) < 1e-9
                    and abs(m.absorptance - A) < 1e-9 and abs(m.roughness - s) < 1e-9
                    and abs(m.retro - q) < 1e-9 and abs(m.retro_half_angle_deg - half) < 1e-9):
                return i
        idx = self.lib.add(name, R, A, s, self._custom_color, retro=q, retro_half_angle_deg=half)
        self.lib[idx].note = "Custom"
        self.refresh_material_combos()
        self.face_mat_combo.blockSignals(True)
        self.face_mat_combo.setCurrentIndex(idx)
        self.face_mat_combo.blockSignals(False)
        self.custom_name.setText(self.lib[idx].name)  # shows the de-duplicated name, e.g. "Foil (2)"
        return idx

    def assign_material(self):
        if not self._sel_count():
            QtWidgets.QMessageBox.information(self, "No faces selected",
                "Click a surface in the 3D view to select it first, or use Assign to whole object.")
            return
        idx = self._material_to_assign()
        for oid, faces in self.selection.items():
            obj = self._obj_by_id(oid)
            if obj is not None and faces:
                obj.face_mat[list(faces)] = idx
        self.clear_selection(refresh=False)
        self._on_object_selected()
        self._schedule()

    def assign_material_to_object(self):
        o = self.current_object()
        if o is None:
            QtWidgets.QMessageBox.information(self, "No object", "Select an object in the geometry list first.")
            return
        o.set_material(self._material_to_assign())
        self.selection.pop(id(o), None)
        self._update_sel_label()
        self._on_object_selected()
        self._schedule()

    # ------------------------------------------------------------------ #
    # Compute & display
    # ------------------------------------------------------------------ #
    def compute(self):
        self._debounce.stop()
        target, spans = geo.build_target(self.visible_objects(), self.lib)
        if target is None:
            self._gen += 1
            self.result, self.base_result, self.spans = None, None, []
            self.results_tab.show_empty("No geometry in the scene.")
            self.summary.setText("—")
            self.summary_sub.setText("No geometry in the scene")
            self.refresh_scene()
            return
        radar, aim = self.radar_settings(), self.aim_point()
        parts = [o for o in self.attachments if o.visible]
        base_key = base_target = None
        if self.cb_biker.isChecked() and parts:
            base_key = cmp.baseline_key(radar, [o.face_mat for o in self.biker], cmp.library_values(self.lib), aim)
            if self.baseline.get(base_key) is None:
                base_target, _ = geo.build_target(self.biker, self.lib)
        self._gen += 1
        self.solver.request(SolveRequest(
            self._gen, target, radar, aim, base_target, base_key,
            extra=dict(spans=spans, parts={o.name for o in parts}, biker=self.cb_biker.isChecked())))
        self.summary_sub.setText("Solving…" + (" (and the biker alone, for comparison)" if base_target is not None else ""))

    def _on_solved(self, reply: SolveReply):
        req = reply.request
        if reply.base_result is not None and req.base_key:
            self.baseline.put(req.base_key, reply.base_result)
        if req.gen != self._gen:
            return  # the scene changed while this was solving; a newer request is queued
        if reply.error:
            self.summary_sub.setText("Solve failed.")
            QtWidgets.QMessageBox.critical(self, "Solve failed", reply.error[-2000:])
            return
        ex = req.extra
        res = reply.result
        base = self.baseline.get(req.base_key) if req.base_key else None
        if base is None:
            if not ex["biker"]:
                reason = "The baseline biker is hidden, so there's nothing to compare against. Tick “Show baseline biker”."
            else:
                reason = "No attachments in the scene yet, so this is the biker alone. Add or import a part to see what it changes."
        else:
            reason = ""
        self.result, self.base_result, self.spans = res, base, ex["spans"]
        self._result_radar = req.radar
        self.results_tab.show_result(res, base, req.radar, ex["parts"], reason)
        self.solve_time.setText(f"Last solve: {reply.seconds:.2f} s")
        self._update_summary()
        self._paint()

    def _update_summary(self):
        res, base = self.result, self.base_result
        if res is None:
            return
        unit = self.results_tab.unit
        if base is not None:
            d = cmp.delta_db(res.rcs_m2, base.rcs_m2)
            color = VIS_COLORS[cmp.visibility(d)]
            self.summary.setText(f"<span style='color:{color}'>{d:+.1f} dB</span> vs. biker alone · "
                                 f"{cmp.fmt_rcs(res.rcs_m2, unit)}")
        else:
            self.summary.setText(f"{cmp.fmt_rcs(res.rcs_m2, unit)} total RCS")
        self.summary_sub.setText(cmp.sentence_case(cmp.band_text(self._result_radar)))

    def show_docs(self, page: str, heading: str = ""):
        self.tabs.setCurrentWidget(self.docs_tab)
        self.docs_tab.show_page(page, heading)

    def _show_object_by_name(self, name: str):
        """Results table click: select the object, switch to the 3D view and flash it."""
        obj = next((o for o in self.visible_objects() if o.name == name), None)
        if obj is None:
            return
        rows = [self.obj_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.obj_list.count())]
        if obj in rows:
            self.obj_list.setCurrentRow(rows.index(obj))
        self.tabs.setCurrentIndex(0)
        self._flash_obj = obj
        self._paint()
        self._flash_timer.start()

    def _end_flash(self):
        self._flash_obj = None
        self._paint()

    def _check_engine(self):
        name, ok = ray_engine()
        if not ok:
            msg = ("Embree not found. Solves will take 20 s or more instead of under a second. "
                   f"See Setup in the README (ray engine: {name}).")
            self.engine_banner.setText(msg)
            self.engine_banner.setVisible(True)
            self.statusBar().showMessage(msg)

    def closeEvent(self, ev):
        from .sweeps import shutdown_pool

        self.compare_tab._cancel()
        self.solver.shutdown()
        shutdown_pool()
        super().closeEvent(ev)

    # ------------------------------------------------------------------ #
    # Inputs for the Compare tab
    # ------------------------------------------------------------------ #
    def _sweep_inputs(self):
        target, _ = geo.build_target(self.visible_objects(), self.lib)
        if target is None:
            raise ValueError("There's no geometry in the scene to sweep.")
        base_t = None
        if self.cb_biker.isChecked() and any(o.visible for o in self.attachments):
            base_t, _ = geo.build_target(self.biker, self.lib)
        return target, base_t, self.aim_point(), self.radar_settings()

    def _chart_inputs(self):
        parts = [o for o in self.attachments if o.visible]
        if not parts:
            raise ValueError("Import or add a part (and leave it ticked) to compare against the biker alone.")
        for o in self.biker:
            o.visible = True  # the baseline is always the full biker, even if it's hidden in the view
        try:
            base_t, _ = geo.build_target(self.biker, self.lib)
            with_t, _ = geo.build_target(self.biker + parts, self.lib)
            aim = self._centre(self.biker)  # same aim for both runs, so the only difference is the parts
        finally:
            self.objects()  # restore biker visibility from the checkbox
        return base_t, with_t, aim, self.radar_settings(), [o.name for o in parts]

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Radar Detector")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
