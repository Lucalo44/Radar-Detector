"""File menu: new, open and save projects, and the GUI settings stored in them. Mixed into MainWindow."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui, QtWidgets

from . import geometry as geo
from .materials import MaterialLibrary
from .widgets import _dist_to_slider


class ProjectMixin:
    # ------------------------------------------------------------------ #
    # Projects
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        menu = self.menuBar().addMenu("File")
        for text, key, fn in (("New Project", QtGui.QKeySequence.New, self.new_project),
                              ("Open Project…", QtGui.QKeySequence.Open, self.open_project),
                              ("Save Project", QtGui.QKeySequence.Save, self.save_project),
                              ("Save Project As…", QtGui.QKeySequence.SaveAs, self.save_project_as)):
            act = QtGui.QAction(text, self)
            act.setShortcut(key)
            act.triggered.connect(fn)
            menu.addAction(act)
        self._update_title()

    def _update_title(self):
        name = self.project_path.stem if self.project_path else "Untitled"
        self.setWindowTitle(f"{name} — Radar Detector (77 GHz bicycle RCS)")

    def _gather_settings(self) -> dict:
        s = {
            "advanced": self.advanced,
            "show_biker": self.cb_biker.isChecked(),
            "distance_m": self.dist_spin.value(),
            "beam_width_deg": self.beam_spin.value(),
            "azimuth_deg": self.az_spin.value(),
            "elevation_deg": self.el_spin.value(),
            "multibounce": self.mb_check.isChecked(),
            "max_bounces": self.bounce_spin.value(),
            "ray_spacing_mm": self.spacing_spin.value(),
            "auto_update": self.auto_check.isChecked(),
            "surface_edge_angle_deg": self.grow_angle.value(),
            "bandwidth_ghz": self.bw_spin.value(),
            "n_freq": self.nf_spin.value(),
            "aspect_average": self.aspect_check.isChecked(),
            "heat_map": self.heat_check.isChecked(),
            "rcs_unit": self.results_tab.unit,
        }
        if not self._first_render:
            s["camera"] = [list(map(float, v)) for v in self.plotter.camera_position]
        return s

    def _apply_settings(self, s: dict):
        widgets = (self.rb_simple, self.rb_adv, self.cb_biker, self.dist_spin, self.dist_slider, self.beam_spin,
                   self.az_spin, self.el_spin, self.mb_check, self.bounce_spin, self.spacing_spin, self.auto_check,
                   self.grow_angle, self.bw_spin, self.nf_spin, self.aspect_check, self.heat_check)
        for w in widgets:
            w.blockSignals(True)
        try:
            d = self._default_settings
            get = lambda k: s.get(k, d.get(k))  # noqa: E731  (older files may lack newer keys)
            (self.rb_adv if get("advanced") else self.rb_simple).setChecked(True)
            self.cb_biker.setChecked(bool(get("show_biker")))
            self.dist_spin.setValue(float(get("distance_m")))
            self.dist_slider.setValue(_dist_to_slider(float(get("distance_m"))))
            self.beam_spin.setValue(float(get("beam_width_deg")))
            self.az_spin.setValue(float(get("azimuth_deg")))
            self.el_spin.setValue(float(get("elevation_deg")))
            self.mb_check.setChecked(bool(get("multibounce")))
            self.bounce_spin.setValue(int(get("max_bounces")))
            self.spacing_spin.setValue(float(get("ray_spacing_mm")))
            self.auto_check.setChecked(bool(get("auto_update")))
            self.grow_angle.setValue(float(get("surface_edge_angle_deg")))
            self.bw_spin.setValue(float(get("bandwidth_ghz")))
            self.nf_spin.setValue(int(get("n_freq")))
            self.aspect_check.setChecked(bool(get("aspect_average")))
            self.heat_check.setChecked(bool(get("heat_map")))
            self.results_tab.set_unit(str(get("rcs_unit")))
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _load_state(self, lib, attachments, biker_face_mats: dict, settings: dict):
        self.lib = lib
        self.biker = geo.build_biker(self.lib)
        for o in self.biker:
            fm = biker_face_mats.get(o.name)
            if fm is not None and len(fm) == o.n_faces:
                o.face_mat = fm
        self.attachments = attachments
        self.selection.clear()
        self._adj_cache.clear()
        self._apply_settings(settings)
        self.refresh_material_combos()
        self.apply_mode()  # refreshes the object list and schedules a compute
        self._update_sel_label()
        self._first_render = "camera" not in settings
        if not self._first_render:
            self.refresh_scene()
            self.plotter.camera_position = [tuple(v) for v in settings["camera"]]
            self.plotter.render()

    def new_project(self):
        if QtWidgets.QMessageBox.question(self, "New project", "Start a new project? Unsaved changes will be lost.") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.project_path = None
        self._load_state(MaterialLibrary(), [], {}, {k: v for k, v in self._default_settings.items() if k != "camera"})
        self._update_title()

    def open_project(self):
        from .project import EXTENSION, load_project

        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", str(Path.home()),
                                                        f"Radar Detector project (*{EXTENSION});;All files (*)")
        if not path:
            return
        try:
            proj = load_project(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Open failed", f"Could not open {path}:\n{e}")
            return
        self.project_path = Path(path)
        self._load_state(proj.lib, proj.attachments, proj.biker_face_mats, proj.settings)
        self._update_title()

    def save_project(self):
        if self.project_path is None:
            self.save_project_as()
        else:
            self._write_project(self.project_path)

    def save_project_as(self):
        from .project import EXTENSION

        start = str(self.project_path or Path.home() / f"radar_project{EXTENSION}")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", start,
                                                        f"Radar Detector project (*{EXTENSION})")
        if path:
            self._write_project(Path(path))

    def _write_project(self, path: Path):
        from .project import save_project

        try:
            self.project_path = save_project(path, self.lib, self.biker, self.attachments, self._gather_settings())
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Could not save {path}:\n{e}")
            return
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.project_path}", 4000)
