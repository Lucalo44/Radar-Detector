"""The 3D view: actors, colouring, the radar overlay, camera views and zoom. Mixed into MainWindow."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker

from .compare import heat_values
from .widgets import _heat_rgb

HIGHLIGHT = np.array([255, 214, 0], dtype=np.uint8)


class SceneMixin:
    """Expects MainWindow's plotter, scene state and radar_settings()/aim_point()/visible_objects()."""

    def _pick(self, x: int, y: int):
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None, -1
        actor, cell = picker.GetActor(), picker.GetCellId()
        if actor is None or cell < 0:
            return None, -1
        return self.actor_map.get(actor.GetAddressAsString("vtkObject")), cell

    def _zoom_at_cursor(self, factor: float):
        """Dolly toward (factor > 1) or away from the point under the cursor, keeping that point fixed on screen."""
        ren = self.plotter.renderer
        cam = ren.GetActiveCamera()
        x, y = self.plotter.iren.interactor.GetEventPosition()
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        if picker.Pick(x, y, 0, ren) and picker.GetCellId() >= 0:
            target = np.array(picker.GetPickPosition())
        else:  # empty space: use the point under the cursor at the focal-plane depth
            ren.SetWorldPoint(*cam.GetFocalPoint(), 1.0)
            ren.WorldToDisplay()
            z = ren.GetDisplayPoint()[2]
            ren.SetDisplayPoint(x, y, z)
            ren.DisplayToWorld()
            w = np.array(ren.GetWorldPoint())
            target = w[:3] / w[3]
        pos, fp = np.array(cam.GetPosition()), np.array(cam.GetFocalPoint())
        cam.SetPosition(*(target + (pos - target) / factor))
        cam.SetFocalPoint(*(target + (fp - target) / factor))
        ren.ResetCameraClippingRange()
        self.plotter.render()

    # ------------------------------------------------------------------ #
    # Scene rendering
    # ------------------------------------------------------------------ #
    def refresh_scene(self):
        """Sync the 3D view with the scene. Mesh actors are reused (only their colours are rewritten) unless an
        object moved; the radar, cone and road are rebuilt only when the radar or aim point change."""
        p = self.plotter
        cam = None if self._first_render else p.camera_position
        visible = self.visible_objects()
        vis_ids = {id(o) for o in visible}
        for oid in list(self._scene_actors):
            if oid not in vis_ids:
                p.remove_actor(self._scene_actors.pop(oid)[0], render=False)
        self.actor_map = {}
        for obj in visible:
            src = obj.world_mesh()
            entry = self._scene_actors.get(id(obj))
            if entry is None or entry[2] is not src:
                if entry is not None:
                    p.remove_actor(entry[0], render=False)
                m = src.copy()
                m.cell_data["colors"] = np.zeros((m.n_cells, 3), dtype=np.uint8)
                actor = p.add_mesh(m, scalars="colors", rgb=True, ambient=0.25, diffuse=0.8, specular=0.15,
                                   render=False, reset_camera=False)
                entry = (actor, m, src)
                self._scene_actors[id(obj)] = entry
            self.actor_map[entry[0].GetAddressAsString("vtkObject")] = obj
        self._paint(render=False)
        self._update_radar_actors()
        if self._first_render:
            self._first_render = False
            self.set_view("iso")
        elif cam is not None:
            p.camera_position = cam
        p.render()

    def _paint(self, render: bool = True):
        """Write face colours: materials, or the heat map when it's on and a current result exists."""
        _, _, _, rgb_lib = self.lib.arrays()
        heat = self.heat_check.isChecked() and self.result is not None
        span_map = {id(o): (a, b) for o, a, b in self.spans}
        top = face_db = None
        if heat:
            vals = heat_values(self.result, [(o.name, a, b) for o, a, b in self.spans])
            face_db = 10 * np.log10(np.maximum(vals, 1e-12))
            top = float(face_db.max())
        for obj in self.visible_objects():
            entry = self._scene_actors.get(id(obj))
            if entry is None:
                continue
            m = entry[1]
            span = span_map.get(id(obj))
            if heat and span is not None and span[1] - span[0] == m.n_cells:
                colors = _heat_rgb(face_db[span[0]:span[1]], top)
            else:
                colors = rgb_lib[obj.face_mat].copy()
            sel = self.selection.get(id(obj))
            if sel:
                colors[list(sel)] = HIGHLIGHT
            if self._flash_obj is obj:
                colors[:] = HIGHLIGHT
            m.cell_data["colors"][:] = colors
            m.Modified()
        self.heat_legend.set_top(top)
        if render:
            self.plotter.render()

    def _update_radar_actors(self):
        radar, aim = self.radar_settings(), self.aim_point()
        key = (radar.distance_m, radar.azimuth_deg, radar.elevation_deg, radar.beam_width_deg, tuple(np.round(aim, 6)))
        if key == self._radar_key:
            return
        self._radar_key = key
        p = self.plotter
        p_r = radar.position(aim)
        bore = (aim - p_r) / radar.distance_m
        L = radar.distance_m * 1.08
        half = np.radians(radar.beam_width_deg / 2)
        cone = pv.Cone(center=p_r + bore * L / 2, direction=-bore, height=L, radius=L * np.tan(half), resolution=64, capping=False)
        kw = dict(pickable=False, render=False, reset_camera=False)
        p.add_mesh(cone, color="#eb6834", opacity=0.10, name="radar_cone", **kw)
        p.add_mesh(pv.Cube(center=p_r, x_length=0.18, y_length=0.18, z_length=0.10), color="#c0392b", name="radar_box", **kw)
        p.add_mesh(pv.Line(p_r, aim), color="#c0392b", line_width=2, name="radar_line", **kw)
        p.add_mesh(pv.Sphere(radius=0.025, center=aim), color="#c0392b", name="radar_aim", **kw)
        p.add_point_labels([0.5 * (p_r + aim)], [f"{radar.distance_m:.1f} m"], font_size=16, point_size=0,
                           shape_color="white", shape_opacity=0.8, text_color="#0b0b0b", always_visible=True,
                           pickable=False, name="radar_dist", render=False, reset_camera=False)
        p.add_point_labels([p_r + np.array([0, 0, 0.25])], ["Radar 77 GHz"], font_size=12, point_size=0,
                           shape_opacity=0.0, text_color="#c0392b", always_visible=True, pickable=False,
                           name="radar_label", render=False, reset_camera=False)
        c = 0.5 * (p_r + aim)
        size = max(4.0, 1.3 * np.linalg.norm(p_r[:2] - aim[:2]) + 3)
        ground = pv.Plane(center=(c[0], c[1], 0), direction=(0, 0, 1), i_size=size, j_size=size, i_resolution=20, j_resolution=20)
        p.add_mesh(ground, color="#d9d9d4", opacity=0.5, show_edges=True, edge_color="#c4c4be", name="road", **kw)
        p.add_text(f"Radar distance: {radar.distance_m:.1f} m   ·   bearing {radar.azimuth_deg:.0f}°   ·   cone {radar.beam_width_deg:.0f}°",
                   position="upper_left", font_size=11, color="#0b0b0b", name="radar_text")

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
