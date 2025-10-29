import math
import os
from typing import Dict, Optional

import cv2
import numpy as np
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from .camera_backends import best_available_source
from .timeline import Session
from .utils import new_session_dir, default_sessions_root, ensure_dir, imread_color, imwrite_safe
from .diffview import ImageView
from .align import orb_similarity_transform, ecc_refine, largest_change_component, build_centering_transform, compose_magenta


class SliderControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)
    editingStarted = QtCore.Signal()
    editingFinished = QtCore.Signal()

    def __init__(self, minimum: float, maximum: float, *, decimals: int = 2, step: Optional[float] = None,
                 log_scale: bool = False, unit: str = "", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum if maximum > minimum else minimum + 1.0
        self._log = log_scale and self._min > 0 and self._max > 0
        self._decimals = decimals
        self._unit = unit
        self._updating = False

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(self._min, self._max)
        if unit:
            self.spin.setSuffix(f" {unit}")
        if step:
            self.spin.setSingleStep(step)
        else:
            span = max(1e-6, self._max - self._min)
            self.spin.setSingleStep(span / 100.0)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self.editingStarted)
        self.slider.sliderReleased.connect(self.editingFinished)
        self.spin.valueChanged.connect(self._on_spin_changed)

    def _value_to_slider(self, value: float) -> int:
        value = float(value)
        if self._log:
            log_min = math.log10(self._min)
            log_max = math.log10(self._max)
            log_val = math.log10(max(min(value, self._max), self._min))
            pos = int(round((log_val - log_min) / (log_max - log_min) * 1000)) if log_max != log_min else 0
        else:
            pos = int(round((value - self._min) / (self._max - self._min) * 1000)) if self._max != self._min else 0
        return max(0, min(1000, pos))

    def _slider_to_value(self, pos: int) -> float:
        ratio = pos / 1000.0
        if self._log:
            log_min = math.log10(self._min)
            log_max = math.log10(self._max)
            value = 10 ** (log_min + ratio * (log_max - log_min))
        else:
            value = self._min + ratio * (self._max - self._min)
        return max(self._min, min(self._max, value))

    def set_value(self, value: float):
        if value is None:
            return
        self._updating = True
        try:
            slider_pos = self._value_to_slider(value)
            self.slider.setValue(slider_pos)
            self.spin.setValue(float(value))
        finally:
            self._updating = False

    def value(self) -> float:
        return float(self.spin.value())

    def _on_slider_changed(self, pos: int):
        if self._updating:
            return
        value = self._slider_to_value(pos)
        self._updating = True
        try:
            self.spin.setValue(value)
        finally:
            self._updating = False
        self.valueChanged.emit(float(value))

    def _on_spin_changed(self, value: float):
        if self._updating:
            return
        self._updating = True
        try:
            self.slider.setValue(self._value_to_slider(value))
        finally:
            self._updating = False
        self.valueChanged.emit(float(value))


class CameraControlPanel(QtWidgets.QGroupBox):
    CONTROL_SPECS = [
        {"name": "AeEnable", "label": "Auto Exposure", "type": "toggle"},
        {"name": "ExposureTime", "label": "Exposure (µs)", "type": "slider", "log": True, "unit": "µs", "decimals": 0, "step": 100.0},
        {"name": "AnalogueGain", "label": "Analogue Gain", "type": "slider", "min": 1.0, "max": 16.0, "decimals": 2, "step": 0.05},
        {"name": "Brightness", "label": "Brightness", "type": "slider", "min": -1.0, "max": 1.0, "decimals": 2, "step": 0.05},
        {"name": "Contrast", "label": "Contrast", "type": "slider", "min": 0.0, "max": 32.0, "decimals": 2, "step": 0.1},
        {"name": "Saturation", "label": "Saturation", "type": "slider", "min": 0.0, "max": 32.0, "decimals": 2, "step": 0.1},
        {"name": "Sharpness", "label": "Sharpness", "type": "slider", "min": 0.0, "max": 16.0, "decimals": 2, "step": 0.1},
        {"name": "AwbEnable", "label": "Auto White Balance", "type": "toggle"},
        {"name": "AwbMode", "label": "White Balance Mode", "type": "combo"},
        {"name": "ColourTemperature", "label": "Colour Temperature (K)", "type": "slider", "min": 2500.0, "max": 8000.0, "decimals": 0, "step": 50.0},
    ]

    DEPENDENCIES = {
        "ExposureTime": {"AeEnable": False},
        "AwbMode": {"AwbEnable": False},
        "ColourTemperature": {"AwbEnable": False},
    }

    AWB_LABELS = {
        0: "Auto",
        1: "Tungsten",
        2: "Fluorescent",
        3: "Indoor",
        4: "Daylight",
        5: "Cloudy",
    }

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("Camera Controls", parent)
        self._source = None
        self._control_widgets: Dict[str, QtWidgets.QWidget] = {}
        self._spec_by_name = {spec["name"]: spec for spec in self.CONTROL_SPECS}

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self.auto_btn = QtWidgets.QPushButton("Auto Optimise")
        self.auto_btn.clicked.connect(self._on_auto_optimize)
        lay.addWidget(self.auto_btn)

        self.form = QtWidgets.QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(6)
        lay.addLayout(self.form)

        self.status_label = QtWidgets.QLabel("Camera controls available once the HQ camera is running.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        lay.addWidget(self.status_label)
        lay.addStretch(1)

        self.setEnabled(False)

    # ------------------------------------------------------------------ lifecycle
    def set_source(self, source):
        if source and getattr(source, "supports_manual_controls", lambda: False)():
            self._source = source
            self.setEnabled(True)
            self.setVisible(True)
            self.status_label.setText("Adjust camera settings or let Auto Optimise pick a baseline.")
            self._rebuild_controls()
            self.refresh_from_source()
        else:
            self._source = None
            self.setEnabled(False)
            self._clear_controls()
            self.status_label.setText("Camera controls are unavailable for this source.")
            self.setVisible(False)

    # ------------------------------------------------------------------ UI setup
    def _clear_controls(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        for widget in self._control_widgets.values():
            widget.deleteLater()
        self._control_widgets.clear()

    def _rebuild_controls(self):
        self._clear_controls()
        if not self._source:
            return
        for spec in self.CONTROL_SPECS:
            name = spec["name"]
            info = getattr(self._source, "control_info", lambda _: {})(name) or {}
            widget = None
            if spec["type"] == "toggle":
                widget = QtWidgets.QCheckBox()
                widget.stateChanged.connect(lambda state, n=name: self._on_toggle(n, state == QtCore.Qt.Checked))
            elif spec["type"] == "slider":
                minimum = info.get("min", spec.get("min"))
                maximum = info.get("max", spec.get("max"))
                if minimum is None or maximum is None:
                    continue
                slider = SliderControl(
                    float(minimum),
                    float(maximum),
                    decimals=int(spec.get("decimals", 2)),
                    step=spec.get("step"),
                    log_scale=spec.get("log", False),
                    unit=spec.get("unit", ""),
                )
                slider.valueChanged.connect(lambda value, n=name: self._on_slider(n, value))
                widget = slider
            elif spec["type"] == "combo":
                values = spec.get("choices") or info.get("values")
                if not values:
                    continue
                combo = QtWidgets.QComboBox()
                for value in values:
                    label = self.AWB_LABELS.get(int(value), str(value))
                    combo.addItem(label, value)
                combo.currentIndexChanged.connect(lambda _idx, n=name, c=combo: self._on_combo(n, c))
                widget = combo
            if widget is None:
                continue
            self._control_widgets[name] = widget
            self.form.addRow(spec["label"] + ":", widget)
        self._update_dependencies()

    # ------------------------------------------------------------------ interactions
    def _on_slider(self, name: str, value: float):
        if not self._source:
            return
        spec = self._spec_by_name.get(name, {})
        if spec.get("decimals", 2) == 0 or spec.get("type") == "slider" and spec.get("round", False):
            value_to_send = int(round(value))
        else:
            value_to_send = float(value)
        if getattr(self._source, "apply_control", lambda *_: False)(name, value_to_send):
            self.status_label.setText(f"Set {spec.get('label', name)} to {value_to_send}.")
        else:
            self.status_label.setText(f"Failed to update {spec.get('label', name)}.")
        self.refresh_from_source()

    def _on_toggle(self, name: str, value: bool):
        if not self._source:
            return
        spec = self._spec_by_name.get(name, {})
        if getattr(self._source, "apply_control", lambda *_: False)(name, bool(value)):
            state = "On" if value else "Off"
            self.status_label.setText(f"{spec.get('label', name)}: {state}.")
        else:
            self.status_label.setText(f"Failed to update {spec.get('label', name)}.")
        self._update_dependencies()
        self.refresh_from_source()

    def _on_combo(self, name: str, combo: QtWidgets.QComboBox):
        if not self._source:
            return
        spec = self._spec_by_name.get(name, {})
        value = combo.currentData()
        if getattr(self._source, "apply_control", lambda *_: False)(name, value):
            self.status_label.setText(f"{spec.get('label', name)}: {combo.currentText()}.")
        else:
            self.status_label.setText(f"Failed to update {spec.get('label', name)}.")
        self.refresh_from_source()

    def _on_auto_optimize(self):
        if not self._source:
            return
        if getattr(self._source, "auto_optimize", lambda: False)():
            self.status_label.setText("Auto optimisation applied.")
            self.refresh_from_source()
        else:
            self.status_label.setText("Auto optimisation failed for this camera.")

    def _update_dependencies(self):
        for child, deps in self.DEPENDENCIES.items():
            widget = self._control_widgets.get(child)
            if not widget:
                continue
            enabled = True
            for parent, expected in deps.items():
                parent_widget = self._control_widgets.get(parent)
                if isinstance(parent_widget, QtWidgets.QCheckBox):
                    current = parent_widget.isChecked()
                else:
                    current = None
                if expected is False and current:
                    enabled = False
                elif expected is True and current is not True:
                    enabled = False
            widget.setEnabled(enabled and self.isEnabled())

    # ------------------------------------------------------------------ refresh
    def refresh_from_source(self):
        if not self._source:
            return
        for name, widget in self._control_widgets.items():
            value = getattr(self._source, "current_control_value", lambda *_: None)(name)
            if value is None:
                # Fall back to default metadata
                info = getattr(self._source, "control_info", lambda *_: {})(name) or {}
                value = info.get("default")
            if value is None:
                continue
            if isinstance(widget, SliderControl):
                widget.set_value(float(value))
            elif isinstance(widget, QtWidgets.QCheckBox):
                block = widget.blockSignals(True)
                widget.setChecked(bool(value))
                widget.blockSignals(block)
            elif isinstance(widget, QtWidgets.QComboBox):
                block = widget.blockSignals(True)
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                widget.blockSignals(block)
        self._update_dependencies()

class LiveTab(QtWidgets.QWidget):
    capture_signal = QtCore.Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.src = None
        self.timer = QtCore.QTimer(self, interval=0)
        self.timer.timeout.connect(self._on_tick)
        self.frame = None
        self._metadata_counter = 0

        # UI
        self.image_view = ImageView()
        self.control_panel = CameraControlPanel()
        self.control_panel.setMinimumWidth(260)
        self.control_panel.setMaximumWidth(380)
        self.btn_start = QtWidgets.QPushButton("Start Camera")
        self.btn_full = QtWidgets.QPushButton("Fullscreen")
        self.btn_capture = QtWidgets.QPushButton("Capture Step")
        self.session_dir_label = QtWidgets.QLabel("")
        self.btn_change_dir = QtWidgets.QPushButton("Change Session Dir")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_full)
        top.addStretch(1)
        top.addWidget(self.btn_change_dir)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.btn_capture)
        bottom.addStretch(1)
        bottom.addWidget(self.session_dir_label)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.image_view)
        self.splitter.addWidget(self.control_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(1, False)
        self.splitter.setSizes([900, 300])

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.splitter, 1)
        lay.addLayout(bottom)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_full.clicked.connect(self.toggle_fullscreen)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_change_dir.clicked.connect(self.change_session_dir)

        self._fullscreen = False
        self._session_dir = None
        self.control_panel.set_source(None)

    def set_session_dir(self, path: str):
        self._session_dir = path
        self.session_dir_label.setText(path)

    def start_camera(self):
        if self.src:
            return
        self.src = best_available_source(prefer_pi=True)
        self.control_panel.set_source(self.src)
        self.timer.start()

    def stop_camera(self):
        self.timer.stop()
        if self.src:
            self.src.stop()
            self.src = None
        self.control_panel.set_source(None)

    def toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        w = self.window().windowHandle()
        if self._fullscreen:
            self.window().showFullScreen()
        else:
            self.window().showNormal()

    def _on_tick(self):
        if not self.src: return
        frame = self.src.read()
        if frame is None: return
        self.frame = frame
        self.image_view.set_image(frame)
        self._metadata_counter = (self._metadata_counter + 1) % 15
        if self._metadata_counter == 0:
            self.control_panel.refresh_from_source()

    def capture(self):
        if self.frame is None: return
        self.capture_signal.emit(self.frame.copy())

    def change_session_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pick session root", default_sessions_root())
        if d:
            self.set_session_dir(d)

class RebuildTab(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session

        # UI
        self.parts_list = QtWidgets.QListWidget()
        self.descriptor = QtWidgets.QLineEdit()
        self.btn_generate = QtWidgets.QPushButton("Generate Composite")
        self.img_view = ImageView()

        self.btn_blink = QtWidgets.QPushButton("Blink")
        self.btn_fade = QtWidgets.QPushButton("Show Fade Slider")
        self.fade_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fade_slider.setRange(0,100); self.fade_slider.setValue(50)
        self.fade_slider.setVisible(False)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Parts (Step k → k+1):"))
        left.addWidget(self.parts_list, 1)
        left.addWidget(QtWidgets.QLabel("Descriptor:"))
        left.addWidget(self.descriptor)
        left.addWidget(self.btn_generate)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.img_view, 1)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.btn_blink)
        h.addWidget(self.btn_fade)
        h.addWidget(self.fade_slider, 1)
        right.addLayout(h)

        lay = QtWidgets.QHBoxLayout(self)
        wleft = QtWidgets.QWidget(); wleft.setLayout(left)
        wleft.setMaximumWidth(300)
        lay.addWidget(wleft)
        lay.addLayout(right, 1)

        self.btn_generate.clicked.connect(self.generate_for_selected)
        self.btn_blink.clicked.connect(self.toggle_blink)
        self.btn_fade.clicked.connect(self.toggle_fade)
        self.fade_slider.valueChanged.connect(self.on_fade_change)

        self.A = None; self.B = None; self.A_v=None; self.B_v=None
        self.refresh_pairs()

    def refresh_pairs(self):
        self.parts_list.clear()
        for k, A, B in self.session.pairs():
            self.parts_list.addItem(f"Part #{k} — Step {k} → {k+1}")

    def selected_index(self):
        row = self.parts_list.currentRow()
        if row < 0: return None
        return row + 1  # step index k

    def toggle_blink(self):
        on = self.img_view._blink is False
        self.img_view.start_blink(on)

    def toggle_fade(self):
        vis = not self.fade_slider.isVisible()
        self.fade_slider.setVisible(vis)

    def on_fade_change(self, v):
        self.img_view.set_fade_alpha(v/100.0)

    def generate_for_selected(self):
        k = self.selected_index()
        if k is None:
            QtWidgets.QMessageBox.information(self, "No selection", "Pick a part pair from the list.")
            return
        A_path = self.session.step_path(k)
        B_path = self.session.step_path(k+1)
        A = imread_color(A_path)
        B = imread_color(B_path)
        if A is None or B is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to load step images.")
            return

        # downscale for alignment speed, but keep scale factor
        scale = 1.0
        maxdim = max(A.shape[1], A.shape[0])
        if maxdim > 1280:
            scale = 1280.0 / maxdim
            A_small = cv2.resize(A, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            B_small = cv2.resize(B, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            A_small, B_small = A, B

        Ag = cv2.cvtColor(A_small, cv2.COLOR_BGR2GRAY)
        Bg = cv2.cvtColor(B_small, cv2.COLOR_BGR2GRAY)

        # Stage A: global align
        res = orb_similarity_transform(Ag, Bg, max_features=2000, ratio=0.75, ransac_thresh=2.0)
        if not res.success:
            QtWidgets.QMessageBox.warning(self, "Alignment", "Feature alignment failed; showing unaligned diff.")
            T = np.float32([[1,0,0],[0,1,0]])
        else:
            T = res.T
        B_warp_small = cv2.warpAffine(B_small, T, (A_small.shape[1], A_small.shape[0]))

        # Optional ECC refine on small images
        try:
            T_ref = ecc_refine(Ag, cv2.cvtColor(B_warp_small, cv2.COLOR_BGR2GRAY), T)
            B_warp_small = cv2.warpAffine(B_small, T_ref, (A_small.shape[1], A_small.shape[0]))
            T = T_ref
        except Exception:
            pass

        # Stage B: change localization on small
        m, bbox, centroid = largest_change_component(A_small, B_warp_small, thresh=25, kernel=3)
        if bbox is None:
            # fallback center crop
            h, w = A_small.shape[:2]
            bbox = (w//3, h//3, w//3, h//3)
            centroid = (w/2, h/2)

        # Build centering transform (no canonical rotation for simplicity here; set to 0)
        V_small = build_centering_transform(A_small.shape, centroid, bbox, target_fill=0.55, rotation_deg=0.0)

        # Re-run alignment and V on full-res
        # Upscale the transforms to full-res coordinates if needed
        if scale != 1.0:
            # Adjust T and V to full-resolution scale
            S = np.array([[scale,0,0],[0,scale,0],[0,0,1]], dtype=np.float32)
            S_inv = np.linalg.inv(S)
            T33 = np.eye(3, dtype=np.float32); T33[:2,:] = T
            V33 = np.eye(3, dtype=np.float32); V33[:2,:] = V_small
            T_full = (S_inv @ T33 @ S)[:2,:]
            V_full = (S_inv @ V33 @ S)[:2,:]
        else:
            T_full = T; V_full = V_small

        B_warp = cv2.warpAffine(B, T_full, (A.shape[1], A.shape[0]))
        A_v = cv2.warpAffine(A, V_full, (A.shape[1], A.shape[0]))
        B_v = cv2.warpAffine(B_warp, V_full, (A.shape[1], A.shape[0]))

        composite, mask = compose_magenta(A_v, B_v, thresh=25)

        # Save composite
        desc = self.descriptor.text().strip() or "part"
        comp_path = Path(self.session.composites_dir) / f"Part_{k:02d}_{desc.replace(' ','_')}_composite.png"
        imwrite_safe(str(comp_path), composite)

        # Update viewer
        self.A, self.B = A, B
        self.A_v, self.B_v = A_v, B_v
        self.img_view.set_image(composite)
        self.img_view.set_blink_pair(A_v, B_v)
        self.img_view.set_fade_pair(A_v, B_v)

        QtWidgets.QMessageBox.information(self, "Done", f"Composite saved:{comp_path}")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, session_dir=None):
        super().__init__()
        self.setWindowTitle("PiScope Studio — Prototype")
        self.resize(1280, 800)

        self.session_dir = session_dir or new_session_dir()
        self.session = Session(self.session_dir)

        self.tabs = QtWidgets.QTabWidget()
        self.live = LiveTab()
        self.rebuild = RebuildTab(self.session)

        self.tabs.addTab(self.live, "Live")
        self.tabs.addTab(self.rebuild, "Rebuild")

        self.setCentralWidget(self.tabs)

        # Wiring
        self.live.set_session_dir(self.session_dir)
        self.live.capture_signal.connect(self.on_capture)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("F"), self, activated=self.toggle_fullscreen)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.live.capture)
        QtGui.QShortcut(QtGui.QKeySequence("B"), self, activated=self.rebuild.toggle_blink)
        QtGui.QShortcut(QtGui.QKeySequence("V"), self, activated=self.rebuild.toggle_fade)

        # Menu
        m = self.menuBar().addMenu("Session")
        act_new = m.addAction("New Session..."); act_new.triggered.connect(self.new_session)
        act_open = m.addAction("Open Session..."); act_open.triggered.connect(self.open_session)

    def on_capture(self, img):
        sid, path = self.session.add_step(img)
        self.statusBar().showMessage(f"Captured step {sid}: {path}", 4000)
        self.rebuild.refresh_pairs()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def new_session(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pick sessions root", default_sessions_root())
        if not d: return
        sdir = new_session_dir(d)
        self._load_session(sdir)

    def open_session(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open session", default_sessions_root())
        if not d: return
        self._load_session(d)

    def _load_session(self, sdir):
        self.session_dir = sdir
        self.session = Session(self.session_dir)
        self.live.set_session_dir(self.session_dir)
        self.rebuild.session = self.session
        self.rebuild.refresh_pairs()

def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
