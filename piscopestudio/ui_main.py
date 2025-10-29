import os, cv2, numpy as np
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from .camera_backends import best_available_source
from .timeline import Session
from .utils import new_session_dir, default_sessions_root, ensure_dir, imread_color, imwrite_safe
from .diffview import ImageView
from .align import orb_similarity_transform, ecc_refine, largest_change_component, build_centering_transform, compose_magenta

class LiveTab(QtWidgets.QWidget):
    capture_signal = QtCore.Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.src = None
        self.timer = QtCore.QTimer(self, interval=0)
        self.timer.timeout.connect(self._on_tick)
        self.frame = None

        # UI
        self.image_view = ImageView()
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

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.image_view, 1)
        lay.addLayout(bottom)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_full.clicked.connect(self.toggle_fullscreen)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_change_dir.clicked.connect(self.change_session_dir)

        self._fullscreen = False
        self._session_dir = None

    def set_session_dir(self, path: str):
        self._session_dir = path
        self.session_dir_label.setText(path)

    def start_camera(self):
        if self.src:
            return
        self.src = best_available_source(prefer_pi=True)
        self.timer.start()

    def stop_camera(self):
        self.timer.stop()
        if self.src:
            self.src.stop()
            self.src = None

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
