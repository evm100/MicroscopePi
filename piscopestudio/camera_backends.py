import cv2
import numpy as np

class FrameSource:
    """Abstract frame source."""
    def start(self): ...
    def read(self): ...
    def stop(self): ...

class USBCamera(FrameSource):
    def __init__(self, index=0, width=1920, height=1080, fps=60):
        self.index=index; self.width=width; self.height=height; self.fps=fps
        self.cap=None

    def start(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        if self.width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps: self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read(self):
        if not self.cap: return None
        ok, frame = self.cap.read()
        if not ok: return None
        return frame

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap=None

class Picamera2Source(FrameSource):
    """Optional: only works if Picamera2 is installed."""
    def __init__(self, size=(1920,1080), fps=60):
        self.size=size; self.fps=fps; self.cam=None

    def start(self):
        try:
            from picamera2 import Picamera2
        except Exception:
            return False
        self.Picamera2 = Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(main={ "size": self.size, "format": "RGB888"}, controls={"FrameRate": self.fps})
        self.cam.configure(cfg)
        self.cam.start()
        return True

    def read(self):
        if not self.cam: return None
        return self.cam.capture_array("main")

    def stop(self):
        if self.cam:
            self.cam.stop()
            self.cam=None

def best_available_source(prefer_pi=True):
    if prefer_pi:
        src = Picamera2Source()
        if src.start():
            return src
    # Fallback to USB camera index 0
    src = USBCamera(0)
    src.start()
    return src
