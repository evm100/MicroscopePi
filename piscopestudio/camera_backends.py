from typing import Any, Dict, Optional

import cv2
import numpy as np

try:  # Optional dependency on libcamera for enumerations
    from libcamera import controls as libcamera_controls  # type: ignore
except Exception:  # pragma: no cover - libcamera is only available on Pi
    libcamera_controls = None

class FrameSource:
    """Abstract frame source."""
    def start(self): ...
    def read(self): ...
    def stop(self): ...

    # Camera control capabilities -------------------------------------------------
    def supports_manual_controls(self) -> bool:
        return False

    def control_info(self, name: str) -> Dict[str, Any]:
        """Return metadata about a camera control."""
        return {}

    def apply_control(self, name: str, value: Any) -> bool:
        return False

    def auto_optimize(self) -> bool:
        return False

    def current_control_value(self, name: str) -> Optional[Any]:
        return None

    def read_metadata(self) -> Dict[str, Any]:
        return {}

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
        self._controls_info: Dict[str, Dict[str, Any]] = {}
        self._last_controls: Dict[str, Any] = {}

    def start(self):
        try:
            from picamera2 import Picamera2
        except Exception:
            return False
        self.Picamera2 = Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": self.size, "format": "RGB888"},
            controls={"FrameRate": self.fps},
        )
        self.cam.configure(cfg)
        self.cam.start()
        self._controls_info = self._extract_control_info()
        return True

    def read(self):
        if not self.cam: return None
        return self.cam.capture_array("main")

    def stop(self):
        if self.cam:
            self.cam.stop()
            self.cam=None

    # ------------------------------------------------------------------ controls
    def supports_manual_controls(self) -> bool:
        return self.cam is not None

    def _extract_control_info(self) -> Dict[str, Dict[str, Any]]:
        info: Dict[str, Dict[str, Any]] = {}
        if not self.cam:
            return info
        try:
            ctrl_dict = getattr(self.cam, "camera_controls", {})
        except Exception:
            ctrl_dict = {}
        for key, ctrl in ctrl_dict.items():
            details: Dict[str, Any] = {}
            for attr in ("min", "max", "default", "values", "step"):
                try:
                    details[attr] = getattr(ctrl, attr)
                except Exception:
                    continue
            info[key] = details
        return info

    def control_info(self, name: str) -> Dict[str, Any]:
        base = self._controls_info.get(name, {}).copy()
        if name == "AwbMode" and libcamera_controls and "values" not in base:
            try:
                base["values"] = [
                    libcamera_controls.AwbModeEnum.Auto,
                    libcamera_controls.AwbModeEnum.Tungsten,
                    libcamera_controls.AwbModeEnum.Fluorescent,
                    libcamera_controls.AwbModeEnum.Indoor,
                    libcamera_controls.AwbModeEnum.Daylight,
                    libcamera_controls.AwbModeEnum.Cloudy,
                ]
            except Exception:
                pass
        return base

    def apply_control(self, name: str, value: Any) -> bool:
        if not self.cam:
            return False
        try:
            self.cam.set_controls({name: value})
            self._last_controls[name] = value
            return True
        except Exception:
            return False

    def auto_optimize(self) -> bool:
        if not self.cam:
            return False
        controls: Dict[str, Any] = {}
        # Enable automatic algorithms when available
        if "AeEnable" in self._controls_info:
            controls["AeEnable"] = True
        if "AwbEnable" in self._controls_info:
            controls["AwbEnable"] = True
        # Reset primary sliders to defaults when known
        for key in ("ExposureTime", "AnalogueGain", "Brightness", "Contrast", "Saturation", "Sharpness"):
            info = self._controls_info.get(key)
            if info and "default" in info:
                controls[key] = info["default"]
        try:
            if controls:
                self.cam.set_controls(controls)
                self._last_controls.update(controls)
            return True
        except Exception:
            return False

    def current_control_value(self, name: str) -> Optional[Any]:
        metadata = self.read_metadata()
        if metadata and name in metadata:
            return metadata.get(name)
        return self._last_controls.get(name)

    def read_metadata(self) -> Dict[str, Any]:
        if not self.cam:
            return {}
        try:
            meta = self.cam.capture_metadata()
            return meta or {}
        except Exception:
            return {}

def best_available_source(prefer_pi=True):
    if prefer_pi:
        src = Picamera2Source()
        if src.start():
            return src
    # Fallback to USB camera index 0
    src = USBCamera(0)
    src.start()
    return src
