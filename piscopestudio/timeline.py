import json, os
from pathlib import Path
from .utils import ensure_dir, auto_step_name, imwrite_safe

class Session:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.steps = []  # list of {id, filename, note}
        self.calibration = {"px_per_mm": None, "canonical_rotation_deg": 0.0}
        self._load()

    @property
    def composites_dir(self):
        p = Path(self.root_dir) / "composites"
        ensure_dir(p)
        return str(p)

    def _session_json_path(self):
        return str(Path(self.root_dir) / "session.json")

    def _load(self):
        p = self._session_json_path()
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
            self.steps = data.get("steps", [])
            self.calibration = data.get("calibration", self.calibration)

    def save(self):
        p = self._session_json_path()
        data = {
            "steps": self.steps,
            "calibration": self.calibration
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    def add_step(self, img, note:str=""):
        # Assign next id
        sid = len(self.steps) + 1
        fname = auto_step_name(sid)
        path = str(Path(self.root_dir) / fname)
        imwrite_safe(path, img)
        self.steps.append({"id": sid, "filename": fname, "note": note})
        self.save()
        return sid, path

    def step_path(self, sid_or_index):
        if isinstance(sid_or_index, int) and 1 <= sid_or_index <= len(self.steps):
            fname = self.steps[sid_or_index-1]["filename"]
            return str(Path(self.root_dir) / fname)
        raise ValueError("Invalid step id")

    def pairs(self):
        # Return (k, pathA, pathB) for step k -> k+1
        out = []
        for k in range(1, len(self.steps)):
            A = self.step_path(k)
            B = self.step_path(k+1)
            out.append((k, A, B))
        return out
