import os, time, cv2, numpy as np
from pathlib import Path
from datetime import datetime

def ts():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)

def default_sessions_root():
    return str(Path.home() / "PiScopeSessions")

def new_session_dir(root=None):
    root = root or default_sessions_root()
    ensure_dir(root)
    sdir = Path(root) / ts()
    ensure_dir(sdir)
    ensure_dir(sdir / "composites")
    return str(sdir)

def auto_step_name(step_index: int):
    return f"Step_{step_index:02d}_{ts()}.jpg"

def imread_color(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def imwrite_safe(path, img):
    # Support for non-ASCII paths on Windows too
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        encode = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        data = cv2.imencode(".jpg", img, encode)[1]
    elif ext in [".png"]:
        data = cv2.imencode(".png", img)[1]
    else:
        data = cv2.imencode(".png", img)[1]
        path = os.path.splitext(path)[0] + ".png"
    data.tofile(path)
    return path
