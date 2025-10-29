# PiScope Studio — StepView Timeline Prototype

A working prototype of a **watchmaker-focused microscope UI** for Raspberry Pi 5 (or any Linux PC with a USB microscope).
It provides: full-screen live view, **StepView Timeline** captures, and a **Rebuild/Compare** feature that auto-aligns two
consecutive steps, locates the **changed region**, and generates a **magenta overlay composite** centered and zoomed on the removed/added part.

This prototype is **non-AI** and uses classical computer vision (OpenCV). It supports USB cameras out of the box and
*optionally* the Raspberry Pi HQ Camera via `Picamera2` if installed.

## ✨ Features

- **Live View** (USB camera by default; optional Pi HQ camera via Picamera2)
- **One‑click Step Capture** with auto‑naming in a session folder
- **StepView Timeline**: ordered steps for disassembly/reassembly
- **Rebuild / Compare** per consecutive step pair:
  - Global **subject alignment** (similarity transform, ORB + RANSAC + optional ECC refinement)
  - **Change localization** via absdiff + threshold + morphology
  - **Centered, zoomed view** on the changed region (canonical rotation optional)
  - **Magenta overlay** composite (Before image with change highlighted)
  - Viewer with **blink**, **fade slider**, and **split** modes
- **Descriptors** per part (user-provided text)
- **Session autosave** (`session.json`)

## 🧱 Project Layout

```
PiScopeStudio/
├── piscopestudio/
│   ├── __init__.py
│   ├── app.py                # Entry point
│   ├── camera_backends.py    # USB/OpenCV capture; optional Picamera2
│   ├── timeline.py           # Session & step management
│   ├── align.py              # Similarity alignment + change detection + view framing
│   ├── diffview.py           # UI widgets for compare/fade/blink + magenta composite
│   ├── ui_main.py            # MainWindow, Live tab, Rebuild tab
│   ├── utils.py              # Helpers (paths, timestamps, image io)
│   └── resources/            # (placeholder for icons)
├── requirements.txt
├── README.md
└── run.sh
```

## 🚀 Quick Start (USB camera, Linux desktop or Pi)

1) Install dependencies:
```bash
sudo apt update
sudo apt install -y python3-opencv python3-pyside6
# (optional, for Pi HQ camera)
sudo apt install -y python3-picamera2
```

Or use a virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the app:
```bash
cd PiScopeStudio
python3 -m piscopestudio.app
# or
./run.sh
```

3) Controls:
- **Live tab**
  - **Capture Step**: saves a still to the current session folder and appends it to the timeline.
  - **Fullscreen** toggle.
- **Rebuild tab**
  - Select a **pair** (Step k → k+1), enter/edit a **descriptor**, and **Generate Composite**.
  - Use **Blink**, **Fade**, or **Split** to inspect Before vs After aligned views.
  - The composite (Before + magenta) is saved in the session’s `composites/` folder.

## 📸 Sessions & Files

By default, a new session is created under:
```
~/PiScopeSessions/YYYY-MM-DD_HHMMSS/
  ├── session.json
  ├── Step_01_YYYYMMDD_HHMMSS.jpg
  ├── Step_02_YYYYMMDD_HHMMSS.jpg
  └── composites/
      └── Part_01_composite.png
```

You can change the session directory from the **Live** tab.

## 🧠 How the Compare/Rebuild Works

Given two consecutive steps (A=before, B=after):

1. **Global alignment** (subject = movement):
   - ORB keypoints + descriptor matching + Lowe ratio filter
   - `estimateAffinePartial2D(..., RANSAC)` for a similarity transform
   - optional **ECC** refinement

2. **Change localization**:
   - `absdiff(A, B_aligned)` → grayscale → threshold → morphology
   - pick **largest component** → centroid + bounding box

3. **View framing**:
   - Build a transform that rotates to **canonical upright** (optional), translates the centroid to the center,
     and **scales** so the box fills ~55% of the viewport.

4. **Magenta composite**:
   - Recompute diff on the framed pair and alpha-blend magenta on the Before image to highlight the change.
   - Annotate with **part number** and **descriptor**.

### Fallbacks / Edge Cases
- If alignment fails (glare, weak texture), the UI offers to run with default framing (no rotation) or re-try with different sensitivity.
- Background masking can be added to improve robustness (not enabled by default in this prototype).

## 🧪 Tested Platforms

- Ubuntu 22.04 + USB webcam
- Raspberry Pi OS (Bookworm) + USB microscope (UVC)
- Raspberry Pi 5 + HQ Camera (requires Picamera2; tested with apt package)

> Note: If Picamera2 isn’t available, the app uses the USB/OpenCV backend automatically.

## ⌨️ Hotkeys (prototype)

- `F` — Toggle fullscreen
- `Space` — Capture Step (Live tab)
- `B` — Blink toggle (Rebuild tab)
- `V` — Show Fade slider (Rebuild tab)

## 📝 License

MIT License — do whatever you want, attribution appreciated.


## 🧭 Roadmap

- Manual 3‑point align (wizard) as a robust fallback
- Measurement tools + calibration
- PDF report exporter
- Illumination controls (GPIO/PWM)
- EDF / Stitch (batch tools, off the main UI thread)

---

**Author:** Prototype generated by ChatGPT (PiScope Studio concept). Feel free to extend!

