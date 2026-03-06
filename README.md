# Women Body Blur — User Guide

Full-body detection and blurring pipeline for selectively anonymizing specific individuals in video. Uses **YOLOv8 instance segmentation** with **ByteTrack** multi-object tracking to produce pixel-accurate body masks, then applies Gaussian blur to designated track IDs while preserving audio.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.12+ |
| GPU | Any CUDA-capable NVIDIA GPU | RTX 3050 or higher |
| VRAM | 2 GB | 4 GB+ |
| CUDA Toolkit | 11.8 | 12.x |
| Disk space | ~500 MB (model + packages) | — |

All Python dependencies are **auto-installed** by Cell 1 of the notebook. No manual `pip install` is required.

### Packages installed automatically

- `torch` + `torchvision` (CUDA build)
- `opencv-python`
- `numpy`
- `matplotlib`
- `ultralytics` (YOLOv8)
- `lapx` (ByteTrack dependency)
- `moviepy` (audio merging)

---

## Project Structure

```
women-face-blur/
├── main.ipynb                         # Main notebook (6 cells)
├── README.md                          # This user guide
├── TECHNICAL.md                       # Technical deep-dive
├── ovaloffice.mp4                     # Input video (user-provided)
├── ovaloffice_women_blurred.mp4       # Output video (generated)
└── models/
    └── yolov8m-seg.pt                 # YOLOv8m-seg weights (~50 MB, auto-downloaded)
```

---

## Quick Start

1. Place your input video as `ovaloffice.mp4` in the project root.
2. Open `main.ipynb` in VS Code or JupyterLab.
3. **Run all cells** (Cells 1–6) sequentially.
4. Cell 5 displays annotated sample frames — identify the track IDs of the people you want to blur.
5. Set `WOMEN_TRACK_IDS = {id1, id2}` in Cell 6 with the correct IDs.
6. Run Cell 6 to process the video. Output is saved as `ovaloffice_women_blurred.mp4` with audio.

---

## Step-by-Step Walkthrough

### Cell 1 — Environment Setup

Verifies/installs all dependencies. If a CPU-only PyTorch is detected, it automatically reinstalls the CUDA build and invalidates the module cache so no kernel restart is needed.

**Output to look for:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

If CUDA shows `False`, check that your NVIDIA drivers are up to date.

---

### Cell 2 — Load YOLOv8m-seg Model

Loads the YOLOv8 medium segmentation model onto the GPU. On first run, the weights file (~50 MB) is downloaded automatically and cached to `models/yolov8m-seg.pt`.

---

### Cell 3 — Person Tracking Utilities

Defines `track_persons_frame()` — the core function that runs YOLOv8-seg with ByteTrack on a single frame. Returns a list of detected persons, each with:

- `box` — bounding box coordinates `(x1, y1, x2, y2)`
- `confidence` — detection confidence score
- `track_id` — persistent integer ID across frames
- `mask` — binary segmentation mask (pixel-level body outline)

---

### Cell 4 — Blur Utilities

Defines two blur functions:

- **`blur_person_mask()`** — Applies Gaussian blur only to pixels inside the segmentation mask. This is the primary method and produces clean, body-shaped blur.
- **`blur_person_box()`** — Fallback that blurs a rectangular bounding box region when no mask is available.

---

### Cell 5 — Diagnostic Scan

Runs tracking on the first ~90 frames of the video and displays 4 annotated sample frames with:

- Green bounding boxes around each detected person
- Green mask overlays showing the segmentation area
- Track ID labels (`ID=1`, `ID=2`, etc.)

**Action required:** Look at the output images, identify which track IDs correspond to the people you want to blur, then set those IDs in Cell 6.

---

### Cell 6 — Video Processing + Audio Merge

The main processing cell. Set `WOMEN_TRACK_IDS` at the top of the cell, then run it.

**What it does:**

1. Reads every frame from the input video.
2. Runs person detection + tracking on each frame.
3. For any detected person whose `track_id` is in `WOMEN_TRACK_IDS`, applies pixel-accurate Gaussian blur using the segmentation mask.
4. Writes blurred frames to a temporary file via OpenCV.
5. Uses `moviepy` to merge the original audio track onto the blurred video.
6. Deletes the temporary file, leaving only the final output.

**Progress output:**
```
[██████████████████████████████] 100.0%  frame 1210/1210  3.4 fps  ETA 0s
```

---

## Configuration

All tunable parameters are in Cell 6:

| Parameter | Default | Description |
|---|---|---|
| `WOMEN_TRACK_IDS` | `{8, 9}` | Set of track IDs to blur. Change these to match your video. |
| `PERSON_CONF` | `0.35` | Minimum detection confidence. Lower = more sensitive, higher = fewer false positives. |
| `BLUR_KSIZE` | `(99, 99)` | Gaussian blur kernel size. Larger = stronger blur. Must be odd numbers. |
| `TRACKER_CONFIG` | `"bytetrack.yaml"` | Tracking algorithm config. ByteTrack is the default and recommended. |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA available: False` | Update NVIDIA GPU drivers. Ensure CUDA toolkit is installed. |
| Track IDs don't match between Cell 5 and Cell 6 | Both cells reset the tracker (`model.predictor = None`). Always run Cell 5 before Cell 6 in order. Do not run other tracking code between them. |
| Blur misses the person on some frames | Lower `PERSON_CONF` (e.g., `0.25`). The person may be partially occluded. |
| Output video has no audio | Ensure `moviepy` is installed (Cell 1 handles this). Check that the input video actually has an audio track. |
| `ModuleNotFoundError: No module named 'lapx'` | Run Cell 1 again — it auto-installs `lapx`. |
| Out of GPU memory | Close other GPU applications. Or switch to `yolov8s-seg.pt` (smaller model, less accurate). |