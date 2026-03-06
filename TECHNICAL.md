# Technical Reference — Women Body Blur Pipeline

Detailed technical documentation covering architecture, algorithms, data flow, and implementation internals.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Model Details](#model-details)
4. [Tracking Algorithm](#tracking-algorithm)
5. [Segmentation Mask Processing](#segmentation-mask-processing)
6. [Blur Implementation](#blur-implementation)
7. [Audio Pipeline](#audio-pipeline)
8. [Performance Characteristics](#performance-characteristics)
9. [API Reference](#api-reference)
10. [Dependencies & Compatibility](#dependencies--compatibility)

---

## Architecture Overview

```
 Input Video (.mp4)
       │
       ▼
 ┌─────────────────────────────────────────────────────┐
 │  YOLOv8m-seg  (Instance Segmentation)               │
 │  ┌───────────────────┐    ┌──────────────────────┐  │
 │  │ Backbone: CSPNet   │───▶│ Seg Head: Proto masks │  │
 │  │ Neck: PANet        │    │ + mask coefficients   │  │
 │  │ Det Head: Detect   │    └──────────────────────┘  │
 │  └───────────────────┘                               │
 └──────────────┬──────────────────────────────────────┘
                │  Per-frame detections (class=0 "person")
                ▼
 ┌─────────────────────────────────────────────────────┐
 │  ByteTrack  (Multi-Object Tracker)                   │
 │  - Kalman filter prediction                          │
 │  - IoU-based association (high + low conf)           │
 │  - Persistent track ID assignment                    │
 └──────────────┬──────────────────────────────────────┘
                │  Tracked persons: {track_id, box, mask, conf}
                ▼
 ┌─────────────────────────────────────────────────────┐
 │  Selective Blur (mask-based Gaussian)                │
 │  - Filter by WOMEN_TRACK_IDS                        │
 │  - Resize mask to frame resolution                   │
 │  - Apply GaussianBlur only within masked pixels      │
 └──────────────┬──────────────────────────────────────┘
                │  Blurred frames
                ▼
 ┌─────────────────────────────────────────────────────┐
 │  Video Writer (OpenCV) → Temp .mp4                   │
 │  Audio Merge  (moviepy) → Final .mp4 with sound      │
 └─────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Frame Extraction

OpenCV's `VideoCapture` reads frames sequentially from the input `.mp4`. Each frame is a BGR `numpy.ndarray` of shape `(H, W, 3)` with `dtype=uint8`.

```python
cap = cv2.VideoCapture(str(VIDEO_INPUT_PATH))
ret, frame = cap.read()  # frame.shape = (1080, 1920, 3)
```

### Stage 2: Instance Segmentation + Tracking

Each frame is passed to `model.track()` with the following parameters:

```python
results = model.track(
    frame_bgr,
    classes=[0],            # COCO class 0 = "person"
    conf=0.35,              # Minimum confidence threshold
    persist=True,           # Maintain tracks across frames
    tracker="bytetrack.yaml",
    verbose=False,
)
```

The `persist=True` flag instructs the tracker to maintain state between calls — each person gets a stable integer `track_id` that persists across frames as long as the person remains visible.

**Output per detection:**

| Field | Type | Description |
|---|---|---|
| `boxes.xyxy[i]` | `Tensor[4]` | Bounding box `(x1, y1, x2, y2)` in pixel coords |
| `boxes.conf[i]` | `Tensor[1]` | Detection confidence `[0, 1]` |
| `boxes.id[i]` | `Tensor[1]` | ByteTrack-assigned persistent track ID |
| `masks.data[i]` | `Tensor[H', W']` | Binary segmentation mask at model resolution |

### Stage 3: Selective Blur

Only detections whose `track_id ∈ WOMEN_TRACK_IDS` are blurred. The blur is applied using the segmentation mask, not the bounding box, for pixel-accurate body anonymization.

### Stage 4: Video Write + Audio Merge

Blurred frames are written to a temporary `.mp4` via OpenCV's `VideoWriter` (codec: `mp4v`). Then `moviepy` merges the audio track from the original input onto the blurred video using `libx264` + `aac` codecs.

---

## Model Details

### YOLOv8m-seg

| Property | Value |
|---|---|
| Architecture | YOLOv8 Medium with Segmentation head |
| Backbone | CSPDarknet (Cross-Stage Partial) |
| Neck | PANet (Path Aggregation Network) |
| Input resolution | 640×640 (auto-resized internally) |
| Parameters | ~27M |
| Weights file | `models/yolov8m-seg.pt` (~52 MB) |
| Training data | COCO 2017 (80 classes, 330K images) |
| Relevant class | Class 0 — `person` |

The model outputs both bounding boxes and per-instance segmentation masks. The mask head generates 32 prototype masks and per-detection mask coefficients, which are linearly combined to produce the final binary mask.

### Why YOLOv8m-seg over alternatives

| Approach | Limitation |
|---|---|
| Face detection (Caffe SSD) | Fails when face is occluded, turned, or partially visible |
| Gender classification (CaffeNet) | Low accuracy on non-frontal faces, cannot detect body |
| Pose estimation (OpenPose) | No segmentation mask — only keypoints |
| YOLOv8-det (detection only) | Bounding box blur includes background pixels |
| **YOLOv8-seg (chosen)** | **Pixel-accurate body mask even with partial occlusion** |

---

## Tracking Algorithm

### ByteTrack

ByteTrack performs multi-object tracking by associating detections across frames using a two-stage matching strategy:

1. **High-confidence matching:** Detections with `conf ≥ threshold` are matched to existing tracks using IoU (Intersection over Union) similarity via the Hungarian algorithm.

2. **Low-confidence recovery:** Unmatched tracks are re-associated with remaining low-confidence detections. This recovers occluded or partially visible persons that would otherwise be lost.

3. **Track lifecycle:**
   - **New track:** Created when a detection cannot be matched to any existing track.
   - **Active track:** Updated with each successful association.
   - **Lost track:** Maintained for a grace period (default: 30 frames) without association before deletion.

**State estimation** uses a Kalman filter predicting `(x_center, y_center, aspect_ratio, height, vx, vy, va, vh)`.

The `lapx` package provides an optimized C implementation of the Linear Assignment Problem (LAP) solver used by the Hungarian algorithm.

### Track ID Stability

Track IDs are deterministic given the same input and model state. To ensure consistency between the diagnostic scan (Cell 5) and the processing pass (Cell 6), the tracker state is reset before each:

```python
model.predictor = None  # Clears internal tracker state
```

---

## Segmentation Mask Processing

The model outputs masks at its internal resolution (typically 160×160 for 640×640 input). These are resized to the original frame resolution before application:

```python
# mask.shape = (160, 160), frame.shape = (1080, 1920, 3)
mask_resized = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
mask_bool = mask_resized > 0.5  # Binary threshold
```

`INTER_NEAREST` interpolation preserves hard edges in the mask. A threshold of `0.5` converts soft mask probabilities to a binary mask.

---

## Blur Implementation

### Mask-based blur (primary)

```python
def blur_person_mask(frame, mask, ksize=(99, 99), sigma=0):
    # 1. Resize mask from model resolution to frame resolution
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_resized > 0.5

    # 2. Blur the entire frame
    blurred = cv2.GaussianBlur(frame, ksize, sigma)

    # 3. Composite: original pixels outside mask, blurred pixels inside
    result = frame.copy()
    result[mask_bool] = blurred[mask_bool]
    return result
```

The approach blurs the full frame and then composites only the masked region. This is more efficient than extracting/blurring/reinserting irregular regions.

### Gaussian blur parameters

| Parameter | Value | Effect |
|---|---|---|
| `ksize` | `(99, 99)` | Kernel size. Larger = stronger blur. Must be odd. |
| `sigma` | `0` | Auto-calculated from `ksize`: `sigma = 0.3 * ((ksize-1) * 0.5 - 1) + 0.8` |

With `ksize=(99, 99)`, each pixel is averaged over a 99×99 neighborhood, rendering facial features and body details unrecognizable.

### Box-based blur (fallback)

Used only when the segmentation mask is unavailable (rare edge case). Applies Gaussian blur to the bounding box region with optional padding (default: 10% expansion on each side).

---

## Audio Pipeline

OpenCV's `VideoWriter` does not support audio tracks. The pipeline uses a two-pass approach:

1. **Pass 1 (OpenCV):** Write blurred frames to `_temp_no_audio.mp4` with `mp4v` codec.
2. **Pass 2 (moviepy):** Read the temp video and the original input, merge the original's audio track onto the blurred video, encode with `libx264` (video) + `aac` (audio), write to final output.
3. **Cleanup:** Delete the temporary file.

```python
from moviepy import VideoFileClip

video_clip = VideoFileClip(str(temp_video))
original_clip = VideoFileClip(str(VIDEO_INPUT_PATH))
final_clip = video_clip.with_audio(original_clip.audio)
final_clip.write_videofile(str(output), codec="libx264", audio_codec="aac")
```

`moviepy` uses `imageio-ffmpeg` (bundled ffmpeg binary) internally — no system-level ffmpeg installation is required.

---

## Performance Characteristics

Benchmarked on NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), 1920×1080 input @ 25 fps:

| Metric | Value |
|---|---|
| Processing speed | ~3.4 fps (inference + blur + write) |
| Total time (1210 frames) | ~6 minutes |
| GPU VRAM usage | ~1.5 GB |
| Peak RAM usage | ~2 GB |
| Output file size | ~25 MB (libx264 compression) |

### Bottleneck analysis

- **~70%** — YOLOv8 inference (GPU-bound)
- **~15%** — Mask resize + Gaussian blur (CPU-bound)
- **~10%** — Video I/O (disk-bound)
- **~5%** — Tracking (CPU, negligible)

---

## API Reference

### `track_persons_frame(frame_bgr, conf=0.4, tracker="bytetrack.yaml")`

Runs YOLOv8-seg with ByteTrack on a single BGR frame.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `frame_bgr` | `np.ndarray` | — | BGR image, shape `(H, W, 3)`, `dtype=uint8` |
| `conf` | `float` | `0.4` | Minimum detection confidence |
| `tracker` | `str` | `"bytetrack.yaml"` | Tracker config file |

**Returns:** `List[Dict]` — each dict contains:

| Key | Type | Description |
|---|---|---|
| `box` | `Tuple[int, int, int, int]` | `(x1, y1, x2, y2)` bounding box |
| `confidence` | `float` | Detection confidence |
| `track_id` | `int` | Persistent track ID (`-1` if tracking failed) |
| `mask` | `np.ndarray | None` | Segmentation mask at model resolution |

---

### `blur_person_mask(frame, mask, ksize=(99, 99), sigma=0)`

Applies Gaussian blur to pixels inside a segmentation mask.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `frame` | `np.ndarray` | — | BGR image, shape `(H, W, 3)` |
| `mask` | `np.ndarray` | — | Binary mask at model resolution |
| `ksize` | `Tuple[int, int]` | `(99, 99)` | Gaussian kernel size (odd numbers) |
| `sigma` | `float` | `0` | Gaussian sigma (0 = auto) |

**Returns:** `np.ndarray` — frame with masked region blurred.

---

### `blur_person_box(frame, box, ksize=(99, 99), sigma=0, padding=0.1)`

Fallback rectangular blur within a bounding box.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `frame` | `np.ndarray` | — | BGR image |
| `box` | `Tuple[int, int, int, int]` | — | `(x1, y1, x2, y2)` |
| `ksize` | `Tuple[int, int]` | `(99, 99)` | Gaussian kernel size |
| `sigma` | `float` | `0` | Gaussian sigma |
| `padding` | `float` | `0.1` | Fractional expansion of the box (0.1 = 10%) |

**Returns:** `np.ndarray` — frame with box region blurred.

---

## Dependencies & Compatibility

### Python packages

| Package | Version tested | Purpose |
|---|---|---|
| `torch` | 2.10.0+cu128 | Deep learning runtime (CUDA) |
| `torchvision` | — | Torch vision utilities |
| `ultralytics` | 8.4.21 | YOLOv8 model loading, inference, tracking |
| `opencv-python` | 4.13.0 | Video I/O, image processing, Gaussian blur |
| `numpy` | — | Array operations |
| `matplotlib` | — | Diagnostic frame visualization |
| `moviepy` | 2.2.1 | Audio track merging |
| `imageio-ffmpeg` | — | Bundled ffmpeg binary (moviepy dependency) |
| `lapx` | — | LAP solver for ByteTrack (C extension) |

### CUDA compatibility

| CUDA version | PyTorch index URL |
|---|---|
| 11.8 | `https://download.pytorch.org/whl/cu118` |
| 12.1 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | `https://download.pytorch.org/whl/cu124` |
| **12.8** | `https://download.pytorch.org/whl/cu128` (default in Cell 1) |

To change the CUDA version, modify the `--index-url` in Cell 1.

### moviepy API note

This project uses **moviepy v2.x**. The import path is:

```python
from moviepy import VideoFileClip          # v2.x ✓
# NOT: from moviepy.editor import VideoFileClip  # v1.x ✗
```

The method for attaching audio is `.with_audio()` (v2.x), not `.set_audio()` (v1.x).
