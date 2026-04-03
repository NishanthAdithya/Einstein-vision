# EinsteinVision — Project Notes & Discussion Log
> WPI RBE 549 Project 3 | Spring 2026
> Deadlines: Phase 1 → March 28 | Phase 2 → April 6 | Phase 3 + EC → April 13

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Previous Group Analysis](#2-previous-group-analysis)
3. [Our Proposed Pipeline](#3-our-proposed-pipeline)
4. [Model Selection Decisions](#4-model-selection-decisions)
5. [State Estimation & Tracking](#5-state-estimation--tracking)
6. [Open Design Questions](#6-open-design-questions)
7. [Phase Task Breakdown](#7-phase-task-breakdown)
8. [Discussion Log](#8-discussion-log)

---

## 1. Project Overview

Build a 3D visualization dashboard inspired by Tesla FSD. Input: videos from 4 cameras of a 2023 Tesla Model S (front, back, left_repeater, right_repeater) across 13 scenes. Output: Blender-rendered video with 3D objects, lanes, and cognitive features.

**Data available:**
- `P3Data/Sequences/scene1..13/` — Raw + Undistorted MP4s (4 cameras each)
- `P3Data/Calib/` — `calibration.mat` with intrinsics, extrinsics, distortion per camera
- `P3Data/Assets/` — Blender models: sedan, SUV, truck, pickup, motorcycle, bicycle, pedestrian, traffic signal, stop sign, speed limit sign, cones, dustbin

---

## 2. Previous Group Analysis

### What All Groups Did (Standard Approaches)
| Component | Common Tools Used | Notes |
|-----------|-----------------|-------|
| Rendering | Blender Python API (`bpy`) | Universal — frames → JSON → Blender |
| Depth | MiDaS → ZoeDepth → Marigold | Quality improved over time |
| Lane detection | CLRNet / Mask RCNN / LaneNet | All struggled on city scenes |
| Object detection | YOLOv8/v9 → DETIC | DETIC won for broad class coverage |
| Vehicle pose (yaw) | YOLO3D | Unreliable on monocular; frequent failures |
| Human pose | OpenPifPaf / OSX / yolo_nas_pose | OSX (obj mesh output) was cleanest |
| Traffic light color | HSV thresholding inside bbox | YCbCr also used for brake lights |
| Optical flow | RAFT | For parked vs. moving classification |
| Tracking | **None** | ← Major gap; all did per-frame detection |

### Key Lessons From Previous Groups
- **YOLO3D fails** for monocular data (trained on stereo KITTI) — yaw estimates unreliable when vehicles close/occluded
- **Marigold > ZoeDepth > MiDaS** for depth quality on this dataset (Group 6 confirmed)
- **City scene lanes are hard** — faded markings defeat CLRNet; Mask RCNN + polynomial was more robust
- **DETIC** (Facebook Detectron2) handles all required classes (cones, dustbins, fire hydrants, etc.) via LVIS classes
- **Lane rendering in Blender**: use Geometry Nodes with Bezier curves (Group Yijia/Oliver) — much cleaner than raw point meshes
- **OSX** generates `.obj` mesh directly from monocular image → cleanest human pose → Blender integration
- **Ground arrows**: YOLOP (drivable area) + Canny + contour shape analysis (edge count heuristic)
- **Speed bump**: detect speed bump sign via YOLO (GLARE dataset), then place custom asset near sign

### What No Group Did (Our Differentiation Opportunities)
1. **No temporal tracking** — zero groups used ByteTrack/SORT/Kalman → our objects will be smooth vs. everyone's jitter
2. **No Depth Pro** — newest metric depth model (Apple, ICLR 2025), zero-shot absolute scale
3. **No multi-camera fusion** — all used front camera only
4. **No SEA-RAFT** — used vanilla RAFT; SEA-RAFT is faster + more accurate (ECCV 2024)

---

## 3. File & Folder Structure

```
einstein-vision/
├── PROJECT_NOTES.md
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── cameras.yaml          # per-camera intrinsics and pose (K, dist, height, pitch)
│   ├── models.yaml           # model names, weights paths, thresholds — swap models here
│   └── pipeline.yaml         # sample_every_n, phase flag, scene list, paths
│
├── src/
│   ├── io/
│   │   ├── schema.py         # ALL shared dataclasses: BBox, Detection, Lane, FrameData, etc.
│   │   ├── video_reader.py   # VideoReader class + find_video() helper
│   │   ├── calibration_loader.py  # load_camera() / load_all_cameras() from cameras.yaml
│   │   └── serializer.py     # save/load MessagePack (detections) + NPZ (depth, flow)
│   │
│   ├── perception/
│   │   ├── base.py           # Abstract base classes: BaseDepthEstimator, etc.
│   │   ├── factory.py        # build_pipeline(config) → concrete implementations
│   │   ├── depth/
│   │   │   ├── depth_anything.py   # Depth Anything V2 (primary)
│   │   │   └── depth_pro.py        # Depth Pro (stub — drop in if needed)
│   │   ├── detection/
│   │   │   └── yolo_combined.py    # YOLO11 + YOLO-World together
│   │   ├── lanes/
│   │   │   └── yolopv2.py          # YOLOPv2 lane mask + type classification
│   │   ├── pose/
│   │   │   └── rtmw.py             # RTMW via mmpose
│   │   ├── flow/
│   │   │   ├── raft.py             # RAFT (primary)
│   │   │   └── sea_raft.py         # SEA-RAFT (stub — upgrade if time permits)
│   │   └── tracking/
│   │       └── bytetrack.py        # ByteTrack via supervision + EMA smoothing
│   │
│   ├── localization/
│   │   ├── camera.py         # Camera class: K matrix, unproject, IPM homography
│   │   ├── projector.py      # bbox + depth → 3D; lift_detections()
│   │   ├── yaw_estimator.py  # geometric yaw from bbox aspect ratio + depth
│   │   └── ground_plane.py   # IPM: pixel → 3D ground point for lane projection
│   │
│   ├── phase3/
│   │   ├── motion_classifier.py  # parked vs. moving via optical flow
│   │   └── light_detector.py     # brake lights + turn signals via YCbCr
│   │
│   └── utils/
│       ├── geometry.py       # rotation matrices, coord transforms, IoU, NMS
│       └── debug_viz.py      # OpenCV 2D debug overlays
│
├── scripts/
│   ├── run_perception.py     # main pipeline: video → MessagePack + NPZ
│   ├── run_blender.py        # launches headless Blender for a scene
│   └── run_parallel.py       # dispatches all 13 scenes across cluster
│
├── blender/
│   ├── scene_builder.py      # main entry: reads msgpack, iterates frames, renders
│   ├── asset_manager.py      # load .blend assets, spawn at 3D coords with yaw
│   ├── lane_renderer.py      # Bezier curves via Geometry Nodes
│   ├── light_animator.py     # brake/turn light material emission
│   └── camera_setup.py       # configure Blender camera from calibration
│
├── P3Data/                   # (existing, untouched)
├── previous_reports/         # (existing, untouched)
│
└── outputs/
    └── scene{1..13}/
        ├── detections/       # frame_0000.msgpack, frame_0005.msgpack, ...
        ├── depth/            # frame_0000.npz, frame_0005.npz, ...
        ├── flow/             # frame_0000.npz, ... (phase 3 only)
        └── renders/          # frame_0000.png, ... → assembled to video
```

### Model Swapping
To swap any model, change the `model` key in `configs/models.yaml`.
`factory.py` reads that key and returns the correct subclass of the relevant
abstract base class. No other file changes are needed.

---

## 4. Our Proposed Pipeline

```
Input: MP4 Videos (front camera primary, others for multi-cam fusion later)
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Calibration & Frame Extraction       │
│     - Read calibration.mat (scipy.io)    │
│     - Sample 1 frame per 5-10 frames     │
│     - Use Undist/ videos (pre-corrected) │
└────────────────┬────────────────────────┘
                 │
         ┌───────┴──────────┐
         ▼                  ▼
┌──────────────┐   ┌───────────────────┐
│  2a. YOLO11  │   │  2b. Mask RCNN    │
│  (detection) │   │  (lane segments)  │
│  + DETIC for │   │  → polynomial fit │
│  rare objects│   │  → 10 Bezier pts  │
└──────┬───────┘   └────────┬──────────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────────┐
│  3. Depth Pro (metric depth)          │
│     - Absolute depth per pixel        │
│     - No intrinsics needed            │
│     - Fallback: Marigold              │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  4. 3D Localization                   │
│     - Objects: bbox center → depth    │
│       → unproject via K matrix        │
│     - Lanes: IPM homography for       │
│       ground-plane points             │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  5. ByteTrack (tracking + smoothing)  │
│     - Assign stable track IDs         │
│     - Kalman prediction for gaps      │
│     - EMA smoothing on 3D positions   │
└───────────────┬──────────────────────┘
                │
          ┌─────┴──────┐
          ▼            ▼
┌─────────────┐  ┌────────────────────┐
│  6a. Phase3 │  │  6b. Vehicle yaw   │
│  logic:     │  │  (YOLO3D or        │
│  brake/turn │  │   geometric est.)  │
│  SEA-RAFT   │  └────────────────────┘
│  for moving │
│  vs. parked │
└─────┬───────┘
      │
      ▼
┌──────────────────────────────────────┐
│  7. JSON Export                       │
│     Lane JSON + Object JSON per frame │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  8. Blender Python API                │
│     - Place assets at 3D coords       │
│     - Bezier lanes via Geometry Nodes │
│     - Animate brake/turn lights       │
│     - Render frame sequence → video   │
└──────────────────────────────────────┘
```

---

## 4. Model Selection Decisions (Reliability-First)

> Audit criteria: working GitHub + downloadable weights + pip installable or near + community usage.
> Reliability ratings from independent audit: HIGH / MEDIUM-HIGH / MEDIUM / LOW

### Depth Estimation
| Model | Reliability | Decision | Notes |
|-------|------------|----------|-------|
| **Depth Anything V2** (metric outdoor) | **HIGH** | **✅ PRIMARY** | pip+HF, 7.7k stars, metric variant on Virtual KITTI 2 (80m range), transformers integration |
| Depth Pro (Apple) | HIGH | ✅ Fallback | Clean release, pip install, but Python 3.9 pinned, frozen since Aug 2024 |
| UniDepthV2 | MEDIUM | ❌ Skip for now | xFormers/CUDA env brittle, shape mismatch bugs, too new |
| Marigold | MEDIUM | ❌ Skip | Slow, relative depth only |

### Object Detection — Standard Classes
| Model | Reliability | Decision | Notes |
|-------|------------|----------|-------|
| **YOLO11** (ultralytics) | **HIGH** | **✅ PRIMARY** | `pip install ultralytics`, covers cars/pedestrians/traffic lights/stop signs/fire hydrants from COCO |

### Object Detection — Open Vocabulary / Rare Objects
| Model | Reliability | Decision | Notes |
|-------|------------|----------|-------|
| **YOLO-World** (ultralytics) | **HIGH** | **✅ PRIMARY** | Same ultralytics ecosystem, text-prompted open-vocab detection, `pip install ultralytics`. Replaces DETIC entirely. |
| DETIC (Facebook) | LOW-MEDIUM | ❌ Skip | Detectron2 dependency hell on PyTorch 2.x, unmaintained since 2022 |
| GroundingDINO | MEDIUM | ⚠️ Fallback only | More complex setup than YOLO-World |

### Lane Detection
| Model | Reliability | Decision | Notes |
|-------|------------|----------|-------|
| **YOLOPv2** | **HIGH** | **✅ PRIMARY** | pip install, lane mask + drivable area + detection simultaneously, proven in prior groups |
| CLRerNet | MEDIUM-LOW | ❌ Skip | Docker required, **Blackwell GPU broken** (our hardware!), mmdet3.x migration issues |
| CLRNet | MEDIUM | ⚠️ Fallback | More stable than CLRerNet but still Docker-dependent |
| Mask RCNN (debuggercafe) | LOW | ❌ Skip | Blog artifact, paywalled weights, no real repo |
| **Lane type classification** | — | Heuristic post-process | Analyze YOLOPv2 mask: dashed = periodic gaps (contour count), solid = continuous blob |

### Vehicle Yaw / Orientation
| Model | Reliability | Decision | Notes |
|-------|------------|----------|-------|
| MonoDETRNext | UNKNOWN | ❌ Skip | **Paper only — no code released** |
| MonoDETR | MEDIUM-LOW | ❌ Skip | CUDA nvcc compile failures on modern setups, unmaintained since Aug 2023 |
| YOLO3D | LOW | ❌ Skip | Proven to fail monocular in all prior groups |
| **Geometric estimation** | **HIGH** | **✅ PRIMARY** | Use 2D bbox aspect ratio + depth: wide bbox→side-facing (yaw≈90°), square→front/rear-facing (yaw≈0°). Good enough for Blender placement. |
| **YOLO11-pose (car keypoints)** | HIGH | ✅ Phase 2 upgrade | YOLO11 has a pose model; can be fine-tuned on car keypoints to get better yaw |

### Human Pose
| Model | Reliability | Decision | Notes |
|-------|------------|----------|-------|
| **RTMW** (mmpose) | **MEDIUM-HIGH** | **✅ PRIMARY** | Whole-body 70+ mAP, real-time, weights on OpenMMLab CDN. Risk: MMCV version pinning — solve once, reliable after. |
| OSX | MEDIUM | ⚠️ Fallback | 2023, also in mmpose, outputs .obj mesh directly — useful if RTMW→Blender integration is hard |

### Optical Flow (Phase 3)
| Model | Reliability | Decision | Notes |
|-------|------------|----------|-------|
| **RAFT** | **HIGH** | **✅ PRIMARY** | The gold standard, multiple reliable pip-installable implementations, no custom CUDA ops |
| SEA-RAFT | HIGH | ✅ Upgrade if time permits | Weights on HuggingFace, no custom CUDA ops, 2.3x faster than RAFT. Try after RAFT is working. |

### Multi-Object Tracking
| Approach | Reliability | Decision | Notes |
|----------|------------|----------|-------|
| **ByteTrack via `supervision`** | **HIGH** | **✅ PRIMARY** | `pip install supervision`, `sv.ByteTrack()`, 36k stars, actively maintained Mar 2026 |
| SG-LKF | UNKNOWN | ❌ Skip | Aug 2025 paper, almost certainly no public code yet |
| DeepSORT | MEDIUM | ⚠️ Only if re-ID needed | — |

### Data Format
| What | Format | Reason |
|------|--------|--------|
| Per-frame detections (bboxes, 3D coords, track IDs) | **MessagePack** | 2x smaller than JSON, 2x faster parse, `pip install msgpack` |
| Depth maps (HxW float32) | **NumPy NPZ** | 10x faster than JSON, native Python |
| Optical flow fields (HxWx2 float32) | **NumPy NPZ** | Same |
| Blender scene description | **MessagePack** | Blender Python reads via `import msgpack` |

---

## 5. State Estimation & Tracking

### Do We Need Kalman Filters?

**Short answer: Yes, but via ByteTrack, not hand-coded Kalman.**

#### Why Tracking Matters Here
Even though this is offline batch processing (not real-time), tracking is essential for:
1. **Smooth Blender animations** — without tracking, cars jitter frame-to-frame due to noisy depth estimates. Every previous group suffered this problem and none solved it.
2. **Phase 3 turn signal detection** — must observe the same vehicle over multiple frames to detect a blinking pattern (~1-2 Hz blink rate)
3. **Phase 3 brake light detection** — need to know it's the same car before/after braking
4. **Parked vs. moving** — need a track history to classify motion over time (optical flow alone is noisy)
5. **Gap filling** — if YOLO misses a detection for 1-2 frames, Kalman prediction keeps the car visible

#### Why ByteTrack (Not Raw Kalman)
A raw Kalman filter only smooths one object in isolation. ByteTrack solves the full **data association** problem:
- Matches detections across frames using IoU in image space
- Uses Kalman filter internally for state prediction
- Handles both high-confidence and low-confidence detections (the "byte" in ByteTrack)
- Assigns **stable track IDs** — crucial for Phase 3

#### What State Does the Kalman Filter Track?
For our use case:
```
State vector: [X, Y, Z, vX, vY, vZ]  (3D position + velocity)
Measurement:  [X, Y, Z]               (from depth unprojection)
Process model: constant velocity
```
- **Prediction step**: extrapolate position from velocity
- **Update step**: fuse with new depth measurement
- **On miss**: coast for up to N=5 frames before dropping the track

#### Smoothing Strategy for Blender
After ByteTrack assigns IDs, apply **Exponential Moving Average (EMA)** on 3D positions per track:
```python
smoothed_pos = alpha * new_pos + (1 - alpha) * prev_smoothed_pos  # alpha ~0.3-0.5
```
This produces the "floating" smooth movement seen in Tesla FSD, where objects glide rather than teleport.

#### When Kalman is NOT Worth It
- **Static objects** (signs, traffic lights, cones): detect once, place permanently. No tracking needed.
- **Lane lines**: smooth via polynomial coefficient averaging, not Kalman.
- **Phase 1/2 only**: if just getting basic detection working, skip tracking initially. Add ByteTrack in Phase 3.

---

## 6. Open Design Questions

| Question | Current Decision |
|----------|-----------------|
| Camera scope | Front camera only for Phase 1; multi-cam in Phase 3 |
| Depth model | **Depth Anything V2** (metric outdoor, HuggingFace) |
| Object detection | **YOLO11** + **YOLO-World** (same ecosystem, no DETIC) |
| Lane detection | **YOLOPv2** (lane mask); heuristic for lane type classification |
| Vehicle yaw | **Geometric estimation** (bbox aspect ratio + depth). Simple and always works. |
| Human pose | **RTMW** via mmpose |
| Tracking | **ByteTrack** via `supervision` library |
| Optical flow | **RAFT** primary; SEA-RAFT upgrade if time permits |
| Data format | **MessagePack** (detections) + **NPZ** (depth/flow arrays). No JSON. |
| Blender lanes | Geometry Nodes + Bezier curves |
| Parallelization | Single RTX 5090 only — no cluster access currently. run_parallel.py deferred. |

---

## 7. Phase Task Breakdown

### Phase 1 — Basic Features (Due: March 28)
| Task | Approach | Status |
|------|---------|--------|
| Shared data schemas | `schema.py` — BBox, Detection, Lane, PoseResult, CameraConfig, FrameData | ✅ |
| Camera calibration loading | `calibration_loader.py` — load_camera() / load_all_cameras() from cameras.yaml | ✅ |
| Frame extraction | `video_reader.py` — VideoReader + find_video(); 1 per 5 frames, 429 frames/scene | ✅ |
| Serialization layer | `serializer.py` — MessagePack (detections) + NPZ (depth/flow); get_output_paths() | ✅ |
| Perception ABCs | `perception/base.py` — 6 abstract base classes, one per pipeline stage | ✅ |
| Model factory | `perception/factory.py` — build_from_config() / build_all() for all 6 components | ✅ |
| Depth estimation | `depth/depth_anything.py` — Depth Anything V2 Metric Outdoor (HF transformers); predict() + predict_batch() | ✅ |
| Object detection | `detection/yolo_combined.py` — YOLO11 (COCO) + YOLO-World (open vocab); cross-model batched NMS | ✅ |
| Lane detection | `lanes/yolopv2.py` — YOLOPv2; letterbox → mask decode → connectedComponents → poly fit → bezier sample | ✅ |
| Tracking | `tracking/bytetrack.py` — ByteTrack via supervision + EMA bbox smoothing per track ID | ✅ |
| Human pose | `pose/rtmw.py` — RTMW-L via mmpose; inference_topdown; (133, 3) keypoints | ✅ |
| Optical flow | `flow/raft.py` — RAFT-large via torchvision; pad → forward → crop → (H,W,2) float32 | ✅ |
| 3D localization — camera model | `localization/camera.py` — K matrix, unproject_pixel, IPM homography | ✅ |
| 3D localization — projector | `localization/projector.py` — lift_detections(), lift_lane_points() using depth + K | ✅ |
| Geometric yaw estimation | `localization/yaw_estimator.py` — bbox aspect ratio + depth heuristic | ✅ |
| Ground plane / IPM | `localization/ground_plane.py` — IPM + lane type classification (dashed/solid/yellow) | ✅ |
| Utility geometry | `utils/geometry.py` — rotation matrices, coord transforms, IoU | ✅ |
| Debug visualization | `utils/debug_viz.py` — OpenCV 2D overlay for development | ✅ |
| Perception runner script | `scripts/run_perception.py` — video → MessagePack + NPZ for all frames | ✅ |
| Blender launcher script | `scripts/run_blender.py` — finds Blender binary, launches headlessly per scene | ✅ |
| Blender scene builder | `blender/scene_builder.py` — main entry: reads msgpack, iterates frames, renders | ✅ |
| Blender asset manager | `blender/asset_manager.py` — spawn .blend assets at 3D coords with yaw | ✅ |
| Blender lane renderer | `blender/lane_renderer.py` — Geometry Nodes + Bezier curves | ✅ |
| Blender light animator | `blender/light_animator.py` — brake/turn signal emission animation | ✅ |
| Blender camera setup | `blender/camera_setup.py` — calibrated camera from cameras.yaml | ✅ |
| Traffic light color | `perception/traffic_lights/hsv.py` — HSV crop classify per bbox; red/yellow/green | ✅ |
| Install model weights | torch+cu128, ultralytics, transformers, supervision, YOLOPv2.pt in weights/ | ✅ |
| End-to-end smoke test | scene3: 432 frames @ 3.74 fps perception, 432 Blender renders, cars + lanes visible | ✅ |
| Process all 13 scenes | Run perception + Blender on scenes 1–13 | ⬜ |
| Assemble output video | ffmpeg PNGs → MP4 per scene | ⬜ |

### Phase 2 — Advanced Features (Due: April 6)
| Task | Approach | Status |
|------|---------|--------|
| Vehicle subclassification | YOLO11 + YOLO-World open vocab (sedan/SUV/pickup/truck/motorcycle) | ⬜ |
| Vehicle yaw estimation | Geometric: bbox aspect ratio + depth → yaw angle → Blender rotation | ⬜ |
| Speed limit sign OCR | YOLO-World detects sign → EasyOCR on cropped region | ⬜ |
| Traffic light arrow | HSV + shape analysis on detected traffic light bbox | ⬜ |
| Ground arrows | YOLOPv2 drivable area mask + Canny + contour edge count heuristic | ⬜ |
| Misc objects | YOLO-World open vocab (cones, dustbins, fire hydrants, barrels) → place assets | ⬜ |
| Human pose → Blender | RTMW keypoints → skeleton animation in Blender | ⬜ |
| Lane type classification | Post-process YOLOPv2 mask: dashed = periodic gaps, solid = continuous blob | ⬜ |
| Blender Bezier lanes | `blender/lane_renderer.py` — Geometry Nodes with Bezier control points | ✅ |
| Asset manager | `blender/asset_manager.py` — spawn .blend assets at 3D coords with yaw | ✅ |
| Cluster parallelisation | `scripts/run_parallel.py` — deferred (no cluster access currently) | ⏸ |

### Phase 3 — Bells & Whistles (Due: April 13)
| Task | Approach | Status |
|------|---------|--------|
| Brake light detection | `phase3/light_detector.py` — YCbCr on rear 1/3 of car bbox → Cr channel threshold | ⬜ |
| Turn signal detection | HSV on left/right quadrants of rear bbox → blink pattern over N frames | ⬜ |
| Moving vs. parked | `phase3/motion_classifier.py` — RAFT flow magnitude inside bbox vs. background | ⬜ |
| Motion arrows | Velocity vector from track history → 3D arrow in Blender | ⬜ |
| Multi-camera fusion | Transform back/side detections to front camera frame using extrinsics | ⬜ |
| Blender light animator | `blender/light_animator.py` — brake/turn signal material emission animation | ✅ |

### Extra Credit
| Task | Approach | Status |
|------|---------|--------|
| Speed bump detection (+10%) | YOLOv8 (GLARE/Roboflow) for speed bump sign → place asset nearby | ⬜ |
| Collision prediction (+15%) | Extrapolate track trajectories → check intersection → red highlight | ⬜ |

---

## 8. Discussion Log

### Session 1 — March 2026
**Topics discussed:**
- Full project breakdown and pipeline design
- Analysis of 4 previous groups' approaches (all read via PDF)
- Key differentiators identified: ByteTrack tracking, Depth Pro, SEA-RAFT, multi-camera
- Confirmed all groups used Blender Python API with JSON intermediary format
- Confirmed no previous group used temporal tracking — this is our main advantage

**Session 2 — Model reliability audit:**
- Conducted independent audit of all proposed models for code/weight availability
- Major changes from Session 1:
  - Depth Anything V2 replaces UniDepthV2 (UniDepthV2 has xFormers/CUDA env issues, shape bugs)
  - YOLO-World replaces DETIC (same open-vocab capability, pip install, no Detectron2)
  - YOLOPv2 replaces CLRerNet (CLRerNet broken on Blackwell GPUs — our hardware)
  - MonoDETRNext dropped (paper only, no code released)
  - MonoDETR dropped (CUDA nvcc compile fails on modern setups, unmaintained)
  - SG-LKF dropped (Aug 2025 paper, no public code)
  - Geometric yaw estimation adopted — simple, always works, good enough
  - RAFT replaces SEA-RAFT as primary (more proven); SEA-RAFT as upgrade
  - User confirmed: 5090 + Blackwell cluster available
- Principle adopted: reliability > novelty. Use well-tested tools that have code + weights + community.

**Key insight from reports:**
- Group 6 (Mihir/Ashwin) had the best final output (realistic Blender renders with full pipeline)
- Their biggest weakness: no tracking → jittery animations, no stable IDs for brake/turn light detection
- Marigold was their best depth model but slow; Depth Pro should be strictly better

**Decisions made:**
- Primary depth: Depth Pro (Apple ICLR 2025)
- Tracking: ByteTrack (with built-in Kalman, not hand-coded)
- Lane detection: Mask RCNN (proven on this dataset)
- Human pose: OSX (`.obj` output native to Blender)
- Optical flow: SEA-RAFT over vanilla RAFT
- All cameras: start front-only, add multi-cam in Phase 2/3
- Processing: offline batch (same as everyone)
- Blender lanes: Geometry Nodes + Bezier curves

**Open questions to answer next:**
- Do we have a GPU available? Depth Pro needs CUDA
- Do we need to train Mask RCNN on lanes or can we use pre-trained weights?
- What is the Blender version available on the machine?
- Should we build the Blender script as a headless batch renderer?

---

### Session 3 — March 18, 2026
**Topics: Implementation sprint — all I/O and perception modules**

Context compaction occurred mid-session; conversation resumed from summary.

**Implemented (in order):**

1. **`src/io/schema.py`** — Single source of truth for all shared dataclasses: BBox, Detection, Lane, PoseResult, CameraConfig, FrameData. Detection carries phase-1 fields (bbox, confidence, class_name, depth, yaw) and phase-3 fields (is_moving, brake_light_on, turn_signal, velocity_3d) in one place so the schema never needs to be touched again.

2. **`src/io/video_reader.py`** — VideoReader with context manager, sampled iteration, seek-by-index, and properties (fps, total_frames, etc.). find_video() handles timestamp-prefixed filenames via glob. Confirmed: 1280×960, 36fps, 2143 total frames, 429 sampled per scene.

3. **`src/io/calibration_loader.py`** — load_camera() / load_all_cameras() from cameras.yaml. Front camera intrinsics confirmed from prior groups (fx=1594.7, fy=1607.7, cx=655.3, cy=414.4). Back/left/right are placeholders (calibration.mat is MATLAB MCOS format unreadable by scipy.io).

4. **`src/io/serializer.py`** — MessagePack for FrameData (detections/lanes/poses), NPZ for depth maps and flow fields. get_output_paths() returns canonical paths. Round-trip tested: 4511 bytes/frame msgpack, ~4.3 MB/frame depth, ~8.6 MB/frame flow.

5. **`src/perception/base.py`** — 6 ABCs: DepthEstimator, ObjectDetector, LaneDetector, PoseEstimator, Tracker, FlowEstimator. Each has a no-op close() for GPU cleanup.

6. **`src/perception/factory.py`** — build_from_config(path, component) and build_all(path). Heavy model imports are deferred inside each _build_* function so importing factory itself costs nothing.

7. **`src/perception/depth/depth_anything.py`** — Depth Anything V2 Metric Outdoor via HuggingFace transformers. predict() for single frames; predict_batch() for GPU-efficient batch processing. Bicubic interpolation back to original H×W.

8. **`src/perception/detection/yolo_combined.py`** — YOLO11 + YOLO-World fused detector. YOLO11 filters to _COCO_KEEP (9 driving-relevant classes). YOLO-World handles open-vocab classes from config. Cross-model batched NMS via torchvision.ops.batched_nms (class-aware, so adjacent cars don't suppress each other).

9. **`src/perception/lanes/yolopv2.py`** — YOLOPv2 lane detector. Letterbox → model forward → argmax over 2-class lane head → upsample → crop letterbox padding → resize to original resolution → connectedComponentsWithStats → polynomial fit (x = ay² + by + c) → bezier sample → optional depth lift. Weights loaded from local .pt file with descriptive FileNotFoundError if missing. lane_type set to "unknown" (colour classification is Phase 3).

10. **`src/perception/tracking/bytetrack.py`** — ByteTrack via supervision. _apply_ema() maintains per-track smoothed bbox dict; alpha=0.40 means 40% new observation, 60% history. _match_detection() finds original Detection using L∞ distance ≤1px so all original fields are preserved. reset() clears both tracker state and EMA cache between scenes.

11. **`src/perception/pose/rtmw.py`** — RTMW-L via mmpose.apis.init_model + inference_topdown. Filters to class_name == "person". _extract_keypoints() handles (1,K,2) and (K,2) output shapes, zero-pads to 133 keypoints. Output: (133, 3) array [x, y, confidence]. keypoints_3d left None for localization.

12. **`src/perception/flow/raft.py`** — RAFT-large via torchvision.models.optical_flow. _pad_to_divisor() ensures H,W divisible by 8 (no-op for our 960×1280 frames). Weight preset "raft-things" maps to Raft_Large_Weights.C_T_SKHT_V2. Returns (H,W,2) float32 in pixels/frame.

**Environment notes:**
- Package manager: uv v0.10.2 (no pip binary in venv — use `uv pip install`)
- Venv: `/home/csr/git/einstein-vision/.venv/`
- Installed this session: pyyaml, opencv-python-headless
- torch/torchvision/ultralytics/supervision/mmpose not yet installed — needed before running inference

**Design decisions confirmed this session:**
- ABC + factory pattern working as intended — factory imports are free until a concrete model is requested
- YOLOPv2 weights must be downloaded manually (no pip package); clear FileNotFoundError message included
- RTMW config path defaults to mmpose bundled cocktail14 config
- RAFT weights auto-download from torchvision on first run (~300 MB)

**Continued — localization + utils + scripts + blender:**

13. **`src/localization/camera.py`** — Camera class wrapping CameraConfig. Builds 3×3 K matrix. `unproject_pixel(u, v, depth)` → (X, Y, Z) in camera frame. `pixel_to_camera_ray(u, v)` → unit direction. `compute_ipm_homography(height_m, pitch_rad)` builds a 3×3 IPM homography for ground-plane projection. `apply_ipm(points)` transforms (N,2) pixel coords to (N,2) ground-plane metre coords.

14. **`src/localization/projector.py`** — `lift_detections(detections, depth_map, camera)` samples depth at bbox center using a 5×5 median patch (robust to depth noise at edges), unprojects to camera frame (X=right, Y=down, Z=forward), then converts to ego frame (X=right, Y=forward, Z=up). `lift_lane_points(lanes, camera)` applies IPM then adds Z=0 for ground plane. Both functions skip out-of-bounds pixels safely.

15. **`src/localization/yaw_estimator.py`** — `estimate_yaw(detection)` implements the geometric heuristic: aspect_ratio = width/height; if ar > _SIDE_THRESHOLD (1.6) → yaw = π/2 (side-facing); if ar < _FRONT_THRESHOLD (0.9) → yaw = 0.0 (front/rear); else linear interpolation between 0 and π/2. `estimate_yaw_batch(detections)` applies per-detection, modifying yaw in-place, returning the list. _SIDE_THRESHOLD and _FRONT_THRESHOLD are module-level constants.

16. **`src/localization/ground_plane.py`** — `classify_lane_type(mask_segment, frame_bgr, pixels_2d)` performs two checks: (a) colour: sample HSV at up to 200 lane pixels, yellow if median H ∈ [15,40] & S>80; (b) dashed: project mask onto Y axis, threshold, find gaps via `np.diff`, count gaps ≥ `_MIN_GAP_PX`=20 — if ≥2 gaps → dashed, else solid. Returns one of `solid-white`, `dashed-white`, `solid-yellow`, `dashed-yellow`. `classify_lane_types(lanes, lane_mask, frame_bgr)` applies this to every lane, updating lane.lane_type in-place.

17. **`src/utils/geometry.py`** — `rotation_x/y/z(angle)` → 3×3 rotation matrices. `ego_to_blender(xyz)` performs the axis swap (ego Y-forward,Z-up → Blender Y-forward,Z-up = identity, but handles the -Y front-face convention). `iou_2d(b1, b2)` → scalar IoU for two BBox objects. `pairwise_iou(boxes1, boxes2)` → (M,N) float32 matrix via vectorized numpy. `nms(detections, iou_threshold)` → filtered list using greedy NMS on confidence.

18. **`src/utils/debug_viz.py`** — `draw_detections(frame, detections)` draws coloured boxes per class with label (class + confidence + track_id + depth). `draw_lanes(frame, lanes)` overlays lane pixel coords with colour by lane_type. `draw_depth(depth_map)` → BGR heatmap (COLORMAP_MAGMA, 8-bit normalized). `draw_pose(frame, pose_results)` draws COCO-17 skeleton edges (indices hardcoded) for the first 17 of 133 keypoints. All functions return a new frame copy.

19. **`scripts/run_perception.py`** — Main CLI runner. Args: `--scene`, `--camera`, `--config`, `--output-root`, `--phase`, `--force`. Builds the full pipeline via `factory.build_all()`. Iterates VideoReader, skips frames with existing outputs (unless `--force`). Per frame: run detector → tracker → depth → lanes → pose → projector → yaw → serializer. Phase 3: also run flow + motion_classifier. Logs per-frame timing at DEBUG level, per-scene summary at INFO level.

20. **`scripts/run_blender.py`** — Launcher wrapper. Auto-detects Blender binary from 6 common paths (PATH, /usr/bin, /snap/bin, /opt/blender, ~/blender). Reads scene list from pipeline.yaml for `--all` mode. Builds subprocess command with `blender --background --python blender/scene_builder.py -- <args>`. Logs last 2000 chars of stderr on failure. `--continue-on-error` flag allows batch runs to survive partial failures.

21. **`blender/camera_setup.py`** — `configure_camera(camera_cfg, render_resolution)`. Sets scene resolution, render.fps=8, lens from fx/fy via sensor_width formula, shift_x/shift_y from principal point offset relative to image centre. Sets clip_start=0.1, clip_end=200.0.

22. **`blender/asset_manager.py`** — `load_assets(assets_root)` pre-imports all .blend files into the scene library (one-time). `_CLASS_TO_BLEND` maps Detection.class_name → .blend filename. `spawn_object(detection, scale_m)` links the mesh, applies 3D ego-frame position, applies yaw rotation via `rotation_z()`, and returns the Blender object. `clear_frame_objects(objects)` unlinks previously spawned objects before each frame.

23. **`blender/lane_renderer.py`** — `create_lane_curve(lane, name)` builds a NURBS path object from `lane.bezier_points_3d`, assigns a flat emission material coloured by lane_type (white/yellow/orange for double). `render_lanes(lanes)` creates all curves, returns object list. `clear_lanes(objects)` removes them between frames. Uses `bpy.data.curves.new(type='CURVE')` with `splines.new('NURBS')`.

24. **`blender/light_animator.py`** — `set_brake_lights(obj, active)` finds any material whose name contains "brake" or "tail", sets `emission_strength` to 5.0 (active) or 0.1 (inactive). `set_turn_signal(obj, side, active)` does the same for materials named "left_turn" / "right_turn". `animate_lights(obj, detection, frame_idx)` calls both based on `detection.brake_light_on` and `detection.turn_signal`.

25. **`blender/scene_builder.py`** — Main headless entry point. `parse_args()` reads args after `--` separator. `load_scene_data(output_root, scene_id)` globs all .msgpack files for the scene. `render_frame(frame_data, depth, ...)` orchestrates: clear previous objects, spawn assets (asset_manager), render lanes (lane_renderer), animate lights (light_animator), configure camera (camera_setup), set output path, call `bpy.ops.render.render()`. Main loop: iterate frames, load msgpack + depth npz, render, save PNG.

**Infrastructure decisions from this sprint:**
- Blender invoked with `blender --background --python scene_builder.py -- <args>` (the `--` separator is critical)
- scene_builder.py uses `sys.argv[sys.argv.index("--") + 1:]` to parse its args past the Blender args
- All bpy imports are at top of blender/ files only — never in src/ (avoids import errors outside Blender context)
- Asset loading is done once at scene start, not per frame (significant speedup)
- Lane NURBS objects are recreated each frame (cheap operation; reusing caused GC issues in bpy)

**Hardware update:**
- Blackwell cluster not currently accessible — only RTX 5090 on local machine
- `scripts/run_parallel.py` deferred until cluster access is restored
- All 13 scenes will be processed sequentially on the 5090 for now

**Next immediate steps:**
1. Install model weights: torch + CUDA → ultralytics → transformers → supervision → mmpose → YOLOPv2 weights → RAFT weights
2. Smoke test: run one scene end-to-end, verify msgpack outputs, render one frame in Blender
3. Traffic light HSV classifier
4. Full pipeline validation across all 13 scenes

---

### Session 4 — March 18, 2026

**Topics: Traffic light classifier, Blender smoke test, bug fixes**

**Implemented:**

26. **`src/perception/traffic_lights/hsv.py`** — `classify_traffic_lights(detections, frame_bgr)`. Crops each `"traffic light"` bbox, converts to HSV, thresholds red (two hue ranges: 0–10 and 160–179 to handle wrap-around), yellow (18–38), green (40–90). Returns dominant colour if ≥4% of crop pixels match, else None. Integrated into `run_perception.py` as step 4 (after tracking, before 3D lifting). `schema.py` and `serializer.py` already had `traffic_light_state` field from prior session.

**Bugs fixed:**

- **`blender/asset_manager.py` — template objects deleted each frame**: `clear_scene_objects()` removed all non-camera/light objects including cached template objects (hide_render=True). On frame 2+, `_loaded_cache` returned names of deleted objects → `bpy.data.objects.get()` → None → no assets spawned. Fix: skip objects with `hide_render=True` in `clear_scene_objects`.

- **`blender/asset_manager.py` — instances inherited hide_render=True**: `obj.copy()` copies all object properties including `hide_render=True` from the template. All spawned instances were invisible to the renderer. Fix: explicitly set `instance.hide_render = False; instance.hide_viewport = False` after copy.

- **`blender/scene_builder.py` — `hasattr` check always True**: `hasattr(det, "traffic_light_state")` always returns True since it is a dataclass field (even when None). Fixed to `det.traffic_light_state is not None`.

- **`blender/asset_manager.py` — _SCALE_MAP all 1.0**: All assets spawned at native mesh scale. Car local bbox is ~231m long; at scale=1.0 the car was 231m long in world space. Fixed by measuring each asset's local bounding box and computing target_size / max_local_dim. New values: car=0.02, truck=0.001, motorcycle=0.012, bicycle=0.14, person=0.056, traffic_light=1.14, stop_sign=0.37.

- **`blender/camera_setup.py` / `run_blender.py` — BLENDER_EEVEE_NEXT not found**: Blender 5.x renamed the engine to `BLENDER_EEVEE`. Fixed by accepting both names; EEVEE branch now handles `"BLENDER_EEVEE"` and `"BLENDER_EEVEE_NEXT"`.

- **Blender Python missing yaml/msgpack**: snap Blender uses a read-only filesystem; pip installs go nowhere. Fix: `pip install --target ~/.config/blender/5.1/scripts/addons/modules pyyaml msgpack` — this directory is on Blender's sys.path.

- **`blender/asset_manager.py` — wrong rotation offsets**: Car local longest axis is Y (not X as assumed), so base_rz=π/2 was rotating an already-Y-forward car to face sideways. Fix: car base_rz=0. Assets with baked X+90° (motorcycle, bicycle, person, traffic light, stop sign) need base_rx=π/2 to stand upright; added `_ROTATION_OFFSET_MAP` for per-asset corrections.

- **Vehicle yaw set to 0 in Blender**: geometric yaw estimator returns 0–π/2 magnitude only, not heading direction. Without 3D bbox or ego-localization, heading is unknowable. All vehicles now forced to yaw=0.0 in `scene_builder.py` (all face ego-forward +Y). Proper per-vehicle heading deferred to Phase 2.

**Environment notes:**
- Blender 5.1.0 installed at /snap/bin/blender (Python 3.13, read-only snap filesystem)
- pyyaml and msgpack installed to ~/.config/blender/5.1/scripts/addons/modules/

**Smoke test results (scene3):**
- Perception: 432 frames at 3.74 fps, mask coverage 0.61%, 3 lane components per frame typical
- Blender: 432 PNGs rendered with EEVEE at 16 samples; cars visible at correct scale and position, facing forward; white/yellow lane stripes visible in lower frame

**Phase 1 status:**
- All pipeline code complete and validated on scene3
- Remaining: run perception + Blender on all 13 scenes, assemble video with ffmpeg

**Next steps (Phase 1 completion):**
1. `python scripts/run_perception.py --scene scene{1..13}` for remaining 12 scenes
2. `python scripts/run_blender.py --all --engine BLENDER_EEVEE --samples 16`
3. ffmpeg to assemble PNGs → MP4 per scene
4. Begin Phase 2 features (due April 6): vehicle subclassification, speed limit OCR, traffic light arrows, ground arrows, human pose → Blender
