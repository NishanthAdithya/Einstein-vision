#!/usr/bin/env python3
"""run_perception.py — Main perception pipeline.

Processes one scene + camera combination end-to-end:
  video frames → detections → depth → 3D lifting → tracking → pose → lanes
  → serialised MessagePack + NPZ files under outputs/<scene_id>/

Usage:
    python scripts/run_perception.py --scene scene1
    python scripts/run_perception.py --scene scene1 --camera front --phase 2
    python scripts/run_perception.py --scene scene1 --debug   # saves debug PNGs
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

# ── project imports ──────────────────────────────────────────────────────────
from src.io.calibration_loader import load_camera
from src.io.schema import FrameData
from src.io.serializer import get_output_paths, save_depth, save_detections
from src.io.video_reader import VideoReader, find_video
from src.localization.camera import Camera
from src.localization.ground_plane import classify_lane_types
from src.localization.projector import lift_detections, lift_lane_points
from src.localization.yaw_estimator import estimate_yaws
from src.perception.factory import build_from_config
from src.perception.traffic_lights.hsv import (
    classify_traffic_lights,
    classify_traffic_light_arrows,
)
from src.utils.debug_viz import draw_frame_data, draw_flow_arrows
from src.phase3.brake_turn_detector import BrakeTurnDetector
from src.phase3.motion_classifier import classify_motion
from src.phase3.collision_predictor import CollisionPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    pipeline_cfg = _load_yaml(args.pipeline_config)
    models_cfg_path = args.models_config

    camera_name = args.camera or pipeline_cfg.get("camera", "front")
    phase = args.phase or pipeline_cfg.get("phase", 1)
    sample_every_n = pipeline_cfg.get("sample_every_n", 5)
    seq_dir = Path(pipeline_cfg["paths"]["sequences_dir"])
    output_root = Path(args.output_root or pipeline_cfg["paths"]["output_root"])

    max_frames = args.max_frames

    scene_dir = seq_dir / args.scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    log.info("Scene: %s  Camera: %s  Phase: %d", args.scene, camera_name, phase)

    # ── build models ─────────────────────────────────────────────────────────
    log.info("Loading models …")
    depth_model    = build_from_config(models_cfg_path, "depth")
    detector       = build_from_config(models_cfg_path, "detection")
    lane_detector  = build_from_config(models_cfg_path, "lanes")
    tracker        = build_from_config(models_cfg_path, "tracker")

    pose_model: Any = None
    subclassifier: Any = None
    flow_model: Any = None
    brake_turn_detector: Any = None
    collision_predictor: Any = None
    if phase >= 2:
        pose_model = build_from_config(models_cfg_path, "pose")
        subclassifier = _build_subclassifier(models_cfg_path)
    if phase >= 3:
        flow_model = build_from_config(models_cfg_path, "flow")
        brake_turn_detector, collision_predictor = _build_phase3(models_cfg_path)

    # ── camera geometry ───────────────────────────────────────────────────────
    cam_config = load_camera(camera_name, args.cameras_config)
    camera = Camera(cam_config)

    # ── video ─────────────────────────────────────────────────────────────────
    video_path = find_video(scene_dir, camera_name, undistorted=True)
    log.info("Video: %s", video_path)

    tracker.reset()
    prev_frame: Optional[np.ndarray] = None
    n_processed = 0
    t_start = time.perf_counter()

    with VideoReader(video_path, sample_every_n=sample_every_n) as reader:
        log.info("Frames: %d total, %d sampled  (every %d)",
                 reader.total_frames, len(reader), sample_every_n)

        for frame_idx, frame_bgr in reader:
            if max_frames > 0 and frame_idx >= max_frames:
                break

            paths = get_output_paths(output_root, args.scene, frame_idx)

            # Skip already-processed frames unless --force
            if not args.force and paths["detections"].exists():
                n_processed += 1
                prev_frame = frame_bgr
                continue

            timestamp = reader.frame_idx_to_timestamp(frame_idx)

            # ── perception ────────────────────────────────────────────────────

            # 1. Depth
            depth_map = depth_model.predict(frame_bgr)

            # 2. Object detection
            detections = detector.predict(frame_bgr)

            # 3. Tracking (assigns track_id)
            detections = tracker.update(detections, frame_bgr)

            # 4. Traffic light colour classification
            classify_traffic_lights(detections, frame_bgr)

            # 4b. Traffic light arrow detection (Phase 2+)
            if phase >= 2:
                classify_traffic_light_arrows(detections, frame_bgr)

            # 4c. Vehicle sub-classification and speed limit OCR (Phase 2+)
            if subclassifier is not None:
                subclassifier.classify_detections(detections, frame_bgr)

            # 5. 3-D lifting: fills pos_3d + depth
            lift_detections(detections, depth_map, camera)

            # 6. Yaw estimation
            estimate_yaws(detections, cam_config.fx)

            # 7. Lane detection
            lanes = lane_detector.predict(frame_bgr, depth_map=depth_map)

            # 8. Lift lane bezier points to 3-D
            lift_lane_points(lanes, camera, depth_map=depth_map)

            # 9. Lane type classification
            lane_mask = _get_lane_mask(lane_detector)
            if lane_mask is not None:
                classify_lane_types(lanes, lane_mask, frame_bgr)

            # 10. Human pose (Phase 2+)
            poses = []
            if pose_model is not None:
                poses = pose_model.predict(frame_bgr, detections)

            # 11. Optical flow (Phase 3+)
            flow: Optional[np.ndarray] = None
            if flow_model is not None and prev_frame is not None:
                flow = flow_model.predict(prev_frame, frame_bgr)
                save_flow_safe(paths["flow"], flow)

            # 12. Motion classification: parked vs moving, Sampson distance (Phase 3+)
            if phase >= 3 and flow is not None:
                classify_motion(detections, flow, camera)

            # 13. Brake light + turn signal detection (Phase 3+)
            if phase >= 3 and brake_turn_detector is not None:
                brake_turn_detector.update(detections, frame_bgr)

            # 14. Collision prediction — pedestrian / vehicle (Phase 3+, extra credit)
            if phase >= 3 and collision_predictor is not None:
                collision_predictor.update(detections)

            # ── assemble + save ───────────────────────────────────────────────
            frame_data = FrameData(
                frame_idx=frame_idx,
                timestamp=timestamp,
                camera_name=camera_name,
                scene_id=args.scene,
                detections=detections,
                lanes=lanes,
                poses=poses,
            )

            save_detections(paths["detections"], frame_data)
            save_depth(paths["depth"], depth_map)

            # ── optional debug overlay ────────────────────────────────────────
            if args.debug:
                debug_frame = draw_frame_data(frame_bgr, frame_data)
                if flow is not None:
                    debug_frame = draw_flow_arrows(debug_frame, detections, flow)
                render_path = paths["render"].with_suffix(".debug.jpg")
                render_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(render_path), debug_frame)

            prev_frame = frame_bgr
            n_processed += 1

            if n_processed % 50 == 0:
                elapsed = time.perf_counter() - t_start
                fps = n_processed / elapsed
                log.info("  frame %d  |  %d/%d done  |  %.1f frames/s",
                         frame_idx, n_processed, len(reader), fps)

    elapsed = time.perf_counter() - t_start
    log.info("Done: %d frames in %.1fs  (%.2f fps)",
             n_processed, elapsed, n_processed / elapsed)

    # ── cleanup ───────────────────────────────────────────────────────────────
    for model in (depth_model, detector, lane_detector, tracker,
                  pose_model, subclassifier, flow_model,
                  brake_turn_detector, collision_predictor):
        if model is not None:
            model.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_phase3(config_path: str) -> tuple:
    """Instantiate Phase 3 detectors from models.yaml.

    Returns:
        (BrakeTurnDetector, CollisionPredictor) — both ready to use.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    p3 = cfg.get("phase3", {})

    from src.phase3.brake_turn_detector import BrakeTurnDetector
    from src.phase3.collision_predictor import CollisionPredictor

    btd = BrakeTurnDetector(
        history_len  = p3.get("brake_history_len", 8),
        brake_thresh = p3.get("brake_thresh", 0.05),
        turn_thresh  = p3.get("turn_thresh",  0.04),
    )
    cp = CollisionPredictor(
        predict_frames = p3.get("predict_frames",  15),
        warning_dist   = p3.get("warning_dist_m",  5.0),
    )
    return btd, cp


def _build_subclassifier(config_path: str) -> Any:
    """Load the VehicleSubclassifier from models.yaml config.

    Returns None gracefully if the openai-clip package is not installed.
    """
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        sub_cfg = cfg.get("vehicle_subclassifier", {})
        device = sub_cfg.get("device", "cuda")
        from src.perception.detection.vehicle_subclassifier import VehicleSubclassifier
        return VehicleSubclassifier(device=device)
    except ImportError:
        log.warning("openai-clip not installed — vehicle sub-classification disabled. "
                    "Install with: pip install openai-clip")
        return None


def _get_lane_mask(lane_detector: Any):
    """Return the raw binary lane mask if the detector exposes it."""
    return getattr(lane_detector, "_last_mask", None)


def save_flow_safe(path: Path, flow: np.ndarray) -> None:
    from src.io.serializer import save_flow
    save_flow(path, flow)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EinsteinVision perception pipeline")
    p.add_argument("--scene",          required=True,
                   help="Scene ID, e.g. 'scene1'")
    p.add_argument("--camera",         default=None,
                   help="Camera name: front | back | left | right  (default from pipeline.yaml)")
    p.add_argument("--phase",          type=int, default=None,
                   help="Pipeline phase 1/2/3  (default from pipeline.yaml)")
    p.add_argument("--output-root",    default=None,
                   help="Override output root directory")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml",
                   help="Path to pipeline.yaml")
    p.add_argument("--models-config",  default="configs/models.yaml",
                   help="Path to models.yaml")
    p.add_argument("--cameras-config", default="configs/cameras.yaml",
                   help="Path to cameras.yaml")
    p.add_argument("--force",          action="store_true",
                   help="Re-process frames even if output already exists")
    p.add_argument("--debug",          action="store_true",
                   help="Save debug overlay JPEGs alongside each frame")
    p.add_argument("--max-frames",     type=int, default=400,
                   help="Stop after processing this many frames (0 = all, default 400)")
    return p.parse_args()


if __name__ == "__main__":
    main()
