from __future__ import annotations

from typing import Any, Dict

import yaml

from src.perception.base import (
    DepthEstimator,
    FlowEstimator,
    LaneDetector,
    ObjectDetector,
    PoseEstimator,
    Tracker,
)


# ---------------------------------------------------------------------------
# Internal loaders — import heavy deps only when the relevant key is requested
# ---------------------------------------------------------------------------

def _build_depth(cfg: Dict[str, Any]) -> DepthEstimator:
    model = cfg.get("model", "depth_anything_v2")
    if model == "depth_anything_v2":
        from src.perception.depth.depth_anything import DepthAnythingV2Estimator
        return DepthAnythingV2Estimator(
            model_id=cfg["model_id"],
            device=cfg.get("device", "cuda"),
        )
    raise ValueError(f"Unknown depth model: '{model}'")


def _build_detector(cfg: Dict[str, Any]) -> ObjectDetector:
    model = cfg.get("model", "yolo11")
    if model == "yolo11":
        from src.perception.detection.yolo_combined import YoloCombinedDetector
        return YoloCombinedDetector(
            yolo11_weights=cfg["yolo11_weights"],
            yolo_world_weights=cfg["yolo_world_weights"],
            open_vocab_classes=cfg.get("open_vocab_classes", []),
            conf_threshold=cfg.get("conf_threshold", 0.30),
            iou_threshold=cfg.get("iou_threshold", 0.45),
        )
    raise ValueError(f"Unknown detection model: '{model}'")


def _build_lanes(cfg: Dict[str, Any]) -> LaneDetector:
    model = cfg.get("model", "yolopv2")
    if model == "yolopv2":
        from src.perception.lanes.yolopv2 import YOLOPv2LaneDetector
        return YOLOPv2LaneDetector(
            weights_path=cfg.get("weights_path", "weights/yolopv2.pt"),
            min_lane_pixels=cfg.get("min_lane_pixels", 200),
            poly_degree=cfg.get("poly_degree", 2),
            bezier_sample_points=cfg.get("bezier_sample_points", 10),
        )
    if model == "maskrcnn":
        from src.perception.lanes.maskrcnn_lane import MaskRCNNLaneDetector
        return MaskRCNNLaneDetector(
            weights_path=cfg.get("weights_path", "models/maskrcnn_lane.pth"),
            device=cfg.get("device", "cuda"),
            conf_threshold=cfg.get("conf_threshold", 0.50),
            poly_degree=cfg.get("poly_degree", 2),
            bezier_points=cfg.get("bezier_points", 10),
        )
    raise ValueError(f"Unknown lane model: '{model}'")


def _build_tracker(cfg: Dict[str, Any]) -> Tracker:
    model = cfg.get("model", "deepsort")
    if model == "deepsort":
        from src.perception.tracking.deepsort import DeepSORTTracker
        return DeepSORTTracker(
            max_age=cfg.get("max_age", 30),
            n_init=cfg.get("n_init", 3),
            max_cosine_dist=cfg.get("max_cosine_dist", 0.4),
            embedder=cfg.get("embedder", "mobilenet"),
        )
    if model == "bytetrack":
        from src.perception.tracking.bytetrack import ByteTrackTracker
        return ByteTrackTracker(
            frame_rate=cfg.get("frame_rate", 7.2),
            lost_track_buffer=cfg.get("lost_track_buffer", 30),
            track_activation_threshold=cfg.get("track_activation_threshold", 0.25),
            minimum_matching_threshold=cfg.get("minimum_matching_threshold", 0.80),
            ema_alpha=cfg.get("ema_alpha", 0.40),
        )
    raise ValueError(f"Unknown tracker: '{model}'")


def _build_pose(cfg: Dict[str, Any]) -> PoseEstimator:
    model = cfg.get("model", "yolo_pose")
    if model == "yolo_pose":
        from src.perception.pose.yolo_pose import YOLOPoseEstimator
        return YOLOPoseEstimator(
            weights=cfg.get("weights", "yolo11x-pose.pt"),
            device=cfg.get("device", "cuda"),
        )
    if model == "rtmw":
        from src.perception.pose.rtmw import RTMWPoseEstimator
        return RTMWPoseEstimator(
            checkpoint=cfg["checkpoint"],
            device=cfg.get("device", "cuda"),
        )
    raise ValueError(f"Unknown pose model: '{model}'")


def _build_flow(cfg: Dict[str, Any]) -> FlowEstimator:
    model = cfg.get("model", "raft")
    if model == "raft":
        from src.perception.flow.raft import RAFTFlowEstimator
        return RAFTFlowEstimator(
            weights=cfg.get("weights", "raft-things"),
            iters=cfg.get("iters", 20),
            device=cfg.get("device", "cuda"),
        )
    raise ValueError(f"Unknown flow model: '{model}'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BUILDERS = {
    "depth":     _build_depth,
    "detection": _build_detector,
    "lanes":     _build_lanes,
    "tracker":   _build_tracker,
    "pose":      _build_pose,
    "flow":      _build_flow,
}


def build_from_config(config_path: str, component: str):
    """Instantiate a single pipeline component from models.yaml.

    Args:
        config_path: Path to configs/models.yaml
        component:   One of 'depth', 'detection', 'lanes', 'tracker', 'pose', 'flow'

    Returns:
        An instance of the corresponding ABC.
    """
    if component not in _BUILDERS:
        raise KeyError(
            f"Unknown component '{component}'. "
            f"Valid options: {list(_BUILDERS)}"
        )
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)

    section = full_cfg.get(component)
    if section is None:
        raise KeyError(f"Section '{component}' not found in {config_path}")

    return _BUILDERS[component](section)


def build_all(config_path: str) -> Dict[str, Any]:
    """Instantiate every pipeline component defined in models.yaml.

    Returns:
        Dict with keys: depth, detection, lanes, tracker, pose, flow
    """
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)

    result = {}
    for key, builder in _BUILDERS.items():
        section = full_cfg.get(key)
        if section is None:
            raise KeyError(f"Section '{key}' not found in {config_path}")
        result[key] = builder(section)

    return result
