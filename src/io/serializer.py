from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import msgpack
import numpy as np

from src.io.schema import (
    BBox,
    CameraConfig,
    Detection,
    FrameData,
    Lane,
    PoseResult,
)


# ---------------------------------------------------------------------------
# Numpy array helpers
# ---------------------------------------------------------------------------

def _array_to_dict(arr: np.ndarray) -> dict:
    return {
        "data": arr.tobytes(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _dict_to_array(d: dict) -> np.ndarray:
    arr = np.frombuffer(d["data"], dtype=np.dtype(d["dtype"]))
    return arr.reshape(d["shape"])


# ---------------------------------------------------------------------------
# Dataclass <-> dict conversions
# ---------------------------------------------------------------------------

def _bbox_to_dict(b: BBox) -> dict:
    return {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}


def _dict_to_bbox(d: dict) -> BBox:
    return BBox(d["x1"], d["y1"], d["x2"], d["y2"])


def _detection_to_dict(det: Detection) -> dict:
    return {
        "class_name": det.class_name,
        "confidence": det.confidence,
        "bbox": _bbox_to_dict(det.bbox),
        "track_id": det.track_id,
        "pos_3d": list(det.pos_3d) if det.pos_3d is not None else None,
        "depth": det.depth,
        "yaw": det.yaw,
        "is_moving": det.is_moving,
        "brake_light_on": det.brake_light_on,
        "turn_signal": det.turn_signal,
        "traffic_light_state": det.traffic_light_state,
        "velocity_3d": list(det.velocity_3d),
    }


def _dict_to_detection(d: dict) -> Detection:
    pos_3d = tuple(d["pos_3d"]) if d["pos_3d"] is not None else None
    return Detection(
        class_name=d["class_name"],
        confidence=d["confidence"],
        bbox=_dict_to_bbox(d["bbox"]),
        track_id=d["track_id"],
        pos_3d=pos_3d,
        depth=d["depth"],
        yaw=d["yaw"],
        is_moving=d["is_moving"],
        brake_light_on=d["brake_light_on"],
        turn_signal=d["turn_signal"],
        traffic_light_state=d.get("traffic_light_state"),
        velocity_3d=tuple(d["velocity_3d"]),
    )


def _lane_to_dict(lane: Lane) -> dict:
    return {
        "lane_type": lane.lane_type,
        "bezier_points_3d": _array_to_dict(lane.bezier_points_3d),
        "poly_coeffs": _array_to_dict(lane.poly_coeffs),
        "pixels_2d": _array_to_dict(lane.pixels_2d),
    }


def _dict_to_lane(d: dict) -> Lane:
    return Lane(
        lane_type=d["lane_type"],
        bezier_points_3d=_dict_to_array(d["bezier_points_3d"]),
        poly_coeffs=_dict_to_array(d["poly_coeffs"]),
        pixels_2d=_dict_to_array(d["pixels_2d"]),
    )


def _pose_to_dict(pose: PoseResult) -> dict:
    return {
        "bbox": _bbox_to_dict(pose.bbox),
        "keypoints": _array_to_dict(pose.keypoints),
        "keypoints_3d": (
            _array_to_dict(pose.keypoints_3d)
            if pose.keypoints_3d is not None
            else None
        ),
    }


def _dict_to_pose(d: dict) -> PoseResult:
    return PoseResult(
        bbox=_dict_to_bbox(d["bbox"]),
        keypoints=_dict_to_array(d["keypoints"]),
        keypoints_3d=(
            _dict_to_array(d["keypoints_3d"])
            if d["keypoints_3d"] is not None
            else None
        ),
    )


def _frame_to_dict(frame: FrameData) -> dict:
    return {
        "frame_idx": frame.frame_idx,
        "timestamp": frame.timestamp,
        "camera_name": frame.camera_name,
        "scene_id": frame.scene_id,
        "detections": [_detection_to_dict(d) for d in frame.detections],
        "lanes": [_lane_to_dict(l) for l in frame.lanes],
        "poses": [_pose_to_dict(p) for p in frame.poses],
    }


def _dict_to_frame(d: dict) -> FrameData:
    return FrameData(
        frame_idx=d["frame_idx"],
        timestamp=d["timestamp"],
        camera_name=d["camera_name"],
        scene_id=d["scene_id"],
        detections=[_dict_to_detection(det) for det in d["detections"]],
        lanes=[_dict_to_lane(l) for l in d["lanes"]],
        poses=[_dict_to_pose(p) for p in d["poses"]],
    )


# ---------------------------------------------------------------------------
# Public API — detections (MessagePack)
# ---------------------------------------------------------------------------

def save_detections(path: str | Path, frame_data: FrameData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        msgpack.pack(_frame_to_dict(frame_data), f, use_bin_type=True)


def load_detections(path: str | Path) -> FrameData:
    with open(Path(path), "rb") as f:
        data = msgpack.unpack(f, raw=False)
    return _dict_to_frame(data)


# ---------------------------------------------------------------------------
# Public API — depth maps and optical flow (NumPy NPZ)
# ---------------------------------------------------------------------------

def save_depth(path: str | Path, depth: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=depth)


def load_depth(path: str | Path) -> np.ndarray:
    return np.load(str(Path(path)))["depth"]


def save_flow(path: str | Path, flow: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, flow=flow)


def load_flow(path: str | Path) -> np.ndarray:
    return np.load(str(Path(path)))["flow"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_output_paths(
    output_root: str | Path,
    scene_id: str,
    frame_idx: int,
) -> Dict[str, Path]:
    """Return canonical output file paths for a single frame.

    The caller is responsible for creating parent directories, or can
    rely on save_* functions to create them automatically.
    """
    root = Path(output_root) / scene_id
    prefix = f"frame_{frame_idx:06d}"
    return {
        "detections": root / "detections" / f"{prefix}.msgpack",
        "depth":      root / "depth"      / f"{prefix}.npz",
        "flow":       root / "flow"       / f"{prefix}.npz",
        "render":     root / "renders"    / f"{prefix}.png",
    }
