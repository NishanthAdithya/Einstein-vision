from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from src.io.schema import BBox


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

def rotation_x(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix around the X axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0,   c,  -s],
         [0.0,   s,   c]],
        dtype=np.float64,
    )


def rotation_y(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix around the Y axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[  c, 0.0,   s],
         [0.0, 1.0, 0.0],
         [ -s, 0.0,   c]],
        dtype=np.float64,
    )


def rotation_z(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix around the Z axis."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[  c,  -s, 0.0],
         [  s,   c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Coordinate frame transforms
# ---------------------------------------------------------------------------

def ego_to_blender(
    xyz_ego: np.ndarray,
    yaw_ego: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Convert a position from the ego frame to the Blender scene frame.

    Ego frame:    X = right, Y = forward, Z = up  (vehicle body frame)
    Blender frame: X = right, Y = forward, Z = up  (same convention when
                   the scene is set up with Z-up and Y-forward)

    The ego and Blender frames share the same axis orientation, so the
    position transform is the identity.  The yaw angle is negated because
    Blender rotations follow right-hand rule around Z (counter-clockwise
    positive) while our yaw convention is clockwise-positive from the
    forward axis.

    Args:
        xyz_ego:  (3,) or (N, 3) float array of ego-frame positions.
        yaw_ego:  Yaw angle in radians (from yaw_estimator).

    Returns:
        (xyz_blender, yaw_blender): position unchanged, yaw sign-flipped.
    """
    xyz_blender = np.asarray(xyz_ego, dtype=np.float32)
    yaw_blender = -float(yaw_ego)
    return xyz_blender, yaw_blender


def camera_frame_to_ego(
    xyz_cam: np.ndarray,
    pitch_rad: float,
    height_m: float,
) -> np.ndarray:
    """Convert camera-frame points to ego frame without a Camera object.

    Convenience function for callers that only have scalars rather than
    a full CameraConfig.  Use Camera.camera_to_ego() when a Camera object
    is available.

    Camera frame: X = right, Y = down, Z = forward
    Ego frame:    X = right, Y = forward, Z = up

    Args:
        xyz_cam:   (N, 3) or (3,) array in camera frame.
        pitch_rad: Camera pitch (nose-down positive) in radians.
        height_m:  Camera height above the ground plane in metres.

    Returns:
        (N, 3) or (3,) float32 array in ego frame.
    """
    # Axis swap: ego_x = cam_x, ego_y = cam_z, ego_z = -cam_y
    M_swap = np.array(
        [[1.0,  0.0, 0.0],
         [0.0,  0.0, 1.0],
         [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    R_pitch = np.array(
        [[1.0, 0.0,  0.0],
         [0.0,  cp,   sp],
         [0.0, -sp,   cp]],
        dtype=np.float64,
    )
    R = R_pitch @ M_swap

    pts = np.atleast_2d(np.asarray(xyz_cam, dtype=np.float64))
    ego = (R @ pts.T).T
    ego[:, 2] += height_m

    result = ego.astype(np.float32)
    return result.squeeze() if np.asarray(xyz_cam).ndim == 1 else result


# ---------------------------------------------------------------------------
# 2-D bounding box IoU
# ---------------------------------------------------------------------------

def iou_2d(a: BBox, b: BBox) -> float:
    """Intersection-over-Union for two BBox objects."""
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


def _intersection_area(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute an M×N pairwise IoU matrix between two sets of boxes.

    Args:
        boxes_a: (M, 4) float array of [x1, y1, x2, y2] boxes
        boxes_b: (N, 4) float array of [x1, y1, x2, y2] boxes

    Returns:
        (M, N) float32 IoU matrix.
    """
    M, N = len(boxes_a), len(boxes_b)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    # Broadcast intersection
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])   # (M, N)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Misc 3-D helpers
# ---------------------------------------------------------------------------

def euclidean_3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3-D points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns the zero vector if norm is zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v
