from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.io.schema import Detection


# Approximate vehicle dimensions (length, width) in metres.
# Used to convert projected bbox width → yaw angle.
_VEHICLE_DIMS: Dict[str, Tuple[float, float]] = {
    "car":          (4.5, 2.0),
    "truck":        (9.0, 2.5),
    "bus":          (12.0, 2.6),
    "motorcycle":   (2.2, 0.8),
    "bicycle":      (1.8, 0.6),
    "person":       (0.5, 0.5),
}
_DEFAULT_DIMS: Tuple[float, float] = (4.5, 2.0)  # fall back to car

# Only estimate yaw for these classes; others are left at 0.0.
_YAW_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def estimate_yaw(detection: Detection, fx: float) -> float:
    """Estimate the yaw angle of a single vehicle from its bbox footprint.

    Geometric approach:
        When a vehicle of length L and width W is viewed at yaw θ
        (0 = front/rear facing, π/2 = side-on), its projected ground
        footprint width is bounded by:
            proj_min = W   (at θ = 0, only the vehicle width is visible)
            proj_max = L   (at θ = π/2, only the vehicle length is visible)

        We linearly interpolate the measured projected width between
        these bounds to get an estimate of θ.

    Args:
        detection:  A Detection with bbox and depth already populated.
        fx:         Focal length in pixels (from CameraConfig.fx).

    Returns:
        Yaw angle in radians.  0 = front/rear-facing.  π/2 = side-on.
        Returns 0.0 for non-vehicle classes or if depth is not available.
    """
    if detection.class_name not in _YAW_CLASSES:
        return 0.0
    if detection.depth <= 0.0:
        return 0.0

    L, W = _VEHICLE_DIMS.get(detection.class_name, _DEFAULT_DIMS)
    if L <= W:
        return 0.0

    # Physical width of the bbox footprint at the measured depth
    proj_width_m = detection.bbox.width * detection.depth / fx

    # Map [W, L] → [0, π/2], clamped to that range
    t = (proj_width_m - W) / (L - W)
    t = max(0.0, min(1.0, t))
    return t * math.pi / 2.0


def estimate_yaws(detections: List[Detection], fx: float) -> List[Detection]:
    """Estimate and fill the yaw field for every detection in-place.

    Args:
        detections: List of Detection objects with depth already populated
                    (call lift_detections first).
        fx:         Focal length in pixels.

    Returns:
        The same list with the yaw field updated on vehicle detections.
    """
    for det in detections:
        det.yaw = estimate_yaw(det, fx)
    return detections
