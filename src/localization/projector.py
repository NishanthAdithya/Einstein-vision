from __future__ import annotations

from typing import List

import numpy as np

from src.io.schema import Detection, Lane
from src.localization.camera import Camera


# Size of the sampling window around bbox center for depth estimation.
# Taking the median over a patch is more robust than a single-pixel lookup
# because depth predictions are often noisy at object boundaries.
_DEPTH_PATCH_HALF = 4   # 9×9 window

# Typical heights (metres) used for perspective-based depth fallback when
# the depth map returns an unreliable value (< _MIN_RELIABLE_DEPTH).
_MIN_RELIABLE_DEPTH = 1.0   # metres; below this the depth is suspect
_CLASS_HEIGHTS = {
    "car": 1.5, "sedan": 1.5, "hatchback": 1.4, "suv": 1.8, "pickup": 1.9,
    "truck": 3.5, "bus": 3.2, "motorcycle": 1.2, "bicycle": 1.1,
    "person": 1.75, "traffic light": 4.0, "stop sign": 2.5,
    "traffic cone": 0.7, "traffic cylinder": 0.9,
}


def lift_detections(
    detections: List[Detection],
    depth_map: np.ndarray,
    camera: Camera,
) -> List[Detection]:
    """Fill pos_3d, depth, and (ego-frame) position for each detection in-place.

    For each detection the depth is sampled from a small patch centred on the
    bbox centre.  The patch median is used to reduce the effect of boundary
    noise in the depth prediction.

    Args:
        detections: List of Detection objects (bbox must be populated).
        depth_map:  float32 (H, W) metric depth map in metres.
        camera:     Camera object matching the frame.

    Returns:
        The same list with pos_3d and depth fields updated.
    """
    h, w = depth_map.shape[:2]

    for det in detections:
        u, v = det.bbox.center          # (col, row) floats
        depth = _sample_depth(depth_map, u, v, h, w)

        # Fallback: depth map returned an unreliable near-zero value.
        # Estimate distance using the bbox height and known real-world height.
        if depth < _MIN_RELIABLE_DEPTH:
            typical_h = _CLASS_HEIGHTS.get(det.class_name, 1.5)
            bbox_px_h = max(1.0, det.bbox.y2 - det.bbox.y1)
            depth = float(camera.config.fy) * typical_h / bbox_px_h
            depth = max(depth, _MIN_RELIABLE_DEPTH)

        x_cam, y_cam, z_cam = camera.unproject(u, v, depth)
        xyz_cam = np.array([[x_cam, y_cam, z_cam]])
        xyz_ego = camera.camera_to_ego(xyz_cam)  # (1, 3) or (3,)
        xyz_ego = np.atleast_1d(xyz_ego).flatten()

        det.depth = float(depth)
        det.pos_3d = (float(xyz_ego[0]), float(xyz_ego[1]), float(xyz_ego[2]))

    return detections


def lift_lane_points(
    lanes: List[Lane],
    camera: Camera,
    depth_map: np.ndarray | None = None,
) -> List[Lane]:
    """Lift lane bezier points to 3D ego-frame coordinates in-place.

    Two strategies are used, in priority order:
      1. IPM (Inverse Perspective Mapping): assumes the ground is flat and
         uses the camera's height + pitch to map pixel → ego (X, Y) with Z=0.
         This is the primary strategy and works well for near-range lanes.
      2. Depth unprojection: used as a fallback for bezier points where IPM
         returns NaN (e.g. far-away lane segments near the horizon). Requires
         a depth_map to be provided.

    If both strategies fail for a point, it is left as (0, 0, 0).

    Args:
        lanes:     List of Lane objects (pixels_2d and poly_coeffs populated).
        camera:    Camera object matching the frame.
        depth_map: Optional float32 (H, W) depth map for the fallback strategy.

    Returns:
        The same list with bezier_points_3d updated.
    """
    h_img = camera.config.image_height
    w_img = camera.config.image_width

    for lane in lanes:
        bezier_xy = lane.bezier_points_3d[:, :2]   # (N, 2) pixel (x, y) coords
        n = bezier_xy.shape[0]

        # Strategy 1: IPM
        ipm_ego = camera.apply_ipm(bezier_xy)       # (N, 2) — NaN where invalid

        points_3d = np.zeros((n, 3), dtype=np.float32)

        ipm_valid = ~np.isnan(ipm_ego[:, 0])
        points_3d[ipm_valid, 0] = ipm_ego[ipm_valid, 0]   # X_ego
        points_3d[ipm_valid, 1] = ipm_ego[ipm_valid, 1]   # Y_ego
        # Z_ego = 0 for ground-plane assumption (road surface)

        # Strategy 2: depth fallback for NaN rows
        ipm_invalid = ~ipm_valid
        if ipm_invalid.any() and depth_map is not None:
            invalid_uv = bezier_xy[ipm_invalid]            # (M, 2)
            depths = np.array(
                [
                    _sample_depth(depth_map, float(uv[0]), float(uv[1]), h_img, w_img)
                    for uv in invalid_uv
                ],
                dtype=np.float32,
            )
            cam_pts = camera.unproject_batch(invalid_uv, depths)  # (M, 3)
            ego_pts = camera.camera_to_ego(cam_pts)               # (M, 3)
            points_3d[ipm_invalid] = ego_pts

        lane.bezier_points_3d = points_3d

    return lanes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_depth(
    depth_map: np.ndarray,
    u: float,
    v: float,
    h: int,
    w: int,
) -> float:
    """Sample the median depth in a patch centred at (u, v).

    Clamping ensures the patch stays within the image.  The median is used
    instead of the mean to suppress outlier depth values at object edges.
    """
    col = max(0, min(w - 1, int(round(u))))
    row = max(0, min(h - 1, int(round(v))))

    r0 = max(0, row - _DEPTH_PATCH_HALF)
    r1 = min(h, row + _DEPTH_PATCH_HALF + 1)
    c0 = max(0, col - _DEPTH_PATCH_HALF)
    c1 = min(w, col + _DEPTH_PATCH_HALF + 1)

    patch = depth_map[r0:r1, c0:c1]
    return float(np.median(patch))
