"""motion_classifier.py — Classify vehicles as moving or parked.

Phase 3 feature: uses RAFT optical flow + Sampson distance (epipolar geometry)
to separate independently-moving objects from the camera-motion background.

Algorithm
---------
1. Estimate camera-induced background flow by fitting a fundamental matrix F
   to a random sample of sparse flow correspondences using RANSAC.  Static
   background points will be near-perfect inliers; independently-moving
   objects will deviate (high Sampson distance).

2. For each vehicle / pedestrian detection:
   a. Compute the median flow vector inside the bounding box.
   b. Subtract the median background flow to get the residual.
   c. Compute the Sampson distance for the bbox-centre correspondence.
   d. A detection is classified as *moving* when either:
      - the residual flow magnitude exceeds FLOW_THRESH  (simple threshold), OR
      - its Sampson distance exceeds SAMPSON_THRESH (epipolar deviation).

3. Fill ``det.is_moving`` and approximate ``det.velocity_3d`` (ego frame,
   units roughly m/frame) for moving detections.

The Sampson distance is defined as:

    d_s(x, x') = (x'^T F x)^2
                 ─────────────────────────────────────────────────────
                 (Fx)₀² + (Fx)₁² + (F^T x')₀² + (F^T x')₁²

where x and x' are homogeneous image coordinates of a matched point pair.
"""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

from src.io.schema import Detection


# ---------------------------------------------------------------------------
# Classes that can be independently moving
# ---------------------------------------------------------------------------

_MOTION_CLASSES = frozenset({
    "car", "sedan", "hatchback", "suv", "pickup",
    "truck", "bus", "motorcycle", "bicycle", "person",
})

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Residual flow magnitude (pixels / frame) above which an object is moving
_FLOW_RESIDUAL_THRESH: float = 2.0

# Sampson distance above which a correspondence deviates from camera motion
_SAMPSON_THRESH: float = 4.0

# Number of random sample points used to estimate the fundamental matrix
_F_SAMPLE_POINTS: int = 800

# Minimum RANSAC inlier count to trust the estimated F
_F_MIN_INLIERS: int = 50

# Minimum bbox dimension (pixels) to sample flow inside
_MIN_BOX_PX: int = 16

# Default focal length used when no Camera object is available
_DEFAULT_FOCAL: float = 718.0


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def classify_motion(
    detections: List[Detection],
    flow: np.ndarray,
    camera=None,
) -> List[Detection]:
    """Fill ``is_moving`` and approximate ``velocity_3d`` for each detection.

    Args:
        detections: All detections for the current frame (modified in-place).
        flow:       (H, W, 2) float32 optical flow from RAFT — dx, dy in
                    pixels / frame.
        camera:     Optional Camera object for intrinsic-based velocity
                    scaling.  When omitted a default focal length is used.

    Returns:
        The same detection list with ``is_moving`` and ``velocity_3d`` set
        for vehicle / pedestrian detections.
    """
    if flow is None:
        return detections

    H, W = flow.shape[:2]

    # ── Background flow via median over entire frame ──────────────────────
    # This is the "median-of-all-pixels" background estimate — fast and
    # robust to sparse moving objects occupying <50 % of the image.
    bg_dx = float(np.median(flow[:, :, 0]))
    bg_dy = float(np.median(flow[:, :, 1]))

    # ── Fundamental matrix via RANSAC on random sparse correspondences ────
    F_mat, F_valid = _estimate_fundamental_matrix(flow, H, W)

    # ── Per-detection classification ─────────────────────────────────────
    fx = camera.config.fx if camera is not None else _DEFAULT_FOCAL
    fy = camera.config.fy if camera is not None else _DEFAULT_FOCAL

    for det in detections:
        if det.class_name not in _MOTION_CLASSES:
            continue

        x1 = max(0, int(det.bbox.x1))
        y1 = max(0, int(det.bbox.y1))
        x2 = min(W, int(det.bbox.x2))
        y2 = min(H, int(det.bbox.y2))

        if (x2 - x1) < _MIN_BOX_PX or (y2 - y1) < _MIN_BOX_PX:
            det.is_moving = False
            continue

        # Median flow inside the bbox
        patch = flow[y1:y2, x1:x2]
        dx = float(np.median(patch[:, :, 0]))
        dy = float(np.median(patch[:, :, 1]))

        # Residual after removing camera-induced background motion
        res_dx = dx - bg_dx
        res_dy = dy - bg_dy
        residual_mag = math.sqrt(res_dx ** 2 + res_dy ** 2)

        # Sampson distance for the bbox-centre correspondence
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s_dist = 0.0
        if F_valid:
            s_dist = sampson_distance(F_mat, cx, cy, cx + dx, cy + dy)

        det.is_moving = (residual_mag > _FLOW_RESIDUAL_THRESH or
                         s_dist > _SAMPSON_THRESH)

        # Approximate ego-frame velocity from residual pixel flow and depth
        if det.is_moving:
            depth = det.depth if det.depth > 0.5 else 20.0   # fallback 20 m
            # Camera X → ego X (right);  camera Y (down) → ego -Z (up)
            # We project onto the ground plane (vz = 0) for the arrow.
            vx_ego = res_dx * depth / fx   # right/left (m/frame)
            vy_ego = 0.0                   # forward — not estimable from 2D
            vz_ego = 0.0
            det.velocity_3d = (vx_ego, vy_ego, vz_ego)

    return detections


# ---------------------------------------------------------------------------
# Sampson distance
# ---------------------------------------------------------------------------

def sampson_distance(
    F: np.ndarray,
    u1: float, v1: float,
    u2: float, v2: float,
) -> float:
    """Compute the Sampson distance for a matched point pair.

    The Sampson distance is the first-order approximation of the geometric
    epipolar error and is cheap to compute.  Points on stationary background
    will have values near zero; independently-moving points deviate.

    Args:
        F:       3×3 fundamental matrix (float64).
        u1, v1:  Image coordinates of the point in frame t   (pixels).
        u2, v2:  Image coordinates of the point in frame t+1 (pixels).

    Returns:
        Non-negative Sampson distance (float).  Returns 0 when denominator
        is numerically zero (degenerate case).
    """
    x1 = np.array([u1, v1, 1.0])
    x2 = np.array([u2, v2, 1.0])

    Fx1   = F   @ x1   # (3,)
    FTx2  = F.T @ x2   # (3,)

    numerator   = float(x2 @ Fx1) ** 2
    denominator = Fx1[0]**2 + Fx1[1]**2 + FTx2[0]**2 + FTx2[1]**2

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Fundamental matrix estimation
# ---------------------------------------------------------------------------

def _estimate_fundamental_matrix(
    flow: np.ndarray,
    H: int,
    W: int,
) -> tuple[Optional[np.ndarray], bool]:
    """Fit a fundamental matrix F to random flow correspondences via RANSAC.

    Samples _F_SAMPLE_POINTS random pixel positions from the full frame.
    Each pixel (u, v) with flow (dx, dy) contributes the correspondence
    (u, v) ↔ (u+dx, v+dy).

    Returns:
        (F, valid) — F is the 3×3 matrix (or None), valid is True when the
        RANSAC fit succeeded with enough inliers.
    """
    rng = np.random.default_rng(seed=0)
    ys  = rng.integers(0, H, _F_SAMPLE_POINTS)
    xs  = rng.integers(0, W, _F_SAMPLE_POINTS)

    dx = flow[ys, xs, 0]
    dy = flow[ys, xs, 1]

    pts1 = np.stack([xs, ys],           axis=1).astype(np.float32)
    pts2 = np.stack([xs + dx, ys + dy], axis=1).astype(np.float32)

    # Filter out near-zero flow (static pixels are fine but pure zeros add
    # noise to the degenerate-motion case)
    mag = np.sqrt(dx**2 + dy**2)
    valid_mask = mag > 0.1
    if valid_mask.sum() < _F_MIN_INLIERS:
        return None, False

    pts1 = pts1[valid_mask]
    pts2 = pts2[valid_mask]

    try:
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=2.0,
            confidence=0.99,
            maxIters=1000,
        )
    except cv2.error:
        return None, False

    if F is None or F.shape != (3, 3):
        return None, False

    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < _F_MIN_INLIERS:
        return None, False

    return F.astype(np.float64), True
