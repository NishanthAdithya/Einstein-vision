from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.io.schema import Detection


# ---------------------------------------------------------------------------
# HSV thresholds for traffic light colours
# ---------------------------------------------------------------------------

# Red wraps around 0°/180° in OpenCV HSV (0–179), so two ranges are needed.
_RED_H_LO1,  _RED_H_HI1  =   0,  10
_RED_H_LO2,  _RED_H_HI2  = 160, 179
_RED_S_MIN   = 100
_RED_V_MIN   = 80

_YELLOW_H_LO, _YELLOW_H_HI = 18,  38
_YELLOW_S_MIN = 100
_YELLOW_V_MIN = 80

_GREEN_H_LO,  _GREEN_H_HI  = 40,  90
_GREEN_S_MIN  = 60
_GREEN_V_MIN  = 60

# Minimum fraction of crop pixels that must match a colour to report it.
_MIN_RATIO = 0.04

# Minimum crop area; tiny boxes are too noisy to classify reliably.
_MIN_CROP_AREA_PX = 100


def classify_traffic_lights(
    detections: List[Detection],
    frame_bgr: np.ndarray,
) -> List[Detection]:
    """Set traffic_light_state for every traffic-light detection in-place.

    Each bounding box is cropped from the frame, converted to HSV, and the
    dominant colour region (red > yellow > green) is selected.  If no colour
    exceeds _MIN_RATIO the state is left as None.

    Args:
        detections: All detections for the frame (non-traffic-light entries
                    are skipped without modification).
        frame_bgr:  uint8 BGR frame matching the detection coordinates.

    Returns:
        The same list with traffic_light_state updated for traffic lights.
    """
    h_img, w_img = frame_bgr.shape[:2]

    for det in detections:
        if det.class_name != "traffic light":
            continue

        x1 = max(0, int(det.bbox.x1))
        y1 = max(0, int(det.bbox.y1))
        x2 = min(w_img, int(det.bbox.x2))
        y2 = min(h_img, int(det.bbox.y2))

        if (x2 - x1) * (y2 - y1) < _MIN_CROP_AREA_PX:
            continue

        crop_hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        det.traffic_light_state = _dominant_colour(crop_hsv)

    return detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colour_ratio(
    hsv: np.ndarray,
    h_lo: int,
    h_hi: int,
    s_min: int,
    v_min: int,
) -> float:
    """Fraction of pixels in hsv that fall within the given HSV range."""
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_min)
    return float(mask.mean())


def _dominant_colour(crop_hsv: np.ndarray) -> str | None:
    """Return 'red', 'yellow', 'green', or None for the strongest colour."""
    red_ratio = max(
        _colour_ratio(crop_hsv, _RED_H_LO1, _RED_H_HI1, _RED_S_MIN, _RED_V_MIN),
        _colour_ratio(crop_hsv, _RED_H_LO2, _RED_H_HI2, _RED_S_MIN, _RED_V_MIN),
    )
    yellow_ratio = _colour_ratio(
        crop_hsv, _YELLOW_H_LO, _YELLOW_H_HI, _YELLOW_S_MIN, _YELLOW_V_MIN
    )
    green_ratio = _colour_ratio(
        crop_hsv, _GREEN_H_LO, _GREEN_H_HI, _GREEN_S_MIN, _GREEN_V_MIN
    )

    best_ratio = max(red_ratio, yellow_ratio, green_ratio)
    if best_ratio < _MIN_RATIO:
        return None

    if best_ratio == red_ratio:
        return "red"
    if best_ratio == yellow_ratio:
        return "yellow"
    return "green"


# ---------------------------------------------------------------------------
# Phase 2 — Arrow detection
# ---------------------------------------------------------------------------

# Circularity = 4π·area/perimeter² ; a perfect circle ≈ 1.0.
# Real arrows are 0.20–0.50; round bulbs are 0.65–0.95.
_ARROW_CIRCULARITY_MAX = 0.55   # anything above → round bulb, no arrow
_ARROW_MIN_AREA_PX     = 80     # ignore tiny blobs
# Only attempt arrow classification on large-enough crops (pixels).
# Far-away lights are too small for reliable shape analysis.
_ARROW_MIN_CROP_DIM    = 30     # both width AND height must exceed this


def classify_traffic_light_arrows(
    detections: List[Detection],
    frame_bgr: np.ndarray,
) -> List[Detection]:
    """Detect arrow direction for every traffic-light detection in-place.

    Runs only on detections that already have a traffic_light_state set AND
    whose bounding box is large enough for reliable shape analysis.
    Sets det.traffic_light_arrow to 'left', 'right', 'straight', or None.
    """
    h_img, w_img = frame_bgr.shape[:2]

    for det in detections:
        if det.class_name != "traffic light":
            continue
        if det.traffic_light_state is None:
            continue

        x1 = max(0, int(det.bbox.x1))
        y1 = max(0, int(det.bbox.y1))
        x2 = min(w_img, int(det.bbox.x2))
        y2 = min(h_img, int(det.bbox.y2))
        bw, bh = x2 - x1, y2 - y1

        # Skip boxes that are too small — not enough pixels for shape analysis
        if bw < _ARROW_MIN_CROP_DIM or bh < _ARROW_MIN_CROP_DIM:
            continue

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        det.traffic_light_arrow = _detect_arrow(crop_bgr, det.traffic_light_state)

    return detections


def _build_colour_mask(crop_hsv: np.ndarray, colour: str) -> np.ndarray:
    """Return a binary mask for the active traffic light colour."""
    if colour == "red":
        m1 = cv2.inRange(crop_hsv,
                         np.array([_RED_H_LO1, _RED_S_MIN, _RED_V_MIN]),
                         np.array([_RED_H_HI1, 255, 255]))
        m2 = cv2.inRange(crop_hsv,
                         np.array([_RED_H_LO2, _RED_S_MIN, _RED_V_MIN]),
                         np.array([179, 255, 255]))
        return cv2.bitwise_or(m1, m2)
    if colour == "yellow":
        return cv2.inRange(crop_hsv,
                           np.array([_YELLOW_H_LO, _YELLOW_S_MIN, _YELLOW_V_MIN]),
                           np.array([_YELLOW_H_HI, 255, 255]))
    return cv2.inRange(crop_hsv,
                       np.array([_GREEN_H_LO, _GREEN_S_MIN, _GREEN_V_MIN]),
                       np.array([_GREEN_H_HI, 255, 255]))


def _detect_arrow(crop_bgr: np.ndarray, colour: str) -> Optional[str]:
    """Return 'left', 'right', 'straight', or None (round bulb / uncertain).

    Strategy:
    1. Isolate the lit colour region via HSV masking.
    2. Find the largest connected blob.
    3. Circularity test: high circularity → round bulb → return None.
    4. Use image moments (orientation angle via mu20/mu11/mu02) to determine
       the principal axis of the shape.
    5. Near-vertical principal axis → 'straight'; near-horizontal →
       'left' or 'right' based on which side of the crop the tip points to
       (tip = the extreme point farthest from the centroid along the axis).
    """
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = _build_colour_mask(crop_hsv, colour)

    # Morphological clean-up to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < _ARROW_MIN_AREA_PX:
        return None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1e-3:
        return None

    # ── circularity gate ────────────────────────────────────────────────────
    circularity = 4.0 * np.pi * area / (perimeter ** 2)
    if circularity > _ARROW_CIRCULARITY_MAX:
        return None   # round bulb

    # ── principal axis via central moments ──────────────────────────────────
    M = cv2.moments(cnt)
    if M["m00"] < 1e-3:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Covariance components
    mu20 = M["mu20"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]

    # Angle of principal (major) axis in radians, measured from +X axis
    angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)  # −π/2 … +π/2

    abs_angle = abs(angle)   # 0 = horizontal, π/2 = vertical

    # Near-vertical (straight arrow): principal axis mostly along Y
    if abs_angle > np.radians(55):
        return "straight"

    # Near-horizontal (left/right arrow): find which side the tip points to.
    # The "tip" of the arrow is the contour point farthest from the centroid
    # along the horizontal axis.
    pts = cnt.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - cx
    # Point with maximum positive dx → rightward tip → "right" arrow
    # Point with maximum negative dx → leftward tip → "left" arrow
    max_right = dx.max()
    max_left  = -dx.min()

    if max_right > max_left:
        return "right"
    return "left"
