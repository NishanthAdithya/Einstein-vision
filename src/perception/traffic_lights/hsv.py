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

# Circularity = 4π·area/perimeter² ; a perfect circle = 1.0.
# Arrows are elongated/pointed so their circularity is well below this.
_ARROW_CIRCULARITY_MAX = 0.70
_ARROW_MIN_AREA_PX     = 50


def classify_traffic_light_arrows(
    detections: List[Detection],
    frame_bgr: np.ndarray,
) -> List[Detection]:
    """Detect arrow direction for every traffic-light detection in-place.

    Runs only on detections that already have a traffic_light_state set.
    Sets det.traffic_light_arrow to 'left', 'right', 'straight', or None
    (None means the lit region is a solid round bulb, not an arrow).

    Args:
        detections: All detections; non-traffic-light entries are skipped.
        frame_bgr:  uint8 BGR frame.

    Returns:
        The same list with traffic_light_arrow updated.
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
        if (x2 - x1) * (y2 - y1) < _MIN_CROP_AREA_PX:
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
    # green
    return cv2.inRange(crop_hsv,
                       np.array([_GREEN_H_LO, _GREEN_S_MIN, _GREEN_V_MIN]),
                       np.array([_GREEN_H_HI, 255, 255]))


def _detect_arrow(crop_bgr: np.ndarray, colour: str) -> Optional[str]:
    """Return arrow direction ('left'/'right'/'straight') or None.

    Strategy:
    1. Extract the lit colour region as a binary mask.
    2. Find the largest contour.
    3. Compute circularity — below threshold → arrow shape.
    4. Use the orientation of the minimum-area bounding rectangle to
       decide direction: horizontal bounding → left/right (side of
       centroid decides which); vertical bounding → straight.
    """
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = _build_colour_mask(crop_hsv, colour)

    # Morphological clean-up
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

    circularity = 4.0 * np.pi * area / (perimeter ** 2)
    if circularity > _ARROW_CIRCULARITY_MAX:
        return None  # round bulb — no arrow

    # Arrow detected: determine direction from bounding rect orientation
    rect = cv2.minAreaRect(cnt)   # (centre, (w, h), angle)
    rw, rh = rect[1]
    if rw == 0 or rh == 0:
        return "straight"

    aspect = max(rw, rh) / min(rw, rh)
    if aspect < 1.4:
        # Almost square — treat as straight (e.g. U-turn symbol)
        return "straight"

    # Wide rectangle → horizontal arrow (left or right)
    if rw >= rh:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            return "right" if cx > crop_bgr.shape[1] / 2.0 else "left"
        return "straight"

    # Tall rectangle → vertical / straight arrow
    return "straight"
