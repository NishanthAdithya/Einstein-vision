from __future__ import annotations

from typing import List

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
