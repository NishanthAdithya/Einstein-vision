from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.io.schema import Detection


# ---------------------------------------------------------------------------
# HSV thresholds — used only for arrow-shape masking, not colour classification
# ---------------------------------------------------------------------------

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

# Minimum crop area; tiny boxes are too noisy to classify reliably.
_MIN_CROP_AREA_PX = 100

# Minimum mean-V of the brightest strip to consider the light on.
_MIN_V_THRESHOLD  = 60


def classify_traffic_lights(
    detections: List[Detection],
    frame_bgr: np.ndarray,
) -> List[Detection]:
    """Classify traffic light colour and arrow using a 3-strip brightness method.

    The bounding box crop is divided into three equal horizontal strips:
        top    → red
        middle → yellow
        bottom → green

    The strip with the highest mean brightness (V channel) is the active light.
    Arrow detection also runs on that strip.

    Args:
        detections: All detections for the frame.
        frame_bgr:  uint8 BGR frame matching the detection coordinates.

    Returns:
        The same list with traffic_light_state and traffic_light_arrow updated.
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

        crop = frame_bgr[y1:y2, x1:x2]
        h, w = crop.shape[:2]

        # ── 3-strip split (top=red, middle=yellow, bottom=green) ────────────
        t1, t2 = h // 3, 2 * h // 3
        strips = [crop[0:t1, :], crop[t1:t2, :], crop[t2:, :]]
        _COLOURS = ("red", "yellow", "green")

        # Score each strip by mean brightness of its V channel
        scores = [
            float(cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
            if s.size > 0 else 0.0
            for s in strips
        ]

        best = int(np.argmax(scores))
        if scores[best] < _MIN_V_THRESHOLD:
            continue   # light appears off

        det.traffic_light_state = _COLOURS[best]

        # ── arrow detection on the winning strip ────────────────────────────
        active_strip = strips[best]
        sh, sw = active_strip.shape[:2]
        if sw >= _ARROW_MIN_CROP_DIM and sh >= _ARROW_MIN_CROP_DIM:
            det.traffic_light_arrow = _detect_arrow(active_strip, det.traffic_light_state)

    return detections


def classify_traffic_light_arrows(
    detections: List[Detection],
    frame_bgr: np.ndarray,
) -> List[Detection]:
    """No-op: arrow detection is now integrated into classify_traffic_lights."""
    return detections


# ---------------------------------------------------------------------------
# Arrow detection helpers
# ---------------------------------------------------------------------------

# Circularity = 4π·area/perimeter² ; a perfect circle ≈ 1.0.
# Real arrows are 0.20–0.50; round bulbs are 0.65–0.95.
_ARROW_CIRCULARITY_MAX = 0.55
_ARROW_MIN_AREA_PX     = 80
_ARROW_MIN_CROP_DIM    = 18   # both width AND height of strip must exceed this


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
       'left' or 'right' based on which side of the strip the tip points to.
    """
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = _build_colour_mask(crop_hsv, colour)

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
        return None   # round bulb

    M = cv2.moments(cnt)
    if M["m00"] < 1e-3:
        return None

    cx = M["m10"] / M["m00"]

    mu20 = M["mu20"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]

    angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)

    if abs(angle) > np.radians(55):
        return "straight"

    pts = cnt.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - cx
    if dx.max() > -dx.min():
        return "right"
    return "left"
