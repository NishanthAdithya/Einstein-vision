"""brake_turn_detector.py — Brake light and turn signal detection.

Phase 3 feature: analyses the rear region of tracked vehicles to detect
activated brake lights (red glow) and turn indicators (amber blinking).

A temporal history buffer keyed by track_id tracks per-light colour scores
over successive frames so blinking turn signals can be distinguished from
brief reflections.
"""
from __future__ import annotations

import collections
from typing import Dict, List

import cv2
import numpy as np

from src.io.schema import Detection


# ---------------------------------------------------------------------------
# Vehicle classes that can show brake / turn lights
# ---------------------------------------------------------------------------

_VEHICLE_CLASSES = frozenset({
    "car", "sedan", "hatchback", "suv", "pickup",
    "truck", "bus", "motorcycle", "bicycle",
})

# ---------------------------------------------------------------------------
# HSV colour thresholds
# ---------------------------------------------------------------------------

# Brake light — bright saturated red
_BRAKE_H_LO1, _BRAKE_H_HI1 = 0,   10
_BRAKE_H_LO2, _BRAKE_H_HI2 = 160, 179
_BRAKE_S_MIN = 80
_BRAKE_V_MIN = 120   # must be bright

# Turn signal — amber/orange
_TURN_H_LO,  _TURN_H_HI  = 8,  28
_TURN_S_MIN  = 110
_TURN_V_MIN  = 120

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

# Fraction of analysed-crop pixels that must be the target colour
_BRAKE_DENSITY_THRESH = 0.05   # 5 %
_TURN_DENSITY_THRESH  = 0.04   # 4 %

# Frames of history to keep per track
_HISTORY_LEN = 8

# Minimum bbox dimension (px) to bother analysing
_MIN_BOX_PX = 20


class BrakeTurnDetector:
    """Detect brake lights and turn signals from vehicle bounding box crops.

    Brake lights:
        Analyses a horizontal strip covering the bottom 40 % of the bbox
        (the taillight region on most passenger vehicles).  Activated when
        the red-pixel density exceeds ``brake_thresh``.

    Turn signals:
        Analyses the left and right 30 % vertical strips of the lower 60 % of
        the bbox.  Uses a temporal history to distinguish sustained amber glow
        from single-frame reflections.

    Args:
        history_len:  Number of frames to keep in the temporal buffer.
        brake_thresh: Min red-pixel fraction to classify brake light as on.
        turn_thresh:  Min amber-pixel fraction to classify a turn signal.
    """

    def __init__(
        self,
        history_len: int = _HISTORY_LEN,
        brake_thresh: float = _BRAKE_DENSITY_THRESH,
        turn_thresh: float = _TURN_DENSITY_THRESH,
    ) -> None:
        self._history_len  = history_len
        self._brake_thresh = brake_thresh
        self._turn_thresh  = turn_thresh
        # track_id → deque of (brake_score, left_score, right_score)
        self._history: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=history_len)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame_bgr: np.ndarray,
    ) -> List[Detection]:
        """Fill ``brake_light_on`` and ``turn_signal`` for each vehicle.

        Args:
            detections: All detections for the current frame.
            frame_bgr:  uint8 BGR frame matching the detection coordinates.

        Returns:
            The same list with fields filled in-place.
        """
        h_img, w_img = frame_bgr.shape[:2]

        for det in detections:
            if det.class_name not in _VEHICLE_CLASSES:
                continue

            x1 = max(0, int(det.bbox.x1))
            y1 = max(0, int(det.bbox.y1))
            x2 = min(w_img, int(det.bbox.x2))
            y2 = min(h_img, int(det.bbox.y2))

            bw, bh = x2 - x1, y2 - y1
            if bw < _MIN_BOX_PX or bh < _MIN_BOX_PX:
                continue

            # ── Brake light: bottom 40 % of bbox ─────────────────────────
            brake_top = int(y1 + 0.60 * bh)
            brake_crop = frame_bgr[brake_top:y2, x1:x2]
            brake_score = _red_density(brake_crop)

            # ── Turn signals: left / right 30 % strips, bottom 60 % ──────
            strip_w    = max(1, int(0.30 * bw))
            vert_top   = int(y1 + 0.40 * bh)
            left_crop  = frame_bgr[vert_top:y2, x1:x1 + strip_w]
            right_crop = frame_bgr[vert_top:y2, x2 - strip_w:x2]
            left_score  = _amber_density(left_crop)
            right_score = _amber_density(right_crop)

            # ── Temporal history ──────────────────────────────────────────
            tid = det.track_id if det.track_id is not None else id(det)
            self._history[tid].append((brake_score, left_score, right_score))
            hist = list(self._history[tid])

            # ── Classify ──────────────────────────────────────────────────
            det.brake_light_on = brake_score > self._brake_thresh
            det.turn_signal     = _classify_turn(hist, self._turn_thresh)

        return detections

    def close(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Colour-density helpers
# ---------------------------------------------------------------------------

def _red_density(crop_bgr: np.ndarray) -> float:
    """Fraction of pixels that are bright red (brake-light colour)."""
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv,
                     np.array([_BRAKE_H_LO1, _BRAKE_S_MIN, _BRAKE_V_MIN]),
                     np.array([_BRAKE_H_HI1, 255, 255]))
    m2 = cv2.inRange(hsv,
                     np.array([_BRAKE_H_LO2, _BRAKE_S_MIN, _BRAKE_V_MIN]),
                     np.array([179, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    total = mask.shape[0] * mask.shape[1]
    return float(np.count_nonzero(mask)) / total if total > 0 else 0.0


def _amber_density(crop_bgr: np.ndarray) -> float:
    """Fraction of pixels that are amber/orange (turn-signal colour)."""
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([_TURN_H_LO, _TURN_S_MIN, _TURN_V_MIN]),
                       np.array([_TURN_H_HI, 255, 255]))
    total = mask.shape[0] * mask.shape[1]
    return float(np.count_nonzero(mask)) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Turn-signal temporal classifier
# ---------------------------------------------------------------------------

def _classify_turn(history: list, threshold: float) -> str:
    """Return 'left', 'right', or 'none' from the temporal score buffer.

    A signal is classified as active when its mean score across the history
    exceeds half the threshold (catches intermittent blinks) AND at least one
    frame in the recent history exceeds the full threshold (rules out very
    faint ambient colour).
    """
    if len(history) < 2:
        return "none"

    left_scores  = np.array([h[1] for h in history])
    right_scores = np.array([h[2] for h in history])

    # Require that the mean exceeds the threshold and at least one frame
    # clearly shows the colour.
    left_active  = (left_scores.mean()  > threshold * 0.5 and
                    left_scores.max()   > threshold)
    right_active = (right_scores.mean() > threshold * 0.5 and
                    right_scores.max()  > threshold)

    if left_active and right_active:
        # Hazard lights — pick the stronger side (or 'none' if equal-ish)
        return "left" if left_scores.mean() > right_scores.mean() else "right"
    if left_active:
        return "left"
    if right_active:
        return "right"
    return "none"
