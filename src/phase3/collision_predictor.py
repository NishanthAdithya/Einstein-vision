"""collision_predictor.py — Pedestrian / vehicle collision prediction.

Extra-credit (15 %): predicts future positions of all tracked objects by
fitting a linear velocity from the recent position history, then flags any
pedestrian–vehicle pair whose predicted paths come within WARNING_DIST metres.

Flagged detections receive ``collision_risk = True``; the Blender renderer
highlights them with a red emissive overlay.

Algorithm
---------
1. Every call to ``update()`` appends the current 3-D ego-frame position to a
   per-track rolling buffer (length HISTORY_LEN frames).
2. A linear velocity is estimated as the mean frame-to-frame displacement over
   the buffer (or the most recent step if the buffer has only 2 entries).
3. Each detection's predicted position in PREDICT_FRAMES frames is:
       p_pred = p_now + velocity * PREDICT_FRAMES
4. All (pedestrian, vehicle) pairs are checked.  If any predicted pair falls
   within WARNING_DIST metres the track IDs of both are marked at risk.
5. The velocity estimate is also stored in ``det.velocity_3d`` (m/frame in
   ego frame) so the motion arrows in Blender can use it.
"""
from __future__ import annotations

import collections
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.io.schema import Detection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Frames of 3-D position history to keep per track
_HISTORY_LEN: int = 10

# Number of future frames to project forward for collision checking
_PREDICT_FRAMES: int = 15   # ≈ 2 s at 7.2 fps sampled

# Minimum 3-D distance (metres) at which a pedestrian–vehicle pair triggers
# a collision risk warning
_WARNING_DIST: float = 5.0

# Minimum track history length before we issue a prediction (avoids false
# positives on tracks that just appeared)
_MIN_HISTORY: int = 3

# ---------------------------------------------------------------------------
# Class sets
# ---------------------------------------------------------------------------

_VEHICLE_CLASSES = frozenset({
    "car", "sedan", "hatchback", "suv", "pickup",
    "truck", "bus", "motorcycle", "bicycle",
})

_PEDESTRIAN_CLASSES = frozenset({"person"})


class CollisionPredictor:
    """Predict pedestrian–vehicle collisions and flag at-risk detections.

    Args:
        history_len:    Rolling buffer length per track (frames).
        predict_frames: How many frames ahead to project.
        warning_dist:   Distance threshold in metres for collision warning.
    """

    def __init__(
        self,
        history_len:    int   = _HISTORY_LEN,
        predict_frames: int   = _PREDICT_FRAMES,
        warning_dist:   float = _WARNING_DIST,
    ) -> None:
        self._history_len    = history_len
        self._predict_frames = predict_frames
        self._warning_dist   = warning_dist

        # track_id → deque of (x, y, z) ego-frame positions
        self._pos_history: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=history_len)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update position histories and flag at-risk detections.

        Args:
            detections: All detections for the current frame.

        Returns:
            The same list with ``collision_risk`` and ``velocity_3d`` set
            where appropriate.
        """
        # ── 1. Update position histories ──────────────────────────────────
        for det in detections:
            if det.pos_3d is None or det.track_id is None:
                continue
            self._pos_history[det.track_id].append(tuple(det.pos_3d))

        # ── 2. Estimate velocities and predicted positions ────────────────
        predicted: Dict[int, Tuple[float, float, float]] = {}
        velocities: Dict[int, Tuple[float, float, float]] = {}

        for det in detections:
            if det.track_id is None or det.pos_3d is None:
                continue
            hist = list(self._pos_history[det.track_id])
            if len(hist) < _MIN_HISTORY:
                continue

            vel = _estimate_velocity(hist)
            velocities[det.track_id] = vel

            px, py, pz = det.pos_3d
            vx, vy, vz = vel
            predicted[det.track_id] = (
                px + vx * self._predict_frames,
                py + vy * self._predict_frames,
                pz + vz * self._predict_frames,
            )

        # ── 3. Find at-risk (pedestrian, vehicle) pairs ───────────────────
        ped_ids = [
            det.track_id for det in detections
            if det.class_name in _PEDESTRIAN_CLASSES
            and det.track_id in predicted
        ]
        veh_ids = [
            det.track_id for det in detections
            if det.class_name in _VEHICLE_CLASSES
            and det.track_id in predicted
        ]

        at_risk: set = set()
        for pid in ped_ids:
            for vid in veh_ids:
                dist = _dist3d(predicted[pid], predicted[vid])
                if dist < self._warning_dist:
                    at_risk.add(pid)
                    at_risk.add(vid)

        # ── 4. Fill detection fields ──────────────────────────────────────
        for det in detections:
            if det.track_id is None:
                continue
            if det.track_id in at_risk:
                det.collision_risk = True
            # Overwrite velocity_3d with the smoother history-based estimate
            # (takes priority over the per-frame flow estimate in motion_classifier)
            if det.track_id in velocities:
                det.velocity_3d = velocities[det.track_id]

        return detections

    def close(self) -> None:
        self._pos_history.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_velocity(
    history: List[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Return mean frame-to-frame displacement as a velocity estimate.

    Args:
        history: List of (x, y, z) positions ordered oldest → newest.

    Returns:
        (vx, vy, vz) in metres per frame (ego frame).
    """
    if len(history) < 2:
        return (0.0, 0.0, 0.0)

    pts = np.array(history, dtype=np.float64)
    # Frame-to-frame displacements
    diffs = np.diff(pts, axis=0)   # (N-1, 3)
    mean_vel = diffs.mean(axis=0)
    return (float(mean_vel[0]), float(mean_vel[1]), float(mean_vel[2]))


def _dist3d(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """Euclidean 3-D distance between two points."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
