from __future__ import annotations

from typing import Dict, List

import numpy as np
import supervision as sv

from src.io.schema import BBox, Detection
from src.perception.base import Tracker


class ByteTrackTracker(Tracker):
    """ByteTrack multi-object tracker with EMA bounding-box smoothing.

    Uses the supervision library's ByteTrack wrapper.  After each update,
    the smoothed bbox for each active track is blended with the raw
    detection bbox using an exponential moving average.

    Args:
        frame_rate:                  Effective frame rate (36 / sample_every_n)
        lost_track_buffer:           Frames to keep a lost track alive
        track_activation_threshold:  Min confidence to initialise a new track
        minimum_matching_threshold:  Min IoU for track-detection matching
        ema_alpha:                   EMA weight for the new observation (0–1).
                                     Higher = trust new bbox more.
    """

    def __init__(
        self,
        frame_rate: float = 7.2,
        lost_track_buffer: int = 30,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.80,
        ema_alpha: float = 0.40,
    ) -> None:
        self._ema_alpha = ema_alpha
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=int(round(frame_rate)),
        )
        # track_id (int) → smoothed [x1, y1, x2, y2]
        self._ema_boxes: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame_bgr: np.ndarray,
    ) -> List[Detection]:
        """Assign track IDs and apply EMA smoothing to bounding boxes.

        Detections that ByteTrack does not match to any track (e.g. too low
        confidence to initialise) are dropped from the returned list.

        Args:
            detections:  Raw detections from the object detector.
            frame_bgr:   Current frame (used by supervision internally).

        Returns:
            Subset of detections with track_id set and bbox EMA-smoothed.
        """
        if not detections:
            return []

        sv_dets = _to_sv_detections(detections)
        tracked = self._tracker.update_with_detections(sv_dets)

        if len(tracked) == 0:
            return []

        tracked_ids = tracked.tracker_id          # (M,) int array
        tracked_boxes = tracked.xyxy              # (M, 4) float array
        tracked_confs = tracked.confidence        # (M,) float array
        tracked_classes = tracked.data.get("class_name", [None] * len(tracked))

        result: List[Detection] = []
        for i, tid in enumerate(tracked_ids):
            raw_box = tracked_boxes[i]
            smoothed = self._apply_ema(int(tid), raw_box)
            bbox = BBox(
                x1=float(smoothed[0]),
                y1=float(smoothed[1]),
                x2=float(smoothed[2]),
                y2=float(smoothed[3]),
            )

            # Find the original Detection to preserve all its fields
            orig = _match_detection(detections, raw_box)
            if orig is not None:
                orig.track_id = int(tid)
                orig.bbox = bbox
                result.append(orig)
            else:
                result.append(
                    Detection(
                        class_name=tracked_classes[i] or "unknown",
                        confidence=float(tracked_confs[i]),
                        bbox=bbox,
                        track_id=int(tid),
                    )
                )

        return result

    def reset(self) -> None:
        """Reset track state — call between scenes."""
        self._tracker.reset()
        self._ema_boxes.clear()

    def close(self) -> None:
        self._ema_boxes.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_ema(self, track_id: int, new_box: np.ndarray) -> np.ndarray:
        """Blend new_box with the stored EMA box for this track ID.

        On first appearance the EMA is initialised to new_box exactly.
        """
        if track_id not in self._ema_boxes:
            self._ema_boxes[track_id] = new_box.copy()
        else:
            alpha = self._ema_alpha
            self._ema_boxes[track_id] = (
                alpha * new_box + (1.0 - alpha) * self._ema_boxes[track_id]
            )
        return self._ema_boxes[track_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_sv_detections(detections: List[Detection]) -> sv.Detections:
    """Convert a list of Detection objects to a supervision Detections object."""
    xyxy = np.array(
        [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
        dtype=np.float32,
    )
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    # supervision needs integer class IDs; use a hash so class names map consistently
    class_id = np.array(
        [hash(d.class_name) & 0xFFFF for d in detections], dtype=np.int32
    )
    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={"class_name": [d.class_name for d in detections]},
    )


def _match_detection(
    detections: List[Detection],
    box: np.ndarray,
    tol: float = 1.0,
) -> Detection | None:
    """Return the Detection whose bbox best matches box (within tol pixels).

    ByteTrack may slightly adjust coordinates during matching, so we use
    L-inf distance rather than exact equality.
    """
    best, best_dist = None, float("inf")
    for d in detections:
        raw = np.array([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
        dist = float(np.abs(raw - box).max())
        if dist < best_dist:
            best_dist = dist
            best = d
    return best if best_dist <= tol else None
