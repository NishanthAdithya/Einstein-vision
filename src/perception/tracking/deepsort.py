"""deepsort.py — DeepSORT multi-object tracker.

Uses deep-sort-realtime which bundles a MobileNet appearance embedder.
Provides more consistent track IDs across occlusions than ByteTrack.

Install: pip install deep-sort-realtime
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.io.schema import Detection
from src.perception.base import Tracker


class DeepSORTTracker(Tracker):
    """DeepSORT tracker with appearance re-ID.

    Args:
        max_age:         Frames to keep a lost track alive.
        n_init:          Detections before a track is confirmed.
        max_cosine_dist: Appearance distance threshold.
        embedder:        Feature extractor — 'mobilenet' (default, bundled)
                         or 'clip_RN50' (requires openai-clip).
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_dist: float = 0.4,
        embedder: str = "mobilenet",
    ) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._max_age = max_age
        self._n_init = n_init
        self._max_cosine_dist = max_cosine_dist
        self._embedder = embedder
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_dist,
            embedder=embedder,
            half=False,
            bgr=True,
            embedder_gpu=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame_bgr: np.ndarray,
    ) -> List[Detection]:
        """Update tracker and assign track_ids.

        Args:
            detections: Current-frame detections (no track_id yet).
            frame_bgr:  The BGR frame the detections were made on.

        Returns:
            Same detections with track_id filled for confirmed tracks.
            Unmatched detections have track_id = None.
        """
        if not detections:
            return detections

        # Convert to deep-sort-realtime format: [[x1,y1,w,h], conf, class]
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox.as_xyxy()
            w, h = x2 - x1, y2 - y1
            raw.append(([x1, y1, w, h], det.confidence, det.class_name))

        tracks = self._tracker.update_tracks(raw, frame=frame_bgr)

        # Build a map from detection index to track id via bbox IoU matching
        # deep-sort-realtime returns tracks with .to_ltwh() and .track_id
        confirmed = [t for t in tracks if t.is_confirmed()]

        for det in detections:
            best_iou, best_tid = 0.0, None
            dx1, dy1, dx2, dy2 = det.bbox.as_xyxy()
            for t in confirmed:
                ltwh = t.to_ltwh()
                tx1, ty1 = ltwh[0], ltwh[1]
                tx2, ty2 = tx1 + ltwh[2], ty1 + ltwh[3]
                # IoU
                ix1, iy1 = max(dx1, tx1), max(dy1, ty1)
                ix2, iy2 = min(dx2, tx2), min(dy2, ty2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    continue
                union = (dx2-dx1)*(dy2-dy1) + (tx2-tx1)*(ty2-ty1) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_tid = t.track_id
            if best_iou > 0.3:
                det.track_id = int(best_tid)

        return detections

    def reset(self) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(
            max_age=self._max_age,
            n_init=self._n_init,
            max_cosine_distance=self._max_cosine_dist,
            embedder=self._embedder,
            half=False,
            bgr=True,
            embedder_gpu=True,
        )

    def close(self) -> None:
        pass
