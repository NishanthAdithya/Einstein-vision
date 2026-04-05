"""yolo_pose.py — Pose estimator backed by YOLO11-pose (ultralytics).

Drop-in replacement for RTMWPoseEstimator that requires no mmcv/mmpose.
Uses the 17 COCO body keypoints from YOLO11x-pose.

Install: already covered by `ultralytics` in requirements.txt.
Weights are auto-downloaded on first run.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from src.io.schema import BBox, Detection, PoseResult
from src.perception.base import PoseEstimator

# YOLO11-pose outputs 17 COCO body keypoints.
# We zero-pad to 133 to stay compatible with the schema (RTMW format).
_YOLO_KPT   = 17
_SCHEMA_KPT = 133


class YOLOPoseEstimator(PoseEstimator):
    """Whole-body pose estimator using YOLO11x-pose.

    Outputs keypoints in (133, 3) format matching the schema: the first 17
    entries are COCO body keypoints (x, y, confidence); the remaining 116
    are zero-padded (feet, face, hands not estimated by this model).

    Args:
        weights: YOLO model weights name or path. Defaults to 'yolo11x-pose.pt'
                 which is auto-downloaded from Ultralytics on first use.
        device:  'cuda' or 'cpu'
    """

    def __init__(
        self,
        weights: str = "yolo11x-pose.pt",
        device: str = "cuda",
    ) -> None:
        self._model = YOLO(weights)
        self._model.fuse()
        self._device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        frame_bgr: np.ndarray,
        person_detections: List[Detection],
    ) -> List[PoseResult]:
        """Estimate body keypoints for each detected person.

        Runs YOLO11-pose on the full frame (it detects people itself), then
        matches results back to the person_detections list by IoU.

        Args:
            frame_bgr:         uint8 BGR image.
            person_detections: Detections from the object detector.

        Returns:
            One PoseResult per person detection, in the same order.
        """
        persons = [d for d in person_detections if d.class_name == "person"]
        if not persons:
            return []

        results = self._model.predict(
            frame_bgr,
            conf=0.25,
            verbose=False,
            device=self._device,
        )

        poses: List[PoseResult] = []
        if not results or results[0].keypoints is None:
            # No pose results — return empty PoseResults for each person
            for det in persons:
                poses.append(PoseResult(
                    bbox=det.bbox,
                    keypoints=np.zeros((_SCHEMA_KPT, 3), dtype=np.float32),
                ))
            return poses

        r = results[0]
        yolo_boxes  = r.boxes.xyxy.cpu().numpy()   # (N, 4)
        yolo_kpts   = r.keypoints.data.cpu().numpy()  # (N, 17, 3)

        for det in persons:
            det_box = np.array([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
            best_idx = _best_iou_match(det_box, yolo_boxes)

            kpts_full = np.zeros((_SCHEMA_KPT, 3), dtype=np.float32)
            if best_idx >= 0:
                kpts_full[:_YOLO_KPT] = yolo_kpts[best_idx]  # (17, 3)

            poses.append(PoseResult(
                bbox=det.bbox,
                keypoints=kpts_full,
            ))

        return poses

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_iou_match(box: np.ndarray, candidates: np.ndarray) -> int:
    """Return index of the candidate with highest IoU, or -1 if none."""
    if len(candidates) == 0:
        return -1

    x1 = np.maximum(box[0], candidates[:, 0])
    y1 = np.maximum(box[1], candidates[:, 1])
    x2 = np.minimum(box[2], candidates[:, 2])
    y2 = np.minimum(box[3], candidates[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_cand = (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1])
    union = area_box + area_cand - inter

    iou = np.where(union > 0, inter / union, 0.0)
    best = int(np.argmax(iou))
    return best if iou[best] > 0.3 else -1
