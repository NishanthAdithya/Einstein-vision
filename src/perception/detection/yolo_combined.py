from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchvision.ops as tv_ops
from ultralytics import YOLO

from src.io.schema import BBox, Detection
from src.perception.base import ObjectDetector


# COCO classes from YOLO11 that are relevant to driving scenes.
# All other COCO classes (furniture, food, sports equipment, etc.) are ignored.
_COCO_KEEP = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "fire hydrant",
}

# Normalise YOLO-World free-text class names → canonical asset_manager keys.
_WORLD_NAME_MAP = {
    "traffic cone":     "traffic cone",
    "traffic cylinder": "traffic cylinder",
    "traffic pole":     "traffic pole",
    "dustbin":          "dustbin",
    "trash can":        "dustbin",
    "waste bin":        "dustbin",
    "litter bin":       "dustbin",
    "garbage bin":      "dustbin",
    "barrel":           "barrel",
    "stop sign":        "stop sign",
    "speed limit sign": "speed limit sign",
    "circular speed limit sign": "speed limit sign",
    "road sign with number":     "speed limit sign",
    "road arrow":       "road arrow marking",
    "road arrow marking": "road arrow marking",
    # Phase 3 / extra credit — speed bumps
    "speed bump":       "speed bump",
    "speed hump":       "speed bump",
    "raised crosswalk": "speed bump",
    "road bump":        "speed bump",
}


class YoloCombinedDetector(ObjectDetector):
    """Object detector that fuses YOLO11 (COCO) and YOLO-World (open-vocab).

    YOLO11 handles the standard driving-scene classes. YOLO-World handles
    classes that COCO does not cover (traffic cones, barrels, etc.).
    Overlapping predictions are removed with cross-model NMS.

    Args:
        yolo11_weights:      Path or name of YOLO11 weights, e.g. 'yolo11x.pt'
        yolo_world_weights:  Path or name of YOLO-World weights
        open_vocab_classes:  List of class names for YOLO-World
        conf_threshold:      Minimum confidence to keep a detection
        iou_threshold:       IoU threshold for NMS (within and across models)
    """

    def __init__(
        self,
        yolo11_weights: str,
        yolo_world_weights: str,
        open_vocab_classes: List[str],
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
    ) -> None:
        self._conf = conf_threshold
        self._iou = iou_threshold

        self._yolo11 = YOLO(yolo11_weights)
        self._yolo11.fuse()

        self._yolo_world = YOLO(yolo_world_weights)
        if open_vocab_classes:
            self._yolo_world.set_classes(open_vocab_classes)
        self._open_vocab_classes = open_vocab_classes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Return fused detections for a single BGR frame.

        Returns:
            List of Detection objects. track_id, pos_3d, depth, yaw, and
            phase-3 fields are left at their default values.
        """
        dets_11 = self._run_yolo11(frame_bgr)
        dets_world = self._run_yolo_world(frame_bgr)

        if not dets_11 and not dets_world:
            return []

        combined = dets_11 + dets_world
        return _nms(combined, self._iou)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_yolo11(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self._yolo11.predict(
            frame_bgr,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                name = r.names[int(boxes.cls[i])]
                if name not in _COCO_KEEP:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(
                    Detection(
                        class_name=name,
                        confidence=float(boxes.conf[i]),
                        bbox=BBox(x1, y1, x2, y2),
                    )
                )
        return detections

    def _run_yolo_world(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self._open_vocab_classes:
            return []
        results = self._yolo_world.predict(
            frame_bgr,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                name = r.names[int(boxes.cls[i])]
                name = _WORLD_NAME_MAP.get(name.lower(), name)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(
                    Detection(
                        class_name=name,
                        confidence=float(boxes.conf[i]),
                        bbox=BBox(x1, y1, x2, y2),
                    )
                )
        return detections

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        del self._yolo11, self._yolo_world
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """Apply class-agnostic NMS across a list of Detection objects.

    Uses torchvision batched_nms so that boxes from the same class compete
    separately, which avoids suppressing truly overlapping objects of the
    same type (e.g. two adjacent cars).
    """
    if not detections:
        return []

    boxes = torch.tensor(
        [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
        dtype=torch.float32,
    )
    scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)

    # Encode class as integer for batched_nms
    class_ids = torch.tensor(
        [hash(d.class_name) & 0xFFFF for d in detections],
        dtype=torch.int64,
    )

    keep = tv_ops.batched_nms(boxes, scores, class_ids, iou_threshold)
    return [detections[i] for i in keep.tolist()]
