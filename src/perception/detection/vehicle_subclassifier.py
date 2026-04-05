"""vehicle_subclassifier.py — CLIP zero-shot vehicle sub-type classifier.

Refines YOLO 'car' detections into: sedan, suv, hatchback, pickup.
Also classifies 'speed limit sign' detections into a numeric value.

Requires: pip install openai-clip
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch

from src.io.schema import Detection

# ---------------------------------------------------------------------------
# Vehicle sub-classification
# ---------------------------------------------------------------------------

_VEHICLE_SUBCLASSES = ["sedan", "suv", "hatchback", "pickup"]

_VEHICLE_PROMPTS = [
    "a photo of a sedan car with a separate trunk",
    "a photo of an SUV or crossover with a tall boxy body",
    "a photo of a hatchback car with a rear door above the bumper",
    "a photo of a pickup truck with an open cargo bed",
]

# ---------------------------------------------------------------------------
# Speed limit classification
# ---------------------------------------------------------------------------

_SPEED_LIMITS = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

_SPEED_PROMPTS = [
    f"a speed limit road sign showing {n} mph" for n in _SPEED_LIMITS
]


class VehicleSubclassifier:
    """CLIP-based classifier for vehicle sub-types and speed limit signs.

    One model instance handles both tasks to avoid loading CLIP twice.

    Args:
        device: 'cuda' or 'cpu'
    """

    def __init__(self, device: str = "cuda") -> None:
        import clip  # openai-clip

        self._device = device
        self._model, self._preprocess = clip.load("ViT-B/32", device=device)
        self._model.eval()

        # Pre-encode all text prompts once
        vehicle_tokens = clip.tokenize(_VEHICLE_PROMPTS).to(device)
        speed_tokens   = clip.tokenize(_SPEED_PROMPTS).to(device)
        with torch.no_grad():
            self._vehicle_feats = self._model.encode_text(vehicle_tokens)
            self._vehicle_feats /= self._vehicle_feats.norm(dim=-1, keepdim=True)
            self._speed_feats   = self._model.encode_text(speed_tokens)
            self._speed_feats   /= self._speed_feats.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_detections(
        self,
        detections: List[Detection],
        frame_bgr: np.ndarray,
    ) -> List[Detection]:
        """Fill vehicle_subclass and speed_limit fields in-place.

        Processes every 'car' detection (sets vehicle_subclass) and every
        'speed limit sign' detection (sets speed_limit).  All other
        detections are skipped.

        Args:
            detections: Detections from the object detector + tracker.
            frame_bgr:  The full BGR frame the detections were made on.

        Returns:
            The same list with relevant fields populated.
        """
        from PIL import Image

        h, w = frame_bgr.shape[:2]

        for det in detections:
            if det.class_name not in ("car", "speed limit sign"):
                continue

            # Crop and validate
            x1 = max(0, int(det.bbox.x1))
            y1 = max(0, int(det.bbox.y1))
            x2 = min(w, int(det.bbox.x2))
            y2 = min(h, int(det.bbox.y2))
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                # Crop too small — use defaults
                if det.class_name == "car":
                    det.vehicle_subclass = "sedan"
                continue

            crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(crop_rgb)
            img_t = self._preprocess(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                img_feat = self._model.encode_image(img_t)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

            if det.class_name == "car":
                sim = (img_feat @ self._vehicle_feats.T).softmax(dim=-1)
                idx = int(sim.argmax())
                det.vehicle_subclass = _VEHICLE_SUBCLASSES[idx]

            elif det.class_name == "speed limit sign":
                sim = (img_feat @ self._speed_feats.T).softmax(dim=-1)
                idx = int(sim.argmax())
                det.speed_limit = _SPEED_LIMITS[idx]

        return detections

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
