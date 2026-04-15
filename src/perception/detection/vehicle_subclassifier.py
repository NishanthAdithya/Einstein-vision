"""vehicle_subclassifier.py — CLIP vehicle sub-type classifier + OCR speed limit reader.

Refines YOLO 'car' detections into: sedan, suv, hatchback, pickup.
Reads speed limit numbers from 'speed limit sign' crops using EasyOCR,
falling back to CLIP if no valid number is found.

Requires: pip install openai-clip easyocr
"""
from __future__ import annotations

import re
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
# Speed limit — valid US values
# ---------------------------------------------------------------------------

_SPEED_LIMITS     = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
_SPEED_LIMITS_SET = set(_SPEED_LIMITS)

_SPEED_PROMPTS = [
    f"a speed limit road sign showing {n} mph" for n in _SPEED_LIMITS
]

# Minimum short side of the sign crop (pixels) for OCR to be reliable
_OCR_MIN_DIM = 20

# Speed limit signs are tall-ish white rectangles; signs where the crop is
# very wide relative to height are likely warning/advisory signs (e.g. SPEED
# HUMP diamond), not round speed limit signs — skip OCR for those.
_MAX_ASPECT_RATIO = 2.0   # width / height — round signs are ~1.0


class VehicleSubclassifier:
    """CLIP classifier for vehicles + EasyOCR reader for speed limit signs.

    One instance handles both tasks.

    Args:
        device: 'cuda' or 'cpu'
    """

    def __init__(self, device: str = "cuda") -> None:
        import clip

        self._device = device
        self._model, self._preprocess = clip.load("ViT-B/32", device=device)
        self._model.eval()

        # Pre-encode text prompts
        vehicle_tokens = clip.tokenize(_VEHICLE_PROMPTS).to(device)
        speed_tokens   = clip.tokenize(_SPEED_PROMPTS).to(device)
        with torch.no_grad():
            self._vehicle_feats = self._model.encode_text(vehicle_tokens)
            self._vehicle_feats /= self._vehicle_feats.norm(dim=-1, keepdim=True)
            self._speed_feats   = self._model.encode_text(speed_tokens)
            self._speed_feats   /= self._speed_feats.norm(dim=-1, keepdim=True)

        # Lazy-init EasyOCR reader (first call triggers model download)
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        return self._ocr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_detections(
        self,
        detections: List[Detection],
        frame_bgr: np.ndarray,
    ) -> List[Detection]:
        """Fill vehicle_subclass and speed_limit fields in-place."""
        from PIL import Image

        h, w = frame_bgr.shape[:2]

        for det in detections:
            if det.class_name not in ("car", "speed limit sign"):
                continue

            x1 = max(0, int(det.bbox.x1))
            y1 = max(0, int(det.bbox.y1))
            x2 = min(w, int(det.bbox.x2))
            y2 = min(h, int(det.bbox.y2))
            bw, bh = x2 - x1, y2 - y1
            if bw < 20 or bh < 20:
                if det.class_name == "car":
                    det.vehicle_subclass = "sedan"
                continue

            crop_bgr = frame_bgr[y1:y2, x1:x2]

            if det.class_name == "car":
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                img_t = self._preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._model.encode_image(img_t)
                    feat /= feat.norm(dim=-1, keepdim=True)
                sim = (feat @ self._vehicle_feats.T).softmax(dim=-1)
                det.vehicle_subclass = _VEHICLE_SUBCLASSES[int(sim.argmax())]

            elif det.class_name == "speed limit sign":
                # Skip obviously non-round signs (diamond/warning shapes)
                if bw / max(bh, 1) > _MAX_ASPECT_RATIO:
                    continue

                limit = self._read_speed_limit_ocr(crop_bgr, bw, bh)
                if limit is not None:
                    det.speed_limit = limit
                else:
                    # Fallback: CLIP similarity
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    img_t = self._preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(self._device)
                    with torch.no_grad():
                        feat = self._model.encode_image(img_t)
                        feat /= feat.norm(dim=-1, keepdim=True)
                    sim = (feat @ self._speed_feats.T).softmax(dim=-1)
                    det.speed_limit = _SPEED_LIMITS[int(sim.argmax())]

        return detections

    # ------------------------------------------------------------------
    # OCR helper
    # ------------------------------------------------------------------

    def _read_speed_limit_ocr(
        self, crop_bgr: np.ndarray, bw: int, bh: int
    ) -> Optional[int]:
        """Run EasyOCR on the crop and extract the first valid speed limit number.

        Pre-processes the crop to maximise OCR accuracy:
          1. Upscale small crops to at least 120px on the short side.
          2. Convert to grayscale and threshold to get black-on-white text.
          3. Run EasyOCR with digit allowlist.
          4. Filter results to known US speed limit values.
        """
        if min(bw, bh) < _OCR_MIN_DIM:
            return None

        # ── pre-process ─────────────────────────────────────────────────
        # Upscale for better OCR
        target = 160
        scale  = target / min(bw, bh)
        if scale > 1.0:
            crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Threshold: Otsu on the upper 2/3 of the crop (where the number lives)
        h2 = max(1, int(gray.shape[0] * 2 / 3))
        _, thresh = cv2.threshold(gray[:h2], 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── OCR ─────────────────────────────────────────────────────────
        ocr = self._get_ocr()
        results = ocr.readtext(thresh, allowlist="0123456789", detail=1)

        # Collect all digit strings, pick one matching a valid speed limit
        candidates: list[int] = []
        for _, text, conf in results:
            text = text.strip()
            if not text.isdigit():
                continue
            val = int(text)
            if val in _SPEED_LIMITS_SET:
                candidates.append((conf, val))

        if not candidates:
            return None

        # Return the highest-confidence match
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        del self._model
        self._ocr = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
