from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.perception.base import DepthEstimator


class DepthAnythingV2Estimator(DepthEstimator):
    """Depth Anything V2 Metric Outdoor (Large) depth estimator.

    Produces metric depth maps in metres from a single BGR frame.
    Uses the HuggingFace transformers interface.

    Args:
        model_id: HuggingFace model ID, e.g.
                  'depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf'
        device:   'cuda' or 'cpu'
    """

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(model_id)
            .to(self._device)
            .eval()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 metric depth map (H×W) in metres.

        Args:
            frame_bgr: uint8 BGR image as read by OpenCV.

        Returns:
            float32 ndarray of shape (H, W) with depth in metres.
        """
        h, w = frame_bgr.shape[:2]
        pil = _bgr_to_pil(frame_bgr)

        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            depth_logits = self._model(**inputs).predicted_depth  # (1, H', W')

        depth = F.interpolate(
            depth_logits.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )  # (1, 1, H, W)

        return depth.squeeze().cpu().numpy().astype(np.float32)

    def predict_batch(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """Run depth estimation on a batch of BGR frames.

        More efficient than calling predict() in a loop because the model
        forward pass is batched on the GPU.

        Args:
            frames_bgr: List of uint8 BGR images (must all be the same H×W).

        Returns:
            List of float32 depth maps (H×W) in metres, one per input frame.
        """
        if not frames_bgr:
            return []

        h, w = frames_bgr[0].shape[:2]
        pils = [_bgr_to_pil(f) for f in frames_bgr]

        inputs = self._processor(images=pils, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            depth_logits = self._model(**inputs).predicted_depth  # (B, H', W')

        depth = F.interpolate(
            depth_logits.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )  # (B, 1, H, W)

        depth_np = depth.squeeze(1).cpu().numpy().astype(np.float32)  # (B, H, W)
        return [depth_np[i] for i in range(len(frames_bgr))]

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

def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_bgr[:, :, ::-1])
