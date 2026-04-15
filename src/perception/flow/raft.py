from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.perception.base import FlowEstimator


# Map config weight names to torchvision weight enum values
_WEIGHTS_MAP = {
    "raft-things": Raft_Large_Weights.C_T_SKHT_V2,
    "default":     Raft_Large_Weights.DEFAULT,
}

# RAFT requires spatial dimensions divisible by this
_PAD_DIVISOR = 8


class RAFTFlowEstimator(FlowEstimator):
    """Dense optical flow estimator backed by RAFT (torchvision).

    Uses the large RAFT variant pretrained on FlyingChairs + FlyingThings
    (raft-things).  Weights are downloaded automatically from torchvision
    on first use and cached in ~/.cache/torch.

    Args:
        weights:  Weight preset name — 'raft-things' or 'default'
        iters:    Number of recurrent update iterations (higher = more accurate)
        device:   'cuda' or 'cpu'
    """

    def __init__(
        self,
        weights: str = "raft-things",
        iters: int = 20,
        device: str = "cuda",
    ) -> None:
        self._iters = iters
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        weight_enum = _WEIGHTS_MAP.get(weights)
        if weight_enum is None:
            raise ValueError(
                f"Unknown RAFT weight preset '{weights}'. "
                f"Valid options: {list(_WEIGHTS_MAP)}"
            )

        self._transforms = weight_enum.transforms()
        self._model = (
            raft_large(weights=weight_enum, progress=True)
            .to(self._device)
            .eval()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        frame_prev_bgr: np.ndarray,
        frame_curr_bgr: np.ndarray,
    ) -> np.ndarray:
        """Compute dense optical flow from frame_prev to frame_curr.

        Args:
            frame_prev_bgr: uint8 BGR image (H × W × 3)
            frame_curr_bgr: uint8 BGR image, same shape as frame_prev_bgr

        Returns:
            float32 flow field of shape (H, W, 2).
            Channel 0 = horizontal displacement (dx), channel 1 = vertical (dy),
            both in pixels per frame.
        """
        h, w = frame_prev_bgr.shape[:2]

        t1, t2 = self._preprocess(frame_prev_bgr, frame_curr_bgr)

        with torch.no_grad():
            flow_list = self._model(t1, t2, num_flow_updates=self._iters)

        # flow_list is a list of progressively refined predictions; take the last
        flow = flow_list[-1]  # (1, 2, H_pad, W_pad)

        # Crop away any padding, move to CPU
        flow = flow[:, :, :h, :w]  # (1, 2, H, W)
        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        return flow_np.astype(np.float32)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        del self._model
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        frame_prev_bgr: np.ndarray,
        frame_curr_bgr: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert BGR frames to padded, normalised tensors expected by RAFT.

        Steps:
          1. BGR → RGB
          2. HWC uint8 → CHW float32 in [0, 1]
          3. Apply torchvision weight transforms (normalise to [-1, 1])
          4. Pad spatial dims to the nearest multiple of _PAD_DIVISOR

        Returns:
            Two (1, 3, H_pad, W_pad) tensors on self._device.
        """
        t1 = _bgr_to_tensor(frame_prev_bgr)
        t2 = _bgr_to_tensor(frame_curr_bgr)

        # torchvision transforms expect (C, H, W) in [0, 1] — output is same range
        t1, t2 = self._transforms(t1, t2)

        t1 = _pad_to_divisor(t1, _PAD_DIVISOR).unsqueeze(0).to(self._device)
        t2 = _pad_to_divisor(t2, _PAD_DIVISOR).unsqueeze(0).to(self._device)

        return t1, t2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert a uint8 BGR numpy array to a float32 (3, H, W) tensor in [0, 1]."""
    rgb = frame_bgr[:, :, ::-1].copy()  # BGR → RGB
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return tensor


def _pad_to_divisor(tensor: torch.Tensor, divisor: int) -> torch.Tensor:
    """Pad H and W dimensions (right/bottom) to the nearest multiple of divisor."""
    _, h, w = tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return tensor
    # F.pad expects (left, right, top, bottom) in reverse dim order
    return F.pad(tensor, (0, pad_w, 0, pad_h))
