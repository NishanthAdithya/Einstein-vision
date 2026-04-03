from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.io.schema import Lane
from src.perception.base import LaneDetector


# Input resolution expected by YOLOPv2
_MODEL_SIZE = 640

# Sigmoid threshold for the single-channel lane head output.
# The TorchScript export produces values that cluster at 0.5000 (background)
# with genuine lane predictions above ~0.52.  Using 0.5 floods the mask.
_LANE_SIGMOID_THRESH = 0.52


class YOLOPv2LaneDetector(LaneDetector):
    """Lane line detector backed by YOLOPv2.

    YOLOPv2 is a multi-task network that simultaneously predicts object
    detections, drivable area, and lane line segmentation masks.  We only
    use the lane segmentation head here.

    The model is loaded from a local weights file.  Download yolopv2.pt from
    the CAIC-AD/YOLOPv2 GitHub release page and pass its path as
    ``weights_path``.  If not provided it defaults to
    ``weights/yolopv2.pt`` relative to the repo root.

    Args:
        weights_path:         Path to yolopv2.pt
        min_lane_pixels:      Minimum connected-component size to keep (px²)
        poly_degree:          Degree of the polynomial fit (should be 2)
        bezier_sample_points: Number of points sampled from each poly curve
        device:               'cuda' or 'cpu'
    """

    def __init__(
        self,
        weights_path: str = "weights/yolopv2.pt",
        min_lane_pixels: int = 200,
        poly_degree: int = 2,
        bezier_sample_points: int = 10,
        device: str = "cuda",
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._min_pixels = min_lane_pixels
        self._poly_degree = poly_degree
        self._n_bezier = bezier_sample_points

        self._model = _load_model(weights_path, self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        frame_bgr: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Lane]:
        """Detect lane lines in a single BGR frame.

        bezier_points_3d is populated when depth_map is provided;
        otherwise it is left as zeros and filled by the localization module.

        Args:
            frame_bgr:  uint8 BGR image.
            depth_map:  Optional float32 (H, W) metric depth map in metres.

        Returns:
            List of Lane objects, one per detected lane line.
        """
        h, w = frame_bgr.shape[:2]

        tensor, (pad_top, pad_left, scale) = _preprocess(frame_bgr, _MODEL_SIZE, self._device)

        with torch.no_grad():
            _, _, ll_out = self._model(tensor)

        lane_mask = _decode_lane_mask(ll_out, h, w, pad_top, pad_left, scale)
        self._last_mask = lane_mask  # exposed for classify_lane_types in run_perception
        return _extract_lanes(lane_mask, depth_map, self._min_pixels, self._poly_degree, self._n_bezier)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load YOLOPv2 from a local .pt file.

    Handles two formats:
      - TorchScript archive (torch.jit.save): loaded with torch.jit.load
      - Regular checkpoint dict with 'model' key: loaded with torch.load
    """
    import pathlib
    path = pathlib.Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"YOLOPv2 weights not found: {path}\n"
            "Download yolopv2.pt from the CAIC-AD/YOLOPv2 GitHub releases page\n"
            "and place it at weights/yolopv2.pt (or set weights_path in models.yaml)."
        )
    # TorchScript archives are zip files beginning with PK
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        model = torch.jit.load(str(path), map_location=device)
    else:
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        model = ckpt["model"].float().fuse()
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Pre- and post-processing
# ---------------------------------------------------------------------------

def _preprocess(
    frame_bgr: np.ndarray,
    target_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int, float]]:
    """Letterbox-resize to target_size × target_size and normalise to [0, 1].

    Returns:
        tensor: (1, 3, target_size, target_size) float32 RGB tensor
        (pad_top, pad_left, scale): metadata needed to invert the transform
    """
    h, w = frame_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    rgb = canvas[:, :, ::-1].copy()  # BGR → RGB
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor, (pad_top, pad_left, scale)


def _decode_lane_mask(
    ll_out: torch.Tensor,
    orig_h: int,
    orig_w: int,
    pad_top: int,
    pad_left: int,
    scale: float,
) -> np.ndarray:
    """Convert the raw lane-line head output to a binary mask at original resolution.

    Handles two output formats:
      - (1, 2, H', W'): two-class logits — take argmax over class dim
      - (1, 1, H', W'): single-channel sigmoid — threshold at 0.5

    Returns:
        uint8 binary mask of shape (orig_h, orig_w), values 0 or 1.
    """
    if ll_out.shape[1] == 2:
        mask = torch.argmax(ll_out, dim=1).squeeze(0).float()  # (H', W')
    else:
        mask = (torch.sigmoid(ll_out.squeeze(1).squeeze(0)) > _LANE_SIGMOID_THRESH).float()  # (H', W')

    # Upsample to full model input size
    mask_full = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(_MODEL_SIZE, _MODEL_SIZE),
        mode="nearest",
    ).squeeze().byte()

    mask_np = mask_full.cpu().numpy()

    # Crop out the letterbox padding
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    cropped = mask_np[pad_top : pad_top + new_h, pad_left : pad_left + new_w]

    # Resize back to original resolution
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Lane line extraction
# ---------------------------------------------------------------------------

def _extract_lanes(
    lane_mask: np.ndarray,
    depth_map: Optional[np.ndarray],
    min_pixels: int,
    poly_degree: int,
    n_bezier: int,
) -> List[Lane]:
    """Find individual lane lines in a binary mask and fit polynomials.

    Each connected component becomes one Lane object.  The polynomial is
    fit as x = f(y) so that vertical lanes are handled without singularities.

    bezier_points_3d is set to zeros unless a depth_map is provided, in
    which case each sampled pixel is back-projected using the depth value.
    Full 3-D lifting using camera intrinsics is done by the localization
    module — here we just store (col, row, depth) as a placeholder.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        lane_mask, connectivity=8
    )

    lanes: List[Lane] = []

    for label in range(1, n_labels):  # skip background (0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_pixels:
            continue

        ys, xs = np.where(labels == label)  # row, col pixel coords

        pixels_2d = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2)

        poly_coeffs = _fit_poly(xs, ys, poly_degree)

        bezier_pts = _sample_bezier(poly_coeffs, ys, n_bezier)  # (n_bezier, 2)

        if depth_map is not None:
            points_3d = _lift_to_3d(bezier_pts, depth_map)
        else:
            points_3d = np.zeros((n_bezier, 3), dtype=np.float32)

        lanes.append(
            Lane(
                lane_type="unknown",  # colour classification handled separately
                bezier_points_3d=points_3d,
                poly_coeffs=poly_coeffs,
                pixels_2d=pixels_2d,
            )
        )

    return lanes


def _fit_poly(xs: np.ndarray, ys: np.ndarray, degree: int) -> np.ndarray:
    """Fit x = a*y^2 + b*y + c and return coefficients [a, b, c].

    Falls back to a flat line if fitting fails.
    """
    try:
        coeffs = np.polyfit(ys, xs, degree).astype(np.float32)
    except (np.linalg.LinAlgError, ValueError):
        coeffs = np.zeros(degree + 1, dtype=np.float32)
    return coeffs  # shape (degree+1,) = (3,) for degree=2


def _sample_bezier(
    poly_coeffs: np.ndarray,
    ys: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Sample n_points evenly along the polynomial x = f(y).

    Returns:
        float32 array of shape (n_points, 2) as (x, y) pixel coordinates.
    """
    y_min, y_max = float(ys.min()), float(ys.max())
    y_sample = np.linspace(y_min, y_max, n_points, dtype=np.float32)
    x_sample = np.polyval(poly_coeffs, y_sample).astype(np.float32)
    return np.stack([x_sample, y_sample], axis=1)  # (n_points, 2)


def _lift_to_3d(pixels_xy: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """Sample depth at each pixel and store (col, row, depth) as a placeholder.

    Real camera-frame lifting is done by localization/projector.py using the
    camera intrinsics.  This gives the localization module enough info to
    complete the projection without re-running lane detection.

    Args:
        pixels_xy:  (N, 2) float32 array of (x, y) = (col, row) pixel coords
        depth_map:  (H, W) float32 depth map in metres

    Returns:
        (N, 3) float32 array of (col, row, depth_metres)
    """
    h, w = depth_map.shape[:2]
    cols = np.clip(pixels_xy[:, 0].round().astype(np.int32), 0, w - 1)
    rows = np.clip(pixels_xy[:, 1].round().astype(np.int32), 0, h - 1)
    depths = depth_map[rows, cols].astype(np.float32)
    return np.stack([pixels_xy[:, 0], pixels_xy[:, 1], depths], axis=1)
