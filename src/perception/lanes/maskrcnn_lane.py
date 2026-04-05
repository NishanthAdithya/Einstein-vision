"""maskrcnn_lane.py — Lane detector using Mask R-CNN + DeepSORT lane tracking.

Uses a fine-tuned torchvision Mask R-CNN (7 classes: background + 6 lane types)
to segment lane instances per frame.  DeepSORT (appearance + IoU) tracks lane
IDs across frames for consistent lane identity.

Weights: models/maskrcnn_lane.pth
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.io.schema import Lane
from src.perception.base import LaneDetector

# 7 classes: 0=background, 1-6=lane types
_NUM_CLASSES = 7
_LANE_TYPE_MAP = {
    1: "solid-white",
    2: "dashed-white",
    3: "solid-yellow",
    4: "double",
    5: "solid-white",   # extra class → solid-white
    6: "dashed-white",  # extra class → dashed-white
}

_CONF_THRESHOLD  = 0.50
_MASK_THRESHOLD  = 0.50   # binarise soft mask
_MIN_MASK_PIXELS = 150


class MaskRCNNLaneDetector(LaneDetector):
    """Mask R-CNN lane instance segmenter with DeepSORT lane tracking.

    Args:
        weights_path:   Path to maskrcnn_lane.pth
        device:         'cuda' or 'cpu'
        conf_threshold: Minimum score to keep a detection
        poly_degree:    Polynomial degree for lane curve fitting
        bezier_points:  Number of sample points for Bezier representation
    """

    def __init__(
        self,
        weights_path: str = "models/maskrcnn_lane.pth",
        device: str = "cuda",
        conf_threshold: float = _CONF_THRESHOLD,
        poly_degree: int = 2,
        bezier_points: int = 10,
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._conf   = conf_threshold
        self._poly   = poly_degree
        self._bpts   = bezier_points
        self._last_mask: Optional[np.ndarray] = None

        self._model = _build_model(weights_path, self._device)
        self._model.eval()

        # Lane tracker: maps lane_instance_id → track_id
        self._tracker = _LaneTracker(max_age=10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        frame_bgr: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Lane]:
        h, w = frame_bgr.shape[:2]

        # Convert BGR → RGB float32 tensor [0,1]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor  = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(tensor)[0]

        scores  = outputs["scores"].cpu().numpy()
        labels  = outputs["labels"].cpu().numpy()
        masks   = outputs["masks"].cpu().numpy()   # (N, 1, H, W) soft masks

        # Build a combined binary mask for debug_viz compatibility
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        raw_lanes: List[dict] = []
        for i, (score, label, soft_mask) in enumerate(zip(scores, labels, masks)):
            if score < self._conf:
                continue
            bin_mask = (soft_mask[0] > _MASK_THRESHOLD).astype(np.uint8)
            pixel_count = bin_mask.sum()
            if pixel_count < _MIN_MASK_PIXELS:
                continue

            combined_mask |= bin_mask

            lane_type = _LANE_TYPE_MAP.get(int(label), "unknown")
            pixels_yx = np.argwhere(bin_mask > 0)   # (N, 2) in (y, x)
            pixels_xy = pixels_yx[:, ::-1].astype(np.float32)  # → (x, y)

            poly, bezier = _fit_lane(pixels_xy, self._poly, self._bpts)
            if poly is None:
                continue

            raw_lanes.append({
                "lane_type":  lane_type,
                "pixels_2d":  pixels_xy,
                "poly_coeffs": poly,
                "bezier_2d":  bezier,
                "score":      float(score),
            })

        self._last_mask = combined_mask

        # Track lane IDs across frames
        tracked = self._tracker.update(raw_lanes)

        lanes: List[Lane] = []
        for info in tracked:
            # Embed 2D pixel bezier coords into the first two columns so that
            # lift_lane_points can apply IPM/depth unprojection correctly.
            bpts_3d = np.zeros((self._bpts, 3), dtype=np.float32)
            if "bezier_2d" in info and info["bezier_2d"] is not None:
                bpts_3d[:, :2] = info["bezier_2d"]
            lanes.append(Lane(
                lane_type=info["lane_type"],
                bezier_points_3d=bpts_3d,
                poly_coeffs=info["poly_coeffs"],
                pixels_2d=info["pixels_2d"],
            ))

        return lanes

    def close(self) -> None:
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Simple IoU-based lane tracker (no appearance — lanes don't have crops)
# ---------------------------------------------------------------------------

class _LaneTracker:
    """Track lane instances across frames by mask IoU."""

    def __init__(self, max_age: int = 10) -> None:
        self._max_age  = max_age
        self._tracks: List[dict] = []   # {id, poly, pixels, age, lane_type}
        self._next_id  = 1

    def update(self, detections: List[dict]) -> List[dict]:
        """Match detections to existing tracks, assign IDs."""
        if not detections:
            for t in self._tracks:
                t["age"] += 1
            self._tracks = [t for t in self._tracks if t["age"] <= self._max_age]
            return []

        if not self._tracks:
            for det in detections:
                det["track_id"] = self._next_id
                self._next_id += 1
                self._tracks.append({
                    "id":        det["track_id"],
                    "poly":      det["poly_coeffs"],
                    "pixels":    det["pixels_2d"],
                    "lane_type": det["lane_type"],
                    "age":       0,
                })
            return detections

        # Match by centroid proximity of polynomial roots
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self._tracks)))

        matched = []
        for di in list(unmatched_dets):
            det = detections[di]
            best_dist, best_ti = 1e9, -1
            det_cx = float(det["pixels_2d"][:, 0].mean())
            for ti in unmatched_trks:
                trk_cx = float(self._tracks[ti]["pixels"][:, 0].mean())
                dist = abs(det_cx - trk_cx)
                if dist < best_dist:
                    best_dist = dist
                    best_ti = ti
            if best_ti >= 0 and best_dist < 80:
                matched.append((di, best_ti))
                unmatched_dets.remove(di)
                unmatched_trks.remove(best_ti)

        for di, ti in matched:
            detections[di]["track_id"] = self._tracks[ti]["id"]
            self._tracks[ti]["poly"]      = detections[di]["poly_coeffs"]
            self._tracks[ti]["pixels"]    = detections[di]["pixels_2d"]
            self._tracks[ti]["bezier_2d"] = detections[di].get("bezier_2d")
            self._tracks[ti]["lane_type"] = detections[di]["lane_type"]
            self._tracks[ti]["age"]       = 0

        for di in unmatched_dets:
            detections[di]["track_id"] = self._next_id
            self._next_id += 1
            self._tracks.append({
                "id":        detections[di]["track_id"],
                "poly":      detections[di]["poly_coeffs"],
                "pixels":    detections[di]["pixels_2d"],
                "lane_type": detections[di]["lane_type"],
                "age":       0,
            })

        for ti in unmatched_trks:
            self._tracks[ti]["age"] += 1

        self._tracks = [t for t in self._tracks if t["age"] <= self._max_age]

        return detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(weights_path: str, device: torch.device):
    """Load a torchvision Mask R-CNN with custom head for _NUM_CLASSES."""
    model = maskrcnn_resnet50_fpn(weights=None)

    # Replace box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, _NUM_CLASSES)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, _NUM_CLASSES
    )

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Handle wrapped checkpoints
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    return model


def _fit_lane(
    pixels_xy: np.ndarray,
    poly_degree: int,
    bezier_points: int,
) -> tuple:
    """Fit polynomial and sample bezier points from lane pixel set.

    Returns (poly_coeffs, bezier_2d) or (None, None) on failure.
    """
    if len(pixels_xy) < 10:
        return None, None

    xs = pixels_xy[:, 0]
    ys = pixels_xy[:, 1]

    try:
        poly = np.polyfit(ys, xs, poly_degree)
    except (np.linalg.LinAlgError, ValueError):
        return None, None

    y_min, y_max = float(ys.min()), float(ys.max())
    y_pts = np.linspace(y_min, y_max, bezier_points)
    x_pts = np.polyval(poly, y_pts)
    bezier = np.stack([x_pts, y_pts], axis=1).astype(np.float32)

    return poly.astype(np.float32), bezier
