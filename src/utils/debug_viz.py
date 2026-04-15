from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.io.schema import Detection, FrameData, Lane, PoseResult


# ---------------------------------------------------------------------------
# Colour palette — one BGR colour per class, cycling as needed
# ---------------------------------------------------------------------------

_PALETTE = [
    (0,   200, 255),   # orange
    (50,  255,  50),   # green
    (255,  50,  50),   # blue
    (255,  50, 255),   # magenta
    (50,  255, 255),   # yellow
    (50,   50, 255),   # red
    (200, 200,  50),   # cyan-ish
]

_LANE_COLOURS = {
    "solid-white":  (255, 255, 255),
    "dashed-white": (180, 180, 180),
    "solid-yellow": (0,   200, 220),
    "double":       (0,   140, 255),
    "unknown":      (120, 120, 120),
}

# COCO whole-body skeleton connections (pairs of keypoint indices, body only)
_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    min_confidence: float = 0.0,
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame.

    Args:
        frame:          uint8 BGR image.
        detections:     List of Detection objects.
        min_confidence: Skip detections below this threshold.

    Returns:
        A new uint8 BGR image with overlays drawn.
    """
    out = frame.copy()
    for det in detections:
        if det.confidence < min_confidence:
            continue

        colour = _class_colour(det.class_name)
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox.as_xyxy())
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label_parts = [det.class_name]
        if det.track_id is not None:
            label_parts.append(f"#{det.track_id}")
        label_parts.append(f"{det.confidence:.2f}")
        if det.depth > 0:
            label_parts.append(f"{det.depth:.1f}m")
        label = "  ".join(label_parts)

        _put_label(out, label, x1, y1, colour)

    return out


def draw_lanes(
    frame: np.ndarray,
    lanes: List[Lane],
    thickness: int = 3,
) -> np.ndarray:
    """Draw lane pixel overlays and polynomial curves on a copy of the frame.

    Args:
        frame:     uint8 BGR image.
        lanes:     List of Lane objects with pixels_2d populated.
        thickness: Line thickness in pixels.

    Returns:
        A new uint8 BGR image with lane overlays.
    """
    out = frame.copy()
    for lane in lanes:
        colour = _LANE_COLOURS.get(lane.lane_type, _LANE_COLOURS["unknown"])

        # Draw sampled pixels as small dots
        if lane.pixels_2d is not None and len(lane.pixels_2d) > 0:
            for pt in lane.pixels_2d[::5]:   # subsample for speed
                cx, cy = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(out, (cx, cy), 2, colour, -1)

        # Draw polynomial curve
        if lane.poly_coeffs is not None and lane.pixels_2d is not None and len(lane.pixels_2d) > 1:
            ys = lane.pixels_2d[:, 1]
            y_min, y_max = int(ys.min()), int(ys.max())
            y_pts = np.linspace(y_min, y_max, 50)
            x_pts = np.polyval(lane.poly_coeffs, y_pts)
            pts = np.stack([x_pts, y_pts], axis=1).astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), colour, thickness)

    return out


def draw_depth(
    depth_map: np.ndarray,
    max_depth: float = 80.0,
    colormap: int = cv2.COLORMAP_PLASMA,
) -> np.ndarray:
    """Convert a float32 metric depth map to a colourised uint8 BGR image.

    Args:
        depth_map: float32 (H, W) depth in metres.
        max_depth: Depths beyond this are clipped (controls colour range).
        colormap:  OpenCV colormap constant.

    Returns:
        uint8 BGR image suitable for display or saving.
    """
    clipped = np.clip(depth_map, 0.0, max_depth)
    normalised = (clipped / max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(normalised, colormap)


def draw_poses(
    frame: np.ndarray,
    poses: List[PoseResult],
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Draw body keypoints and skeleton connections on a copy of the frame.

    Only the first 17 COCO body keypoints are drawn (body skeleton).
    Face and hand keypoints are omitted to avoid visual clutter.

    Args:
        frame:          uint8 BGR image.
        poses:          List of PoseResult objects.
        conf_threshold: Skip keypoints below this confidence.

    Returns:
        A new uint8 BGR image with pose overlays.
    """
    out = frame.copy()
    for pose in poses:
        kpts = pose.keypoints  # (133, 3): x, y, conf
        body = kpts[:17]       # first 17 are COCO body keypoints

        # Draw skeleton connections
        for i, j in _SKELETON_PAIRS:
            if body[i, 2] >= conf_threshold and body[j, 2] >= conf_threshold:
                p1 = (int(body[i, 0]), int(body[i, 1]))
                p2 = (int(body[j, 0]), int(body[j, 1]))
                cv2.line(out, p1, p2, (50, 255, 50), 2)

        # Draw keypoint dots
        for kpt in body:
            if kpt[2] >= conf_threshold:
                cx, cy = int(kpt[0]), int(kpt[1])
                cv2.circle(out, (cx, cy), 4, (0, 200, 255), -1)

    return out


def draw_frame_data(frame: np.ndarray, fd: FrameData) -> np.ndarray:
    """Convenience wrapper — draws detections, lanes, and poses in one call."""
    out = draw_detections(frame, fd.detections)
    out = draw_lanes(out, fd.lanes)
    out = draw_poses(out, fd.poses)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_colour(class_name: str) -> tuple:
    idx = hash(class_name) % len(_PALETTE)
    return _PALETTE[idx]


def _put_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """Draw a filled-background label above the point (x, y)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y0 = max(y - 4, th + baseline)
    cv2.rectangle(img, (x, y0 - th - baseline), (x + tw, y0 + baseline), colour, -1)
    cv2.putText(img, text, (x, y0), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
