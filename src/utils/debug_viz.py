from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.io.schema import Detection, FrameData, Lane, PoseResult


# ---------------------------------------------------------------------------
# 3-D bounding box constants
# ---------------------------------------------------------------------------

# Typical (length, width, height) in metres per class
_CLASS_3D_DIMS = {
    # Vehicles
    "car":              (4.5, 1.8, 1.5),
    "sedan":            (4.5, 1.8, 1.5),
    "hatchback":        (4.2, 1.7, 1.4),
    "suv":              (4.8, 1.9, 1.8),
    "pickup":           (5.2, 1.9, 1.7),
    "truck":            (8.0, 2.5, 3.5),
    "bus":              (12.0, 2.5, 3.2),
    "motorcycle":       (2.2, 0.8, 1.2),
    "bicycle":          (1.8, 0.6, 1.0),
    # People
    "person":           (0.5, 0.4, 1.75),
    # Traffic infrastructure
    "traffic light":    (0.5, 0.4, 3.9),
    "traffic cone":     (0.4, 0.4, 0.7),
    "traffic cylinder": (0.4, 0.4, 1.0),
    "traffic pole":     (0.15, 0.15, 3.0),
    "dustbin":          (0.5, 0.5, 1.0),
    "barrel":           (0.6, 0.6, 0.9),
    "stop sign":        (0.6, 0.1, 2.5),
    "speed limit sign": (0.5, 0.1, 2.5),
    "fire hydrant":     (0.4, 0.4, 0.6),
    "speed bump":       (3.5, 0.6, 0.12),
}

# Bottom face: 0-3, top face: 4-7 (corners in local box frame)
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]

# Classes that can move — flow arrows are only drawn for these
_FLOW_CLASSES = {
    "car", "sedan", "hatchback", "suv", "pickup",
    "truck", "bus", "motorcycle", "bicycle", "person",
}

# Minimum flow magnitude (pixels/frame) to draw an arrow on a vehicle
_FLOW_MIN_MAG = 1.5


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
        if det.collision_risk:
            colour = (0, 0, 255)   # bright red for collision risk
        elif det.traffic_light_state == "red":
            colour = (0, 0, 220)
        elif det.traffic_light_state == "yellow":
            colour = (0, 200, 220)
        elif det.traffic_light_state == "green":
            colour = (0, 200, 50)
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox.as_xyxy())
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label_parts = [det.class_name]
        if det.track_id is not None:
            label_parts.append(f"#{det.track_id}")
        label_parts.append(f"{det.confidence:.2f}")
        if det.depth > 0:
            label_parts.append(f"{det.depth:.1f}m")
        if det.traffic_light_state:
            label_parts.append(det.traffic_light_state.upper())
        if det.traffic_light_arrow:
            label_parts.append(det.traffic_light_arrow.upper())
        if det.speed_limit is not None:
            label_parts.append(f"{det.speed_limit}mph")
        # Phase 3 annotations
        if det.brake_light_on:
            label_parts.append("BRAKE")
        if det.turn_signal and det.turn_signal != "none":
            label_parts.append(f"TURN-{det.turn_signal.upper()}")
        if det.is_moving is not None:
            label_parts.append("MOV" if det.is_moving else "PARK")
        if det.collision_risk:
            label_parts.append("COLLISION!")
        label = "  ".join(label_parts)

        _put_label(out, label, x1, y1, colour)

        # Draw directional arrow inside the active-colour strip of the bbox
        if det.traffic_light_arrow:
            _strip = {"red": 0, "yellow": 1, "green": 2}
            strip_idx = _strip.get(det.traffic_light_state, 1)
            bh = y2 - y1
            strip_cy = y1 + int(bh * (strip_idx + 0.5) / 3)
            _draw_arrow_on_box(out, x1, y1, x2, y2, det.traffic_light_arrow,
                               colour, cy_override=strip_cy)

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


def draw_3d_boxes(
    frame: np.ndarray,
    detections: List[Detection],
    camera,
) -> np.ndarray:
    """Project and draw 3-D wireframe bounding boxes for vehicle detections.

    Uses each detection's pos_3d and yaw together with typical class dimensions
    to construct the 8 corners of a 3-D box in the ego frame, then projects
    them to image coordinates using the camera model.

    Args:
        frame:      uint8 BGR image.
        detections: List of Detection objects (pos_3d + yaw must be set).
        camera:     Camera object with ego_to_image() method.

    Returns:
        New uint8 BGR image with 3-D box wireframes drawn.
    """
    out = frame.copy()
    for det in detections:
        if det.pos_3d is None or det.class_name not in _CLASS_3D_DIMS:
            continue
        dims = _CLASS_3D_DIMS[det.class_name]
        corners = _box_corners_ego(det.pos_3d, det.yaw, dims)
        uvs = camera.ego_to_image(corners)   # (8, 2)

        colour = _class_colour(det.class_name)
        for i, j in _BOX_EDGES:
            if np.isnan(uvs[i]).any() or np.isnan(uvs[j]).any():
                continue
            p1 = (int(np.clip(uvs[i, 0], -10000, 10000)),
                  int(np.clip(uvs[i, 1], -10000, 10000)))
            p2 = (int(np.clip(uvs[j, 0], -10000, 10000)),
                  int(np.clip(uvs[j, 1], -10000, 10000)))
            cv2.line(out, p1, p2, colour, 1)
    return out


def draw_flow_arrows(
    frame: np.ndarray,
    detections: List[Detection],
    flow: np.ndarray,
    scale: float = 4.0,
) -> np.ndarray:
    """Draw motion arrows on vehicle detections using an optical flow field.

    For each vehicle detection the mean flow vector inside its bounding box
    is computed.  If the magnitude exceeds _FLOW_MIN_MAG, an arrow is drawn
    from the bbox centre in the direction of motion.

    Args:
        frame:      uint8 BGR image.
        detections: List of Detection objects.
        flow:       (H, W, 2) float32 flow field (dx, dy in pixels/frame).
        scale:      Arrow length multiplier for visibility.

    Returns:
        New uint8 BGR image with flow arrows drawn.
    """
    out = frame.copy()
    h_img, w_img = frame.shape[:2]

    for det in detections:
        if det.class_name not in _FLOW_CLASSES:
            continue
        x1 = max(0, int(det.bbox.x1))
        y1 = max(0, int(det.bbox.y1))
        x2 = min(w_img, int(det.bbox.x2))
        y2 = min(h_img, int(det.bbox.y2))
        if x2 <= x1 or y2 <= y1:
            continue

        patch = flow[y1:y2, x1:x2]
        dx = float(np.median(patch[:, :, 0]))
        dy = float(np.median(patch[:, :, 1]))
        mag = (dx ** 2 + dy ** 2) ** 0.5

        if mag < _FLOW_MIN_MAG:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ex = int(cx + dx * scale)
        ey = int(cy + dy * scale)
        colour = _class_colour(det.class_name)
        cv2.arrowedLine(out, (cx, cy), (ex, ey), colour, 2, tipLength=0.3)

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


def _draw_arrow_on_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    arrow: str,
    colour: tuple,
    cy_override: int | None = None,
) -> None:
    """Draw a bold directional arrow centred inside a bounding box."""
    cx = (x1 + x2) // 2
    cy = cy_override if cy_override is not None else (y1 + y2) // 2
    r  = max(8, min((x2 - x1), (y2 - y1)) // 3)   # arrow half-length

    _DIR = {
        "straight": (cx,      cy + r, cx,      cy - r),   # tail, tip
        "left":     (cx + r,  cy,     cx - r,  cy),
        "right":    (cx - r,  cy,     cx + r,  cy),
        "uturn":    (cx,      cy - r, cx,      cy + r),
    }
    coords = _DIR.get(arrow)
    if coords is None:
        return
    tx, ty, hx, hy = coords
    cv2.arrowedLine(img, (tx, ty), (hx, hy), colour, 2, tipLength=0.4)


def _box_corners_ego(
    pos_3d: tuple,
    yaw: float,
    dims: tuple,
) -> np.ndarray:
    """Return (8, 3) array of 3-D box corners in the ego frame."""
    l, w, h = dims
    # Box is centred on pos_3d (depth estimation gives the bbox-centre 3D point).
    # Z corners span [-h/2, +h/2] so the box is centred at pos_3d height.
    local = np.array([
        [ w/2,  l/2, -h/2], [-w/2,  l/2, -h/2],
        [-w/2, -l/2, -h/2], [ w/2, -l/2, -h/2],
        [ w/2,  l/2,  h/2], [-w/2,  l/2,  h/2],
        [-w/2, -l/2,  h/2], [ w/2, -l/2,  h/2],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return ((R @ local.T).T + np.array(pos_3d)).astype(np.float32)


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
