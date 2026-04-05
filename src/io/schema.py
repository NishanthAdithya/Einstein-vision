from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0.0 else 0.0

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height

    def clamp(self, img_width: int, img_height: int) -> BBox:
        return BBox(
            max(0.0, self.x1),
            max(0.0, self.y1),
            min(float(img_width), self.x2),
            min(float(img_height), self.y2),
        )


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: BBox

    # Filled by tracking module
    track_id: Optional[int] = None

    # Filled by localization module
    pos_3d: Optional[Tuple[float, float, float]] = None
    depth: float = 0.0
    yaw: float = 0.0

    # Filled by phase 2 modules
    vehicle_subclass: Optional[str] = None     # 'sedan' | 'suv' | 'hatchback' | 'pickup'
    traffic_light_arrow: Optional[str] = None  # 'left' | 'right' | 'straight'
    speed_limit: Optional[int] = None          # e.g. 25, 35, 45, 55, 65

    # Filled by phase 3 modules
    is_moving: Optional[bool] = None
    brake_light_on: Optional[bool] = None
    turn_signal: Optional[str] = None          # 'left' | 'right' | 'none'
    traffic_light_state: Optional[str] = None  # 'red' | 'yellow' | 'green'
    velocity_3d: Tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )


@dataclass
class Lane:
    lane_type: str                             # 'solid-white' | 'dashed-white' | 'solid-yellow' | 'double'
    bezier_points_3d: np.ndarray               # (N, 3) 3D control points in ego frame, metres
    poly_coeffs: np.ndarray                    # (3,) coefficients [a, b, c] of x = ay^2 + by + c
    pixels_2d: np.ndarray                      # (N, 2) corresponding pixel coordinates


@dataclass
class PoseResult:
    bbox: BBox
    keypoints: np.ndarray                      # (133, 3): x, y, confidence — RTMW whole-body
    keypoints_3d: Optional[np.ndarray] = None  # (133, 3) in camera frame, metres
    pos_3d: Optional[Tuple[float, float, float]] = None  # 3D root position in ego frame
    depth: float = 0.0                         # depth of person in metres


@dataclass
class CameraConfig:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    height_above_ground_m: float
    pitch_deg: float
    dist_coeffs: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )


@dataclass
class FrameData:
    frame_idx: int
    timestamp: float
    camera_name: str
    scene_id: str
    detections: List[Detection] = field(default_factory=list)
    lanes: List[Lane] = field(default_factory=list)
    poses: List[PoseResult] = field(default_factory=list)
