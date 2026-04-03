from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from src.io.schema import Detection, FrameData, Lane, PoseResult


class DepthEstimator(ABC):
    """Produce a per-pixel depth map from a single BGR frame."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 depth map with the same H×W as the input, in metres."""
        ...

    def close(self) -> None:
        """Release any held resources (GPU memory, file handles, etc.)."""


class ObjectDetector(ABC):
    """Detect objects in a single BGR frame."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Return a list of Detection objects (track_id, pos_3d etc. left unfilled)."""
        ...

    def close(self) -> None: ...


class LaneDetector(ABC):
    """Detect lane lines in a single BGR frame."""

    @abstractmethod
    def predict(
        self,
        frame_bgr: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Lane]:
        """Return a list of Lane objects.

        depth_map may be provided so implementations can lift 2-D pixels to 3-D.
        """
        ...

    def close(self) -> None: ...


class PoseEstimator(ABC):
    """Estimate whole-body pose keypoints for detected persons."""

    @abstractmethod
    def predict(
        self,
        frame_bgr: np.ndarray,
        person_detections: List[Detection],
    ) -> List[PoseResult]:
        """Return one PoseResult per person crop (bbox, keypoints, keypoints_3d)."""
        ...

    def close(self) -> None: ...


class Tracker(ABC):
    """Assign stable track IDs to per-frame detections."""

    @abstractmethod
    def update(self, detections: List[Detection], frame_bgr: np.ndarray) -> List[Detection]:
        """Return the same detections with track_id fields populated.

        Detections that do not match an active track are dropped or returned
        with track_id=None, depending on the implementation.
        """
        ...

    def reset(self) -> None:
        """Reset internal track state (call between scenes)."""

    def close(self) -> None: ...


class FlowEstimator(ABC):
    """Compute dense optical flow between consecutive frames."""

    @abstractmethod
    def predict(
        self,
        frame_prev_bgr: np.ndarray,
        frame_curr_bgr: np.ndarray,
    ) -> np.ndarray:
        """Return a float32 flow field of shape (H, W, 2) in pixels per frame."""
        ...

    def close(self) -> None: ...
