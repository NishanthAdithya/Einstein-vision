from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np


# Maps the logical camera name to the suffix used in video filenames.
CAMERA_SUFFIX = {
    "front": "front",
    "back": "back",
    "left": "left_repeater",
    "right": "right_repeater",
}


def find_video(scene_dir: str | Path, camera: str, undistorted: bool = True) -> Path:
    """Return the video path for a given scene directory and camera name.

    Args:
        scene_dir: Path to a scene folder, e.g. P3Data/Sequences/scene1
        camera: Logical camera name — 'front', 'back', 'left', or 'right'
        undistorted: If True, look in the Undist/ subfolder; otherwise Raw/
    """
    scene_dir = Path(scene_dir)
    suffix = CAMERA_SUFFIX[camera]
    subfolder = "Undist" if undistorted else "Raw"
    video_dir = scene_dir / subfolder

    pattern = f"*-{suffix}_undistort.mp4" if undistorted else f"*-{suffix}.mp4"
    matches = sorted(video_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No video matching '{pattern}' in {video_dir}"
        )
    return matches[0]


class VideoReader:
    """Iterates over sampled frames from a single video file.

    Yields (frame_idx, frame_bgr) tuples for every nth frame,
    where frame_idx is the original index in the full video.
    """

    def __init__(self, video_path: str | Path, sample_every_n: int = 5):
        self._path = Path(video_path)
        self._sample_every_n = sample_every_n
        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self._path}")

    # -- Properties ----------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    # -- Frame access --------------------------------------------------------

    def frame_idx_to_timestamp(self, frame_idx: int) -> float:
        return frame_idx / self.fps if self.fps > 0 else 0.0

    def sampled_indices(self) -> List[int]:
        return list(range(0, self.total_frames, self._sample_every_n))

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Seek to and return a specific frame by its original index."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {self._path}")
        return frame

    # -- Iteration -----------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_idx, frame_bgr) for every nth frame."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame_idx % self._sample_every_n == 0:
                yield frame_idx, frame
            frame_idx += 1

    def __len__(self) -> int:
        return len(self.sampled_indices())

    def __repr__(self) -> str:
        return (
            f"VideoReader('{self._path.name}', "
            f"{self.width}x{self.height}, "
            f"{self.fps:.1f}fps, "
            f"sample_every={self._sample_every_n})"
        )

    # -- Resource management -------------------------------------------------

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        self.close()
