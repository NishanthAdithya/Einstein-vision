from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.io.schema import BBox, Detection, PoseResult
from src.perception.base import PoseEstimator


# RTMW whole-body keypoint count (COCO-WholeBody format)
_N_KEYPOINTS = 133

# Config shipped inside mmpose for RTMW-L (256×192 input)
_RTMW_CONFIG = (
    "configs/wholebody_2d_keypoint/rtmpose/cocktail14/"
    "rtmw-l_8xb64-270e_cocktail14-256x192.py"
)


class RTMWPoseEstimator(PoseEstimator):
    """Whole-body pose estimator backed by RTMW-L via mmpose.

    Estimates 133 keypoints per person in COCO-WholeBody format:
    body (17) + feet (6) + face (68) + hands (42).

    The model is loaded lazily from an mmpose config + checkpoint file.
    Download the checkpoint with::

        mim download mmpose \\
            --config rtmw-l_8xb64-270e_cocktail14-256x192 \\
            --dest weights/

    and set ``checkpoint`` in configs/models.yaml to the downloaded path.

    Args:
        checkpoint:  Path to the RTMW-L .pth checkpoint file.
        config:      Path to the mmpose Python config.  Defaults to the
                     RTMW-L cocktail14 config bundled with mmpose.
        device:      'cuda' or 'cpu'
    """

    def __init__(
        self,
        checkpoint: str,
        config: str = _RTMW_CONFIG,
        device: str = "cuda",
    ) -> None:
        from mmpose.apis import init_model

        self._model = init_model(config, checkpoint, device=device)
        self._device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        frame_bgr: np.ndarray,
        person_detections: List[Detection],
    ) -> List[PoseResult]:
        """Estimate whole-body keypoints for each detected person.

        Only detections with class_name == 'person' are processed.
        keypoints_3d is left as None and filled by the localization module.

        Args:
            frame_bgr:         uint8 BGR image.
            person_detections: All detections from the object detector.
                               Non-person detections are silently ignored.

        Returns:
            One PoseResult per person crop, in the same order as the
            filtered person_detections list.
        """
        persons = [d for d in person_detections if d.class_name == "person"]
        if not persons:
            return []

        bboxes = np.array(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in persons],
            dtype=np.float32,
        )

        from mmpose.apis import inference_topdown

        results = inference_topdown(self._model, frame_bgr, bboxes, bbox_format="xyxy")

        poses: List[PoseResult] = []
        for det, result in zip(persons, results):
            kpts, scores = _extract_keypoints(result)
            keypoints = np.concatenate(
                [kpts, scores[:, np.newaxis]], axis=1
            ).astype(np.float32)  # (133, 3)

            poses.append(
                PoseResult(
                    bbox=det.bbox,
                    keypoints=keypoints,
                    keypoints_3d=None,
                )
            )

        return poses

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        import torch
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keypoints(
    result,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull keypoint coordinates and scores out of an mmpose PoseDataSample.

    Returns:
        kpts:   (133, 2) float32 array of (x, y) pixel coordinates
        scores: (133,)   float32 array of per-keypoint confidence scores

    If the result contains fewer than 133 keypoints (e.g. a partial crop),
    the missing entries are zero-padded.
    """
    instances = result.pred_instances

    kpts = np.array(instances.keypoints, dtype=np.float32)    # (1, K, 2) or (K, 2)
    scores = np.array(instances.keypoint_scores, dtype=np.float32)  # (1, K) or (K,)

    # Squeeze batch dimension if present
    if kpts.ndim == 3:
        kpts = kpts[0]
    if scores.ndim == 2:
        scores = scores[0]

    # Pad to _N_KEYPOINTS if needed
    n = kpts.shape[0]
    if n < _N_KEYPOINTS:
        kpts = np.pad(kpts, ((0, _N_KEYPOINTS - n), (0, 0)))
        scores = np.pad(scores, (0, _N_KEYPOINTS - n))

    return kpts[:_N_KEYPOINTS], scores[:_N_KEYPOINTS]
