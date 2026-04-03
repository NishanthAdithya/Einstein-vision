from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.io.schema import Lane


# ---------------------------------------------------------------------------
# Lane type classification thresholds
# ---------------------------------------------------------------------------

# HSV range for yellow lane markings
_YELLOW_H_LO, _YELLOW_H_HI = 18, 38
_YELLOW_S_MIN = 80
_YELLOW_V_MIN = 100

# White lane markings have low saturation and high value
_WHITE_S_MAX = 60
_WHITE_V_MIN = 160

# A component is "dashed" when the fraction of empty rows exceeds this
_DASHED_GAP_RATIO = 0.25

# A component is classified as "double" when its pixel width at the
# median row exceeds this many pixels (rough proxy for two parallel stripes)
_DOUBLE_WIDTH_PX = 60


def classify_lane_types(
    lanes: List[Lane],
    lane_mask: np.ndarray,
    frame_bgr: np.ndarray,
) -> List[Lane]:
    """Classify each lane as solid-white, dashed-white, solid-yellow, or double.

    Updates lane.lane_type in-place for every lane in the list.

    Args:
        lanes:      Lane objects whose pixels_2d field is populated.
        lane_mask:  Binary uint8 mask (H, W) from the lane detector.
        frame_bgr:  The original uint8 BGR frame (same H, W).

    Returns:
        The same list with lane_type updated.
    """
    h, w = frame_bgr.shape[:2]
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    for lane in lanes:
        if lane.pixels_2d is None or len(lane.pixels_2d) == 0:
            lane.lane_type = "unknown"
            continue

        cols = np.clip(lane.pixels_2d[:, 0].round().astype(np.int32), 0, w - 1)
        rows = np.clip(lane.pixels_2d[:, 1].round().astype(np.int32), 0, h - 1)

        is_yellow = _is_yellow(frame_hsv, rows, cols)
        is_dashed = _is_dashed(rows)
        is_double = _is_double(lane_mask, rows, cols)

        if is_double:
            lane.lane_type = "double"
        elif is_yellow:
            lane.lane_type = "solid-yellow"
        elif is_dashed:
            lane.lane_type = "dashed-white"
        else:
            lane.lane_type = "solid-white"

    return lanes


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _is_yellow(
    frame_hsv: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> bool:
    """Return True if the majority of sampled pixels are yellow in HSV."""
    h_vals = frame_hsv[rows, cols, 0].astype(np.float32)
    s_vals = frame_hsv[rows, cols, 1].astype(np.float32)
    v_vals = frame_hsv[rows, cols, 2].astype(np.float32)

    yellow_mask = (
        (h_vals >= _YELLOW_H_LO) & (h_vals <= _YELLOW_H_HI) &
        (s_vals >= _YELLOW_S_MIN) &
        (v_vals >= _YELLOW_V_MIN)
    )
    return bool(yellow_mask.mean() > 0.30)


def _is_dashed(rows: np.ndarray) -> bool:
    """Return True if the lane has significant vertical gaps (dashed pattern).

    Strategy: build a row-presence bitmap over the lane's vertical extent,
    then measure the fraction of empty rows.  A continuous (solid) lane
    has few or no empty rows; a dashed lane has many.
    """
    y_min, y_max = int(rows.min()), int(rows.max())
    span = y_max - y_min + 1
    if span < 10:
        return False

    occupied = np.zeros(span, dtype=bool)
    occupied[rows - y_min] = True

    gap_ratio = (~occupied).sum() / span
    return bool(gap_ratio > _DASHED_GAP_RATIO)


def _is_double(
    lane_mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> bool:
    """Return True if the lane component spans a width suggesting two stripes.

    At the median row of the component, measure the horizontal extent of
    connected foreground pixels.  A double-line marking is much wider than
    a single stripe.
    """
    median_row = int(np.median(rows))
    row_slice = lane_mask[median_row, :]

    # Find runs of foreground pixels in this row
    nonzero_cols = np.where(row_slice > 0)[0]
    if len(nonzero_cols) == 0:
        return False

    # Check if this component's columns are among the nonzero pixels
    comp_cols_at_row = cols[rows == median_row]
    if len(comp_cols_at_row) == 0:
        return False

    span_px = comp_cols_at_row.max() - comp_cols_at_row.min() + 1
    return bool(span_px >= _DOUBLE_WIDTH_PX)
