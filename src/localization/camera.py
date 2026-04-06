from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from src.io.schema import CameraConfig


class Camera:
    """Encapsulates camera intrinsics and provides geometric operations.

    Wraps a CameraConfig and exposes:
    - The 3×3 intrinsic matrix K and its inverse
    - Single-pixel and batched unprojection (2D pixel + depth → 3D camera frame)
    - IPM (Inverse Perspective Mapping) homography for ground-plane projection

    Coordinate conventions:
        Camera frame: X = right, Y = down, Z = forward (into scene)
        Ego frame:    X = right, Y = forward, Z = up
    """

    def __init__(self, config: CameraConfig) -> None:
        self._cfg = config
        self._K = np.array(
            [
                [config.fx, 0.0,        config.cx],
                [0.0,       config.fy,  config.cy],
                [0.0,       0.0,        1.0      ],
            ],
            dtype=np.float64,
        )
        self._K_inv = np.linalg.inv(self._K)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return self._K

    @property
    def K_inv(self) -> np.ndarray:
        """Inverse of K."""
        return self._K_inv

    @property
    def config(self) -> CameraConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Unprojection — pixel + depth → 3D camera frame
    # ------------------------------------------------------------------

    def unproject(self, u: float, v: float, depth: float) -> Tuple[float, float, float]:
        """Unproject a single pixel to the camera frame.

        Args:
            u:     Pixel column (x)
            v:     Pixel row    (y)
            depth: Metric depth in metres (Z in camera frame)

        Returns:
            (X, Y, Z) in the camera frame in metres.
            X = right, Y = down, Z = forward.
        """
        X = (u - self._cfg.cx) * depth / self._cfg.fx
        Y = (v - self._cfg.cy) * depth / self._cfg.fy
        return float(X), float(Y), float(depth)

    def unproject_batch(
        self,
        uvs: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        """Unproject N pixels to 3D camera-frame points.

        Args:
            uvs:    (N, 2) float array of (u, v) pixel coordinates
            depths: (N,)   float array of metric depths in metres

        Returns:
            (N, 3) float32 array of (X, Y, Z) camera-frame points.
        """
        u = uvs[:, 0]
        v = uvs[:, 1]
        X = (u - self._cfg.cx) * depths / self._cfg.fx
        Y = (v - self._cfg.cy) * depths / self._cfg.fy
        return np.stack([X, Y, depths], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Coordinate frame transform: camera → ego
    # ------------------------------------------------------------------

    def camera_to_ego(self, xyz_cam: np.ndarray) -> np.ndarray:
        """Rotate camera-frame points into the ego (vehicle body) frame.

        Camera: X=right, Y=down, Z=forward
        Ego:    X=right, Y=forward, Z=up

        The rotation accounts for the camera's mount height above the ground
        and its pitch angle (nose-down tilt).

        Args:
            xyz_cam: (N, 3) or (3,) float array in camera frame

        Returns:
            Same shape float32 array in ego frame.
        """
        R = self._camera_to_ego_rotation()
        pts = np.atleast_2d(xyz_cam).astype(np.float64)
        ego = (R @ pts.T).T
        # Translate: camera sits height_above_ground_m above the ground plane
        ego[:, 2] += self._cfg.height_above_ground_m
        result = ego.astype(np.float32)
        return result.squeeze() if xyz_cam.ndim == 1 else result

    def _camera_to_ego_rotation(self) -> np.ndarray:
        """Build the rotation matrix from camera frame to ego frame.

        Applies:
          1. Swap axes: cam(X,Y,Z) → ego base (X, -Z, Y) to align
             Y-forward and Z-up conventions.
          2. Pitch correction: rotate around X axis by -pitch_deg so that
             the ground plane maps correctly.
        """
        pitch_rad = math.radians(self._cfg.pitch_deg)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

        # Axis swap: ego_x = cam_x, ego_y = cam_z, ego_z = -cam_y
        M_swap = np.array(
            [
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        # Pitch rotation around the (new) X axis
        R_pitch = np.array(
            [
                [1.0, 0.0,  0.0],
                [0.0,  cp,  sp],
                [0.0, -sp,  cp],
            ],
            dtype=np.float64,
        )
        return R_pitch @ M_swap

    def ego_to_image(self, xyz_ego: np.ndarray) -> np.ndarray:
        """Project ego-frame 3D points to image pixel coordinates (u, v).

        Args:
            xyz_ego: (N, 3) or (3,) float array in ego frame.

        Returns:
            (N, 2) float32 array of (u, v) pixel coordinates.
            Rows where the point is behind the camera are set to NaN.
        """
        R_inv = self._camera_to_ego_rotation().T   # ego → camera rotation

        pts = np.atleast_2d(xyz_ego).astype(np.float64)
        shifted = pts.copy()
        shifted[:, 2] -= self._cfg.height_above_ground_m   # undo height offset
        cam = (R_inv @ shifted.T).T                         # (N, 3) camera frame

        uv = np.full((len(cam), 2), np.nan, dtype=np.float32)
        valid = cam[:, 2] > 0.1   # in front of camera
        if valid.any():
            u = self._cfg.fx * cam[valid, 0] / cam[valid, 2] + self._cfg.cx
            v = self._cfg.fy * cam[valid, 1] / cam[valid, 2] + self._cfg.cy
            uv[valid] = np.stack([u, v], axis=1).astype(np.float32)

        return uv.squeeze() if np.asarray(xyz_ego).ndim == 1 else uv

    # ------------------------------------------------------------------
    # IPM homography for the ground plane
    # ------------------------------------------------------------------

    def ipm_homography(self) -> np.ndarray:
        """Compute the 3×3 homography that maps image pixels to ground-plane points.

        The ground plane is defined as Z=0 in the ego frame (road surface).
        Returns H such that:
            [X_ego, Y_ego, 1]^T  ∝  H @ [u, v, 1]^T

        Only valid for pixels that project onto the flat ground plane.
        Points above the horizon (sky, buildings) will map to large/invalid coords.

        Returns:
            (3, 3) float64 homography matrix.
        """
        h = self._cfg.height_above_ground_m
        pitch_rad = math.radians(self._cfg.pitch_deg)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

        fx, fy = self._cfg.fx, self._cfg.fy
        cx, cy = self._cfg.cx, self._cfg.cy

        # For a flat ground plane at height h below the camera:
        # A pixel (u, v) with normalised coords (x_n, y_n) = ((u-cx)/fx, (v-cy)/fy)
        # hits the ground at depth Z = h / (y_n * cp + sp)
        # Full derivation: camera-frame ray direction dotted with ground normal.
        #
        # We encode this as a homography by working with homogeneous coords.
        # H maps [u, v, 1] → [X_ego, Y_ego, scale] with X_ego/scale, Y_ego/scale
        # being the ego-frame ground coords.

        # Derivation:
        #   A pixel (u,v) defines a ray direction d_ego = R @ K_inv @ [u,v,1].
        #   Camera origin in ego frame: (0, 0, h).
        #   Ray: P(t) = (0,0,h) + t * d_ego.
        #   Ground plane: P_z = 0  →  t = -h / d_ego[2].
        #   Ground point: X_ego = t * d_ego[0] = -h * A[0]@p / (A[2]@p)
        #                 Y_ego = t * d_ego[1] = -h * A[1]@p / (A[2]@p)
        #   where A = R @ K_inv and p = [u, v, 1].
        #
        # In homogeneous form H @ p = [num_x, num_y, denom]:
        #   H = [[ h * A[0] ],
        #        [ h * A[1] ],
        #        [    -A[2] ]]
        # so that X_ego = num_x / denom = h*A[0]@p / (-A[2]@p) = -h*A[0]@p / A[2]@p.
        # Valid ground-plane pixels have d_ego[2] = A[2]@p < 0, hence denom > 0.
        R = self._camera_to_ego_rotation()
        A = R @ self._K_inv  # (3, 3)

        H = np.array(
            [
                h * A[0],
                h * A[1],
                -A[2],
            ],
            dtype=np.float64,
        )
        return H

    def apply_ipm(self, uvs: np.ndarray) -> np.ndarray:
        """Map image pixels to ego-frame ground-plane coordinates via IPM.

        Args:
            uvs: (N, 2) float array of (u, v) pixel coordinates

        Returns:
            (N, 2) float32 array of (X_ego, Y_ego) in metres.
            Rows where the pixel is at or above the horizon are set to NaN.
        """
        H = self.ipm_homography()
        ones = np.ones((uvs.shape[0], 1), dtype=np.float64)
        hom = np.hstack([uvs.astype(np.float64), ones])  # (N, 3)
        mapped = (H @ hom.T).T                             # (N, 3)

        scale = mapped[:, 2:3]  # (N, 1) — zero/negative means above horizon
        valid = scale[:, 0] > 1e-6

        xy = np.full((uvs.shape[0], 2), np.nan, dtype=np.float32)
        xy[valid] = (mapped[valid, :2] / scale[valid]).astype(np.float32)
        return xy
