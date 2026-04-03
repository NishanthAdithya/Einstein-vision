from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from src.io.schema import CameraConfig


_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "cameras.yaml"


def load_camera(
    name: str,
    config_path: str | Path = _DEFAULT_CONFIG,
) -> CameraConfig:
    """Load intrinsics and pose for a single camera.

    Args:
        name: Camera name — 'front', 'back', 'left', or 'right'
        config_path: Path to cameras.yaml

    Returns:
        CameraConfig populated from the YAML file
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if name not in data:
        raise KeyError(
            f"Camera '{name}' not found in {config_path}. "
            f"Available: {list(data.keys())}"
        )

    entry = data[name]
    return CameraConfig(
        name=name,
        fx=float(entry["fx"]),
        fy=float(entry["fy"]),
        cx=float(entry["cx"]),
        cy=float(entry["cy"]),
        image_width=int(entry["image_width"]),
        image_height=int(entry["image_height"]),
        height_above_ground_m=float(entry["height_above_ground_m"]),
        pitch_deg=float(entry["pitch_deg"]),
        dist_coeffs=[float(v) for v in entry.get("dist_coeffs", [0.0] * 5)],
    )


def load_all_cameras(
    config_path: str | Path = _DEFAULT_CONFIG,
) -> Dict[str, CameraConfig]:
    """Load intrinsics for all four cameras.

    Returns:
        Dict mapping camera name to CameraConfig
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return {name: load_camera(name, config_path) for name in data}
