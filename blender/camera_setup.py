"""camera_setup.py — Configure the Blender camera from CameraConfig data.

Run inside Blender Python (bpy available).  Called once per scene.
"""

from __future__ import annotations

import math
from typing import Dict, Any

import bpy


def setup_camera(cam_data: Dict[str, Any], image_width: int, image_height: int) -> bpy.types.Object:
    """Create or replace the scene camera using intrinsics from cameras.yaml.

    Args:
        cam_data:     Dict with keys fx, fy, cx, cy, image_width, image_height,
                      height_above_ground_m, pitch_deg  (matches CameraConfig fields).
        image_width:  Render width in pixels.
        image_height: Render height in pixels.

    Returns:
        The Blender camera object.
    """
    scene = bpy.context.scene
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height
    scene.render.resolution_percentage = 100

    # Remove existing camera if present
    if "EinsteinCamera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["EinsteinCamera"], do_unlink=True)

    cam_bpy = bpy.data.cameras.new("EinsteinCamera")
    cam_obj = bpy.data.objects.new("EinsteinCamera", cam_bpy)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # ── intrinsics → Blender sensor model ────────────────────────────────────
    # Blender uses sensor_width + focal_length (mm) to represent the K matrix.
    # We fix sensor_width = image_width pixels (i.e. 1 pixel = 1 mm) and set
    # focal_length = fx pixels.  This gives the correct horizontal FoV.
    fx = cam_data["fx"]
    fy = cam_data["fy"]
    cx = cam_data["cx"]
    cy = cam_data["cy"]

    cam_bpy.sensor_fit = "HORIZONTAL"
    cam_bpy.sensor_width = float(image_width)   # virtual sensor in mm
    cam_bpy.lens = float(fx)                     # focal length in mm

    # Shift principal point away from centre
    # Blender shift_x/y are in units of sensor half-width/half-height
    cam_bpy.shift_x = (cx - image_width  / 2.0) / image_width
    cam_bpy.shift_y = (cy - image_height / 2.0) / image_height

    # ── extrinsics: mount pose ────────────────────────────────────────────────
    # Camera sits height_above_ground_m above Z=0, tilted nose-down by pitch_deg.
    # In Blender: camera looks down -Z by default; we rotate it to look along +Y
    # (ego forward) and then apply the pitch.
    h = cam_data["height_above_ground_m"]
    pitch_deg = cam_data["pitch_deg"]

    cam_obj.location = (0.0, 0.0, h)
    # Base rotation: point camera along +Y (forward) with Z-up
    # Blender camera default looks at -Z; rotate 90° around X to look at +Y
    cam_obj.rotation_euler = (
        math.radians(90.0 - pitch_deg),  # X: tilt nose down
        0.0,                              # Y
        0.0,                              # Z
    )

    return cam_obj


def set_render_settings(
    samples: int = 64,
    engine: str = "CYCLES",
    use_gpu: bool = True,
) -> None:
    """Configure render engine and quality settings.

    Args:
        samples:  Number of render samples (lower = faster, higher = quality).
        engine:   'CYCLES' or 'BLENDER_EEVEE_NEXT'.
        use_gpu:  Enable GPU rendering in Cycles.
    """
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.image_settings.file_format = "PNG"

    if engine == "CYCLES":
        scene.cycles.samples = samples
        if use_gpu:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            for dev in prefs.devices:
                dev.use = True
            scene.cycles.device = "GPU"
    elif engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        scene.eevee.taa_render_samples = samples
