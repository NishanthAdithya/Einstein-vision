"""asset_manager.py — Load .blend assets and place them in the scene.

Run inside Blender Python (bpy available).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import bpy


# ---------------------------------------------------------------------------
# Mapping: detection class_name → .blend file (relative to P3Data/Assets/)
# ---------------------------------------------------------------------------

_ASSET_MAP: Dict[str, str] = {
    # Vehicles
    "car":          "Vehicles/SedanAndHatchback.blend",
    "truck":        "Vehicles/Truck.blend",
    "bus":          "Vehicles/Truck.blend",           # closest available
    "motorcycle":   "Vehicles/Motorcycle.blend",
    "bicycle":      "Vehicles/Bicycle.blend",
    # People
    "person":       "Pedestrain.blend",
    # Traffic furniture
    "traffic light": "TrafficSignal.blend",
    "stop sign":    "StopSign.blend",
    "traffic cone": "TrafficConeAndCylinder.blend",
    "dustbin":      "Dustbin.blend",
    "barrel":       "TrafficConeAndCylinder.blend",
    "fire hydrant": "TrafficConeAndCylinder.blend",   # placeholder
}

# Per-asset base rotation (rx, ry, rz) in radians that restores the asset to
# a canonical upright, +Y-forward pose.  Measured by inspecting the baked
# rotation stored on the mesh object inside each .blend file:
#   car:                    (0, 0, +90°) baked  → needs (0, 0, π/2) offset
#   motorcycle/bicycle/
#   pickup/person/signs:    (90°, 0, 0)  baked  → needs (π/2, 0, 0) offset
#   truck/suv:              no baked rotation   → (0, 0, 0)
# The predicted yaw is added to the rz component on top of this base.
_ROTATION_OFFSET_MAP: Dict[str, Tuple[float, float, float]] = {
    "car":           (0.0,            0.0, 0.0),
    "truck":         (0.0,            0.0, 0.0),
    "bus":           (0.0,            0.0, 0.0),
    "motorcycle":    (math.pi / 2,    0.0, 0.0),
    "bicycle":       (math.pi / 2,    0.0, 0.0),
    "person":        (math.pi / 2,    0.0, 0.0),
    "traffic light": (math.pi / 2,    0.0, 0.0),
    "stop sign":     (math.pi / 2,    0.0, 0.0),
}

# Scale factors to normalise each asset to real-world metres.
# These override any scale baked into the .blend file.
# Derived by measuring the asset's local bounding box and dividing the
# target real-world dimension (longest axis) by the local max dimension.
_SCALE_MAP: Dict[str, float] = {
    "car":           0.020,   # local 231m → ~4.5m
    "truck":         0.001,   # local 8937m → ~8.5m
    "bus":           0.001,   # uses Truck asset
    "motorcycle":    0.012,   # local 178m → ~2.2m
    "bicycle":       0.14,    # local 12.7m → ~1.8m
    "person":        0.022,   # local 31m (Z) → ~1.75m (empirically halved after render check)
    "traffic light": 1.14,    # local 3.5m → ~4.0m
    "stop sign":     0.37,    # local 6.7m → ~2.5m
    "traffic cone":  1.0,
    "dustbin":       1.0,
    "barrel":        1.0,
    "fire hydrant":  0.6,
}

# Cache of already-loaded library objects: blend_path → object name in scene
_loaded_cache: Dict[str, str] = {}

# Object names that persist across frames and must not be deleted by
# clear_scene_objects (e.g. ground plane created once in main()).
_STATIC_OBJECTS: frozenset = frozenset({"GroundPlane"})


def spawn_asset(
    class_name: str,
    pos_ego: Tuple[float, float, float],
    yaw_rad: float,
    assets_root: str | Path,
    track_id: Optional[int] = None,
) -> Optional[bpy.types.Object]:
    """Spawn one asset at the given ego-frame position.

    Args:
        class_name:  Detection class name (e.g. 'car', 'person').
        pos_ego:     (X, Y, Z) in ego frame, metres.
        yaw_rad:     Yaw rotation in Blender convention (negative of ego yaw).
        assets_root: Path to P3Data/Assets/.
        track_id:    Optional track ID used to name the object for animation.

    Returns:
        The spawned Blender object, or None if no asset exists for this class.
    """
    blend_rel = _ASSET_MAP.get(class_name)
    if blend_rel is None:
        return None

    blend_path = Path(assets_root) / blend_rel
    obj = _get_or_load_asset(blend_path, class_name, Path(assets_root))
    if obj is None:
        return None

    # Duplicate the template object for this instance
    instance = obj.copy()
    instance.data = obj.data.copy() if obj.data else None
    bpy.context.scene.collection.objects.link(instance)

    # Unhide — template has hide_render=True; instances must be visible
    instance.hide_render = False
    instance.hide_viewport = False

    # Name for easy identification and animation lookup
    suffix = f"_{track_id}" if track_id is not None else f"_{id(instance)}"
    instance.name = f"{class_name}{suffix}"

    # Place at ego position
    x, y, z = pos_ego
    instance.location = (x, y, z)

    # Restore the asset's canonical orientation then add predicted yaw on Z.
    # Each asset has a different baked rotation that must be preserved so the
    # model is upright and forward-facing before our yaw correction is applied.
    base_rx, base_ry, base_rz = _ROTATION_OFFSET_MAP.get(class_name, (0.0, 0.0, 0.0))
    instance.rotation_euler = (base_rx, base_ry, base_rz + yaw_rad)

    scale = _SCALE_MAP.get(class_name, 1.0)
    instance.scale = (scale, scale, scale)

    return instance


def clear_scene_objects(prefix: str = "") -> None:
    """Remove all non-camera, non-light objects whose names start with prefix.

    Call between frames to clear the previous frame's assets.
    Pass prefix="" to remove everything except cameras and lights.
    Template objects (hide_render=True) are preserved so the asset cache
    remains valid across frames.
    """
    keep_types = {"CAMERA", "LIGHT"}
    to_remove = [
        obj for obj in bpy.data.objects
        if obj.type not in keep_types
        and obj.name.startswith(prefix)
        and not obj.hide_render          # preserve cached template objects
        and obj.name not in _STATIC_OBJECTS  # preserve ground plane etc.
    ]
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_load_asset(
    blend_path: Path,
    class_name: str,
    assets_root: Path,
) -> Optional[bpy.types.Object]:
    """Load the first mesh object from a .blend file, caching the result."""
    key = str(blend_path)

    if key in _loaded_cache:
        name = _loaded_cache[key]
        return bpy.data.objects.get(name)

    if not blend_path.exists():
        import warnings
        warnings.warn(f"Asset not found: {blend_path}")
        return None

    with bpy.data.libraries.load(str(blend_path), link=False) as (src, dst):
        dst.objects = list(src.objects)

    # Find the first mesh object
    mesh_obj = None
    for obj in dst.objects:
        if obj is not None and obj.type == "MESH":
            bpy.context.scene.collection.objects.link(obj)
            obj.hide_render = True   # template is invisible — only instances render
            obj.hide_viewport = True
            mesh_obj = obj
            break

    if mesh_obj is None:
        return None

    # Per-asset post-load setup
    if class_name == "stop sign":
        _apply_stop_sign_texture(mesh_obj, assets_root)

    _loaded_cache[key] = mesh_obj.name
    return mesh_obj


def _apply_stop_sign_texture(obj: bpy.types.Object, assets_root: Path) -> None:
    """Apply StopSignImage.png to the stop sign mesh's material.

    The mesh already has a UVMap unwrapped at export time, so we only need
    to plug the image into the material's Principled BSDF Base Color input.
    """
    img_path = assets_root / "StopSignImage.png"
    if not img_path.exists() or not obj.data or not obj.data.materials:
        return

    img = bpy.data.images.get("StopSignImage")
    if img is None:
        img = bpy.data.images.load(str(img_path))
        img.name = "StopSignImage"

    mat = obj.data.materials[0]
    if mat is None:
        return

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex  = nodes.new("ShaderNodeTexImage")
    tex.image = img

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.8

    out  = nodes.new("ShaderNodeOutputMaterial")

    links.new(tex.outputs["Color"],  bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"],  out.inputs["Surface"])
