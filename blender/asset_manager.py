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
    # Vehicles — generic
    "car":              "Vehicles/SedanAndHatchback.blend",
    "truck":            "Vehicles/Truck.blend",
    "bus":              "Vehicles/Truck.blend",           # closest available
    "motorcycle":       "Vehicles/Motorcycle.blend",
    "bicycle":          "Vehicles/Bicycle.blend",
    # Phase 2 — vehicle sub-classes
    "sedan":            "Vehicles/SedanAndHatchback.blend",
    "hatchback":        "Vehicles/SedanAndHatchback.blend",
    "suv":              "Vehicles/SUV.blend",
    "pickup":           "Vehicles/PickupTruck.blend",
    # People
    "person":           "Pedestrain.blend",
    # Traffic furniture
    "traffic light":    "TrafficSignal.blend",
    "stop sign":        "StopSign.blend",
    "traffic cone":     "TrafficConeAndCylinder.blend",
    "traffic cylinder": "TrafficConeAndCylinder.blend",
    "traffic pole":     "TrafficAssets.blend",
    "dustbin":          "Dustbin.blend",
    "barrel":           "TrafficConeAndCylinder.blend",
    "fire hydrant":     "TrafficConeAndCylinder.blend",   # placeholder
    # Phase 2 — road signs
    "speed limit sign": "SpeedLimitSign.blend",
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
    # Vehicles: assets baked with side-on orientation, rotate 90° on Z to face forward
    "car":              (0.0,         0.0, math.pi / 2),
    "sedan":            (0.0,         0.0, math.pi / 2),
    "hatchback":        (0.0,         0.0, math.pi / 2),
    "suv":              (0.0,         0.0, 0.0),
    "pickup":           (0.0,         0.0, math.pi / 2),
    "truck":            (0.0,         0.0, math.pi / 2),
    "bus":              (0.0,         0.0, math.pi / 2),
    "motorcycle":       (math.pi / 2, 0.0, math.pi / 2),
    "bicycle":          (math.pi / 2, 0.0, math.pi / 2),
    # People: upright with 90° X to stand up, face toward ego
    "person":           (math.pi / 2, 0.0, math.pi),
    # Traffic light: upright, faces toward ego.
    # TrafficSignal.blend local axes: Y=tall (3.5m), X=face (normals ±X), Z=lateral.
    # Target: local Y→world Z (up), local X→world -Y (toward camera).
    # Blender XYZ euler: R = Rz @ Ry @ Rx
    # (π/2, 0, -π/2): Rz(-π/2)@Rx(π/2) → local X→world -Y ✓, local Y→world Z ✓
    "traffic light":    (math.pi / 2, 0.0, -math.pi / 2),
    # Signs: upright, face toward ego
    "stop sign":        (math.pi / 2, 0.0, math.pi),
    "speed limit sign": (math.pi / 2, 0.0, math.pi),
    "traffic pole":     (math.pi / 2, 0.0, 0.0),
}

# Scale factors to normalise each asset to real-world metres.
# These override any scale baked into the .blend file.
# Derived by measuring the asset's local bounding box and dividing the
# target real-world dimension (longest axis) by the local max dimension.
_SCALE_MAP: Dict[str, float] = {
    "car":              0.020,   # local 231m → ~4.5m
    "sedan":            0.020,
    "hatchback":        0.020,
    "suv":              3.35,    # Jeep_3_: built-in scale 3.354, nose along local +Y (4.27 m long)
    "pickup":           2.5,     # Cube placeholder: 2 m sides → ~5 m world
    "truck":            0.001,   # local 8937m → ~8.5m
    "bus":              0.001,   # uses Truck asset
    "motorcycle":       0.012,   # local 178m → ~2.2m
    "bicycle":          0.14,    # local 12.7m → ~1.8m
    "person":           0.022,   # local 31m (Z) → ~1.75m
    "traffic light":    1.14,    # local 3.5m → ~4.0m
    "stop sign":        0.37,    # local 6.7m → ~2.5m
    "speed limit sign": 0.37,    # same pole height as stop sign
    "traffic pole":     1.0,
    "traffic cone":     1.0,
    "traffic cylinder": 1.0,
    "dustbin":          1.0,
    "barrel":           1.0,
    "fire hydrant":     0.6,
}

# Cache of already-loaded library objects: blend_path → list of mesh object names
_loaded_cache: Dict[str, List[str]] = {}

# For assets with multiple mesh parts (e.g. Dustbin has wheels + lid + body),
# load ALL mesh parts so the full asset assembles at the spawn location.
# For assets where only one specific mesh is useful, name it here.
# None = load all meshes; a string = load only that named mesh.
_ASSET_MESH_NAME: Dict[str, Optional[str]] = {
    "dustbin":          None,          # all 3 parts (wheels, lid, body)
    "traffic pole":     "Cylinder",    # Cylinder from TrafficAssets.blend
    "traffic cone":     "absperrhut",  # German for traffic cone
    "traffic cylinder": "absperrhut",  # same asset, cone shape
}

# Classes whose Z position should be clamped to ground (Z=0) when spawning,
# since their asset origin is at the base and pos_3d gives the bbox centre height.
_GROUND_CLAMP_CLASSES = frozenset({
    "car", "sedan", "hatchback", "suv", "pickup", "truck", "bus",
    "motorcycle", "bicycle", "person",
    "dustbin", "traffic cone", "traffic cylinder", "barrel", "fire hydrant",
    "traffic pole",
})

# Object names that persist across frames and must not be deleted by
# clear_scene_objects (e.g. ground plane created once in main()).
_STATIC_OBJECTS: frozenset = frozenset({"GroundPlane"})

# Vehicle class names that receive per-track-ID colouring
_VEHICLE_CLASSES = frozenset({
    "car", "sedan", "hatchback", "suv", "pickup",
    "truck", "bus", "motorcycle", "bicycle",
})

# Visually distinct hues (0–1) cycled by track_id for vehicle colouring
_TRACK_HUES = [0.0, 0.08, 0.18, 0.33, 0.50, 0.58, 0.67, 0.75, 0.83, 0.92]


def spawn_asset(
    class_name: str,
    pos_ego: Tuple[float, float, float],
    yaw_rad: float,
    assets_root: str | Path,
    track_id: Optional[int] = None,
    speed_limit: Optional[int] = None,
) -> Optional[bpy.types.Object]:
    """Spawn one asset at the given ego-frame position.

    Args:
        class_name:  Detection class name (e.g. 'car', 'sedan', 'suv').
        pos_ego:     (X, Y, Z) in ego frame, metres.
        yaw_rad:     Yaw rotation in Blender convention.
        assets_root: Path to P3Data/Assets/.
        track_id:    Optional track ID used to name the object for animation.
        speed_limit: Detected speed limit value (only used when class_name is
                     'speed limit sign'); triggers number texture generation.

    Returns:
        The spawned Blender object, or None if no asset exists for this class.
    """
    blend_rel = _ASSET_MAP.get(class_name)
    if blend_rel is None:
        return None

    blend_path = Path(assets_root) / blend_rel
    template_names = _get_or_load_asset(blend_path, class_name, Path(assets_root))
    if not template_names:
        return None

    suffix = f"_{track_id}" if track_id is not None else f"_{id(object())}"
    base_rx, base_ry, base_rz = _ROTATION_OFFSET_MAP.get(class_name, (0.0, 0.0, 0.0))
    scale = _SCALE_MAP.get(class_name, 1.0)

    # Ground-based objects always sit at Z=0 regardless of pos_3d.z
    x, y, z = pos_ego
    if class_name in _GROUND_CLAMP_CLASSES:
        z = 0.0

    first_instance = None
    for i, tpl_name in enumerate(template_names):
        tpl = bpy.data.objects.get(tpl_name)
        if tpl is None:
            continue

        instance = tpl.copy()
        instance.data = tpl.data.copy() if tpl.data else None
        bpy.context.scene.collection.objects.link(instance)

        instance.hide_render = False
        instance.hide_viewport = False
        instance.name = f"{class_name}{suffix}" if i == 0 else f"{class_name}{suffix}_p{i}"
        instance.location = (x, y, z)
        instance.rotation_euler = (base_rx, base_ry, base_rz + yaw_rad)
        instance.scale = (scale, scale, scale)

        if first_instance is None:
            first_instance = instance

    if first_instance is None:
        return None

    # Phase 2 — apply speed limit number texture on the sign instance
    if class_name == "speed limit sign" and speed_limit is not None:
        _apply_speed_limit_texture(first_instance, speed_limit)

    # Apply per-track colour to vehicles so each tracked object is distinguishable
    if class_name in _VEHICLE_CLASSES and track_id is not None:
        _apply_track_colour(first_instance, track_id)

    return first_instance


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
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        # Free the copied mesh data block immediately to prevent unbounded accumulation
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_load_asset(
    blend_path: Path,
    class_name: str,
    assets_root: Path,
) -> List[str]:
    """Load mesh object(s) from a .blend file and return their names (cached).

    For most assets, loads the first mesh only.  For multi-part assets (e.g.
    Dustbin), loads all mesh parts so the full model assembles at spawn time.
    Use _ASSET_MESH_NAME to control which meshes are loaded per class.
    """
    key = str(blend_path)

    if key in _loaded_cache:
        return _loaded_cache[key]

    if not blend_path.exists():
        import warnings
        warnings.warn(f"Asset not found: {blend_path}")
        _loaded_cache[key] = []
        return []

    with bpy.data.libraries.load(str(blend_path.resolve()), link=False) as (src, dst):
        dst.objects = list(src.objects)

    mesh_override = _ASSET_MESH_NAME.get(class_name, "FIRST")  # sentinel = take first

    mesh_objs = []
    for obj in dst.objects:
        if obj is None or obj.type != "MESH":
            continue
        if mesh_override == "FIRST":
            # Default: load only the first mesh found
            bpy.context.scene.collection.objects.link(obj)
            obj.hide_render = True
            obj.hide_viewport = True
            mesh_objs.append(obj)
            break
        elif mesh_override is None:
            # Load ALL mesh parts (e.g. dustbin: wheels + lid + body)
            bpy.context.scene.collection.objects.link(obj)
            obj.hide_render = True
            obj.hide_viewport = True
            mesh_objs.append(obj)
        elif obj.name == mesh_override:
            # Load only the specifically named mesh
            bpy.context.scene.collection.objects.link(obj)
            obj.hide_render = True
            obj.hide_viewport = True
            mesh_objs.append(obj)
            break

    if not mesh_objs:
        _loaded_cache[key] = []
        return []

    # Per-asset post-load setup
    if class_name == "stop sign":
        _apply_stop_sign_texture(mesh_objs[0], assets_root)

    names = [obj.name for obj in mesh_objs]
    _loaded_cache[key] = names
    return names


def _apply_speed_limit_texture(obj: bpy.types.Object, speed_limit: int) -> None:
    """Generate a number texture for the speed limit sign and apply it.

    Uses PIL to render the number onto a white background, loads the image
    into Blender, and wires it into the sign mesh's material.

    Args:
        obj:         The speed limit sign instance (already duplicated).
        speed_limit: Integer speed limit to display, e.g. 35.
    """
    import os
    import tempfile

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return  # Pillow not available inside Blender — skip texture

    if not obj.data or not obj.data.materials:
        return

    # ── generate texture ───────────────────────────────────────────────────
    size = 512
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Draw a red border (speed limit signs have a red ring)
    border = 30
    draw.ellipse([border, border, size - border, size - border],
                 outline="red", width=20)

    # Draw the number
    text = str(speed_limit)
    font_size = 180
    font = None
    for font_name in ("arial.ttf", "DejaVuSans-Bold.ttf", "FreeSansBold.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    bbox_t = draw.textbbox((0, 0), text, font=font)
    tw = bbox_t[2] - bbox_t[0]
    th = bbox_t[3] - bbox_t[1]
    draw.text(((size - tw) // 2, (size - th) // 2), text, fill="black", font=font)

    # ── load into Blender ──────────────────────────────────────────────────
    img_name = f"SpeedLimit_{speed_limit}"
    bpy_img = bpy.data.images.get(img_name)
    if bpy_img is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        bpy_img = bpy.data.images.load(tmp.name)
        bpy_img.name = img_name
        os.unlink(tmp.name)

    # ── wire into material ─────────────────────────────────────────────────
    mat = obj.data.materials[0]
    if mat is None:
        return
    mat = mat.copy()                          # don't modify the template's material
    obj.data.materials[0] = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex  = nodes.new("ShaderNodeTexImage")
    tex.image = bpy_img
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.6
    out  = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex.outputs["Color"],  bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"],  out.inputs["Surface"])


def _apply_track_colour(obj: bpy.types.Object, track_id: int) -> None:
    """Apply a stable per-track-ID colour to a vehicle mesh instance.

    Creates one shared Principled BSDF material per track_id (cached in
    bpy.data.materials) and assigns it to the instance's first material slot.
    Colours are drawn from _TRACK_HUES cycled by track_id, giving each tracked
    vehicle a distinct paint colour that is consistent across all frames.
    """
    import colorsys

    mat_name = f"Vehicle_Track_{track_id}"
    if mat_name not in bpy.data.materials:
        hue = _TRACK_HUES[track_id % len(_TRACK_HUES)]
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.80)

        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value  = (r, g, b, 1.0)
        bsdf.inputs["Roughness"].default_value   = 0.4
        bsdf.inputs["Metallic"].default_value    = 0.1
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    mat = bpy.data.materials[mat_name]
    if obj.data and hasattr(obj.data, "materials"):
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat


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
