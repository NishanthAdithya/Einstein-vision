"""light_animator.py — Animate brake lights and turn signals via material emission.

Run inside Blender Python (bpy available).  Phase 3 feature.
"""

from __future__ import annotations

import math
from typing import Optional

import bpy


# Emission strength values
_BRAKE_STRENGTH_ON  = 8.0
_BRAKE_STRENGTH_OFF = 0.5
_TURN_STRENGTH_ON   = 6.0
_TURN_STRENGTH_OFF  = 0.3

_BULB_COLOURS = {
    "red":    (1.0, 0.05, 0.05, 1.0),
    "yellow": (1.0, 0.80, 0.00, 1.0),
    "green":  (0.05, 1.0, 0.15, 1.0),
}

# World-Z offset from obj.location.z (housing base) to each bulb centre.
# Housing is ~3.92 m tall (scale 1.14 × local Y range 3.44 m).
# Bulbs are equally spaced: centres at 1/6, 3/6, 5/6 of housing height.
# 3.92 / 6 = 0.65, 3.92 / 2 = 1.96, 5 * 3.92 / 6 = 3.27
_BULB_OFFSETS = {"red": 3.27, "yellow": 1.96, "green": 0.65}
_BULB_RADIUS  = 0.42   # metres — sized to fill the housing aperture


def apply_brake_light(obj: bpy.types.Object, is_on: bool) -> None:
    """Set the brake-light emission strength on a vehicle object."""
    mat = _find_or_create_light_material(obj, "BrakeLight", (1.0, 0.05, 0.05, 1.0))
    _set_emission_strength(
        mat,
        _BRAKE_STRENGTH_ON if is_on else _BRAKE_STRENGTH_OFF,
    )


def apply_turn_signal(
    obj: bpy.types.Object,
    turn_signal: Optional[str],
    frame: int,
    blink_hz: float = 1.5,
    fps: float = 36.0,
) -> None:
    """Animate a turn signal blink on a vehicle object."""
    if not turn_signal or turn_signal == "none":
        return

    period_frames = fps / blink_hz
    blink_on = (frame % int(round(period_frames))) < int(round(period_frames / 2))

    colour = (1.0, 0.45, 0.0, 1.0)  # amber
    mat = _find_or_create_light_material(obj, f"TurnSignal_{turn_signal}", colour)
    _set_emission_strength(
        mat,
        _TURN_STRENGTH_ON if blink_on else _TURN_STRENGTH_OFF,
    )


def set_traffic_light_colour(
    obj: bpy.types.Object,
    state: str,
) -> None:
    """Set the active light colour on a traffic signal asset.

    Strategy 1: if the asset has named material slots ('red'/'yellow'/'green'),
    set emission strength on each slot.

    Strategy 2 (fallback for assets with no bulb materials, e.g. TrafficSignal.blend):
    apply a pre-built emission material directly to the mesh, tinting the entire
    housing with the detected state colour.  Three shared materials are created
    once and reused across instances.

    Args:
        obj:   The Blender traffic light object.
        state: 'red', 'yellow', or 'green'.
    """
    if state not in _BULB_COLOURS:
        state = "red"

    # ── strategy 1: named material slots in the asset ────────────────────────
    found_any = False
    if obj.data and hasattr(obj.data, "materials"):
        for slot in obj.data.materials:
            if slot is None:
                continue
            slot_lower = slot.name.lower()
            for bulb in ("red", "yellow", "green"):
                if bulb in slot_lower:
                    found_any = True
                    _ensure_emission_nodes(slot)
                    if bulb == state:
                        _set_emission_colour(slot, _BULB_COLOURS[bulb])
                        _set_emission_strength(slot, 8.0)
                    else:
                        _set_emission_colour(slot, (0.02, 0.02, 0.02, 1.0))
                        _set_emission_strength(slot, 0.1)

    if found_any:
        return

    # ── strategy 2: dark housing + glowing bulb sphere in front of face ─────────
    # Apply a shared dark-gray material to the whole housing so it reads as a
    # black box.  Then add/reuse a small emissive sphere parented to the object
    # at the active bulb position in front of the face (world -Y direction after
    # our rotation fix maps local X → world -Y).
    _apply_dark_housing(obj)
    _ensure_bulb_sphere(obj, state)


def _apply_dark_housing(obj: bpy.types.Object) -> None:
    """Assign a shared dark-gray matte material to a traffic light housing."""
    mat_name = "TL_Housing"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.04, 0.04, 0.04, 1.0)
        bsdf.inputs["Roughness"].default_value  = 0.9
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat = bpy.data.materials["TL_Housing"]
    if obj.data and hasattr(obj.data, "materials"):
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat


def _ensure_bulb_sphere(obj: bpy.types.Object, state: str) -> None:
    """Create or update an emissive sphere at the active bulb position.

    The sphere is placed in world space directly in front of the signal face
    (offset -0.6 m in world Y, which is the camera direction after our
    rotation fix) at the correct vertical offset for the active state.
    The sphere is NOT parented so it stays at world coords each frame.
    Old spheres from previous frames are cleaned up by clear_scene_objects.
    """
    sphere_name = f"{obj.name}_bulb"
    # Remove any existing bulb sphere for this TL instance
    old = bpy.data.objects.get(sphere_name)
    if old is not None:
        mesh = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    colour = _BULB_COLOURS.get(state, _BULB_COLOURS["red"])
    z_off  = _BULB_OFFSETS.get(state, 0.0)
    x, y, z = obj.location

    # Housing front face is at obj.y - 1.08 m (local X=0.95 × scale 1.14 → world -Y).
    # Place sphere 1.3 m in front to sit visibly outside the housing.
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=_BULB_RADIUS, segments=12, ring_count=8,
        location=(x, y - 1.3, z + z_off),
    )
    sphere = bpy.context.active_object
    sphere.name = sphere_name

    mat_name = f"TL_Bulb_{state}"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value    = colour
        emit.inputs["Strength"].default_value = 12.0
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat = bpy.data.materials[mat_name]
    if not sphere.data.materials:
        sphere.data.materials.append(mat)
    else:
        sphere.data.materials[0] = mat


def set_traffic_light_arrow(
    obj: bpy.types.Object,
    arrow: str,
    state: str,
) -> None:
    """Overlay a directional arrow disc on the active traffic light bulb.

    Creates (or reuses) a flat arrow-shaped plane child object positioned at
    the active bulb, oriented to face the camera (+Y direction).

    Args:
        obj:   The Blender traffic light object.
        arrow: 'left', 'right', 'straight', or 'uturn'.
        state: Active light state used to find the correct bulb position.
    """
    if not arrow or arrow == "none":
        return
    if state not in _BULB_OFFSETS:
        state = "red"

    # Place arrow plane in WORLD space, co-located with the bulb sphere.
    # After the TL rotation (π/2, 0, -π/2), the face direction is world -Y,
    # so we push the plane 1.35 m in world -Y (same as the bulb sphere) and
    # at the correct world Z for the active bulb.
    x, y, z = obj.location
    z_off    = _BULB_OFFSETS[state]
    world_loc = (x, y - 1.35, z + z_off)

    arrow_obj_name = f"{obj.name}_arrow"
    # Previous-frame arrow is already removed by clear_scene_objects.
    bpy.ops.mesh.primitive_plane_add(size=0.35, location=world_loc)
    arrow_obj = bpy.context.active_object
    arrow_obj.name = arrow_obj_name
    # No parent — world-space placement avoids TL rotation confusion.

    # Rotate plane so its face (normal +Z by default) points toward camera (-Y).
    arrow_obj.rotation_euler = (math.pi / 2, 0.0, 0.0)

    # Assign an arrow emission material
    _arrow_rot = {
        "left":     0.0,
        "right":    math.pi,
        "straight": math.pi / 2,
        "uturn":    -math.pi / 2,
    }
    colour = _BULB_COLOURS.get(state, _BULB_COLOURS["red"])
    mat_name = f"TL_Arrow_{arrow}_{state}"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = colour
        emit.inputs["Strength"].default_value = 6.0
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat = bpy.data.materials[mat_name]
    if not arrow_obj.data.materials:
        arrow_obj.data.materials.append(mat)
    else:
        arrow_obj.data.materials[0] = mat


# ---------------------------------------------------------------------------
# Collision-risk highlight (extra credit)
# ---------------------------------------------------------------------------

_COLLISION_STRENGTH_ON = 10.0


def apply_collision_highlight(obj: bpy.types.Object) -> None:
    """Overlay a bright red emissive material on a collision-risk object.

    Creates (or reuses) a shared 'CollisionRisk' material and appends it to
    the object's material slots so the red glow blends with the base material.

    Args:
        obj: The Blender vehicle or pedestrian object.
    """
    mat_name = "CollisionRisk"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value    = (1.0, 0.02, 0.02, 1.0)  # bright red
        emit.inputs["Strength"].default_value = _COLLISION_STRENGTH_ON
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(emit.outputs["Emission"], out.inputs["Surface"])

    mat = bpy.data.materials[mat_name]
    if obj.data and hasattr(obj.data, "materials"):
        # Append so the red glow mixes with the base vehicle colour
        if mat_name not in [m.name for m in obj.data.materials if m]:
            obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _find_or_create_light_material(
    obj: bpy.types.Object,
    mat_name: str,
    colour: tuple,
) -> bpy.data.materials:
    """Return an existing emission material by name, or create and attach it."""
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        if mat_name not in [m.name for m in obj.data.materials if m]:
            obj.data.materials.append(mat)
        return mat

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = colour
    emit.inputs["Strength"].default_value = 0.5

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    if obj.data and hasattr(obj.data, "materials"):
        obj.data.materials.append(mat)

    return mat


def _ensure_emission_nodes(mat) -> None:
    """Convert a material to an emission node setup if not already."""
    mat.use_nodes = True
    tree = mat.node_tree
    emit = next((n for n in tree.nodes if n.type == "EMISSION"), None)
    if emit:
        return
    nodes = tree.nodes
    links = tree.links
    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 0.5
    links.new(emit.outputs["Emission"], out.inputs["Surface"])


def _set_emission_colour(mat, colour: tuple) -> None:
    """Set the Emission colour on a material."""
    if not mat.use_nodes:
        return
    for node in mat.node_tree.nodes:
        if node.type == "EMISSION":
            node.inputs["Color"].default_value = colour
            return


def _set_emission_strength(mat: bpy.data.materials, strength: float) -> None:
    """Set the Emission > Strength input on a node-based material."""
    if not mat.use_nodes:
        return
    for node in mat.node_tree.nodes:
        if node.type == "EMISSION":
            node.inputs["Strength"].default_value = strength
            return
