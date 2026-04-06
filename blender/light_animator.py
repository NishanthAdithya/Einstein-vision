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

# Vertical offset from the detected object centre (metres) per bulb colour.
# Still used by set_traffic_light_arrow to position the arrow overlay.
_BULB_OFFSETS = {"red": 0.38, "yellow": 0.0, "green": -0.38}
_BULB_RADIUS  = 0.22   # metres — used to position arrow overlays


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

    # ── strategy 2: apply coloured emission material to the mesh ─────────────
    # Three shared materials (one per state) are created once and reused.
    # Each instance already has a COPY of the mesh data (from spawn_asset),
    # so assigning a shared material here is safe.
    mat_name = f"TL_State_{state}"
    if mat_name not in bpy.data.materials:
        colour = _BULB_COLOURS[state]
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Dark diffuse housing + strong emission tint
        mix   = nodes.new("ShaderNodeMixShader")
        diff  = nodes.new("ShaderNodeBsdfDiffuse")
        diff.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
        emit  = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value    = colour
        emit.inputs["Strength"].default_value = 6.0
        mix.inputs["Fac"].default_value = 0.65   # 65% emission, 35% dark diffuse
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(diff.outputs["BSDF"],       mix.inputs[1])
        links.new(emit.outputs["Emission"],   mix.inputs[2])
        links.new(mix.outputs["Shader"],      out.inputs["Surface"])

    mat = bpy.data.materials[mat_name]
    if obj.data and hasattr(obj.data, "materials"):
        if not obj.data.materials:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat


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

    arrow_obj_name = f"{obj.name}_arrow"
    arrow_obj = bpy.data.objects.get(arrow_obj_name)

    if arrow_obj is None:
        # Create a thin rectangular plane representing the arrow
        bpy.ops.mesh.primitive_plane_add(size=0.3, location=(0, 0, 0))
        arrow_obj = bpy.context.active_object
        arrow_obj.name = arrow_obj_name
        arrow_obj.parent = obj
        arrow_obj.matrix_parent_inverse = obj.matrix_world.inverted()

    # Position the arrow at the active bulb's location (parent-relative)
    z_offset = _BULB_OFFSETS[state]
    arrow_obj.location = (0.0, -_BULB_RADIUS - 0.02, z_offset)
    # Face toward camera (-Y in Blender, but object is already rotated)
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
