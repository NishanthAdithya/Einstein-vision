"""light_animator.py — Animate brake lights and turn signals via material emission.

Run inside Blender Python (bpy available).  Phase 3 feature.
"""

from __future__ import annotations

from typing import Optional

import bpy


# Emission strength values
_BRAKE_STRENGTH_ON  = 8.0
_BRAKE_STRENGTH_OFF = 0.5
_TURN_STRENGTH_ON   = 6.0
_TURN_STRENGTH_OFF  = 0.3


def apply_brake_light(obj: bpy.types.Object, is_on: bool) -> None:
    """Set the brake-light emission strength on a vehicle object.

    Looks for a material slot whose name contains 'brake' or 'rear' (case-
    insensitive).  Falls back to creating a simple red emissive material on
    the last material slot if none is found.

    Args:
        obj:   The Blender vehicle object.
        is_on: True = brake lights active (bright red), False = off (dim).
    """
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
    fps: float = 7.2,
) -> None:
    """Animate a turn signal blink on a vehicle object.

    The signal blinks at blink_hz Hz.  At the effective frame rate (fps),
    this determines whether the light is on or off for a given frame.

    Args:
        obj:         The Blender vehicle object.
        turn_signal: 'left', 'right', 'none', or None.
        frame:       Blender frame number (used to compute blink phase).
        blink_hz:    Blink frequency in Hz.
        fps:         Effective frame rate (36 / sample_every_n).
    """
    if not turn_signal or turn_signal == "none":
        return

    # Compute blink state: on/off alternates at blink_hz
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

    Args:
        obj:   The Blender traffic light object.
        state: 'red', 'yellow', or 'green'.
    """
    colours = {
        "red":    (1.0, 0.05, 0.05, 1.0),
        "yellow": (1.0, 0.80, 0.00, 1.0),
        "green":  (0.05, 1.0, 0.15, 1.0),
    }
    colour = colours.get(state, colours["red"])
    mat = _find_or_create_light_material(obj, f"TrafficLight_{state}", colour)
    _set_emission_strength(mat, 5.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_or_create_light_material(
    obj: bpy.types.Object,
    mat_name: str,
    colour: tuple,
) -> bpy.data.materials:
    """Return an existing emission material by name, or create and attach it."""
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        # Ensure it is attached to this object
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
    emit.inputs["Strength"].default_value = 0.5   # starts dim

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    if obj.data and hasattr(obj.data, "materials"):
        obj.data.materials.append(mat)

    return mat


def _set_emission_strength(mat: bpy.data.materials, strength: float) -> None:
    """Set the Emission > Strength input on a node-based material."""
    if not mat.use_nodes:
        return
    for node in mat.node_tree.nodes:
        if node.type == "EMISSION":
            node.inputs["Strength"].default_value = strength
            return
