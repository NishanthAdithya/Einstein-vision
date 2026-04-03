"""lane_renderer.py — Draw lane lines as Bezier curves via Geometry Nodes.

Run inside Blender Python (bpy available).
"""

from __future__ import annotations

from typing import List, Tuple

import bpy
import mathutils
import numpy as np

from src.io.schema import Lane


# Lane colour in linear RGB (Blender uses linear, not sRGB)
_LANE_COLOURS: dict = {
    "solid-white":  (0.9, 0.9, 0.9, 1.0),
    "dashed-white": (0.7, 0.7, 0.7, 1.0),
    "solid-yellow": (1.0, 0.8, 0.0, 1.0),
    "double":       (1.0, 0.8, 0.0, 1.0),
    "unknown":      (0.5, 0.5, 0.5, 1.0),
}

_LANE_WIDTH = 0.15   # metres — visual width of the rendered stripe


def render_lanes(lanes: List[Lane], frame_idx: int) -> List[bpy.types.Object]:
    """Create one Bezier-curve object per lane line.

    Each lane's bezier_points_3d (N, 3) becomes the control points of a
    Blender Bezier spline.  The curve is extruded slightly and given a
    flat emission material so it reads clearly against the ground.

    Args:
        lanes:      List of Lane objects with bezier_points_3d populated.
        frame_idx:  Used to name objects for per-frame cleanup.

    Returns:
        List of created Blender curve objects.
    """
    created = []
    for i, lane in enumerate(lanes):
        pts = lane.bezier_points_3d  # (N, 3)
        if pts is None or len(pts) < 2:
            continue

        obj = _make_bezier_curve(pts, f"Lane_{frame_idx}_{i}")
        _apply_lane_material(obj, lane.lane_type)
        created.append(obj)

    return created


def clear_lanes(frame_idx: int) -> None:
    """Remove all lane curve objects created for a given frame."""
    prefix = f"Lane_{frame_idx}_"
    to_remove = [
        obj for obj in bpy.data.objects
        if obj.name.startswith(prefix)
    ]
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bezier_curve(points_3d: np.ndarray, name: str) -> bpy.types.Object:
    """Create a Blender Bezier curve object from an (N, 3) array of control points.

    Auto-handles are used so Blender computes smooth tangents automatically.
    """
    curve_data = bpy.data.curves.new(name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = _LANE_WIDTH / 2.0
    curve_data.bevel_resolution = 0
    curve_data.fill_mode = "FULL"

    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points_3d) - 1)

    for i, (x, y, z) in enumerate(points_3d):
        bp = spline.bezier_points[i]
        bp.co = mathutils.Vector((x, y, z))
        bp.handle_left_type  = "AUTO"
        bp.handle_right_type = "AUTO"

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _apply_lane_material(obj: bpy.types.Object, lane_type: str) -> None:
    """Create and assign a flat emission material in the lane's colour."""
    colour = _LANE_COLOURS.get(lane_type, _LANE_COLOURS["unknown"])
    mat_name = f"LaneMat_{lane_type}"

    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = colour
        emit.inputs["Strength"].default_value = 2.0

        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(emit.outputs["Emission"], out.inputs["Surface"])
    else:
        mat = bpy.data.materials[mat_name]

    obj.data.materials.clear()
    obj.data.materials.append(mat)
