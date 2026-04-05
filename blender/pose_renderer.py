"""pose_renderer.py — Draw pedestrian skeletons in Blender from COCO keypoints.

Run inside Blender Python (bpy available).
Each PoseResult's keypoints (133, 3) are used to draw a stick-figure skeleton
using curve objects.  The first 17 keypoints are COCO body joints.
"""

from __future__ import annotations

import math
from typing import List, Optional

import bpy
import mathutils

# COCO 17-keypoint skeleton connections
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9),                             # left arm
    (6, 8), (8, 10),                            # right arm
    (5, 11), (6, 12),                           # torso sides
    (11, 12),                                   # hips
    (11, 13), (13, 15),                         # left leg
    (12, 14), (14, 16),                         # right leg
]

_CONF_THRESHOLD = 0.25
_BONE_RADIUS    = 0.04   # metres — thickness of each bone tube
_BONE_COLOUR    = (0.1, 0.9, 0.3, 1.0)   # bright green RGBA

# Track curve object names created this frame so we can clear them next frame
_pose_object_names: List[str] = []


def clear_poses() -> None:
    """Remove all skeleton curve objects from the previous frame."""
    global _pose_object_names
    to_remove = [
        bpy.data.objects[n]
        for n in _pose_object_names
        if n in bpy.data.objects
    ]
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    _pose_object_names = []


def render_poses(poses, frame_idx: int) -> None:
    """Draw stick-figure skeletons for a list of PoseResult objects.

    Args:
        poses:      List of PoseResult with keypoints (133, 3) in pixel coords.
                    NOTE: keypoints are in 2D pixel space here — we need the
                    3D position from the Detection's pos_3d to place the figure,
                    and scale keypoints relative to bounding box size.
        frame_idx:  Current frame index (used for unique naming).
    """
    global _pose_object_names

    mat = _get_or_create_bone_material()

    for p_idx, pose in enumerate(poses):
        kpts = pose.keypoints   # (133, 3): x, y, conf  — pixel coords
        body = kpts[:17]        # COCO body only

        # Get 3D root position from pose bbox centre — project to ground plane
        # We use the 2D keypoints scaled relative to a unit skeleton then
        # placed at the 3D location stored in pose.pos_3d if available,
        # otherwise place at origin (will be updated once we wire pos_3d).
        root_3d = getattr(pose, "pos_3d", None)
        if root_3d is None:
            # Fall back: estimate from hip keypoints if confident
            hip_l, hip_r = body[11], body[12]
            if hip_l[2] < _CONF_THRESHOLD and hip_r[2] < _CONF_THRESHOLD:
                continue
            # No 3D info — skip (can't place in world)
            continue

        rx, ry, rz = float(root_3d[0]), float(root_3d[1]), float(root_3d[2])

        # Estimate person height from bbox to scale skeleton
        bbox = pose.bbox
        person_height_px = bbox.height
        # Assume average person 1.75 m; scale pixel height to metres
        # Use depth-derived scale: depth * height_px / fy ≈ real height
        depth = getattr(pose, "depth", None) or rz
        fy = 800.0   # approximate — will be close enough for stick figure
        if depth > 0 and person_height_px > 0:
            scale = depth * person_height_px / fy
        else:
            scale = 1.75

        # Build normalised skeleton from pixel keypoints
        # Map pixel keypoints relative to bbox centre, then scale to metres
        cx_px = (bbox.x1 + bbox.x2) / 2.0
        cy_px = (bbox.y1 + bbox.y2) / 2.0

        def kpt_to_3d(kpt):
            """Convert 2D pixel keypoint to 3D Blender position."""
            dx = (kpt[0] - cx_px) / person_height_px * scale if person_height_px > 0 else 0
            dy_px = (cy_px - kpt[1]) / person_height_px * scale if person_height_px > 0 else 0
            return (rx + dx, ry, rz + dy_px)

        # Draw each bone as a separate curve segment
        for i, j in _SKELETON:
            ki, kj = body[i], body[j]
            if ki[2] < _CONF_THRESHOLD or kj[2] < _CONF_THRESHOLD:
                continue

            p1 = kpt_to_3d(ki)
            p2 = kpt_to_3d(kj)

            obj = _create_bone(p1, p2, mat)
            name = f"Pose_{frame_idx}_{p_idx}_{i}_{j}"
            obj.name = name
            _pose_object_names.append(name)

        # Draw keypoint spheres at joints
        for k_idx, kpt in enumerate(body):
            if kpt[2] < _CONF_THRESHOLD:
                continue
            pos = kpt_to_3d(kpt)
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=_BONE_RADIUS * 1.5,
                location=pos,
                segments=6,
                ring_count=4,
            )
            sphere = bpy.context.active_object
            name = f"PoseJoint_{frame_idx}_{p_idx}_{k_idx}"
            sphere.name = name
            if sphere.data.materials:
                sphere.data.materials[0] = mat
            else:
                sphere.data.materials.append(mat)
            _pose_object_names.append(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_bone(
    p1: tuple,
    p2: tuple,
    mat,
) -> bpy.types.Object:
    """Create a cylinder between two 3D points representing a bone."""
    import numpy as np

    a = mathutils.Vector(p1)
    b = mathutils.Vector(p2)
    mid = (a + b) / 2.0
    length = (b - a).length

    if length < 1e-4:
        length = 0.01

    bpy.ops.mesh.primitive_cylinder_add(
        radius=_BONE_RADIUS,
        depth=length,
        location=mid,
        vertices=6,
    )
    obj = bpy.context.active_object

    # Orient cylinder along the bone direction
    direction = (b - a).normalized()
    up = mathutils.Vector((0, 0, 1))
    if abs(direction.dot(up)) > 0.999:
        up = mathutils.Vector((0, 1, 0))
    rot = up.rotation_difference(direction)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rot

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return obj


def _get_or_create_bone_material():
    """Return a shared bright-green emissive material for all bones."""
    name = "PoseBoneMat"
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value    = _BONE_COLOUR
    emit.inputs["Strength"].default_value = 3.0

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    return mat
