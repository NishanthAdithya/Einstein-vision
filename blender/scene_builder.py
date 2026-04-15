"""scene_builder.py — Main Blender entry point for EinsteinVision rendering.

Run headlessly via:
    blender --background --python blender/scene_builder.py -- \\
        --scene scene1 \\
        --output-root outputs \\
        --assets-root P3Data/Assets \\
        --cameras-config configs/cameras.yaml \\
        [--phase 3]

The script:
  1. Reads the serialised FrameData MessagePack files for every sampled frame.
  2. Clears and rebuilds the Blender scene for each frame.
  3. Spawns vehicle/pedestrian assets at their 3-D ego-frame positions.
  4. Draws lane lines as Bezier curves.
  5. Applies brake light / turn signal emission (Phase 3).
  6. Renders the frame to a PNG.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import bpy

# ---------------------------------------------------------------------------
# Resolve project root + vendor (pyyaml/msgpack bundled for Blender's Python)
# Must happen before any project imports.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_VENDOR_DIR = _SCRIPT_DIR / "vendor"
for _p in (str(_PROJECT_ROOT), str(_VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml

from src.io.calibration_loader import load_camera
from src.io.schema import FrameData
from src.io.serializer import get_output_paths, load_detections

# Blender sub-modules (relative to blender/)
sys.path.insert(0, str(_SCRIPT_DIR))
import asset_manager
import camera_setup
import lane_renderer
import light_animator
import pose_renderer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve relative paths against the project root (Blender may change CWD)
    output_root   = (_PROJECT_ROOT / args.output_root).resolve()
    assets_root   = (_PROJECT_ROOT / args.assets_root).resolve()
    scene_id      = args.scene
    phase         = args.phase
    render_engine = args.engine
    samples       = args.samples

    cam_cfg = load_camera("front", str(_PROJECT_ROOT / args.cameras_config))

    # ── one-time Blender scene setup ─────────────────────────────────────────
    _reset_scene()
    _create_ground_plane()
    cam_obj = camera_setup.setup_camera(
        cam_data={
            "fx": cam_cfg.fx, "fy": cam_cfg.fy,
            "cx": cam_cfg.cx, "cy": cam_cfg.cy,
            "height_above_ground_m": cam_cfg.height_above_ground_m,
            "pitch_deg": cam_cfg.pitch_deg,
        },
        image_width=cam_cfg.image_width,
        image_height=cam_cfg.image_height,
    )
    camera_setup.set_render_settings(
        samples=samples,
        engine=render_engine,
        use_gpu=True,
    )
    _setup_world_lighting()

    # ── collect frame list ───────────────────────────────────────────────────
    det_dir = output_root / scene_id / "detections"
    if not det_dir.exists():
        print(f"[scene_builder] No detections found at {det_dir} — run perception first.")
        return

    msgpack_files = sorted(det_dir.glob("frame_*.msgpack"))
    if args.max_frames > 0:
        msgpack_files = msgpack_files[:args.max_frames]
    print(f"[scene_builder] {len(msgpack_files)} frames to render for {scene_id}")

    renders_dir = output_root / scene_id / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    for msgpack_path in msgpack_files:
        frame_data = load_detections(msgpack_path)
        frame_idx  = frame_data.frame_idx

        render_path = renders_dir / f"frame_{frame_idx:06d}.png"
        if render_path.exists() and not args.force:
            continue

        _render_frame(
            frame_data=frame_data,
            render_path=render_path,
            assets_root=assets_root,
            phase=phase,
        )
        print(f"[scene_builder] Rendered frame {frame_idx:06d} → {render_path.name}")

    print(f"[scene_builder] Done — {scene_id}")


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

def _render_frame(
    frame_data: FrameData,
    render_path: Path,
    assets_root: Path,
    phase: int,
) -> None:
    """Clear the scene, rebuild it from FrameData, and render one PNG."""
    frame_idx = frame_data.frame_idx

    # Remove all previous frame objects (keep camera + lights)
    asset_manager.clear_scene_objects()
    lane_renderer.clear_lanes(frame_idx)
    pose_renderer.clear_poses()

    # ── build set of person bbox centres that have matching pose data ────────
    _posed_person_centres: set = set()
    if phase >= 2 and frame_data.poses:
        for pose in frame_data.poses:
            px = (pose.bbox.x1 + pose.bbox.x2) / 2
            py = (pose.bbox.y1 + pose.bbox.y2) / 2
            _posed_person_centres.add((round(px, 1), round(py, 1)))

    # ── spawn detection assets ────────────────────────────────────────────────
    for det in frame_data.detections:
        if det.pos_3d is None:
            continue

        # Skip person mesh when pose_renderer will draw the skeleton instead
        if det.class_name == "person" and phase >= 2:
            dpx = round((det.bbox.x1 + det.bbox.x2) / 2, 1)
            dpy = round((det.bbox.y1 + det.bbox.y2) / 2, 1)
            if (dpx, dpy) in _posed_person_centres:
                continue

        from src.utils.geometry import ego_to_blender

        # Use estimated yaw from Phase 1/2 geometry
        xyz_b, yaw_b = ego_to_blender(det.pos_3d, det.yaw)

        # Phase 2 — use vehicle sub-class for the asset lookup so the
        # correct 3-D model (sedan/suv/pickup/hatchback) is spawned.
        effective_class = (
            det.vehicle_subclass
            if det.class_name == "car" and det.vehicle_subclass
            else det.class_name
        )

        obj = asset_manager.spawn_asset(
            class_name=effective_class,
            pos_ego=tuple(xyz_b),
            yaw_rad=yaw_b,
            assets_root=assets_root,
            track_id=det.track_id,
            speed_limit=det.speed_limit,
        )

        if obj is None:
            continue

        # Phase 3 — brake lights + turn signals
        if phase >= 3:
            if det.brake_light_on is not None:
                light_animator.apply_brake_light(obj, det.brake_light_on)
            if det.turn_signal:
                light_animator.apply_turn_signal(
                    obj, det.turn_signal, frame=frame_idx, fps=36.0
                )
            # Motion direction arrow for moving vehicles
            if det.is_moving:
                _spawn_motion_arrow(obj, det.velocity_3d, yaw_b)
            # Collision risk highlight (extra credit)
            if det.collision_risk:
                light_animator.apply_collision_highlight(obj)

        # Traffic light colour + arrow + pole (Phase 2+)
        if det.class_name == "traffic light":
            state = det.traffic_light_state or "red"
            light_animator.set_traffic_light_colour(obj, state)
            if det.traffic_light_arrow:
                light_animator.set_traffic_light_arrow(
                    obj, det.traffic_light_arrow, state
                )
            _spawn_tl_pole(obj)

    # ── speed bumps (Phase 3, extra credit) ──────────────────────────────────
    if phase >= 3:
        _SPEED_BUMP_CLASSES = frozenset({
            "speed bump", "speed hump", "raised crosswalk", "road bump",
        })
        for det in frame_data.detections:
            if det.class_name.lower() in _SPEED_BUMP_CLASSES and det.pos_3d is not None:
                from src.utils.geometry import ego_to_blender as _e2b
                xyz_b, _ = _e2b(det.pos_3d, 0.0)
                _spawn_speed_bump(tuple(xyz_b))

    # ── pedestrian poses (Phase 2+) ───────────────────────────────────────────
    if phase >= 2 and frame_data.poses:
        # Inject pos_3d and depth from matched person detections into each pose
        person_dets = [d for d in frame_data.detections if d.class_name == "person" and d.pos_3d]
        from src.utils.geometry import ego_to_blender as _e2b
        for pose in frame_data.poses:
            # Match by bbox centre proximity
            px = (pose.bbox.x1 + pose.bbox.x2) / 2
            py = (pose.bbox.y1 + pose.bbox.y2) / 2
            best, best_dist = None, 1e9
            for det in person_dets:
                dx = (det.bbox.x1 + det.bbox.x2) / 2 - px
                dy = (det.bbox.y1 + det.bbox.y2) / 2 - py
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best = det
            if best and best_dist < 200 ** 2:
                xyz_b, _ = _e2b(best.pos_3d, 0.0)
                pose.pos_3d = tuple(xyz_b)
                pose.depth = best.depth
        pose_renderer.render_poses(frame_data.poses, frame_idx)

    # ── lane lines ────────────────────────────────────────────────────────────
    lane_renderer.render_lanes(frame_data.lanes, frame_idx)

    # ── render ────────────────────────────────────────────────────────────────
    bpy.context.scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _spawn_motion_arrow(
    obj: bpy.types.Object,
    velocity_3d: tuple,
    yaw: float,
) -> None:
    """Place a directional cone arrow near a moving vehicle.

    The arrow points in the vehicle's direction of travel (estimated from
    optical-flow velocity and yaw).  It is placed at ground level slightly
    in front of the vehicle so it is clearly visible.

    Args:
        obj:         The Blender vehicle object (used for position).
        velocity_3d: (vx, vy, vz) ego-frame velocity in m/frame.
        yaw:         Vehicle yaw in radians (Blender convention).
    """
    import math as _math

    x, y, z = obj.location

    # Use yaw-based forward direction (more stable than noisy pixel flow)
    vx, vy, _ = velocity_3d
    speed_hint = _math.sqrt(vx ** 2 + vy ** 2)

    # Arrow direction: primarily along yaw (forward), with lateral component
    # from horizontal flow residual when significant.
    fwd_x = _math.cos(yaw)   # Blender yaw: CCW from +Y
    fwd_y = _math.sin(yaw)

    # Slightly ahead of vehicle to avoid z-fighting with the mesh
    offset = 2.5
    tip_x = x + fwd_x * offset
    tip_y = y + fwd_y * offset

    # Cone pointing in yaw direction, lying on the ground
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.4,
        radius2=0.0,
        depth=1.2,
        location=(tip_x, tip_y, 0.6),
        rotation=(math.pi / 2, 0.0, -yaw + math.pi / 2),
    )
    arrow = bpy.context.active_object
    arrow.name = f"{obj.name}_motion_arrow"

    # Bright cyan emissive material — distinct from brake/turn colours
    mat_name = "MotionArrow"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value    = (0.0, 0.9, 1.0, 1.0)   # cyan
        emit.inputs["Strength"].default_value = 6.0
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat = bpy.data.materials[mat_name]
    if not arrow.data.materials:
        arrow.data.materials.append(mat)
    else:
        arrow.data.materials[0] = mat


def _spawn_speed_bump(pos_ego: tuple) -> None:
    """Place a yellow-striped speed bump rectangle on the ground plane.

    The speed bump is a flat rectangular prism (3.5 m wide × 0.6 m deep ×
    0.12 m tall) centred at ``pos_ego`` with a yellow/white alternating
    stripe emission material for high visibility.

    Args:
        pos_ego: (X, Y, Z) ego-frame / Blender-frame position in metres.
    """
    x, y, _ = pos_ego   # force Z = 0 (ground)

    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(x, y, 0.06),
    )
    bump = bpy.context.active_object
    bump.name = f"SpeedBump_{x:.1f}_{y:.1f}"
    bump.scale = (3.5, 0.6, 0.12)
    bpy.ops.object.transform_apply(scale=True)

    mat_name = "SpeedBumpMat"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Alternating yellow/white emission using a Checker Texture node
        checker = nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = 6.0
        checker.inputs["Color1"].default_value = (1.0, 0.85, 0.0, 1.0)  # yellow
        checker.inputs["Color2"].default_value = (1.0, 1.0,  1.0, 1.0)  # white

        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Strength"].default_value = 3.0
        links.new(checker.outputs["Color"], emit.inputs["Color"])

        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(emit.outputs["Emission"], out.inputs["Surface"])

    bump.data.materials.append(bpy.data.materials[mat_name])


def _spawn_tl_pole(tl_obj: bpy.types.Object) -> None:
    """Add a slim pole cylinder from the ground up to the traffic light base.

    The pole is placed at the same (X, Y) as the traffic light, running from
    Z=0 to Z=tl_obj.location.z - 0.6 (leaving room for the housing body).
    """
    x, y, z = tl_obj.location
    pole_height = max(0.5, z - 0.6)
    pole_z_centre = pole_height / 2.0

    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.06,
        depth=pole_height,
        location=(x, y, pole_z_centre),
    )
    pole = bpy.context.active_object
    pole.name = f"{tl_obj.name}_pole"

    # Dark gray pole material (shared)
    mat_name = "TL_Pole"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.08, 1.0)
        bsdf.inputs["Roughness"].default_value  = 0.8
        out = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    pole.data.materials.append(bpy.data.materials["TL_Pole"])


def _create_ground_plane() -> None:
    """Add a large flat ground plane at Z=0 with a dark asphalt material.

    Named 'GroundPlane' so asset_manager.clear_scene_objects() preserves it
    across frames.  The plane is sized to cover the full forward field of view
    (300 m deep, 200 m wide) centred 100 m ahead of the ego vehicle.
    """
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 100.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "GroundPlane"
    plane.scale = (100.0, 150.0, 1.0)   # 200 m wide × 300 m deep
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("AsphaltMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.06, 0.06, 0.07, 1.0)
    bsdf.inputs["Roughness"].default_value  = 0.95

    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    plane.data.materials.append(mat)


def _reset_scene() -> None:
    """Delete all mesh/curve objects; keep default camera placeholder."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Clean up orphaned data blocks
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.curves):
        if block.users == 0:
            bpy.data.curves.remove(block)


def _setup_world_lighting() -> None:
    """Set a neutral sky / ambient light for the scene."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.30, 0.32, 0.38, 1.0)  # cool overcast sky
        bg_node.inputs["Strength"].default_value = 0.5

    # Add a warm sun lamp for directional lighting + contrast
    if "EinsteinSun" not in bpy.data.objects:
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 20))
        sun = bpy.context.active_object
        sun.name = "EinsteinSun"
        sun.data.energy = 2.5
        sun.data.color = (1.0, 0.92, 0.80)          # warm white
        sun.rotation_euler = (math.radians(50), 0, math.radians(30))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    # Blender passes its own args before '--'; we only care about what's after
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="EinsteinVision Blender renderer")
    p.add_argument("--scene",           required=True)
    p.add_argument("--output-root",     default="outputs")
    p.add_argument("--assets-root",     default="P3Data/Assets")
    p.add_argument("--cameras-config",  default="configs/cameras.yaml")
    p.add_argument("--phase",           type=int, default=1)
    p.add_argument("--engine",          default="BLENDER_EEVEE",
                   choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
    p.add_argument("--samples",         type=int, default=8)
    p.add_argument("--force",           action="store_true")
    p.add_argument("--max-frames",      type=int, default=400,
                   help="Only render first N frames (0 = all, default 400)")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
