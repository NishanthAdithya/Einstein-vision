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

    # ── spawn detection assets ────────────────────────────────────────────────
    for det in frame_data.detections:
        if det.pos_3d is None:
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

        # Traffic light colour + arrow (Phase 2+)
        if det.class_name == "traffic light":
            state = det.traffic_light_state or "red"
            light_animator.set_traffic_light_colour(obj, state)
            if det.traffic_light_arrow:
                light_animator.set_traffic_light_arrow(
                    obj, det.traffic_light_arrow, state
                )

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
        bg_node.inputs["Color"].default_value = (0.25, 0.28, 0.35, 1.0)  # darker overcast sky
        bg_node.inputs["Strength"].default_value = 0.6

    # Add a sun lamp for directional lighting
    if "EinsteinSun" not in bpy.data.objects:
        bpy.ops.object.light_add(type="SUN", location=(0, 0, 20))
        sun = bpy.context.active_object
        sun.name = "EinsteinSun"
        sun.data.energy = 2.0
        sun.rotation_euler = (math.radians(60), 0, math.radians(45))


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
    p.add_argument("--max-frames",      type=int, default=0,
                   help="Only render first N frames (0 = all)")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
