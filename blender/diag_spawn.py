import sys
sys.path.insert(0, r"C:\Users\adith\Downloads\Einstein-vision")
sys.path.insert(0, r"C:\Users\adith\Downloads\Einstein-vision\blender")
sys.path.insert(0, r"C:\Users\adith\Downloads\Einstein-vision\blender\vendor")

import bpy
from pathlib import Path

import asset_manager
from src.io.serializer import load_detections
from src.utils.geometry import ego_to_blender

fd = load_detections(r"C:\Users\adith\Downloads\Einstein-vision\outputs\scene4\detections\frame_000100.msgpack")
assets_root = Path(r"C:\Users\adith\Downloads\Einstein-vision\P3Data\Assets")
spawned = 0
skipped = 0
for det in fd.detections:
    if det.pos_3d is None:
        continue
    ec = det.vehicle_subclass if det.class_name == "car" and det.vehicle_subclass else det.class_name
    xyz_b, yaw_b = ego_to_blender(det.pos_3d, det.yaw)
    obj = asset_manager.spawn_asset(ec, tuple(xyz_b), yaw_b, assets_root, det.track_id)
    if obj is not None:
        spawned += 1
        print(f"SPAWNED: {ec} loc={[round(v,1) for v in obj.location]} scale={list(obj.scale)}")
    else:
        skipped += 1
        print(f"SKIPPED: {ec}")
print(f"Total SPAWNED={spawned} SKIPPED={skipped}")
