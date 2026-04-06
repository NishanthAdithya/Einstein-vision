#!/usr/bin/env python3
"""make_debug_video.py — Overlay detections, 3D boxes, RAFT flow arrows, lanes, and poses.

Usage:
    python scripts/make_debug_video.py --scene scene4
    python scripts/make_debug_video.py --scene scene4 --no-flow --no-3d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.io.calibration_loader import load_camera
from src.io.serializer import load_detections
from src.io.video_reader import VideoReader, find_video
from src.localization.camera import Camera
from src.utils.debug_viz import (
    draw_3d_boxes,
    draw_flow_arrows,
    draw_frame_data,
)


def main() -> None:
    args = _parse_args()
    cfg = yaml.safe_load(open(args.pipeline_config))

    camera_name = args.camera or cfg.get("camera", "front")
    seq_dir     = Path(cfg["paths"]["sequences_dir"])
    output_dir  = Path(cfg["paths"]["output_root"]) / args.scene
    det_dir     = output_dir / "detections"
    video_path  = find_video(seq_dir / args.scene, camera_name, undistorted=True)

    msgpack_files = sorted(det_dir.glob("frame_*.msgpack"))
    if not msgpack_files:
        print(f"No detections found in {det_dir} — run perception first.")
        return

    frame_map = {int(p.stem.split("_")[1]): p for p in msgpack_files}

    # ── camera model for 3-D box projection ─────────────────────────────────
    cam_cfg = load_camera(camera_name, args.cameras_config)
    camera  = Camera(cam_cfg)

    # ── RAFT optical flow model ──────────────────────────────────────────────
    raft = None
    if not args.no_flow:
        try:
            from src.perception.flow.raft import RAFTFlowEstimator
            print("Loading RAFT …")
            raft = RAFTFlowEstimator(weights="raft-things", iters=6, device="cuda")
            print("RAFT ready.")
        except Exception as e:
            print(f"RAFT unavailable ({e}) — skipping flow arrows.")

    # ── video writer ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = output_dir / f"debug_{camera_name}.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )

    sample_every_n = cfg.get("sample_every_n", 1)
    prev_frame = None
    n_written  = 0

    with VideoReader(video_path, sample_every_n=sample_every_n) as reader:
        print(f"Processing {len(reader)} frames from {video_path.name} …")
        for frame_idx, frame_bgr in reader:
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            if frame_idx not in frame_map:
                prev_frame = frame_bgr
                continue

            fd = load_detections(frame_map[frame_idx])

            # Base overlay: bbox + labels + lanes + poses
            annotated = draw_frame_data(frame_bgr, fd)

            # 3-D bounding boxes
            if not args.no_3d:
                annotated = draw_3d_boxes(annotated, fd.detections, camera)

            # RAFT optical flow arrows
            if raft is not None and prev_frame is not None:
                try:
                    flow = raft.predict(prev_frame, frame_bgr)
                    annotated = draw_flow_arrows(annotated, fd.detections, flow)
                except Exception:
                    pass

            writer.write(annotated)
            n_written += 1
            prev_frame = frame_bgr

            if n_written % 50 == 0:
                print(f"  {n_written} frames written …")

    writer.release()
    if raft is not None:
        raft.close()
    print(f"Saved {n_written} frames to: {out_path}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",            required=True)
    p.add_argument("--camera",           default=None)
    p.add_argument("--fps",              type=float, default=36.0)
    p.add_argument("--max-frames",       type=int,   default=400)
    p.add_argument("--no-flow",          action="store_true",
                   help="Skip RAFT optical flow arrows")
    p.add_argument("--no-3d",            action="store_true",
                   help="Skip 3-D bounding box projection")
    p.add_argument("--cameras-config",   default="configs/cameras.yaml")
    p.add_argument("--pipeline-config",  default="configs/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    main()
