#!/usr/bin/env python3
"""make_debug_video.py — Overlay detections, lanes, and poses onto a video.

Usage:
    python scripts/make_debug_video.py --scene scene1
    python scripts/make_debug_video.py --scene scene1 --camera front --fps 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from src.io.serializer import load_detections
from src.io.video_reader import VideoReader, find_video
from src.utils.debug_viz import draw_frame_data


def main() -> None:
    args = _parse_args()
    cfg = yaml.safe_load(open(args.pipeline_config))

    camera     = args.camera or cfg.get("camera", "front")
    seq_dir    = Path(cfg["paths"]["sequences_dir"])
    output_dir = Path(cfg["paths"]["output_root"]) / args.scene
    det_dir    = output_dir / "detections"
    video_path = find_video(seq_dir / args.scene, camera, undistorted=True)

    msgpack_files = sorted(det_dir.glob("frame_*.msgpack"))
    if not msgpack_files:
        print(f"No detections found in {det_dir} — run perception first.")
        return

    # Build index: frame_idx → msgpack path
    frame_map = {}
    for p in msgpack_files:
        idx = int(p.stem.split("_")[1])
        frame_map[idx] = p

    # Read one frame to get video dimensions
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = output_dir / f"debug_{camera}.mp4"
    fps = args.fps
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    sample_every_n = cfg.get("sample_every_n", 5)
    n_written = 0

    with VideoReader(video_path, sample_every_n=sample_every_n) as reader:
        print(f"Reading {len(reader)} sampled frames from {video_path.name} ...")
        for frame_idx, frame_bgr in reader:
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            if frame_idx not in frame_map:
                continue
            fd = load_detections(frame_map[frame_idx])
            annotated = draw_frame_data(frame_bgr, fd)
            writer.write(annotated)
            n_written += 1

    writer.release()
    print(f"Saved {n_written} frames to: {out_path}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene",           required=True)
    p.add_argument("--camera",          default=None)
    p.add_argument("--fps",             type=float, default=36.0)
    p.add_argument("--max-frames",      type=int,   default=400)
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    return p.parse_args()


if __name__ == "__main__":
    main()
