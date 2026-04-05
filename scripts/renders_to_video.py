#!/usr/bin/env python3
"""renders_to_video.py — Compile Blender PNG renders into an MP4 video.

Usage:
    python scripts/renders_to_video.py --scene scene4
    python scripts/renders_to_video.py --scene scene4 --fps 7.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    args = _parse_args()
    renders_dir = Path("outputs") / args.scene / "renders"
    frames = sorted(renders_dir.glob("frame_*.png"))

    if not frames:
        print(f"No PNG frames found in {renders_dir}")
        return

    # Read first frame to get dimensions
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    out_path = Path("outputs") / args.scene / f"render_{args.scene}.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )

    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        if img is not None:
            writer.write(img)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frames)} frames written...")

    writer.release()
    print(f"Done: {len(frames)} frames -> {out_path}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--fps",   type=float, default=36.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
