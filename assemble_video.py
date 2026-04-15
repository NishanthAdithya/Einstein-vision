#!/usr/bin/env python3
"""
Einstein Vision - Video Assembly Script
Cuts clips from all sources and assembles the final 2-min pitch video.
Run: python assemble_video.py
Output: einstein_vision_final.mp4
"""

import subprocess
import os
from pathlib import Path

BASE  = Path("c:/Users/adith/Downloads/Einstein-vision")
CLIPS_DIR = BASE / "video_clips"
CLIPS_DIR.mkdir(exist_ok=True)
FINAL = BASE / "einstein_vision_final.mp4"

W, H     = 1920, 1080
BAR_H    = 90          # cinematic letterbox bar height (top + bottom)
FADE_DUR = 0.35        # crossfade duration in seconds

# ---------------------------------------------------------------------------
# Clip list: (relative_source, start_sec, duration_sec, label)
# label="" → no text overlay (cold open, outro)
# ---------------------------------------------------------------------------
CLIPS = [
    # ── Cold Open: raw stock city footage, no text ──────────────────────────
    ("19224196-uhd_3840_2160_25fps.mp4",                          0.0,  11.0, ""),

    # ── Detection + Tracking ────────────────────────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene1_front_perception.mp4",     5.0,   7.0, "Object Detection  +  Multi-Object Tracking"),

    # ── Blender 3D Reveal ───────────────────────────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene1_chase.mp4",               10.0,  10.0, "Einstein Vision"),

    # ── Traffic Lights + Lanes ──────────────────────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene10_front_perception.mp4",    5.0,   8.0, "Traffic Lights  |  Lanes  |  3D Lifting"),

    # ── 3D scene from different angle ───────────────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene10_chase.mp4",              10.0,   8.0, "3D Scene Reconstruction"),

    # ── Depth + debug overlays ──────────────────────────────────────────────
    ("outputs/scene1/debug_front.mp4",                            8.0,   7.0, "Monocular Depth Estimation"),

    # ── Blender full render ─────────────────────────────────────────────────
    ("outputs/scene5/render_scene5.mp4",                         12.0,   8.0, "Blender World Model"),

    # ── Optical flow + motion classification ────────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene6_front_perception.mp4",     4.0,   7.0, "Optical Flow  +  Motion Classification"),

    # ── Scene6 3D chase ─────────────────────────────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene6_chase.mp4",                5.0,   7.0, "Vehicle Sub-classification"),

    # ── Collision prediction scene ───────────────────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene13_chase.mp4",               5.0,   8.0, "Collision Prediction  +  Brake Detection"),

    # ── Render scene4 ───────────────────────────────────────────────────────
    ("outputs/scene4/render_scene4.mp4",                          0.0,  10.0, "Full Pipeline"),

    # ── Outro: Blender late-angle + title card ───────────────────────────────
    ("Group5_p3ph2/Group5_p3ph2/scene1_chase.mp4",               45.0,  11.0, "Einstein Vision  |  RBE 549  |  Spring 2026"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vf_for_clip(label: str, duration: float, is_first: bool, is_last: bool,
                is_low_fps: bool) -> str:
    """Build the full -vf filter string for one clip."""
    parts = []

    # 1. Scale to fill 1920x1080, crop centre (handles both 4:3 and 16:9 sources)
    parts.append(
        f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={W}:{H}"
    )

    # 2. Frame rate — use blend-mode motion interpolation for very low fps sources
    if is_low_fps:
        parts.append("minterpolate=fps=25:mi_mode=blend")
    else:
        parts.append("fps=25")

    # 3. Cinematic letterbox bars (drawn OVER the image)
    parts.append(f"drawbox=x=0:y=0:w={W}:h={BAR_H}:color=black:t=fill")
    parts.append(f"drawbox=x=0:y={H - BAR_H}:w={W}:h={BAR_H}:color=black:t=fill")

    # 4. Colour grade — cool/cinematic tone
    parts.append("eq=contrast=1.12:saturation=0.82:brightness=-0.02:gamma_b=0.93")

    # 5. Text label (centered, just above bottom bar)
    if label:
        text_y   = H - BAR_H - 58
        fade_in  = 0.4
        fade_out = 0.4
        alpha = (
            f"if(lt(t,{fade_in}),t/{fade_in},"
            f"if(gt(t,{duration - fade_out}),({duration}-t)/{fade_out},1))"
        )
        safe = label.replace("'", "\u2019").replace(":", "\\:")
        parts.append(
            f"drawtext="
            f"font='Arial Bold':"
            f"text='{safe}':"
            f"fontsize=40:"
            f"fontcolor=white:"
            f"x=(w-text_w)/2:"
            f"y={text_y}:"
            f"alpha='{alpha}'"
        )

    # 6. Fade in / out
    fi_dur = 0.6 if is_first else FADE_DUR
    fo_dur = 0.8 if is_last  else FADE_DUR
    parts.append(f"fade=t=in:st=0:d={fi_dur}")
    parts.append(f"fade=t=out:st={duration - fo_dur}:d={fo_dur}")

    return ",".join(parts)


def process_clip(idx: int, source: str, start: float, duration: float,
                 label: str, is_first: bool, is_last: bool) -> Path | None:
    src = BASE / source
    if not src.exists():
        print(f"  [SKIP] Not found: {src}")
        return None

    out = CLIPS_DIR / f"clip_{idx:02d}.mp4"
    if out.exists():
        print(f"  [CACHED] clip_{idx:02d}.mp4")
        return out

    # Chase videos are 6 fps — flag for motion interpolation
    low_fps = "chase" in source

    vf = vf_for_clip(label, duration, is_first, is_last, low_fps)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-an",
        "-r", "25",
        str(out),
    ]

    tag = f"clip_{idx:02d}  [{source.split('/')[-1]}  {start:.0f}s–{start+duration:.0f}s]  '{label}'"
    print(f"  Processing {tag}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-600:]}")
        return None
    print(f"  Done -> {out.name}")
    return out


def build_title_card(text: str, duration: float, out: Path) -> Path | None:
    """Generate a plain black title card with centred white text."""
    if out.exists():
        return out
    vf = (
        f"color=c=black:size={W}x{H}:rate=25:duration={duration},"
        f"drawtext=font='Arial Bold':text='{text}':fontsize=72:"
        f"fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:"
        f"alpha='if(lt(t,0.5),t/0.5,if(gt(t,{duration-0.5}),({duration}-t)/0.5,1))',"
        f"fade=t=out:st={duration-0.5}:d=0.5"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Title card error: {result.stderr[-400:]}")
        return None
    return out


def concatenate(clip_paths: list[Path], output: Path) -> None:
    list_file = CLIPS_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.as_posix()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "17",
        "-an",
        str(output),
    ]
    print(f"\nConcatenating {len(clip_paths)} clips …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Concat error:\n{result.stderr[-800:]}")
    else:
        size_mb = output.stat().st_size / 1_048_576
        print(f"\nDone!  ->  {output}  ({size_mb:.1f} MB)")
        print("Next step: import into CapCut and add background music.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Einstein Vision — Video Assembler")
    print("=" * 60)

    ready: list[Path] = []
    n = len(CLIPS)

    for i, (src, start, dur, label) in enumerate(CLIPS):
        out = process_clip(i, src, start, dur, label,
                           is_first=(i == 0), is_last=(i == n - 1))
        if out:
            ready.append(out)

    if not ready:
        print("No clips processed — aborting.")
        return

    total_s = sum(d for _, _, d, _ in CLIPS[:len(ready)])
    print(f"\n{len(ready)}/{n} clips ready  ({total_s:.0f}s ~ {int(total_s)//60}m {int(total_s)%60}s)")
    concatenate(ready, FINAL)


if __name__ == "__main__":
    main()
