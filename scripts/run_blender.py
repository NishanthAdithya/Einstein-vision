#!/usr/bin/env python3
"""run_blender.py — Launch the Blender headless renderer for one or more scenes.

Finds the Blender executable automatically, then calls:

    blender --background --python blender/scene_builder.py -- <args>

Usage:
    # Single scene
    python scripts/run_blender.py --scene scene1

    # All 13 scenes sequentially
    python scripts/run_blender.py --all

    # Phase 3, EEVEE, 32 samples
    python scripts/run_blender.py --scene scene3 --phase 3 \\
        --engine BLENDER_EEVEE_NEXT --samples 32

    # Override Blender path (e.g. custom install)
    python scripts/run_blender.py --scene scene1 --blender /opt/blender/blender

Installing Blender (if not present):
    sudo snap install blender --classic
    # or download from blender.org and set --blender to the binary path
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Common install locations, tried in order
_BLENDER_SEARCH_PATHS = [
    "blender",                          # on PATH (snap / apt / manual)
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    "/snap/bin/blender",
    "/opt/blender/blender",
    str(Path.home() / "blender/blender"),
]

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_SCENE_BUILDER = _PROJECT_ROOT / "blender" / "scene_builder.py"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    blender_bin = _find_blender(args.blender)
    log.info("Blender: %s", blender_bin)

    pipeline_cfg = _load_yaml(args.pipeline_config)

    if args.all:
        scenes = pipeline_cfg.get("scenes", [])
    else:
        scenes = [args.scene]

    if not scenes:
        log.error("No scenes specified — use --scene <id> or --all")
        sys.exit(1)

    log.info("Scenes to render: %s", scenes)

    total_start = time.perf_counter()
    failed = []

    for scene_id in scenes:
        log.info("── Rendering %s ──", scene_id)
        t0 = time.perf_counter()
        ok = _run_blender(blender_bin, scene_id, args)
        elapsed = time.perf_counter() - t0

        if ok:
            log.info("  %s done in %.1fs", scene_id, elapsed)
        else:
            log.error("  %s FAILED after %.1fs", scene_id, elapsed)
            failed.append(scene_id)
            if not args.continue_on_error:
                break

    total = time.perf_counter() - total_start
    log.info("Total time: %.1fs", total)

    if failed:
        log.error("Failed scenes: %s", failed)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _run_blender(blender_bin: str, scene_id: str, args: argparse.Namespace) -> bool:
    """Launch Blender for a single scene. Returns True on success."""
    cmd = [
        blender_bin,
        "--background",
        "--python", str(_SCENE_BUILDER),
        "--",                              # separator: Blender args / script args
        "--scene",         scene_id,
        "--output-root",   args.output_root,
        "--assets-root",   args.assets_root,
        "--cameras-config", args.cameras_config,
        "--phase",         str(args.phase),
        "--engine",        args.engine,
        "--samples",       str(args.samples),
    ]
    if args.force:
        cmd.append("--force")

    log.debug("CMD: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            check=False,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=None if args.verbose else subprocess.PIPE,
        )
    except FileNotFoundError:
        log.error("Blender executable not found: %s", blender_bin)
        return False

    if result.returncode != 0:
        if not args.verbose and result.stderr:
            log.error("Blender stderr:\n%s", result.stderr.decode(errors="replace")[-2000:])
        return False

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_blender(override: str | None) -> str:
    """Return the path to the Blender executable."""
    if override:
        if not Path(override).exists() and not shutil.which(override):
            raise FileNotFoundError(f"Blender not found at: {override}")
        return override

    for candidate in _BLENDER_SEARCH_PATHS:
        found = shutil.which(candidate) or (Path(candidate).exists() and candidate)
        if found:
            return candidate

    raise FileNotFoundError(
        "Blender executable not found. Install with:\n"
        "  sudo snap install blender --classic\n"
        "or download from blender.org and pass --blender /path/to/blender"
    )


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EinsteinVision Blender launcher")

    scene_grp = p.add_mutually_exclusive_group(required=True)
    scene_grp.add_argument("--scene", help="Single scene ID, e.g. 'scene1'")
    scene_grp.add_argument("--all",   action="store_true",
                           help="Render all scenes listed in pipeline.yaml")

    p.add_argument("--blender",         default=None,
                   help="Path to Blender executable (auto-detected if omitted)")
    p.add_argument("--output-root",     default="outputs")
    p.add_argument("--assets-root",     default="P3Data/Assets")
    p.add_argument("--cameras-config",  default="configs/cameras.yaml")
    p.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    p.add_argument("--phase",           type=int, default=1)
    p.add_argument("--engine",          default="CYCLES",
                   choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
    p.add_argument("--samples",         type=int, default=64)
    p.add_argument("--force",           action="store_true",
                   help="Re-render frames that already have output PNGs")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going if one scene fails (default: stop)")
    p.add_argument("--verbose", "-v",   action="store_true",
                   help="Show Blender stdout/stderr")
    return p.parse_args()


if __name__ == "__main__":
    main()
