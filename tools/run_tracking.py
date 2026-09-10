#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import (
    DEFAULT_CONFIG,
    PipelineError,
    SIMPLETRACK_ROOT,
    TRACKING_CLASSES,
    assert_preprocessed_data,
    base_environment,
    canonical_method,
    experiment_name,
    python_command,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official SimpleTrack at nuScenes 2 Hz.")
    parser.add_argument("--det-name", required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--result-folder", type=Path, required=True)
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name")
    parser.add_argument("--debug-scene")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        method = canonical_method(args.det_name)
        if args.process < 1:
            raise PipelineError("--process musi być dodatnie.")
        if not args.config.is_file():
            raise PipelineError(f"Brak configu: {args.config}")
        scenes = assert_preprocessed_data(args.data_folder, args.debug_scene)
        det_dir = args.data_folder / "detection" / method / "dets"
        missing_dets = [scene for scene in scenes if not (det_dir / f"{scene}.npz").is_file()]
        if missing_dets:
            raise PipelineError(f"Brak preprocessingu detekcji dla {len(missing_dets)} scen.")

        name = args.name or experiment_name(method, args.debug_scene)
        script = SIMPLETRACK_ROOT / "tools" / "main_nuscenes.py"
        command = python_command(
            script,
            "--name",
            name,
            "--det_name",
            method,
            "--config_path",
            args.config.resolve(),
            "--result_folder",
            args.result_folder.resolve(),
            "--data_folder",
            args.data_folder.resolve(),
            "--process",
            args.process,
            "--obj_types",
            ",".join(TRACKING_CLASSES),
        )
        run_command(
            command,
            cwd=SIMPLETRACK_ROOT,
            dry_run=args.dry_run,
            env=base_environment(),
        )
        if not args.dry_run:
            summary = args.result_folder / name / "summary"
            missing = [
                f"{obj_type}/{scene}.npz"
                for obj_type in TRACKING_CLASSES
                for scene in scenes
                if not (summary / obj_type / f"{scene}.npz").is_file()
            ]
            if missing:
                raise PipelineError(
                    "SimpleTrack nie utworzył kompletu wyników; pierwsze braki: "
                    + ", ".join(missing[:8])
                )
            print(f"Tracking gotowy: {args.result_folder / name}")
        return 0
    except (PipelineError, OSError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
