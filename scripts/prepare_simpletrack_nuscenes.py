#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import (
    DATA_COMPONENTS,
    PipelineError,
    SIMPLETRACK_ROOT,
    assert_preprocessed_data,
    base_environment,
    check_nuscenes_root,
    ensure_simpletrack_checkout,
    python_command,
    run_command,
    utc_now,
    validation_scene_names,
    write_json,
)


PREPROCESSORS = (
    ("token_info", "token_info.py"),
    ("ts_info", "time_stamp.py"),
    ("calib_info", "sensor_calibration.py"),
    ("ego_info", "ego_pose.py"),
    ("gt_info", "gt_info.py"),
    ("pc", "raw_pc.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official SimpleTrack nuScenes preprocessing for 2 Hz keyframes only."
    )
    parser.add_argument("nuscenes_root", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--debug-scene", help="prepare only one validation scene")
    parser.add_argument("--force", action="store_true", help="rerun existing component files")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def component_ready(folder: Path, key: str, scene: str | None) -> bool:
    relative, extension = DATA_COMPONENTS[key]
    component = folder / relative
    if scene:
        return (component / f"{scene}{extension}").is_file()
    expected = validation_scene_names()
    present = {path.stem for path in component.glob(f"*{extension}")} if component.is_dir() else set()
    return present == expected


def main() -> int:
    args = parse_args()
    try:
        ensure_simpletrack_checkout()
        check_nuscenes_root(args.nuscenes_root)
        args.output_folder.mkdir(parents=True, exist_ok=True)
        script_folder = SIMPLETRACK_ROOT / "preprocessing" / "nuscenes_data"
        completed = []
        for key, filename in PREPROCESSORS:
            if not args.force and component_ready(args.output_folder, key, args.debug_scene):
                print(f"Pominięto gotowy komponent: {key}")
                completed.append(key)
                continue
            command = python_command(
                script_folder / filename,
                "--raw_data_folder",
                args.nuscenes_root.resolve(),
                "--data_folder",
                args.output_folder.resolve(),
                "--mode",
                "2hz",
            )
            if args.debug_scene:
                command.extend(("--scene", args.debug_scene))
            run_command(
                command,
                cwd=script_folder,
                dry_run=args.dry_run,
                env=base_environment(),
            )
            if not args.dry_run and not component_ready(
                args.output_folder, key, args.debug_scene
            ):
                raise PipelineError(f"Preprocessor {filename} nie utworzył komponentu {key}.")
            completed.append(key)

        if not args.dry_run:
            scenes = assert_preprocessed_data(args.output_folder, args.debug_scene)
            write_json(
                args.output_folder / ".simpletrack_2hz_preprocess.json",
                {
                    "created_at": utc_now(),
                    "nuscenes_root": str(args.nuscenes_root.resolve()),
                    "output_folder": str(args.output_folder.resolve()),
                    "split": "val",
                    "mode": "2hz_keyframes",
                    "debug_scene": args.debug_scene,
                    "components": completed,
                    "scene_count_checked": len(scenes),
                },
            )
            print(f"Preprocessing 2 Hz gotowy: {len(scenes)} scen w {args.output_folder}")
        return 0
    except (PipelineError, OSError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
