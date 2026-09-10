#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import (
    PipelineError,
    SIMPLETRACK_ROOT,
    assert_preprocessed_data,
    base_environment,
    canonical_method,
    check_nuscenes_root,
    python_command,
    run_command,
    scene_tokens,
    write_json,
)
from simpletrack_pipeline.validation import (
    subset_detection_file,
    validate_detection_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and preprocess a nuScenes detection JSON with upstream detection.py."
    )
    parser.add_argument("--det-name", required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--simpletrack-data", type=Path, required=True)
    parser.add_argument(
        "--velocity-policy", choices=("require", "ignore"), default="require"
    )
    parser.add_argument("--debug-scene")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--skip-validation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        method = canonical_method(args.det_name)
        check_nuscenes_root(args.nuscenes_root)
        scenes = assert_preprocessed_data(args.simpletrack_data, args.debug_scene)
        if not args.input_json.is_file():
            raise PipelineError(f"Brak JSON detekcji: {args.input_json}")

        if not args.skip_validation:
            report = validate_detection_file(
                args.input_json,
                args.nuscenes_root,
                velocity_policy=args.velocity_policy,
            )
            if args.validation_report:
                write_json(args.validation_report, report)
            if not report["valid"]:
                preview = "; ".join(report["errors"][:5])
                raise PipelineError(f"Niepoprawny JSON detekcji: {preview}")

        input_path = args.input_json.resolve()
        if args.debug_scene:
            tokens = scene_tokens(args.simpletrack_data, args.debug_scene)[args.debug_scene]
            input_path = (
                args.simpletrack_data
                / "detection"
                / method
                / f"{args.debug_scene}_input.json"
            )
            if not args.dry_run:
                subset_detection_file(args.input_json, input_path, tokens)

        script = SIMPLETRACK_ROOT / "preprocessing" / "nuscenes_data" / "detection.py"
        command = python_command(
            script,
            "--raw_data_folder",
            args.nuscenes_root.resolve(),
            "--data_folder",
            args.simpletrack_data.resolve(),
            "--det_name",
            method,
            "--file_path",
            input_path,
            "--mode",
            "2hz",
        )
        if args.velocity_policy == "require":
            command.append("--velo")
        run_command(
            command,
            cwd=script.parent,
            dry_run=args.dry_run,
            env=base_environment(),
        )
        if not args.dry_run:
            det_folder = args.simpletrack_data / "detection" / method / "dets"
            missing = [scene for scene in scenes if not (det_folder / f"{scene}.npz").is_file()]
            if missing:
                raise PipelineError(
                    f"Preprocessing detekcji nie utworzył {len(missing)} plików scen."
                )
            print(f"Detekcje {method} gotowe dla {len(scenes)} scen: {det_folder}")
        return 0
    except (PipelineError, OSError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
