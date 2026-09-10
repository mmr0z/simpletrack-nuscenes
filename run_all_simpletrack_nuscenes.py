#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simpletrack_pipeline.common import PROJECT_ROOT, experiment_name, python_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the identical SimpleTrack 2 Hz setup for BEVFusion, MVP and LCF3D."
    )
    parser.add_argument("--bevfusion", type=Path, required=True)
    parser.add_argument("--mvp", type=Path, required=True)
    parser.add_argument("--lcf3d", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tracking_results"))
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--velocity-policy", choices=("require", "ignore"), default="require")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--render-curves", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detection_paths = {
        "bevfusion": args.bevfusion,
        "mvp": args.mvp,
        "lcf3d": args.lcf3d,
    }
    for method, detections in detection_paths.items():
        command: list[object] = [
            PROJECT_ROOT / "run_simpletrack_nuscenes.py",
            "--det-name",
            method,
            "--detections",
            detections,
            "--nuscenes-root",
            args.nuscenes_root,
            "--data-folder",
            args.data_folder,
            "--output",
            args.output,
            "--process",
            args.process,
            "--velocity-policy",
            args.velocity_policy,
        ]
        if args.no_evaluate:
            command.append("--no-evaluate")
        if args.render_curves:
            command.append("--render-curves")
        run_command(python_command(*command), cwd=PROJECT_ROOT)

    if not args.no_evaluate:
        compare: list[object] = [
            PROJECT_ROOT / "tools" / "compare_tracking_results.py",
            "--output-dir",
            args.output,
        ]
        for method in detection_paths:
            compare.extend(
                (
                    "--input",
                    f"{method}={args.output / experiment_name(method)}",
                )
            )
        run_command(python_command(*compare), cwd=PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
