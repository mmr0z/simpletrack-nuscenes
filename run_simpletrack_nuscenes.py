#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

from simpletrack_pipeline.common import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    PipelineError,
    SIMPLETRACK_ROOT,
    TRACKING_CLASSES,
    assert_preprocessed_data,
    canonical_method,
    check_nuscenes_root,
    dependency_snapshot,
    ensure_simpletrack_checkout,
    experiment_name,
    input_signature,
    make_data_view,
    python_command,
    run_command,
    sha256_file,
    simpletrack_commit,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end BEVFusion/MVP/LCF3D -> SimpleTrack 2 Hz -> nuScenes tracking evaluation."
    )
    parser.add_argument("--det-name", required=True, choices=("bevfusion", "mvp", "lcf3d"))
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tracking_results"))
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--velocity-policy",
        choices=("require", "ignore"),
        default="require",
        help="Use the same value for all three methods; ignore never synthesizes velocity.",
    )
    parser.add_argument("--debug-scene", help="run one val scene and skip benchmark evaluation")
    parser.add_argument(
        "--require-preprocessed",
        action="store_true",
        help="fail instead of invoking the official 2 Hz preprocessing when data are absent",
    )
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--render-curves", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"completed_steps": []}
    from simpletrack_pipeline.common import read_json

    value = read_json(path)
    if not isinstance(value, dict):
        raise PipelineError(f"Niepoprawny stan pipeline: {path}")
    value.setdefault("completed_steps", [])
    return value


def mark_step(path: Path, state: dict[str, Any], step: str) -> None:
    if step not in state["completed_steps"]:
        state["completed_steps"].append(step)
    state["updated_at"] = utc_now()
    write_json(path, state)


def methodology(args: argparse.Namespace, method: str) -> dict[str, Any]:
    dependencies = dependency_snapshot()
    return {
        "detector": method,
        "experiment_name": experiment_name(method, args.debug_scene),
        "input_detection": input_signature(args.detections),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "simpletrack_commit": simpletrack_commit(),
        "simpletrack_root": str(SIMPLETRACK_ROOT.resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "dependencies": dependencies,
        "nuscenes_devkit_version": dependencies.get("nuscenes-devkit"),
        "split": "val",
        "mode": "2hz_keyframes",
        "tracking_classes": list(TRACKING_CLASSES),
        "velocity_policy": args.velocity_policy,
        "debug_scene": args.debug_scene,
        "processes": args.process,
        "nuscenes_root": str(args.nuscenes_root.resolve()),
        "simpletrack_data": str(args.data_folder.resolve()),
        "arguments": vars(args) | {
            key: str(value)
            for key, value in vars(args).items()
            if isinstance(value, Path)
        },
        "upstream_changes": [
            "detection.py reads velocity only when --velo is selected",
            "2 Hz preprocessing scripts accept optional --scene for smoke tests",
            "np.int compatibility alias replaced with builtin int",
        ],
    }


def verify_existing_info(path: Path, current: dict[str, Any]) -> None:
    if not path.is_file():
        return
    from simpletrack_pipeline.common import read_json

    previous = read_json(path)
    keys = ("detector", "input_detection", "config_sha256", "velocity_policy", "debug_scene")
    mismatch = [key for key in keys if previous.get(key) != current.get(key)]
    if mismatch:
        raise PipelineError(
            "Katalog eksperymentu zawiera wynik o innej proweniencji ("
            + ", ".join(mismatch)
            + "). Wybierz inny --output."
        )


def main() -> int:
    args = parse_args()
    info_path: Path | None = None
    info: dict[str, Any] | None = None
    try:
        method = canonical_method(args.det_name)
        ensure_simpletrack_checkout()
        check_nuscenes_root(args.nuscenes_root)
        if not args.detections.is_file():
            raise PipelineError(f"Brak detekcji: {args.detections}")
        if not args.config.is_file():
            raise PipelineError(f"Brak configu SimpleTrack: {args.config}")
        if args.process < 1:
            raise PipelineError("--process musi być dodatnie.")

        name = experiment_name(method, args.debug_scene)
        experiment_dir = args.output.resolve() / name
        info_path = experiment_dir / "experiment_info.json"
        state_path = experiment_dir / "pipeline_state.json"
        info = methodology(args, method)
        info["started_at"] = utc_now()
        info["status"] = "dry_run" if args.dry_run else "running"
        if not args.dry_run:
            experiment_dir.mkdir(parents=True, exist_ok=True)
            verify_existing_info(info_path, info)
            write_json(info_path, info)
        state = load_state(state_path) if not args.dry_run else {"completed_steps": []}

        validator_report = experiment_dir / "input_validation.json"
        if "validation" not in state["completed_steps"]:
            run_command(
                python_command(
                    PROJECT_ROOT / "tools" / "validate_nuscenes_detections.py",
                    "--input-json",
                    args.detections.resolve(),
                    "--nuscenes-root",
                    args.nuscenes_root.resolve(),
                    "--velocity-policy",
                    args.velocity_policy,
                    "--report",
                    validator_report,
                ),
                cwd=PROJECT_ROOT,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                mark_step(state_path, state, "validation")
        else:
            print("Pominięto ukończoną walidację wejścia.")

        prepared = True
        try:
            assert_preprocessed_data(args.data_folder, args.debug_scene)
        except PipelineError:
            prepared = False
        if not prepared:
            if args.require_preprocessed:
                raise PipelineError("Brak kompletnego preprocessingu, a podano --require-preprocessed.")
            run_command(
                python_command(
                    PROJECT_ROOT / "scripts" / "prepare_simpletrack_nuscenes.py",
                    args.nuscenes_root.resolve(),
                    args.data_folder.resolve(),
                    *(("--debug-scene", args.debug_scene) if args.debug_scene else ()),
                ),
                cwd=PROJECT_ROOT,
                dry_run=args.dry_run,
            )
        if not args.dry_run:
            assert_preprocessed_data(args.data_folder, args.debug_scene)
            mark_step(state_path, state, "nuscenes_preprocessing_checked")

        data_view = experiment_dir / "simpletrack_data"
        if not args.dry_run:
            make_data_view(args.data_folder, data_view, args.debug_scene)

        if "detection_preprocessing" not in state["completed_steps"]:
            run_command(
                python_command(
                    PROJECT_ROOT / "tools" / "run_simpletrack_detection_preprocess.py",
                    "--det-name",
                    method,
                    "--input-json",
                    args.detections.resolve(),
                    "--nuscenes-root",
                    args.nuscenes_root.resolve(),
                    "--simpletrack-data",
                    data_view,
                    "--velocity-policy",
                    args.velocity_policy,
                    "--skip-validation",
                    *(("--debug-scene", args.debug_scene) if args.debug_scene else ()),
                ),
                cwd=PROJECT_ROOT,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                mark_step(state_path, state, "detection_preprocessing")
        else:
            print("Pominięto ukończony preprocessing detekcji.")

        if "tracking" not in state["completed_steps"]:
            run_command(
                python_command(
                    PROJECT_ROOT / "tools" / "run_tracking.py",
                    "--det-name",
                    method,
                    "--data-folder",
                    data_view,
                    "--result-folder",
                    args.output.resolve(),
                    "--process",
                    args.process,
                    "--config",
                    args.config.resolve(),
                    "--name",
                    name,
                    *(("--debug-scene", args.debug_scene) if args.debug_scene else ()),
                ),
                cwd=PROJECT_ROOT,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                mark_step(state_path, state, "tracking")
        else:
            print("Pominięto ukończony tracking.")

        tracking_json = experiment_dir / "results" / "tracking_result.json"
        if "conversion" not in state["completed_steps"]:
            run_command(
                python_command(
                    PROJECT_ROOT / "tools" / "convert_tracking_results.py",
                    "--name",
                    name,
                    "--data-folder",
                    data_view,
                    "--result-folder",
                    args.output.resolve(),
                ),
                cwd=PROJECT_ROOT,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                mark_step(state_path, state, "conversion")
        else:
            print("Pominięto ukończoną konwersję.")

        evaluate = not args.no_evaluate and not args.debug_scene
        evaluation_dir = experiment_dir / "evaluation"
        evaluation_artifacts = (
            evaluation_dir / "metrics_summary.json",
            evaluation_dir / "metrics_details.json",
            evaluation_dir / "metrics_readable.json",
            evaluation_dir / "metrics_global.csv",
            evaluation_dir / "metrics_per_class.csv",
            evaluation_dir / "metrics.txt",
        )
        if (
            evaluate
            and "evaluation" not in state["completed_steps"]
            and all(path.is_file() for path in evaluation_artifacts)
        ):
            print("Znaleziono kompletne istniejące wyniki evaluatora; pominięto ponowne liczenie.")
            if not args.dry_run:
                mark_step(state_path, state, "evaluation")
        if evaluate and "evaluation" not in state["completed_steps"]:
            evaluate_args: list[object] = [
                PROJECT_ROOT / "tools" / "evaluate_tracking.py",
                tracking_json,
                "--nuscenes-root",
                args.nuscenes_root.resolve(),
                "--split",
                "val",
                "--output-dir",
                evaluation_dir,
            ]
            if args.render_curves:
                evaluate_args.append("--render-curves")
            run_command(
                python_command(*evaluate_args),
                cwd=PROJECT_ROOT,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                mark_step(state_path, state, "evaluation")
        elif args.debug_scene:
            print("Tryb debug-scene: pominięto oficjalną ewaluację pełnego splitu val.")
            if not args.dry_run:
                write_json(
                    experiment_dir / "evaluation_skipped.json",
                    {
                        "reason": "Official val evaluation requires all 150 validation scenes.",
                        "debug_scene": args.debug_scene,
                    },
                )

        if not args.dry_run and info is not None and info_path is not None:
            info["status"] = "completed"
            info["completed_at"] = utc_now()
            info["tracking_json"] = str(tracking_json)
            info["evaluation_dir"] = str(experiment_dir / "evaluation") if evaluate else None
            write_json(info_path, info)
        print(f"Pipeline zakończony: {experiment_dir}")
        return 0
    except Exception as exc:
        if info is not None and info_path is not None and not args.dry_run:
            info["status"] = "failed"
            info["failed_at"] = utc_now()
            info["error"] = str(exc)
            write_json(info_path, info)
        if isinstance(exc, PipelineError):
            print(f"BŁĄD: {exc}", file=sys.stderr)
        else:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
