#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import (
    PROJECT_ROOT,
    PipelineError,
    base_environment,
    check_nuscenes_root,
    run_command,
    write_csv,
    write_json,
)


PRIMARY_METRICS = (
    ("AMOTA", "amota"),
    ("AMOTP", "amotp"),
    ("Recall", "recall"),
    ("MOTAR", "motar"),
    ("MOTA", "mota"),
    ("MOTP", "motp"),
    ("IDS", "ids"),
    ("FP", "fp"),
    ("FN", "fn"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official nuscenes-devkit tracking evaluator and export readable metrics."
    )
    parser.add_argument("tracking_json", type=Path)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-config", type=Path)
    parser.add_argument("--render-curves", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def export_metrics(summary_path: Path, output_dir: Path) -> None:
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Nie można odczytać metrics_summary.json: {exc}") from exc

    global_metrics = {
        label: summary.get(key) for label, key in PRIMARY_METRICS
    }
    all_global = {
        key: value
        for key, value in summary.items()
        if key not in {"label_metrics", "cfg", "meta"}
    }
    label_metrics = summary.get("label_metrics", {})
    classes = sorted(
        {
            class_name
            for values in label_metrics.values()
            if isinstance(values, dict)
            for class_name in values
        }
    )
    per_class: dict[str, dict[str, Any]] = {}
    for class_name in classes:
        per_class[class_name] = {
            metric: values.get(class_name)
            for metric, values in label_metrics.items()
            if isinstance(values, dict)
        }

    readable = {
        "primary_global": global_metrics,
        "all_global": all_global,
        "per_class": per_class,
        "official_metrics_summary": str(summary_path.resolve()),
    }
    write_json(output_dir / "metrics_readable.json", readable)
    write_csv(
        output_dir / "metrics_global.csv",
        [label for label, _ in PRIMARY_METRICS],
        [global_metrics],
    )
    metric_names = sorted(label_metrics)
    write_csv(
        output_dir / "metrics_per_class.csv",
        ["Class", *metric_names],
        [
            {"Class": class_name, **per_class[class_name]}
            for class_name in classes
        ],
    )
    lines = ["nuScenes Tracking metrics (official evaluator)", ""]
    lines.extend(f"{label}: {global_metrics[label]}" for label, _ in PRIMARY_METRICS)
    lines.extend(("", "All global metrics:"))
    lines.extend(f"{key}: {value}" for key, value in sorted(all_global.items()))
    (output_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        if args.split != "val":
            raise PipelineError("Ten pipeline metodologiczny dopuszcza wyłącznie --split val.")
        check_nuscenes_root(args.nuscenes_root, args.version)
        if not args.tracking_json.is_file():
            raise PipelineError(f"Brak Tracking JSON: {args.tracking_json}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "nuscenes_tracking_eval_compat.py"),
            str(args.tracking_json.resolve()),
            "--output_dir",
            str(args.output_dir.resolve()),
            "--eval_set",
            args.split,
            "--dataroot",
            str(args.nuscenes_root.resolve()),
            "--version",
            args.version,
            "--render_curves",
            "1" if args.render_curves else "0",
        ]
        if args.tracking_config:
            command.extend(("--config_path", str(args.tracking_config.resolve())))
        env = base_environment()
        mpl_dir = args.output_dir / ".matplotlib"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_dir.resolve())
        run_command(command, dry_run=args.dry_run, env=env)
        if not args.dry_run:
            summary = args.output_dir / "metrics_summary.json"
            details = args.output_dir / "metrics_details.json"
            if not summary.is_file() or not details.is_file():
                raise PipelineError("Oficjalny evaluator nie zapisał pełnych plików metryk.")
            export_metrics(summary, args.output_dir)
            print(f"Metryki zapisano w: {args.output_dir}")
        return 0
    except (PipelineError, OSError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
