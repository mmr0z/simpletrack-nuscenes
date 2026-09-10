#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import PipelineError, read_json, write_csv, write_json
from tools.evaluate_tracking import PRIMARY_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official nuScenes tracking metrics for multiple SimpleTrack experiments."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="experiment directory or metrics_summary.json; repeat for all methods",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser.parse_args()


def resolve_summary(path: Path) -> Path:
    candidates = (
        path,
        path / "evaluation" / "metrics_summary.json",
        path / "metrics_summary.json",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "metrics_summary.json":
            return candidate
    raise PipelineError(f"Nie znaleziono metrics_summary.json pod {path}")


def load_manifest(summary: Path) -> dict[str, Any] | None:
    candidates = (
        summary.parent.parent / "experiment_info.json",
        summary.parent / "experiment_info.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return read_json(candidate)
    return None


def check_methodology(manifests: list[dict[str, Any] | None], allow: bool) -> None:
    available = [manifest for manifest in manifests if manifest]
    if not available:
        return
    keys = (
        "config_sha256",
        "simpletrack_commit",
        "nuscenes_devkit_version",
        "split",
        "mode",
        "tracking_classes",
        "velocity_policy",
    )
    mismatches = {
        key: sorted({str(manifest.get(key)) for manifest in available})
        for key in keys
        if len({str(manifest.get(key)) for manifest in available}) > 1
    }
    if mismatches and not allow:
        raise PipelineError(
            "Eksperymenty nie używają identycznej metodologii: " + str(mismatches)
        )


def main() -> int:
    args = parse_args()
    try:
        entries = []
        for value in args.input:
            if "=" not in value:
                raise PipelineError(f"Oczekiwano METHOD=PATH, otrzymano {value!r}")
            method, raw_path = value.split("=", 1)
            summary_path = resolve_summary(Path(raw_path))
            summary = read_json(summary_path)
            entries.append((method, summary_path, summary, load_manifest(summary_path)))
        check_methodology([entry[3] for entry in entries], args.allow_mismatch)

        columns = ["Method", *(label for label, _ in PRIMARY_METRICS)]
        global_rows = [
            {
                "Method": method,
                **{label: summary.get(key) for label, key in PRIMARY_METRICS},
            }
            for method, _, summary, _ in entries
        ]
        per_class_rows = []
        for method, _, summary, _ in entries:
            label_metrics = summary.get("label_metrics", {})
            classes = sorted(
                {
                    class_name
                    for values in label_metrics.values()
                    if isinstance(values, dict)
                    for class_name in values
                }
            )
            for class_name in classes:
                per_class_rows.append(
                    {
                        "Method": method,
                        "Class": class_name,
                        **{
                            label: label_metrics.get(key, {}).get(class_name)
                            for label, key in PRIMARY_METRICS
                        },
                    }
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "tracking_comparison.csv", columns, global_rows)
        write_csv(
            args.output_dir / "tracking_comparison_per_class.csv",
            ["Method", "Class", *(label for label, _ in PRIMARY_METRICS)],
            per_class_rows,
        )
        comparison = {
            "global": global_rows,
            "per_class": per_class_rows,
            "sources": {
                method: str(path.resolve()) for method, path, _, _ in entries
            },
        }
        write_json(args.output_dir / "tracking_comparison.json", comparison)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [header, separator]
        for row in global_rows:
            lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
        (args.output_dir / "tracking_comparison.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(f"Porównanie zapisano w: {args.output_dir}")
        return 0
    except (PipelineError, OSError, json.JSONDecodeError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
