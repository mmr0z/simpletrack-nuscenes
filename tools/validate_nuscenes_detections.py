#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import PipelineError, check_nuscenes_root, write_json
from simpletrack_pipeline.validation import validate_detection_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an official nuScenes detection JSON before SimpleTrack."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument(
        "--velocity-policy",
        choices=("require", "ignore"),
        default="require",
        help="require stores detector velocity; ignore consistently omits it without synthesizing values",
    )
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        check_nuscenes_root(args.nuscenes_root, args.version)
        report = validate_detection_file(
            args.input_json,
            args.nuscenes_root,
            velocity_policy=args.velocity_policy,
            version=args.version,
        )
        if args.report:
            write_json(args.report, report)
        for warning in report["warnings"]:
            print(f"UWAGA: {warning}", file=sys.stderr)
        if not report["valid"]:
            for error in report["errors"]:
                print(f"BŁĄD: {error}", file=sys.stderr)
            print(
                f"Walidacja nie powiodła się ({len(report['errors'])} raportowanych błędów).",
                file=sys.stderr,
            )
            return 2
        summary = report["summary"]
        print(
            "JSON poprawny: "
            f"{summary['sample_entries']} wpisów klatek, {summary['detections']} detekcji, "
            f"velocity obecne/brak={summary['velocity_present']}/{summary['velocity_missing']}."
        )
        return 0
    except PipelineError as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
