#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import (
    PipelineError,
    SIMPLETRACK_ROOT,
    TRACKING_CLASSES,
    base_environment,
    python_command,
    run_command,
    scene_tokens,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two official 2 Hz SimpleTrack nuScenes conversion scripts."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--result-folder", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def coordinate_alignment(
    results: dict[str, list[dict]], data_folder: Path, max_samples: int = 2000
) -> dict:
    det_roots = sorted((data_folder / "detection").glob("*/dets"))
    if len(det_roots) != 1:
        return {"checked": False, "reason": f"expected one detection folder, found {len(det_roots)}"}
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scene, tokens in scene_tokens_from_expected.items():
        archive = np.load(det_roots[0] / f"{scene}.npz", allow_pickle=True)
        for index, token in enumerate(tokens):
            candidates[token] = (archive["bboxes"][index], archive["types"][index])

    tracked = [(token, box) for token, boxes in results.items() for box in boxes]
    if len(tracked) > max_samples:
        indices = np.linspace(0, len(tracked) - 1, max_samples, dtype=int)
        tracked = [tracked[index] for index in indices]
    distances: list[float] = []
    without_same_class_detection = 0
    for token, box in tracked:
        bboxes, types = candidates[token]
        matching = [
            index for index, name in enumerate(types) if name == box["tracking_name"]
        ]
        if not matching:
            without_same_class_detection += 1
            continue
        centers = np.asarray([bboxes[index][:2] for index in matching], dtype=float)
        center = np.asarray(box["translation"][:2], dtype=float)
        distances.append(float(np.min(np.linalg.norm(centers - center, axis=1))))
    if not distances:
        return {"checked": False, "reason": "no same-class detections available"}
    values = np.asarray(distances)
    median = float(np.median(values))
    return {
        "checked": True,
        "sampled_tracking_boxes": len(tracked),
        "boxes_compared_to_same_class_detection": len(distances),
        "boxes_without_same_class_detection": without_same_class_detection,
        "nearest_xy_distance_median_m": median,
        "nearest_xy_distance_p95_m": float(np.percentile(values, 95)),
        "nearest_xy_distance_max_m": float(np.max(values)),
        "fraction_within_1m": float(np.mean(values <= 1.0)),
        "fraction_within_5m": float(np.mean(values <= 5.0)),
        "gross_shift_warning": median > 5.0,
        "interpretation": (
            "Tracked global centers are compared with same-frame, same-class input detections. "
            "Small non-zero distances are expected from Kalman prediction and unmatched tracks."
        ),
    }


def validate_tracking_json(path: Path, expected_tokens: set[str], data_folder: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Niepoprawny końcowy Tracking JSON: {exc}") from exc
    results = document.get("results")
    if not isinstance(results, dict):
        raise PipelineError("Końcowy JSON nie zawiera obiektu 'results'.")
    actual = set(results)
    if actual != expected_tokens:
        raise PipelineError(
            f"Niezgodne tokeny Tracking JSON: brak={len(expected_tokens-actual)}, "
            f"nadmiar={len(actual-expected_tokens)}."
        )

    errors: list[str] = []
    ids_by_frame: defaultdict[str, list[int]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    box_count = 0
    max_abs_center = 0.0
    ordered_tokens = [token for tokens in scene_tokens_from_expected.values() for token in tokens]
    frame_index = {token: index for index, token in enumerate(ordered_tokens)}
    for token, boxes in results.items():
        if not isinstance(boxes, list):
            errors.append(f"{token}: wynik nie jest listą")
            continue
        for box in boxes:
            box_count += 1
            required = (
                "sample_token",
                "translation",
                "size",
                "rotation",
                "velocity",
                "tracking_id",
                "tracking_name",
                "tracking_score",
            )
            missing = [field for field in required if field not in box]
            if missing:
                errors.append(f"{token}: brak pól {missing}")
                continue
            if box["sample_token"] != token:
                errors.append(f"{token}: niespójne sample_token w rekordzie")
            if box["tracking_name"] not in TRACKING_CLASSES:
                errors.append(f"{token}: niedozwolona tracking_name {box['tracking_name']!r}")
            values = box["translation"] + box["size"] + box["rotation"] + box["velocity"]
            if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in values):
                errors.append(f"{token}: NaN/Inf albo wartość nieliczbowa")
            if len(box["rotation"]) != 4:
                errors.append(f"{token}: quaternion nie ma długości 4")
            else:
                norm = math.sqrt(sum(float(x) ** 2 for x in box["rotation"]))
                if not 0.99 <= norm <= 1.01:
                    errors.append(f"{token}: quaternion ma normę {norm}")
            max_abs_center = max(max_abs_center, *(abs(float(x)) for x in box["translation"]))
            ids_by_frame[str(box["tracking_id"])].append(frame_index[token])
            class_counts[str(box["tracking_name"])] += 1
    if errors:
        raise PipelineError("Błędy w końcowym Tracking JSON: " + "; ".join(errors[:8]))

    multi_frame = sum(1 for frames in ids_by_frame.values() if len(set(frames)) > 1)
    return {
        "valid": True,
        "sample_count": len(results),
        "frame_count_matches_preprocessing": len(results) == len(expected_tokens),
        "tracking_box_count": box_count,
        "unique_tracking_ids": len(ids_by_frame),
        "ids_observed_in_multiple_frames": multi_frame,
        "class_counts": dict(sorted(class_counts.items())),
        "max_absolute_center_coordinate": max_abs_center,
        "coordinate_check": coordinate_alignment(results, data_folder),
        "coordinate_contract": "No coordinate transform is applied by the pipeline.",
        "official_converter_velocity": (
            "The upstream 2 Hz converter writes [0.0, 0.0] in the required tracking velocity field."
        ),
    }


scene_tokens_from_expected: dict[str, list[str]] = {}


def main() -> int:
    args = parse_args()
    try:
        global scene_tokens_from_expected
        scene_tokens_from_expected = scene_tokens(args.data_folder)
        if not scene_tokens_from_expected:
            raise PipelineError(f"Brak token_info w {args.data_folder}")
        expected = {token for tokens in scene_tokens_from_expected.values() for token in tokens}
        create_script = SIMPLETRACK_ROOT / "tools" / "nuscenes_result_creation.py"
        merge_script = SIMPLETRACK_ROOT / "tools" / "nuscenes_type_merge.py"
        common_args = (
            "--name",
            args.name,
            "--result_folder",
            args.result_folder.resolve(),
        )
        run_command(
            python_command(
                create_script,
                *common_args,
                "--data_folder",
                args.data_folder.resolve(),
                "--obj_types",
                ",".join(TRACKING_CLASSES),
            ),
            cwd=SIMPLETRACK_ROOT,
            dry_run=args.dry_run,
            env=base_environment(),
        )
        run_command(
            python_command(
                merge_script,
                *common_args,
                "--obj_types",
                ",".join(TRACKING_CLASSES),
            ),
            cwd=SIMPLETRACK_ROOT,
            dry_run=args.dry_run,
            env=base_environment(),
        )
        if not args.dry_run:
            result_dir = args.result_folder / args.name / "results"
            upstream_result = result_dir / "results.json"
            final_result = result_dir / "tracking_result.json"
            if not upstream_result.is_file():
                raise PipelineError(f"Brak wyniku merge: {upstream_result}")
            shutil.copy2(upstream_result, final_result)
            report = validate_tracking_json(final_result, expected, args.data_folder)
            write_json(args.result_folder / args.name / "conversion_report.json", report)
            print(f"Tracking JSON gotowy: {final_result}")
        return 0
    except (PipelineError, OSError) as exc:
        print(f"BŁĄD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
