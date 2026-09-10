#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import (
    METHOD_NAMES,
    PROJECT_ROOT,
    PipelineError,
    TRACKING_CLASSES,
    check_nuscenes_root,
    format_command,
    write_csv,
    write_json,
)
from simpletrack_pipeline.visualization import (
    load_tracking_results,
    resolve_tracking_result,
)


METHODS = ("bevfusion", "mvp", "lcf3d")
VULNERABLE_CLASSES = {"pedestrian", "bicycle", "motorcycle"}
HEAVY_CLASSES = {"bus", "truck", "trailer"}
CONTEXT_KEYWORDS = {
    "night": ("night", "dark", "difficult lighting"),
    "rain": ("rain", "water reflection"),
    "intersection": ("intersection", "junction", "roundabout", "red light"),
    "construction": ("construction", "barrier", "cones", "crane", "industrial"),
    "parking": ("parking", "parked"),
    "dense_traffic": ("busy", "dense traffic", "congestion", "many vehicles"),
    "vulnerable_users": ("ped", "people", "jaywalker", "cyclist", "bicycle", "bike"),
    "heavy_vehicles": ("bus", "truck", "trailer", "semi"),
    "ego_stationary": ("stationary ego", "wait", "waiting", "stopped", "stop at"),
    "ego_turning": ("turn", "roundabout", "u turn"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Wybiera kontekstowo zróżnicowane sceny nuScenes na podstawie metadanych "
            "i rozbieżności wyników SimpleTrack. Ranking nie jest oficjalną metryką."
        )
    )
    result.add_argument("--nuscenes-root", required=True, type=Path)
    result.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "tracking_results"
    )
    result.add_argument("--bevfusion", type=Path)
    result.add_argument("--mvp", type=Path)
    result.add_argument("--lcf3d", type=Path)
    result.add_argument("--count", type=int, default=12)
    result.add_argument("--match-distance", type=float, default=2.0)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tracking_visualizations" / "scene_selection",
    )
    result.add_argument(
        "--generate-gifs",
        action="store_true",
        help="Po selekcji wygeneruj wyłącznie GIF-y BEV i CAM_FRONT",
    )
    result.add_argument("--camera-channel", default="CAM_FRONT")
    result.add_argument("--fps", type=float, default=2.0)
    result.add_argument("--dpi", type=int, default=90)
    result.add_argument("--without-gt", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def context_tags(description: str) -> list[str]:
    lowered = description.lower()
    return [
        tag
        for tag, keywords in CONTEXT_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]


def load_validation_scenes(root: Path) -> list[dict[str, Any]]:
    from nuscenes.utils.splits import create_splits_scenes

    version = root / "v1.0-trainval"
    scenes = json.loads((version / "scene.json").read_text(encoding="utf-8"))
    samples = json.loads((version / "sample.json").read_text(encoding="utf-8"))
    logs = json.loads((version / "log.json").read_text(encoding="utf-8"))
    sample_by_token = {sample["token"]: sample for sample in samples}
    log_by_token = {log["token"]: log for log in logs}
    validation = set(create_splits_scenes()["val"])
    output = []
    for scene in scenes:
        if scene["name"] not in validation:
            continue
        tokens = []
        token = scene["first_sample_token"]
        while token:
            tokens.append(token)
            token = sample_by_token[token]["next"]
        output.append(
            {
                "scene": scene["name"],
                "description": scene["description"],
                "location": log_by_token[scene["log_token"]]["location"],
                "tokens": tokens,
                "context_tags": context_tags(scene["description"]),
            }
        )
    return sorted(output, key=lambda row: row["scene"])


def grouped_centers(boxes: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for box in boxes:
        name = box.get("tracking_name")
        translation = box.get("translation")
        if name in TRACKING_CLASSES and isinstance(translation, list) and len(translation) == 3:
            grouped[str(name)].append([float(translation[0]), float(translation[1])])
    return {
        name: np.asarray(centers, dtype=float).reshape(-1, 2)
        for name, centers in grouped.items()
    }


def match_frame(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    max_distance: float,
) -> tuple[int, int, float, int]:
    denominator = 0
    matches = 0
    distance_sum = 0.0
    for name in TRACKING_CLASSES:
        left = first.get(name, np.empty((0, 2)))
        right = second.get(name, np.empty((0, 2)))
        denominator += len(left) + len(right)
        if len(left) == 0 or len(right) == 0:
            continue
        distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        accepted = distances[rows, columns] <= max_distance
        matches += int(accepted.sum())
        distance_sum += float(distances[rows[accepted], columns[accepted]].sum())
    return denominator, matches, distance_sum, matches


def analyze_scene(
    scene: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    match_distance: float,
) -> dict[str, Any]:
    tokens = scene["tokens"]
    method_totals = {method: 0 for method in METHODS}
    method_ids = {method: set() for method in METHODS}
    class_totals = {name: 0 for name in TRACKING_CLASSES}
    match_denominator = 0
    match_count = 0
    matched_distance_sum = 0.0
    matched_distance_count = 0

    for token in tokens:
        grouped = {}
        for method in METHODS:
            boxes = results[method][token]
            method_totals[method] += len(boxes)
            method_ids[method].update(str(box.get("tracking_id")) for box in boxes)
            for box in boxes:
                name = box.get("tracking_name")
                if name in class_totals:
                    class_totals[str(name)] += 1
            grouped[method] = grouped_centers(boxes)
        for first, second in (("bevfusion", "mvp"), ("bevfusion", "lcf3d"), ("mvp", "lcf3d")):
            denominator, matches, distance_sum, distance_count = match_frame(
                grouped[first], grouped[second], match_distance
            )
            match_denominator += denominator
            match_count += matches
            matched_distance_sum += distance_sum
            matched_distance_count += distance_count

    frame_count = len(tokens)
    per_frame = {
        method: method_totals[method] / frame_count for method in METHODS
    }
    method_average = sum(per_frame.values()) / len(METHODS)
    mismatch = (
        1.0 - (2.0 * match_count / match_denominator)
        if match_denominator
        else 0.0
    )
    combined_frames = frame_count * len(METHODS)
    return {
        "scene": scene["scene"],
        "location": scene["location"],
        "description": scene["description"],
        "context_tags": scene["context_tags"],
        "frames": frame_count,
        "disagreement_ratio_2m": mismatch,
        "matched_center_distance_m": (
            matched_distance_sum / matched_distance_count
            if matched_distance_count
            else None
        ),
        "mean_boxes_per_frame": method_average,
        "box_count_spread_per_frame": max(per_frame.values()) - min(per_frame.values()),
        "vulnerable_boxes_per_frame": sum(
            class_totals[name] for name in VULNERABLE_CLASSES
        )
        / combined_frames,
        "heavy_boxes_per_frame": sum(class_totals[name] for name in HEAVY_CLASSES)
        / combined_frames,
        **{f"{method}_boxes_per_frame": per_frame[method] for method in METHODS},
        **{f"{method}_tracks": len(method_ids[method]) for method in METHODS},
    }


def add_normalized_score(rows: list[dict[str, Any]]) -> None:
    weights = {
        "disagreement_ratio_2m": 0.55,
        "mean_boxes_per_frame": 0.15,
        "vulnerable_boxes_per_frame": 0.15,
        "heavy_boxes_per_frame": 0.15,
    }
    ranges = {}
    for key in weights:
        values = [float(row[key]) for row in rows]
        ranges[key] = (min(values), max(values))
    for row in rows:
        score = 0.0
        for key, weight in weights.items():
            low, high = ranges[key]
            normalized = (float(row[key]) - low) / (high - low) if high > low else 0.0
            score += weight * normalized
        row["diagnostic_score"] = score


def select_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()

    def choose(reason: str, predicate: Any, key: str) -> None:
        if len(selected) >= count:
            return
        candidates = [
            row for row in rows if row["scene"] not in selected_names and predicate(row)
        ]
        if not candidates:
            return
        winner = max(candidates, key=lambda row: float(row[key]))
        winner["selection_reason"] = reason
        winner["selection_order"] = len(selected) + 1
        selected.append(winner)
        selected_names.add(winner["scene"])

    choose("największa rozbieżność metod", lambda row: True, "disagreement_ratio_2m")
    choose("scena nocna", lambda row: "night" in row["context_tags"], "diagnostic_score")
    choose("deszcz", lambda row: "rain" in row["context_tags"], "diagnostic_score")
    choose("skrzyżowanie/rondo", lambda row: "intersection" in row["context_tags"], "diagnostic_score")
    choose("roboty drogowe/budowa", lambda row: "construction" in row["context_tags"], "diagnostic_score")
    choose("gęsty ruch", lambda row: "dense_traffic" in row["context_tags"], "diagnostic_score")
    choose("dużo niechronionych uczestników", lambda row: True, "vulnerable_boxes_per_frame")
    choose("dużo ciężkich pojazdów", lambda row: True, "heavy_boxes_per_frame")
    for location in sorted({str(row["location"]) for row in rows}):
        choose(
            f"reprezentacja lokalizacji {location}",
            lambda row, location=location: row["location"] == location,
            "diagnostic_score",
        )
    for row in sorted(rows, key=lambda item: float(item["diagnostic_score"]), reverse=True):
        if len(selected) >= count:
            break
        if row["scene"] not in selected_names:
            row["selection_reason"] = "wysoki wynik diagnostyczny"
            row["selection_order"] = len(selected) + 1
            selected.append(row)
            selected_names.add(row["scene"])
    return selected


def write_reports(
    rows: list[dict[str, Any]], selected: list[dict[str, Any]], output_dir: Path, match_distance: float
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking = sorted(rows, key=lambda row: float(row["diagnostic_score"]), reverse=True)
    for index, row in enumerate(ranking, 1):
        row["diagnostic_rank"] = index
        row["selected"] = row in selected
        row.setdefault("selection_reason", "")
        row.setdefault("selection_order", "")
    serializable = []
    for row in ranking:
        item = dict(row)
        item["context_tags"] = ",".join(item["context_tags"])
        serializable.append(item)
    fieldnames = list(serializable[0])
    write_csv(output_dir / "scene_ranking.csv", fieldnames, serializable)
    write_json(
        output_dir / "scene_ranking.json",
        {
            "note": "Ranking diagnostyczny, nie oficjalna metryka nuScenes.",
            "matching": f"Hungarian, ta sama klasa, odległość XY <= {match_distance:g} m",
            "scenes": ranking,
        },
    )
    write_json(
        output_dir / "selected_scenes.json",
        {
            "count": len(selected),
            "scenes": selected,
        },
    )
    (output_dir / "selected_scenes.txt").write_text(
        "".join(f"{row['scene']}\n" for row in selected), encoding="utf-8"
    )
    lines = [
        "# Reprezentatywne sceny do wizualizacji",
        "",
        "Ranking jest diagnostyczny i nie zastępuje oficjalnych metryk nuScenes.",
        "",
        "| # | Scene | Powód | Lokalizacja | Kontekst | Rozbieżność 2 m | Boxy/klatkę | Opis |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in selected:
        description = str(row["description"]).replace("|", "/")
        lines.append(
            f"| {row['selection_order']} | {row['scene']} | {row['selection_reason']} | "
            f"{row['location']} | {', '.join(row['context_tags'])} | "
            f"{row['disagreement_ratio_2m']:.3f} | {row['mean_boxes_per_frame']:.1f} | {description} |"
        )
    (output_dir / "selected_scenes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_gifs(args: argparse.Namespace, selected: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(selected, 1):
        scene = str(row["scene"])
        base = args.output_dir / "gifs" / scene
        common = [
            "--scene", scene,
            "--nuscenes-root", str(args.nuscenes_root),
            "--results-root", str(args.results_root),
            "--gif-only",
            "--fps", str(args.fps),
            "--dpi", str(args.dpi),
        ]
        if not args.without_gt:
            common.append("--show-gt")
        commands = (
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "visualize_tracking_comparison.py"),
                *common,
                "--output-dir", str(base / "bev"),
            ],
            [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "visualize_tracking_camera.py"),
                *common,
                "--camera-channel", args.camera_channel,
                "--output-dir", str(base / args.camera_channel.lower()),
            ],
        )
        print(f"\n[{index}/{len(selected)}] {scene}: {row['selection_reason']}", flush=True)
        for command in commands:
            print(f"+ {format_command(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run() -> int:
    args = parser().parse_args()
    try:
        check_nuscenes_root(args.nuscenes_root)
        if not 1 <= args.count <= 150:
            raise PipelineError("--count musi należeć do zakresu 1..150.")
        if args.match_distance <= 0 or args.fps <= 0 or args.dpi <= 0:
            raise PipelineError("match-distance, fps i dpi muszą być dodatnie.")
        paths = {
            method: resolve_tracking_result(
                method, getattr(args, method), args.results_root, "scene-0003"
            )
            for method in METHODS
        }
        print("Wczytywanie pełnych Tracking JSON...", flush=True)
        results = load_tracking_results(paths)
        scenes = load_validation_scenes(args.nuscenes_root)
        validation_tokens = {
            token for scene in scenes for token in scene["tokens"]
        }
        for method, method_results in results.items():
            missing = validation_tokens - set(method_results)
            if missing:
                raise PipelineError(
                    f"Ranking wymaga pełnego validation; {METHOD_NAMES[method]} "
                    f"nie zawiera {len(missing)} z {len(validation_tokens)} sample tokenów."
                )
        print(f"Analiza {len(scenes)} scen validation...", flush=True)
        rows = []
        for index, scene in enumerate(scenes, 1):
            rows.append(analyze_scene(scene, results, args.match_distance))
            if index % 10 == 0 or index == len(scenes):
                print(f"  {index}/{len(scenes)}", flush=True)
        add_normalized_score(rows)
        selected = select_rows(rows, args.count)
        write_reports(rows, selected, args.output_dir, args.match_distance)
        print("\nWybrane sceny:")
        for row in selected:
            print(
                f"  {row['selection_order']:2d}. {row['scene']}: {row['selection_reason']} "
                f"(rozbieżność={row['disagreement_ratio_2m']:.3f})"
            )
        if args.generate_gifs:
            generate_gifs(args, selected)
        print(f"\nRaporty: {args.output_dir}")
        return 0
    except (PipelineError, subprocess.CalledProcessError) as exc:
        print(f"BŁĄD: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
