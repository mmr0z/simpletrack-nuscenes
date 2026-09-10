#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/simpletrack_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simpletrack_pipeline.common import (
    METHOD_NAMES,
    PROJECT_ROOT,
    PipelineError,
    TRACKING_CLASSES,
    check_nuscenes_root,
    write_json,
)
from simpletrack_pipeline.visualization import (
    boxes_for_frame,
    camera_frame,
    class_set,
    draw_bev_detection_status,
    draw_camera_detection_status,
    ego_frame,
    gt_boxes,
    load_lidar_points,
    load_tracking_results,
    match_detections_to_ground_truth,
    project_lidar_to_camera,
    render_camera_panel,
    render_panel,
    resolve_tracking_result,
    scene_sample_tokens,
    validate_result_coverage,
)


METHODS = ("bevfusion", "mvp", "lcf3d")
CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
VIEWS = ("BEV", *CAMERA_CHANNELS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Tworzy osobne pliki BEV i kamera dla każdego modelu. "
            "Modele ani widoki nigdy nie są łączone na jednym panelu."
        )
    )
    result.add_argument("--scene", action="append", help="Można powtórzyć")
    result.add_argument("--scene-file", type=Path, help="Plik z jedną nazwą sceny na wiersz")
    result.add_argument("--nuscenes-root", required=True, type=Path)
    result.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "tracking_results"
    )
    result.add_argument("--bevfusion", type=Path)
    result.add_argument("--mvp", type=Path)
    result.add_argument("--lcf3d", type=Path)
    result.add_argument(
        "--method", action="append", choices=METHODS, help="Można powtórzyć"
    )
    result.add_argument(
        "--view",
        action="append",
        type=str.upper,
        choices=VIEWS,
        help="Widok do zapisania; można powtórzyć. Domyślnie: BEV i CAM_FRONT.",
    )
    result.add_argument(
        "--camera-channel",
        type=str.upper,
        choices=CAMERA_CHANNELS,
        default="CAM_FRONT",
        help="Zgodność wsteczna: kamera używana, gdy nie podano --view.",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tracking_visualizations" / "individual_models",
    )
    result.add_argument("--frame-index", type=int, default=0)
    result.add_argument("--view-range", type=float, default=50.0)
    result.add_argument("--history", type=int, default=8)
    result.add_argument("--classes", default=",".join(TRACKING_CLASSES))
    result.add_argument("--min-score", type=float, default=0.0)
    result.add_argument("--color-by", choices=("class", "id"), default="class")
    result.add_argument("--labels", action="store_true")
    result.add_argument("--show-gt", action="store_true")
    result.add_argument(
        "--show-detection-status",
        action="store_true",
        help="Oznacz TP/FP/FN i pokaż zestawienie per klasa (tylko wizualizacja).",
    )
    result.add_argument(
        "--match-distance",
        type=float,
        default=2.0,
        help="Próg odległości centrów dla diagnostycznego TP/FP/FN [m].",
    )
    result.add_argument("--no-lidar", action="store_true")
    result.add_argument("--max-bev-lidar-points", type=int, default=20000)
    result.add_argument("--max-camera-lidar-points", type=int, default=6000)
    result.add_argument("--point-size", type=float, default=1.8)
    result.add_argument("--max-depth", type=float, default=80.0)
    result.add_argument("--gif", action="store_true")
    result.add_argument("--gif-only", action="store_true")
    result.add_argument("--fps", type=float, default=2.0)
    result.add_argument("--dpi", type=int, default=90)
    return result


def requested_scenes(explicit: Sequence[str] | None, scene_file: Path | None) -> list[str]:
    scenes = list(explicit or [])
    if scene_file is not None:
        if not scene_file.is_file():
            raise PipelineError(f"Nie znaleziono pliku scen: {scene_file}")
        scenes.extend(
            line.strip()
            for line in scene_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    scenes = list(dict.fromkeys(scenes))
    if not scenes:
        raise PipelineError("Podaj co najmniej jedno --scene albo --scene-file.")
    return scenes


def selected_views(
    explicit: Sequence[str] | None, camera_channel: str = "CAM_FRONT"
) -> list[str]:
    views = list(explicit or ("BEV", camera_channel))
    return list(dict.fromkeys(view.upper() for view in views))


def _status_box(box: Mapping[str, Any], *, ground_truth: bool) -> dict[str, Any]:
    result = {
        "class": str(box.get("tracking_name", "")),
        "translation": [float(value) for value in box["translation"]],
    }
    if ground_truth:
        result["gt_annotation_token"] = str(box.get("tracking_id", ""))
    else:
        result["tracking_id"] = str(box.get("tracking_id", ""))
        result["tracking_score"] = float(box.get("tracking_score", 0.0))
    return result


def detection_status_record(
    *,
    nusc: Any,
    token: str,
    method_results: Mapping[str, Sequence[Mapping[str, Any]]],
    classes: set[str],
    min_score: float,
    match_distance: float,
    frame_index: int,
) -> dict[str, Any]:
    predictions = boxes_for_frame(method_results, token, classes, min_score)
    ground_truth = gt_boxes(nusc, token, classes)
    match = match_detections_to_ground_truth(
        predictions, ground_truth, match_distance
    )
    matched = []
    for prediction_index, gt_index in match.matched_pairs:
        prediction = predictions[prediction_index]
        gt = ground_truth[gt_index]
        distance = np.linalg.norm(
            np.asarray(prediction["translation"][:2], dtype=float)
            - np.asarray(gt["translation"][:2], dtype=float)
        )
        matched.append(
            {
                "class": str(prediction.get("tracking_name", "")),
                "tracking_id": str(prediction.get("tracking_id", "")),
                "gt_annotation_token": str(gt.get("tracking_id", "")),
                "center_distance_m": float(distance),
                "tracking_score": float(prediction.get("tracking_score", 0.0)),
                "prediction_translation": [
                    float(value) for value in prediction["translation"]
                ],
                "gt_translation": [float(value) for value in gt["translation"]],
            }
        )
    return {
        "frame_index": frame_index,
        "sample_token": token,
        "scope": "whole_keyframe_global_coordinates",
        "match_rule": "same class + Hungarian XY center distance",
        "match_distance_m": match_distance,
        "totals": {"tp": match.tp, "fp": match.fp, "fn": match.fn},
        "per_class": match.per_class,
        "matched": matched,
        "false_positives": [
            _status_box(predictions[index], ground_truth=False)
            for index in match.unmatched_predictions
        ],
        "false_negatives": [
            _status_box(ground_truth[index], ground_truth=True)
            for index in match.unmatched_ground_truth
        ],
    }


def make_bev_figure() -> tuple[Any, Any]:
    return plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)


def make_camera_figure() -> tuple[Any, Any]:
    return plt.subplots(figsize=(12, 7), constrained_layout=True)


def draw_bev_frame(
    figure: Any,
    axis: Any,
    *,
    nusc: Any,
    args: argparse.Namespace,
    scene: str,
    method: str,
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    classes: set[str],
    frame_index: int,
) -> None:
    axis.clear()
    token = tokens[frame_index]
    ground_truth = (
        gt_boxes(nusc, token, classes)
        if args.show_gt or args.show_detection_status
        else []
    )
    predictions = boxes_for_frame(
        results[method], token, classes, args.min_score
    )

    bev_frame = ego_frame(nusc, token)
    bev_lidar = (
        None
        if args.no_lidar
        else bev_frame.lidar_to_ego(
            load_lidar_points(bev_frame.lidar_path, args.max_bev_lidar_points)
        )
    )
    render_panel(
        axis,
        title=f"{METHOD_NAMES[method]} — BEV",
        methods=[method],
        method_results=results,
        tokens=tokens,
        frame_index=frame_index,
        frame=bev_frame,
        classes=classes,
        min_score=args.min_score,
        view_range=args.view_range,
        history_length=args.history,
        color_by=args.color_by,
        lidar_points=bev_lidar,
        ground_truth=ground_truth if args.show_gt else [],
        labels=args.labels,
        legend=True,
    )
    if args.show_detection_status:
        match = match_detections_to_ground_truth(
            predictions, ground_truth, args.match_distance
        )
        draw_bev_detection_status(
            axis,
            predictions=predictions,
            ground_truth=ground_truth,
            match=match,
            frame=bev_frame,
            view_range=args.view_range,
        )
    figure.suptitle(
        f"{METHOD_NAMES[method]} + SimpleTrack 2 Hz — {scene} — "
        f"klatka {frame_index + 1}/{len(tokens)}\n{token}",
        fontsize=14,
    )


def draw_camera_frame(
    figure: Any,
    axis: Any,
    *,
    nusc: Any,
    args: argparse.Namespace,
    scene: str,
    method: str,
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    classes: set[str],
    frame_index: int,
    camera_channel: str,
) -> None:
    axis.clear()
    token = tokens[frame_index]
    ground_truth = (
        gt_boxes(nusc, token, classes)
        if args.show_gt or args.show_detection_status
        else []
    )
    predictions = boxes_for_frame(
        results[method], token, classes, args.min_score
    )

    camera = camera_frame(nusc, token, camera_channel)
    if args.no_lidar:
        from PIL import Image

        with Image.open(camera.image_path) as image_handle:
            image = np.asarray(image_handle.convert("RGB"))
        projected = None
        depths = None
    else:
        projected, depths, image = project_lidar_to_camera(
            nusc, token, camera_channel, args.max_camera_lidar_points
        )
    render_camera_panel(
        axis,
        title=f"{METHOD_NAMES[method]} — {camera_channel}",
        methods=[method],
        method_results=results,
        tokens=tokens,
        frame_index=frame_index,
        frame=camera,
        classes=classes,
        min_score=args.min_score,
        history_length=args.history,
        color_by=args.color_by,
        image=image,
        lidar_projection=projected,
        lidar_depths=depths,
        max_depth=args.max_depth,
        point_size=args.point_size,
        ground_truth=ground_truth if args.show_gt else [],
        labels=args.labels,
        legend=True,
    )
    if args.show_detection_status:
        match = match_detections_to_ground_truth(
            predictions, ground_truth, args.match_distance
        )
        draw_camera_detection_status(
            axis,
            predictions=predictions,
            ground_truth=ground_truth,
            match=match,
            frame=camera,
        )
    figure.suptitle(
        f"{METHOD_NAMES[method]} + SimpleTrack 2 Hz — {scene} — "
        f"klatka {frame_index + 1}/{len(tokens)}\n{token}",
        fontsize=14,
    )


def render_scene_method(
    *,
    nusc: Any,
    args: argparse.Namespace,
    scene: str,
    method: str,
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    classes: set[str],
) -> list[Path]:
    destination = args.output_dir / scene / method
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if not args.gif_only:
        if "BEV" in args.views:
            figure, axis = make_bev_figure()
            draw_bev_frame(
                figure,
                axis,
                nusc=nusc,
                args=args,
                scene=scene,
                method=method,
                results=results,
                tokens=tokens,
                classes=classes,
                frame_index=args.frame_index,
            )
            path = destination / f"frame_{args.frame_index:03d}_bev.png"
            figure.savefig(path, dpi=args.dpi)
            plt.close(figure)
            outputs.append(path)

        for camera_channel in (view for view in args.views if view != "BEV"):
            figure, axis = make_camera_figure()
            draw_camera_frame(
                figure,
                axis,
                nusc=nusc,
                args=args,
                scene=scene,
                method=method,
                results=results,
                tokens=tokens,
                classes=classes,
                frame_index=args.frame_index,
                camera_channel=camera_channel,
            )
            path = destination / (
                f"frame_{args.frame_index:03d}_{camera_channel.lower()}.png"
            )
            figure.savefig(path, dpi=args.dpi)
            plt.close(figure)
            outputs.append(path)

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        if "BEV" in args.views:
            figure, axis = make_bev_figure()

            def update_bev(frame_index: int) -> tuple[Any, ...]:
                draw_bev_frame(
                    figure,
                    axis,
                    nusc=nusc,
                    args=args,
                    scene=scene,
                    method=method,
                    results=results,
                    tokens=tokens,
                    classes=classes,
                    frame_index=frame_index,
                )
                print(
                    f"{scene} / {METHOD_NAMES[method]} / BEV: "
                    f"{frame_index + 1}/{len(tokens)}",
                    flush=True,
                )
                return (axis,)

            animation = FuncAnimation(
                figure, update_bev, frames=len(tokens), blit=False
            )
            path = destination / f"{scene}_{method}_bev.gif"
            animation.save(
                path, writer=PillowWriter(fps=args.fps), dpi=min(args.dpi, 90)
            )
            plt.close(figure)
            outputs.append(path)

        for camera_channel in (view for view in args.views if view != "BEV"):
            figure, axis = make_camera_figure()

            def update_camera(frame_index: int) -> tuple[Any, ...]:
                draw_camera_frame(
                    figure,
                    axis,
                    nusc=nusc,
                    args=args,
                    scene=scene,
                    method=method,
                    results=results,
                    tokens=tokens,
                    classes=classes,
                    frame_index=frame_index,
                    camera_channel=camera_channel,
                )
                print(
                    f"{scene} / {METHOD_NAMES[method]} / {camera_channel}: "
                    f"{frame_index + 1}/{len(tokens)}",
                    flush=True,
                )
                return (axis,)

            animation = FuncAnimation(
                figure, update_camera, frames=len(tokens), blit=False
            )
            path = destination / f"{scene}_{method}_{camera_channel.lower()}.gif"
            animation.save(
                path, writer=PillowWriter(fps=args.fps), dpi=min(args.dpi, 90)
            )
            plt.close(figure)
            outputs.append(path)

    status_path = None
    if args.show_detection_status:
        status_path = destination / "detection_status.json"
        frame_indices = range(len(tokens)) if args.gif else (args.frame_index,)
        write_json(
            status_path,
            {
                "scene": scene,
                "method": method,
                "diagnostic_only": True,
                "frames": [
                    detection_status_record(
                        nusc=nusc,
                        token=tokens[index],
                        method_results=results[method],
                        classes=classes,
                        min_score=args.min_score,
                        match_distance=args.match_distance,
                        frame_index=index,
                    )
                    for index in frame_indices
                ],
            },
        )

    write_json(
        destination / "individual_model_visualization_info.json",
        {
            "scene": scene,
            "method": method,
            "method_display_name": METHOD_NAMES[method],
            "views": args.views,
            "frame_count": len(tokens),
            "show_gt": args.show_gt,
            "show_detection_status": args.show_detection_status,
            "diagnostic_match_distance_m": args.match_distance,
            "detection_status_file": str(status_path) if status_path else None,
            "show_lidar": not args.no_lidar,
            "min_score_visual_only": args.min_score,
            "classes": sorted(classes),
            "generated": [str(path) for path in outputs],
        },
    )
    return outputs


def run() -> int:
    args = parser().parse_args()
    try:
        if args.gif_only:
            args.gif = True
        check_nuscenes_root(args.nuscenes_root)
        scenes = requested_scenes(args.scene, args.scene_file)
        methods = list(dict.fromkeys(args.method or METHODS))
        args.views = selected_views(args.view, args.camera_channel)
        classes = class_set(args.classes.split(","))
        if args.history < 1 or args.view_range <= 0:
            raise PipelineError("history i view-range muszą być dodatnie.")
        if args.fps <= 0 or args.dpi <= 0 or args.point_size <= 0:
            raise PipelineError("fps, dpi i point-size muszą być dodatnie.")
        if args.max_depth <= 1 or not 0.0 <= args.min_score <= 1.0:
            raise PipelineError("max-depth > 1 i min-score w [0, 1] są wymagane.")
        if args.match_distance <= 0:
            raise PipelineError("--match-distance musi być dodatni.")
        paths = {
            method: resolve_tracking_result(
                method, getattr(args, method), args.results_root, scenes[0]
            )
            for method in methods
        }
        results = load_tracking_results(paths)

        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(
            version="v1.0-trainval", dataroot=str(args.nuscenes_root), verbose=False
        )
        generated = []
        for scene_index, scene in enumerate(scenes, 1):
            tokens = scene_sample_tokens(nusc, scene)
            validate_result_coverage(results, tokens)
            if not 0 <= args.frame_index < len(tokens):
                raise PipelineError(
                    f"{scene}: --frame-index={args.frame_index} poza zakresem 0..{len(tokens) - 1}."
                )
            for method_index, method in enumerate(methods, 1):
                print(
                    f"\nScena {scene_index}/{len(scenes)}, model {method_index}/{len(methods)}: "
                    f"{scene} / {METHOD_NAMES[method]}",
                    flush=True,
                )
                generated.extend(
                    render_scene_method(
                        nusc=nusc,
                        args=args,
                        scene=scene,
                        method=method,
                        results=results,
                        tokens=tokens,
                        classes=classes,
                    )
                )
        write_json(
            args.output_dir / "individual_models_batch_info.json",
            {
                "scenes": scenes,
                "methods": methods,
                "views": args.views,
                "results": {method: str(path) for method, path in paths.items()},
                "generated": [str(path) for path in generated],
            },
        )
        print("\nWygenerowane pliki:")
        for path in generated:
            print(f"  {path}")
        return 0
    except PipelineError as exc:
        print(f"BŁĄD: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
