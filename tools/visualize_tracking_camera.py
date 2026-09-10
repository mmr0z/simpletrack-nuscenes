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
    camera_frame,
    class_set,
    gt_boxes,
    load_tracking_results,
    project_lidar_to_camera,
    render_camera_panel,
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Rzutuje punkty LIDAR_TOP i wyniki SimpleTrack trzech metod na obraz kamery nuScenes."
        )
    )
    result.add_argument("--scene", required=True, help="Nazwa sceny, np. scene-0003")
    result.add_argument("--nuscenes-root", required=True, type=Path)
    result.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "tracking_results"
    )
    result.add_argument("--bevfusion", type=Path, help="Jawna ścieżka Tracking JSON")
    result.add_argument("--mvp", type=Path, help="Jawna ścieżka Tracking JSON")
    result.add_argument("--lcf3d", type=Path, help="Jawna ścieżka Tracking JSON")
    result.add_argument("--camera-channel", choices=CAMERA_CHANNELS, default="CAM_FRONT")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--frame-index", type=int, default=0)
    result.add_argument("--history", type=int, default=8)
    result.add_argument("--classes", default=",".join(TRACKING_CLASSES))
    result.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Tylko filtr wizualny; nie zmienia Tracking JSON",
    )
    result.add_argument("--color-by", choices=("class", "id"), default="class")
    result.add_argument("--labels", action="store_true")
    result.add_argument("--show-gt", action="store_true")
    result.add_argument("--no-lidar", action="store_true")
    result.add_argument("--max-lidar-points", type=int, default=8000)
    result.add_argument("--point-size", type=float, default=2.0)
    result.add_argument("--max-depth", type=float, default=80.0)
    result.add_argument("--all-frames", action="store_true")
    result.add_argument("--gif", action="store_true")
    result.add_argument(
        "--gif-only",
        action="store_true",
        help="Zapisz wyłącznie GIF (automatycznie włącza --gif)",
    )
    result.add_argument("--fps", type=float, default=2.0)
    result.add_argument("--dpi", type=int, default=120)
    return result


def make_figure() -> tuple[Any, Any]:
    return plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)


def frame_resources(
    nusc: Any, token: str, args: argparse.Namespace, classes: set[str]
) -> tuple[Any, np.ndarray, np.ndarray | None, np.ndarray | None, Any]:
    from PIL import Image

    frame = camera_frame(nusc, token, args.camera_channel)
    if args.no_lidar:
        with Image.open(frame.image_path) as image_handle:
            image = np.asarray(image_handle.convert("RGB"))
        points = None
        depths = None
    else:
        points, depths, image = project_lidar_to_camera(
            nusc, token, args.camera_channel, args.max_lidar_points
        )
    ground_truth = gt_boxes(nusc, token, classes) if args.show_gt else []
    return frame, image, points, depths, ground_truth


def draw_comparison(
    figure: Any,
    axes: Any,
    *,
    nusc: Any,
    args: argparse.Namespace,
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    classes: set[str],
    frame_index: int,
) -> None:
    for ax in axes.flat:
        ax.clear()
    token = tokens[frame_index]
    frame, image, points, depths, ground_truth = frame_resources(
        nusc, token, args, classes
    )
    common = dict(
        method_results=results,
        tokens=tokens,
        frame_index=frame_index,
        frame=frame,
        classes=classes,
        min_score=args.min_score,
        history_length=args.history,
        color_by=args.color_by,
        image=image,
        lidar_projection=points,
        lidar_depths=depths,
        max_depth=args.max_depth,
        point_size=args.point_size,
        ground_truth=ground_truth,
        labels=args.labels,
        legend=True,
    )
    for ax, method in zip(axes.flat[:3], METHODS):
        render_camera_panel(
            ax, title=METHOD_NAMES[method], methods=[method], **common
        )
    render_camera_panel(
        axes.flat[3], title="Nałożenie metod", methods=METHODS, **common
    )
    figure.suptitle(
        f"SimpleTrack 2 Hz + LIDAR_TOP → {args.camera_channel} — {args.scene} — "
        f"klatka {frame_index + 1}/{len(tokens)}\n{token}",
        fontsize=14,
    )


def save_single_panels(
    *,
    nusc: Any,
    args: argparse.Namespace,
    output_dir: Path,
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    classes: set[str],
) -> list[Path]:
    frame_index = args.frame_index
    token = tokens[frame_index]
    frame, image, points, depths, ground_truth = frame_resources(
        nusc, token, args, classes
    )
    common = dict(
        method_results=results,
        tokens=tokens,
        frame_index=frame_index,
        frame=frame,
        classes=classes,
        min_score=args.min_score,
        history_length=args.history,
        color_by=args.color_by,
        image=image,
        lidar_projection=points,
        lidar_depths=depths,
        max_depth=args.max_depth,
        point_size=args.point_size,
        ground_truth=ground_truth,
        labels=args.labels,
        legend=True,
    )
    outputs = []
    specs = [(method, [method], METHOD_NAMES[method]) for method in METHODS]
    specs.append(("overlay", list(METHODS), "Nałożenie metod"))
    channel = args.camera_channel.lower()
    for filename, methods, title in specs:
        figure, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        render_camera_panel(ax, title=title, methods=methods, **common)
        figure.suptitle(
            f"{args.scene} — {args.camera_channel} — klatka {frame_index + 1}/{len(tokens)}",
            fontsize=13,
        )
        path = output_dir / f"frame_{frame_index:03d}_{channel}_{filename}.png"
        figure.savefig(path, dpi=args.dpi)
        plt.close(figure)
        outputs.append(path)
    return outputs


def run() -> int:
    args = parser().parse_args()
    try:
        if args.gif_only:
            args.gif = True
        check_nuscenes_root(args.nuscenes_root)
        if args.history < 1 or args.fps <= 0 or args.dpi <= 0:
            raise PipelineError("history, fps i dpi muszą być dodatnie.")
        if args.point_size <= 0 or args.max_depth <= 1:
            raise PipelineError("point-size musi być dodatni, a max-depth większy od 1 m.")
        if not 0.0 <= args.min_score <= 1.0:
            raise PipelineError("--min-score musi należeć do [0, 1].")
        classes = class_set(args.classes.split(","))
        paths = {
            method: resolve_tracking_result(
                method, getattr(args, method), args.results_root, args.scene
            )
            for method in METHODS
        }
        results = load_tracking_results(paths)

        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(
            version="v1.0-trainval", dataroot=str(args.nuscenes_root), verbose=False
        )
        tokens = scene_sample_tokens(nusc, args.scene)
        validate_result_coverage(results, tokens)
        if not 0 <= args.frame_index < len(tokens):
            raise PipelineError(
                f"--frame-index={args.frame_index} poza zakresem 0..{len(tokens) - 1}."
            )

        output_dir = args.output_dir or (
            PROJECT_ROOT
            / "tracking_visualizations"
            / args.scene
            / args.camera_channel.lower()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = []
        channel = args.camera_channel.lower()
        if not args.gif_only:
            generated.extend(
                save_single_panels(
                    nusc=nusc,
                    args=args,
                    output_dir=output_dir,
                    results=results,
                    tokens=tokens,
                    classes=classes,
                )
            )

            figure, axes = make_figure()
            draw_comparison(
                figure,
                axes,
                nusc=nusc,
                args=args,
                results=results,
                tokens=tokens,
                classes=classes,
                frame_index=args.frame_index,
            )
            comparison_path = output_dir / f"frame_{args.frame_index:03d}_{channel}_comparison.png"
            figure.savefig(comparison_path, dpi=args.dpi)
            plt.close(figure)
            generated.append(comparison_path)

        if args.all_frames:
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            figure, axes = make_figure()
            for frame_index in range(len(tokens)):
                draw_comparison(
                    figure,
                    axes,
                    nusc=nusc,
                    args=args,
                    results=results,
                    tokens=tokens,
                    classes=classes,
                    frame_index=frame_index,
                )
                frame_path = frames_dir / f"frame_{frame_index:03d}.png"
                figure.savefig(frame_path, dpi=args.dpi)
                print(f"Zapisano {frame_path}", flush=True)
            plt.close(figure)
            generated.append(frames_dir)

        if args.gif:
            from matplotlib.animation import FuncAnimation, PillowWriter

            figure, axes = make_figure()

            def update(frame_index: int) -> tuple[Any, ...]:
                draw_comparison(
                    figure,
                    axes,
                    nusc=nusc,
                    args=args,
                    results=results,
                    tokens=tokens,
                    classes=classes,
                    frame_index=frame_index,
                )
                print(f"Renderowanie GIF: {frame_index + 1}/{len(tokens)}", flush=True)
                return tuple(axes.flat)

            animation = FuncAnimation(figure, update, frames=len(tokens), blit=False)
            gif_path = output_dir / f"{args.scene}_{channel}_comparison.gif"
            animation.save(
                gif_path, writer=PillowWriter(fps=args.fps), dpi=min(args.dpi, 90)
            )
            plt.close(figure)
            generated.append(gif_path)

        write_json(
            output_dir / "camera_visualization_info.json",
            {
                "scene": args.scene,
                "camera_channel": args.camera_channel,
                "frame_count": len(tokens),
                "selected_frame_index": args.frame_index,
                "selected_sample_token": tokens[args.frame_index],
                "results": {method: str(path) for method, path in paths.items()},
                "nuscenes_root": str(args.nuscenes_root.resolve()),
                "classes": sorted(classes),
                "history_frames": args.history,
                "min_score_visual_only": args.min_score,
                "show_gt": args.show_gt,
                "show_lidar": not args.no_lidar,
                "lidar_projection": "official nuscenes-devkit map_pointcloud_to_image",
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
