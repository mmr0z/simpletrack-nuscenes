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
CAMERA_LAYOUT = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Tworzy osobne, dookólne wizualizacje sześciu kamer dla każdego modelu."
    )
    result.add_argument(
        "--scene", action="append", required=True, help="Można powtórzyć dla kilku scen"
    )
    result.add_argument("--nuscenes-root", required=True, type=Path)
    result.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "tracking_results"
    )
    result.add_argument("--bevfusion", type=Path)
    result.add_argument("--mvp", type=Path)
    result.add_argument("--lcf3d", type=Path)
    result.add_argument(
        "--method", action="append", choices=METHODS, help="Domyślnie wszystkie trzy"
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tracking_visualizations" / "surround",
    )
    result.add_argument("--frame-index", type=int, default=0)
    result.add_argument("--history", type=int, default=8)
    result.add_argument("--classes", default=",".join(TRACKING_CLASSES))
    result.add_argument("--min-score", type=float, default=0.0)
    result.add_argument("--color-by", choices=("class", "id"), default="class")
    result.add_argument("--labels", action="store_true")
    result.add_argument("--show-gt", action="store_true")
    result.add_argument("--no-lidar", action="store_true")
    result.add_argument("--max-lidar-points", type=int, default=5000)
    result.add_argument("--point-size", type=float, default=1.5)
    result.add_argument("--max-depth", type=float, default=80.0)
    result.add_argument("--gif", action="store_true")
    result.add_argument(
        "--gif-only", action="store_true", help="Zapisz tylko GIF-y, bez statycznych PNG"
    )
    result.add_argument("--fps", type=float, default=2.0)
    result.add_argument("--dpi", type=int, default=80)
    return result


def make_figure() -> tuple[Any, Any]:
    return plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)


def draw_surround(
    figure: Any,
    axes: Any,
    *,
    nusc: Any,
    args: argparse.Namespace,
    method: str,
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    classes: set[str],
    scene: str,
    frame_index: int,
) -> None:
    token = tokens[frame_index]
    ground_truth = gt_boxes(nusc, token, classes) if args.show_gt else []
    for panel_index, (ax, channel) in enumerate(zip(axes.flat, CAMERA_LAYOUT)):
        ax.clear()
        frame = camera_frame(nusc, token, channel)
        if args.no_lidar:
            from PIL import Image

            with Image.open(frame.image_path) as image_handle:
                image = np.asarray(image_handle.convert("RGB"))
            points = None
            depths = None
        else:
            points, depths, image = project_lidar_to_camera(
                nusc, token, channel, args.max_lidar_points
            )
        render_camera_panel(
            ax,
            title=channel.removeprefix("CAM_").replace("_", " "),
            methods=[method],
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
            legend=panel_index == len(CAMERA_LAYOUT) - 1,
        )
    figure.suptitle(
        f"{METHOD_NAMES[method]} + SimpleTrack 2 Hz — widok dookólny — {scene} — "
        f"klatka {frame_index + 1}/{len(tokens)}\n{token}",
        fontsize=14,
    )


def render_method_scene(
    *,
    nusc: Any,
    args: argparse.Namespace,
    method: str,
    results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    classes: set[str],
    scene: str,
) -> list[Path]:
    destination = args.output_dir / scene / method
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    if not args.gif_only:
        figure, axes = make_figure()
        draw_surround(
            figure,
            axes,
            nusc=nusc,
            args=args,
            method=method,
            results=results,
            tokens=tokens,
            classes=classes,
            scene=scene,
            frame_index=args.frame_index,
        )
        path = destination / f"frame_{args.frame_index:03d}_surround.png"
        figure.savefig(path, dpi=args.dpi)
        plt.close(figure)
        outputs.append(path)

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        figure, axes = make_figure()

        def update(frame_index: int) -> tuple[Any, ...]:
            draw_surround(
                figure,
                axes,
                nusc=nusc,
                args=args,
                method=method,
                results=results,
                tokens=tokens,
                classes=classes,
                scene=scene,
                frame_index=frame_index,
            )
            print(
                f"{scene} / {METHOD_NAMES[method]}: {frame_index + 1}/{len(tokens)}",
                flush=True,
            )
            return tuple(axes.flat)

        animation = FuncAnimation(figure, update, frames=len(tokens), blit=False)
        path = destination / f"{scene}_{method}_surround.gif"
        animation.save(path, writer=PillowWriter(fps=args.fps), dpi=min(args.dpi, 80))
        plt.close(figure)
        outputs.append(path)

    write_json(
        destination / "surround_visualization_info.json",
        {
            "scene": scene,
            "method": method,
            "method_display_name": METHOD_NAMES[method],
            "frame_count": len(tokens),
            "camera_layout": [list(CAMERA_LAYOUT[:3]), list(CAMERA_LAYOUT[3:])],
            "show_gt": args.show_gt,
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
        if args.history < 1 or args.fps <= 0 or args.dpi <= 0:
            raise PipelineError("history, fps i dpi muszą być dodatnie.")
        if args.point_size <= 0 or args.max_depth <= 1:
            raise PipelineError("point-size musi być dodatni, a max-depth większy od 1 m.")
        if not 0.0 <= args.min_score <= 1.0:
            raise PipelineError("--min-score musi należeć do [0, 1].")
        scenes = list(dict.fromkeys(args.scene))
        methods = list(dict.fromkeys(args.method or METHODS))
        classes = class_set(args.classes.split(","))
        paths = {
            method: resolve_tracking_result(
                method, getattr(args, method), args.results_root, scenes[0]
            )
            for method in METHODS
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
                    render_method_scene(
                        nusc=nusc,
                        args=args,
                        method=method,
                        results=results,
                        tokens=tokens,
                        classes=classes,
                        scene=scene,
                    )
                )
        write_json(
            args.output_dir / "surround_batch_info.json",
            {
                "scenes": scenes,
                "methods": methods,
                "camera_layout": [list(CAMERA_LAYOUT[:3]), list(CAMERA_LAYOUT[3:])],
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
