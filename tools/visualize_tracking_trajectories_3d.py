#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/simpletrack_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from simpletrack_pipeline.common import (
    METHOD_NAMES,
    PROJECT_ROOT,
    PipelineError,
    TRACKING_CLASSES,
    check_nuscenes_root,
    write_json,
)
from simpletrack_pipeline.visualization import (
    CLASS_COLORS,
    class_set,
    load_tracking_results,
    resolve_tracking_result,
    scene_sample_tokens,
    stable_track_color,
    validate_result_coverage,
)


METHODS = ("bevfusion", "mvp", "lcf3d")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Wizualizuje same trajektorie SimpleTrack w przestrzeni 3D."
    )
    result.add_argument("--scene", action="append", required=True)
    result.add_argument("--nuscenes-root", required=True, type=Path)
    result.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "tracking_results"
    )
    result.add_argument("--bevfusion", type=Path)
    result.add_argument("--mvp", type=Path)
    result.add_argument("--lcf3d", type=Path)
    result.add_argument("--method", action="append", choices=METHODS)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tracking_visualizations" / "trajectories_3d",
    )
    result.add_argument("--z-axis", choices=("time", "height"), default="time")
    result.add_argument("--classes", default=",".join(TRACKING_CLASSES))
    result.add_argument("--min-score", type=float, default=0.0)
    result.add_argument("--min-track-length", type=int, default=2)
    result.add_argument("--color-by", choices=("id", "class"), default="id")
    result.add_argument("--labels", action="store_true")
    result.add_argument("--elevation", type=float, default=27.0)
    result.add_argument("--azimuth", type=float, default=-58.0)
    result.add_argument("--gif", action="store_true", help="Obrotowy GIF 360 stopni")
    result.add_argument("--gif-only", action="store_true")
    result.add_argument("--rotation-frames", type=int, default=72)
    result.add_argument("--fps", type=float, default=12.0)
    result.add_argument("--dpi", type=int, default=100)
    return result


def scene_timestamps_and_origin(nusc: Any, tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(
        [float(nusc.get("sample", token)["timestamp"]) / 1_000_000.0 for token in tokens]
    )
    first_sample = nusc.get("sample", tokens[0])
    lidar = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", lidar["ego_pose_token"])
    return (
        timestamps - timestamps[0],
        np.asarray(ego["translation"], dtype=float),
        Quaternion(ego["rotation"]).rotation_matrix,
    )


def split_contiguous(observations: Sequence[tuple[int, np.ndarray]]) -> list[list[tuple[int, np.ndarray]]]:
    if not observations:
        return []
    segments = [[observations[0]]]
    for observation in observations[1:]:
        if observation[0] != segments[-1][-1][0] + 1:
            segments.append([])
        segments[-1].append(observation)
    return segments


def collect_tracks(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    tokens: Sequence[str],
    classes: set[str],
    min_score: float,
    min_track_length: int,
) -> list[dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    for frame_index, token in enumerate(tokens):
        for box in results.get(token, []):
            name = box.get("tracking_name")
            score = box.get("tracking_score")
            if name not in classes or not isinstance(score, (int, float)) or score < min_score:
                continue
            tracking_id = str(box.get("tracking_id"))
            track = tracks.setdefault(
                tracking_id,
                {"tracking_id": tracking_id, "tracking_name": str(name), "observations": []},
            )
            track["observations"].append(
                (frame_index, np.asarray(box["translation"], dtype=float))
            )
    return [
        track
        for track in tracks.values()
        if len(track["observations"]) >= min_track_length
    ]


def trajectory_coordinates(
    observations: Sequence[tuple[int, np.ndarray]],
    timestamps: np.ndarray,
    origin: np.ndarray,
    ego_to_global_rotation: np.ndarray,
    z_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame_indices = np.asarray([frame for frame, _ in observations], dtype=int)
    global_points = np.asarray([point for _, point in observations], dtype=float)
    ego_points = (global_points - origin) @ ego_to_global_rotation
    lateral_right = -ego_points[:, 1]
    forward = ego_points[:, 0]
    vertical = timestamps[frame_indices] if z_axis == "time" else ego_points[:, 2]
    return frame_indices, np.column_stack((lateral_right, forward, vertical))


def set_axes_limits(ax: Any, all_points: np.ndarray, z_axis: str) -> None:
    if len(all_points) == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1)
        return
    minima = all_points.min(axis=0)
    maxima = all_points.max(axis=0)
    spans = np.maximum(maxima - minima, [2.0, 2.0, 1.0])
    centers = (minima + maxima) / 2
    xy_span = max(spans[0], spans[1])
    ax.set_xlim(centers[0] - xy_span * 0.55, centers[0] + xy_span * 0.55)
    ax.set_ylim(centers[1] - xy_span * 0.55, centers[1] + xy_span * 0.55)
    z_margin = spans[2] * 0.08
    ax.set_zlim(minima[2] - z_margin, maxima[2] + z_margin)
    visual_z = max(xy_span * (0.48 if z_axis == "time" else 0.30), 1.0)
    ax.set_box_aspect((xy_span, xy_span, visual_z))


def render_trajectory_axis(
    ax: Any,
    *,
    method: str,
    tracks: Sequence[Mapping[str, Any]],
    timestamps: np.ndarray,
    origin: np.ndarray,
    ego_to_global_rotation: np.ndarray,
    z_axis: str,
    color_by: str,
    labels: bool,
    elevation: float,
    azimuth: float,
) -> dict[str, Any]:
    from matplotlib.lines import Line2D

    all_coordinates = []
    segment_count = 0
    observation_count = 0
    classes_present = set()
    for track in tracks:
        name = str(track["tracking_name"])
        tracking_id = str(track["tracking_id"])
        classes_present.add(name)
        color = stable_track_color(tracking_id) if color_by == "id" else CLASS_COLORS[name]
        last_coordinate = None
        for segment in split_contiguous(track["observations"]):
            _, coordinates = trajectory_coordinates(
                segment, timestamps, origin, ego_to_global_rotation, z_axis
            )
            all_coordinates.append(coordinates)
            segment_count += 1
            observation_count += len(coordinates)
            last_coordinate = coordinates[-1]
            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
                color=color,
                linewidth=1.35,
                alpha=0.88,
            )
            ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
                color=[color],
                s=5,
                alpha=0.72,
                depthshade=False,
            )
        if labels and last_coordinate is not None:
            suffix = tracking_id.rsplit("_", 1)[-1]
            ax.text(
                last_coordinate[0],
                last_coordinate[1],
                last_coordinate[2],
                f"{name[:3]}:{suffix}",
                color=color,
                fontsize=5,
            )

    points = np.concatenate(all_coordinates) if all_coordinates else np.empty((0, 3))
    set_axes_limits(ax, points, z_axis)
    ax.set_xlabel("położenie boczne [m]\n(− lewo, + prawo)", fontsize=8)
    ax.set_ylabel("położenie wzdłużne [m]\n(przód +)", fontsize=8)
    ax.set_zlabel("czas od początku [s]" if z_axis == "time" else "wysokość względem ego [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.35)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_title(
        f"{METHOD_NAMES[method]}\ntrajektorie: {len(tracks)}, obserwacje: {observation_count}",
        fontsize=11,
    )
    if color_by == "class":
        handles = [
            Line2D([0], [0], color=CLASS_COLORS[name], linewidth=2, label=name)
            for name in TRACKING_CLASSES
            if name in classes_present
        ]
        ax.legend(handles=handles, fontsize=6.5, loc="upper right")
    else:
        ax.text2D(
            0.02,
            0.97,
            "kolor = tracking_id",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
        )
    return {
        "tracks": len(tracks),
        "segments": segment_count,
        "observations": observation_count,
    }


def render_scene_method(
    *,
    args: argparse.Namespace,
    scene: str,
    method: str,
    tracks: Sequence[Mapping[str, Any]],
    timestamps: np.ndarray,
    origin: np.ndarray,
    ego_rotation: np.ndarray,
) -> tuple[list[Path], dict[str, Any]]:
    destination = args.output_dir / scene / method
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    figure = plt.figure(figsize=(10, 9), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    stats = render_trajectory_axis(
        ax,
        method=method,
        tracks=tracks,
        timestamps=timestamps,
        origin=origin,
        ego_to_global_rotation=ego_rotation,
        z_axis=args.z_axis,
        color_by=args.color_by,
        labels=args.labels,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )
    figure.suptitle(
        f"SimpleTrack 2 Hz — same trajektorie — {scene} — "
        f"Z={'czas' if args.z_axis == 'time' else 'wysokość'}",
        fontsize=14,
    )
    if not args.gif_only:
        png_path = destination / f"{scene}_{method}_trajectories_3d_{args.z_axis}.png"
        figure.savefig(png_path, dpi=args.dpi)
        outputs.append(png_path)
    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        def update(index: int) -> tuple[Any, ...]:
            angle = args.azimuth + 360.0 * index / args.rotation_frames
            ax.view_init(elev=args.elevation, azim=angle)
            print(
                f"{scene} / {METHOD_NAMES[method]}: obrót {index + 1}/{args.rotation_frames}",
                flush=True,
            )
            return (ax,)

        animation = FuncAnimation(
            figure, update, frames=args.rotation_frames, blit=False
        )
        gif_path = destination / f"{scene}_{method}_trajectories_3d_{args.z_axis}.gif"
        animation.save(
            gif_path, writer=PillowWriter(fps=args.fps), dpi=min(args.dpi, 100)
        )
        outputs.append(gif_path)
    plt.close(figure)
    write_json(
        destination / "trajectory_visualization_info.json",
        {
            "scene": scene,
            "method": method,
            "z_axis": args.z_axis,
            "coordinate_system": "first_keyframe_ego_x_forward_y_left_z_up",
            "color_by": args.color_by,
            "min_track_length_visual_only": args.min_track_length,
            "min_score_visual_only": args.min_score,
            "classes": sorted(args.class_set),
            **stats,
            "generated": [str(path) for path in outputs],
        },
    )
    return outputs, stats


def save_comparison(
    *,
    args: argparse.Namespace,
    scene: str,
    tracks_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    timestamps: np.ndarray,
    origin: np.ndarray,
    ego_rotation: np.ndarray,
    methods: Sequence[str],
) -> Path:
    figure = plt.figure(figsize=(18, 6.5), constrained_layout=True)
    for index, method in enumerate(methods, 1):
        ax = figure.add_subplot(1, len(methods), index, projection="3d")
        render_trajectory_axis(
            ax,
            method=method,
            tracks=tracks_by_method[method],
            timestamps=timestamps,
            origin=origin,
            ego_to_global_rotation=ego_rotation,
            z_axis=args.z_axis,
            color_by=args.color_by,
            labels=args.labels,
            elevation=args.elevation,
            azimuth=args.azimuth,
        )
    figure.suptitle(f"SimpleTrack 2 Hz — porównanie trajektorii 3D — {scene}", fontsize=15)
    path = args.output_dir / scene / f"{scene}_trajectories_3d_comparison_{args.z_axis}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=args.dpi)
    plt.close(figure)
    return path


def run() -> int:
    args = parser().parse_args()
    try:
        if args.gif_only:
            args.gif = True
        check_nuscenes_root(args.nuscenes_root)
        if args.min_track_length < 1 or args.rotation_frames < 2:
            raise PipelineError("min-track-length >= 1 i rotation-frames >= 2 są wymagane.")
        if args.fps <= 0 or args.dpi <= 0:
            raise PipelineError("fps i dpi muszą być dodatnie.")
        if not 0.0 <= args.min_score <= 1.0:
            raise PipelineError("--min-score musi należeć do [0, 1].")
        args.class_set = class_set(args.classes.split(","))
        scenes = list(dict.fromkeys(args.scene))
        methods = list(dict.fromkeys(args.method or METHODS))
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
        batch_stats = {}
        for scene_index, scene in enumerate(scenes, 1):
            tokens = scene_sample_tokens(nusc, scene)
            validate_result_coverage(results, tokens)
            timestamps, origin, ego_rotation = scene_timestamps_and_origin(nusc, tokens)
            tracks_by_method = {
                method: collect_tracks(
                    results[method],
                    tokens,
                    args.class_set,
                    args.min_score,
                    args.min_track_length,
                )
                for method in methods
            }
            batch_stats[scene] = {}
            for method_index, method in enumerate(methods, 1):
                print(
                    f"\nScena {scene_index}/{len(scenes)}, model {method_index}/{len(methods)}: "
                    f"{scene} / {METHOD_NAMES[method]}",
                    flush=True,
                )
                paths_created, stats = render_scene_method(
                    args=args,
                    scene=scene,
                    method=method,
                    tracks=tracks_by_method[method],
                    timestamps=timestamps,
                    origin=origin,
                    ego_rotation=ego_rotation,
                )
                generated.extend(paths_created)
                batch_stats[scene][method] = stats
            if not args.gif_only:
                generated.append(
                    save_comparison(
                        args=args,
                        scene=scene,
                        tracks_by_method=tracks_by_method,
                        timestamps=timestamps,
                        origin=origin,
                        ego_rotation=ego_rotation,
                        methods=methods,
                    )
                )
        write_json(
            args.output_dir / "trajectory_batch_info.json",
            {
                "scenes": scenes,
                "methods": methods,
                "z_axis": args.z_axis,
                "stats": batch_stats,
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
