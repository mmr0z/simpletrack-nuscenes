from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pyquaternion import Quaternion

from simpletrack_pipeline.common import METHOD_NAMES, PipelineError, TRACKING_CLASSES, read_json


METHOD_COLORS = {
    "bevfusion": "#0072B2",
    "mvp": "#D55E00",
    "lcf3d": "#009E73",
}
METHOD_LINESTYLES = {
    "bevfusion": "-",
    "mvp": "--",
    "lcf3d": ":",
}
CLASS_COLORS = {
    "car": "#0072B2",
    "bus": "#E69F00",
    "trailer": "#CC79A7",
    "truck": "#D55E00",
    "pedestrian": "#009E73",
    "bicycle": "#56B4E9",
    "motorcycle": "#F0E442",
}
NUSCENES_TRACKING_CATEGORY_MAP = {
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


@dataclass(frozen=True)
class EgoFrame:
    """Transforms needed to express global boxes and LIDAR points in ego coordinates."""

    token: str
    lidar_path: Path | None
    ego_to_global_rotation: np.ndarray
    ego_to_global_translation: np.ndarray
    lidar_to_ego_rotation: np.ndarray
    lidar_to_ego_translation: np.ndarray

    def global_to_ego(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return (points - self.ego_to_global_translation) @ self.ego_to_global_rotation

    def lidar_to_ego(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return points @ self.lidar_to_ego_rotation.T + self.lidar_to_ego_translation


@dataclass(frozen=True)
class CameraFrame:
    """Camera calibration and pose at the image timestamp."""

    token: str
    channel: str
    image_path: Path
    image_size: tuple[int, int]
    intrinsic: np.ndarray
    camera_to_global_rotation: np.ndarray
    camera_to_global_translation: np.ndarray

    def global_to_camera(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return (points - self.camera_to_global_translation) @ self.camera_to_global_rotation

    def project(self, camera_points: np.ndarray) -> np.ndarray:
        camera_points = np.asarray(camera_points, dtype=float)
        projected = camera_points @ self.intrinsic.T
        return projected[:, :2] / projected[:, 2:3]


@dataclass(frozen=True)
class DetectionMatch:
    """Class-aware center-distance association used only for visualization."""

    matched_pairs: tuple[tuple[int, int], ...]
    unmatched_predictions: tuple[int, ...]
    unmatched_ground_truth: tuple[int, ...]
    per_class: Mapping[str, Mapping[str, int]]

    @property
    def tp(self) -> int:
        return len(self.matched_pairs)

    @property
    def fp(self) -> int:
        return len(self.unmatched_predictions)

    @property
    def fn(self) -> int:
        return len(self.unmatched_ground_truth)


def scene_sample_tokens(nusc: Any, scene_name: str) -> list[str]:
    matches = [scene for scene in nusc.scene if scene["name"] == scene_name]
    if not matches:
        raise PipelineError(f"Nie znaleziono sceny {scene_name!r} w nuScenes.")
    if len(matches) != 1:
        raise PipelineError(f"Niejednoznaczna nazwa sceny nuScenes: {scene_name!r}.")

    tokens: list[str] = []
    token = matches[0]["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return tokens


def ego_frame(nusc: Any, sample_token: str) -> EgoFrame:
    sample = nusc.get("sample", sample_token)
    lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", lidar["ego_pose_token"])
    calibrated = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])

    ego_rotation = Quaternion(ego["rotation"]).rotation_matrix
    sensor_rotation = Quaternion(calibrated["rotation"]).rotation_matrix
    return EgoFrame(
        token=sample_token,
        lidar_path=Path(nusc.dataroot) / lidar["filename"],
        ego_to_global_rotation=ego_rotation,
        ego_to_global_translation=np.asarray(ego["translation"], dtype=float),
        lidar_to_ego_rotation=sensor_rotation,
        lidar_to_ego_translation=np.asarray(calibrated["translation"], dtype=float),
    )


def camera_frame(nusc: Any, sample_token: str, channel: str) -> CameraFrame:
    sample = nusc.get("sample", sample_token)
    if channel not in sample["data"]:
        raise PipelineError(f"Brak kanału {channel!r} dla sample {sample_token}.")
    camera = nusc.get("sample_data", sample["data"][channel])
    ego = nusc.get("ego_pose", camera["ego_pose_token"])
    calibrated = nusc.get("calibrated_sensor", camera["calibrated_sensor_token"])
    intrinsic = np.asarray(calibrated.get("camera_intrinsic"), dtype=float)
    if intrinsic.shape != (3, 3):
        raise PipelineError(f"Brak macierzy intrinsics 3x3 dla kanału {channel}.")
    ego_rotation = Quaternion(ego["rotation"]).rotation_matrix
    camera_rotation = Quaternion(calibrated["rotation"]).rotation_matrix
    rotation = ego_rotation @ camera_rotation
    translation = np.asarray(ego["translation"], dtype=float) + ego_rotation @ np.asarray(
        calibrated["translation"], dtype=float
    )
    return CameraFrame(
        token=sample_token,
        channel=channel,
        image_path=Path(nusc.dataroot) / camera["filename"],
        image_size=(int(camera["width"]), int(camera["height"])),
        intrinsic=intrinsic,
        camera_to_global_rotation=rotation,
        camera_to_global_translation=translation,
    )


def box_corners_ego(box: Mapping[str, Any], frame: EgoFrame) -> np.ndarray:
    """Return the four bottom footprint corners in ego coordinates."""

    width, length, _ = (float(value) for value in box["size"])
    local = np.array(
        [
            [length / 2, width / 2, 0.0],
            [length / 2, -width / 2, 0.0],
            [-length / 2, -width / 2, 0.0],
            [-length / 2, width / 2, 0.0],
        ]
    )
    box_rotation = Quaternion(box["rotation"]).rotation_matrix
    global_corners = local @ box_rotation.T + np.asarray(box["translation"], dtype=float)
    return frame.global_to_ego(global_corners)


def box_heading_ego(box: Mapping[str, Any], frame: EgoFrame) -> np.ndarray:
    width, length, _ = (float(value) for value in box["size"])
    del width
    center = np.asarray(box["translation"], dtype=float)
    heading = Quaternion(box["rotation"]).rotation_matrix @ np.array(
        [length / 2, 0.0, 0.0]
    )
    return frame.global_to_ego(np.vstack((center, center + heading)))


def ego_to_bev(points: np.ndarray) -> np.ndarray:
    """Map x-forward/y-left ego coordinates to image x-right/y-forward."""

    points = np.asarray(points)
    return np.column_stack((-points[..., 1], points[..., 0]))


def stable_track_color(tracking_id: str) -> tuple[float, float, float]:
    """Deterministic vivid RGB color for a tracking ID."""

    import colorsys

    digest = hashlib.sha1(tracking_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.60 + digest[2] / 255.0 * 0.25
    value = 0.68 + digest[3] / 255.0 * 0.22
    return colorsys.hsv_to_rgb(hue, saturation, value)


def boxes_for_frame(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    token: str,
    classes: set[str],
    min_score: float,
) -> list[Mapping[str, Any]]:
    selected = []
    for box in results.get(token, []):
        name = box.get("tracking_name")
        score = box.get("tracking_score")
        if name in classes and isinstance(score, (int, float)) and score >= min_score:
            selected.append(box)
    return selected


def match_detections_to_ground_truth(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    distance_threshold: float = 2.0,
) -> DetectionMatch:
    """Match boxes by class and global XY center using Hungarian assignment."""

    if distance_threshold <= 0:
        raise ValueError("distance_threshold musi być dodatni.")
    from scipy.optimize import linear_sum_assignment

    matched: list[tuple[int, int]] = []
    classes = sorted(
        {str(box.get("tracking_name", "")) for box in predictions}
        | {str(box.get("tracking_name", "")) for box in ground_truth}
    )
    per_class: dict[str, dict[str, int]] = {}
    for name in classes:
        prediction_indices = [
            index
            for index, box in enumerate(predictions)
            if box.get("tracking_name") == name
        ]
        gt_indices = [
            index
            for index, box in enumerate(ground_truth)
            if box.get("tracking_name") == name
        ]
        class_pairs: list[tuple[int, int]] = []
        if prediction_indices and gt_indices:
            prediction_xy = np.asarray(
                [predictions[index]["translation"][:2] for index in prediction_indices],
                dtype=float,
            )
            gt_xy = np.asarray(
                [ground_truth[index]["translation"][:2] for index in gt_indices],
                dtype=float,
            )
            distances = np.linalg.norm(
                prediction_xy[:, None, :] - gt_xy[None, :, :], axis=2
            )
            forbidden_cost = max(1_000_000.0, distance_threshold * 1_000_000.0)
            costs = np.where(distances <= distance_threshold, distances, forbidden_cost)
            rows, columns = linear_sum_assignment(costs)
            class_pairs = [
                (prediction_indices[row], gt_indices[column])
                for row, column in zip(rows.tolist(), columns.tolist())
                if distances[row, column] <= distance_threshold
            ]
            matched.extend(class_pairs)
        per_class[name] = {
            "tp": len(class_pairs),
            "fp": len(prediction_indices) - len(class_pairs),
            "fn": len(gt_indices) - len(class_pairs),
        }

    matched_predictions = {prediction for prediction, _ in matched}
    matched_ground_truth = {gt for _, gt in matched}
    return DetectionMatch(
        matched_pairs=tuple(sorted(matched)),
        unmatched_predictions=tuple(
            index for index in range(len(predictions)) if index not in matched_predictions
        ),
        unmatched_ground_truth=tuple(
            index for index in range(len(ground_truth)) if index not in matched_ground_truth
        ),
        per_class=per_class,
    )


def detection_match_text(match: DetectionMatch) -> str:
    class_lines = [
        f"{name}: {counts['tp']}/{counts['fp']}/{counts['fn']}"
        for name, counts in match.per_class.items()
        if counts["tp"] or counts["fp"] or counts["fn"]
    ]
    breakdown = ", ".join(class_lines) if class_lines else "brak obiektów"
    return (
        f"TP {match.tp} | FP {match.fp} | FN {match.fn} (cała klatka 360°)\n"
        f"klasa TP/FP/FN: {breakdown}"
    )


def track_history(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    tokens: Sequence[str],
    frame_index: int,
    tracking_id: str,
    history_length: int,
    frame: EgoFrame,
) -> np.ndarray:
    points = track_history_global(
        results, tokens, frame_index, tracking_id, history_length
    )
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    return ego_to_bev(frame.global_to_ego(points))


def track_history_global(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    tokens: Sequence[str],
    frame_index: int,
    tracking_id: str,
    history_length: int,
) -> np.ndarray:
    start = max(0, frame_index - history_length + 1)
    points = []
    for token in tokens[start : frame_index + 1]:
        match = next(
            (box for box in results.get(token, []) if box.get("tracking_id") == tracking_id),
            None,
        )
        if match is not None:
            points.append(match["translation"])
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def load_lidar_points(path: Path, max_points: int) -> np.ndarray:
    if not path.is_file():
        raise PipelineError(f"Brak pliku chmury LiDAR: {path}")
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 5 != 0:
        raise PipelineError(f"Uszkodzony plik nuScenes LiDAR: {path}")
    points = raw.reshape(-1, 5)[:, :3]
    if max_points > 0 and len(points) > max_points:
        step = int(np.ceil(len(points) / max_points))
        points = points[::step]
    return points


def project_lidar_to_camera(
    nusc: Any,
    sample_token: str,
    channel: str,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the official nuScenes projection, including per-sensor timestamps."""

    sample = nusc.get("sample", sample_token)
    points, depths, image = nusc.explorer.map_pointcloud_to_image(
        sample["data"]["LIDAR_TOP"], sample["data"][channel], min_dist=1.0
    )
    if max_points > 0 and points.shape[1] > max_points:
        step = int(np.ceil(points.shape[1] / max_points))
        points = points[:, ::step]
        depths = depths[::step]
    return points, np.asarray(depths), np.asarray(image.convert("RGB"))


def gt_boxes(nusc: Any, sample_token: str, classes: set[str]) -> list[dict[str, Any]]:
    sample = nusc.get("sample", sample_token)
    boxes = nusc.get_boxes(sample["data"]["LIDAR_TOP"])
    converted = []
    for box in boxes:
        name = NUSCENES_TRACKING_CATEGORY_MAP.get(box.name)
        if name not in classes:
            continue
        converted.append(
            {
                "translation": box.center.tolist(),
                "size": box.wlh.tolist(),
                "rotation": box.orientation.q.tolist(),
                "tracking_name": name,
                "tracking_id": box.token or "gt",
                "tracking_score": 1.0,
            }
        )
    return converted


def validate_result_coverage(
    method_results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
) -> dict[str, list[str]]:
    missing = {
        method: [token for token in tokens if token not in results]
        for method, results in method_results.items()
    }
    incomplete = {method: values for method, values in missing.items() if values}
    if incomplete:
        details = ", ".join(f"{method}: {len(values)}" for method, values in incomplete.items())
        raise PipelineError(f"Brak klatek sceny w Tracking JSON ({details}).")
    return missing


def resolve_tracking_result(
    method: str,
    explicit: Path | None,
    results_root: Path,
    scene: str,
) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        experiment = f"{METHOD_NAMES[method]}_SimpleTrack_2Hz"
        candidates = [
            results_root / experiment / "results" / "tracking_result.json",
            results_root / f"{experiment}_debug_{scene}" / "results" / "tracking_result.json",
        ]
    for candidate in candidates:
        resolved = candidate / "results" / "tracking_result.json" if candidate.is_dir() else candidate
        if resolved.is_file():
            return resolved.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise PipelineError(f"Nie znaleziono wyniku {METHOD_NAMES[method]}: {rendered}")


def load_tracking_results(
    paths: Mapping[str, Path],
) -> dict[str, Mapping[str, Sequence[Mapping[str, Any]]]]:
    loaded = {}
    for method, path in paths.items():
        payload = read_json(path)
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            raise PipelineError(f"Brak obiektu 'results' w {path}")
        loaded[method] = results
    return loaded


def tracking_box_in_camera(box: Mapping[str, Any], frame: CameraFrame) -> Any:
    from nuscenes.utils.data_classes import Box

    center = frame.global_to_camera(np.asarray([box["translation"]], dtype=float))[0]
    camera_from_global = Quaternion(matrix=frame.camera_to_global_rotation.T)
    orientation = camera_from_global * Quaternion(box["rotation"])
    return Box(
        center=center,
        size=np.asarray(box["size"], dtype=float),
        orientation=orientation,
        name=str(box.get("tracking_name", "")),
        token=str(box.get("tracking_id", "")),
    )


def draw_projected_box(
    ax: Any,
    box: Any,
    frame: CameraFrame,
    *,
    color: Any,
    linestyle: str,
    linewidth: float,
    alpha: float,
) -> None:
    corners = box.corners().T
    projected = frame.project(corners)
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for first, second in edges:
        ax.plot(
            projected[[first, second], 0],
            projected[[first, second], 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
    front = projected[[0, 1, 2, 3]].mean(axis=0)
    center = projected.mean(axis=0)
    ax.plot(
        [center[0], front[0]],
        [center[1], front[1]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )


def draw_bev_detection_status(
    ax: Any,
    *,
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    match: DetectionMatch,
    frame: EgoFrame,
    view_range: float,
) -> None:
    """Overlay TP/FP/FN markers on an already rendered BEV panel."""

    from matplotlib.patches import Polygon

    matched_predictions = {prediction for prediction, _ in match.matched_pairs}
    for index, box in enumerate(predictions):
        center_ego = frame.global_to_ego(
            np.asarray([box["translation"]], dtype=float)
        )[0]
        if abs(center_ego[0]) > view_range or abs(center_ego[1]) > view_range:
            continue
        center = ego_to_bev(center_ego.reshape(1, 3))[0]
        if index in matched_predictions:
            ax.scatter(
                center[0], center[1], marker="o", s=24, facecolors="none",
                edgecolors="#00A651", linewidths=1.3, zorder=12,
            )
        else:
            ax.scatter(
                center[0], center[1], marker="x", s=31, c="#FF8C00",
                linewidths=1.6, zorder=12,
            )

    for index in match.unmatched_ground_truth:
        box = ground_truth[index]
        center_ego = frame.global_to_ego(
            np.asarray([box["translation"]], dtype=float)
        )[0]
        if abs(center_ego[0]) > view_range or abs(center_ego[1]) > view_range:
            continue
        corners = ego_to_bev(box_corners_ego(box, frame))
        ax.add_patch(
            Polygon(
                corners, closed=True, fill=False, edgecolor="#E31A1C",
                linewidth=2.0, linestyle="--", alpha=0.95, zorder=11,
            )
        )
        center = ego_to_bev(center_ego.reshape(1, 3))[0]
        ax.scatter(
            center[0], center[1], marker="x", s=35, c="#E31A1C",
            linewidths=1.8, zorder=12,
        )

    ax.text(
        0.01,
        0.99,
        detection_match_text(match)
        + "\n● zielony: wykryty | × pomarańczowy: FP | × czerwony: niewykryty GT",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color="#111111",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#777777", "pad": 3},
        zorder=20,
    )


def draw_camera_detection_status(
    ax: Any,
    *,
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    match: DetectionMatch,
    frame: CameraFrame,
) -> None:
    """Overlay visible TP/FP/FN markers on an already rendered camera panel."""

    from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image

    matched_predictions = {prediction for prediction, _ in match.matched_pairs}
    for index, raw_box in enumerate(predictions):
        box = tracking_box_in_camera(raw_box, frame)
        if not box_in_image(
            box, frame.intrinsic, frame.image_size, vis_level=BoxVisibility.ANY
        ):
            continue
        if index in matched_predictions:
            center = frame.project(box.center.reshape(1, 3))[0]
            ax.scatter(
                center[0], center[1], marker="o", s=25, facecolors="none",
                edgecolors="#00E676", linewidths=1.4, zorder=12,
            )
        else:
            draw_projected_box(
                ax, box, frame, color="#FF8C00", linestyle=":",
                linewidth=2.2, alpha=0.95,
            )

    for index in match.unmatched_ground_truth:
        box = tracking_box_in_camera(ground_truth[index], frame)
        if box_in_image(
            box, frame.intrinsic, frame.image_size, vis_level=BoxVisibility.ANY
        ):
            draw_projected_box(
                ax, box, frame, color="#FF1744", linestyle="--",
                linewidth=2.4, alpha=0.98,
            )

    ax.text(
        0.01,
        0.90,
        detection_match_text(match)
        + "\nzielony: wykryty | pomarańczowy: FP | czerwony: niewykryty GT",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 3},
        zorder=20,
    )


def render_camera_panel(
    ax: Any,
    *,
    title: str,
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    frame_index: int,
    frame: CameraFrame,
    classes: set[str],
    min_score: float,
    history_length: int,
    color_by: str,
    image: np.ndarray,
    lidar_projection: np.ndarray | None,
    lidar_depths: np.ndarray | None,
    max_depth: float,
    point_size: float,
    ground_truth: Sequence[Mapping[str, Any]],
    labels: bool,
    legend: bool,
) -> dict[str, int]:
    from matplotlib.lines import Line2D
    from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image

    ax.imshow(image)
    if lidar_projection is not None and lidar_depths is not None:
        ax.scatter(
            lidar_projection[0],
            lidar_projection[1],
            c=lidar_depths,
            cmap="turbo",
            vmin=1.0,
            vmax=max_depth,
            s=point_size,
            alpha=0.80,
            linewidths=0,
        )
        ax.text(
            0.01,
            0.98,
            f"LiDAR: kolor = głębokość 1–{max_depth:g} m",
            transform=ax.transAxes,
            va="top",
            color="white",
            fontsize=6.5,
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
        )

    for raw_box in ground_truth:
        box = tracking_box_in_camera(raw_box, frame)
        if box_in_image(box, frame.intrinsic, frame.image_size, vis_level=BoxVisibility.ANY):
            draw_projected_box(
                ax,
                box,
                frame,
                color="#F2F2F2",
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
            )

    token = tokens[frame_index]
    overlay = len(methods) > 1
    counts: dict[str, int] = {}
    for method in methods:
        visible_count = 0
        for raw_box in boxes_for_frame(method_results[method], token, classes, min_score):
            box = tracking_box_in_camera(raw_box, frame)
            if not box_in_image(
                box, frame.intrinsic, frame.image_size, vis_level=BoxVisibility.ANY
            ):
                continue
            visible_count += 1
            tracking_id = str(raw_box["tracking_id"])
            name = str(raw_box["tracking_name"])
            color = (
                METHOD_COLORS[method]
                if overlay
                else stable_track_color(tracking_id)
                if color_by == "id"
                else CLASS_COLORS[name]
            )
            linestyle = METHOD_LINESTYLES[method] if overlay else "-"
            draw_projected_box(
                ax,
                box,
                frame,
                color=color,
                linestyle=linestyle,
                linewidth=2.0 if overlay else 1.8,
                alpha=0.95,
            )

            if history_length > 1:
                history_global = track_history_global(
                    method_results[method], tokens, frame_index, tracking_id, history_length
                )
                history_camera = frame.global_to_camera(history_global)
                visible = history_camera[:, 2] > 1.0
                if visible.sum() > 1:
                    history_image = frame.project(history_camera[visible])
                    ax.plot(
                        history_image[:, 0],
                        history_image[:, 1],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.2,
                        alpha=0.65,
                    )
            if labels and not overlay:
                center = frame.project(box.center.reshape(1, 3))[0]
                suffix = tracking_id.rsplit("_", 1)[-1]
                ax.text(
                    center[0],
                    center[1],
                    f"{name[:3]}:{suffix} {float(raw_box['tracking_score']):.2f}",
                    color=color,
                    fontsize=6,
                    bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 0.5},
                    clip_on=True,
                )
        counts[method] = visible_count

    count_text = " | ".join(f"{METHOD_NAMES[m]}: {counts[m]}" for m in methods)
    ax.set_title(f"{title}\n{count_text}", fontsize=10)
    ax.set_xlim(0, frame.image_size[0])
    ax.set_ylim(frame.image_size[1], 0)
    ax.axis("off")
    if legend:
        if overlay:
            handles = [
                Line2D([0], [0], color=METHOD_COLORS[m], linestyle=METHOD_LINESTYLES[m], linewidth=2, label=METHOD_NAMES[m])
                for m in methods
            ]
        elif color_by == "class":
            handles = [
                Line2D([0], [0], color=CLASS_COLORS[name], linewidth=2, label=name)
                for name in TRACKING_CLASSES
                if name in classes
            ]
        else:
            handles = [Line2D([0], [0], color="white", linewidth=2, label="kolor = tracking_id")]
        if ground_truth:
            handles.append(Line2D([0], [0], color="#F2F2F2", linestyle="--", label="GT"))
        legend_object = ax.legend(
            handles=handles, loc="lower right", fontsize=6.5, framealpha=0.72, facecolor="black"
        )
        for label in legend_object.get_texts():
            label.set_color("white")
    return counts


def render_panel(
    ax: Any,
    *,
    title: str,
    methods: Sequence[str],
    method_results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    tokens: Sequence[str],
    frame_index: int,
    frame: EgoFrame,
    classes: set[str],
    min_score: float,
    view_range: float,
    history_length: int,
    color_by: str,
    lidar_points: np.ndarray | None,
    ground_truth: Sequence[Mapping[str, Any]],
    labels: bool,
    legend: bool,
) -> dict[str, int]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle

    ax.set_facecolor("#FAFAFA")
    if lidar_points is not None:
        bev_points = ego_to_bev(lidar_points)
        visible = (
            (np.abs(bev_points[:, 0]) <= view_range)
            & (np.abs(bev_points[:, 1]) <= view_range)
        )
        ax.scatter(
            bev_points[visible, 0],
            bev_points[visible, 1],
            s=0.20,
            c="#777777",
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )

    for box in ground_truth:
        corners = ego_to_bev(box_corners_ego(box, frame))
        ax.add_patch(
            Polygon(corners, closed=True, fill=False, edgecolor="#222222", linewidth=1.0, linestyle="--", alpha=0.75)
        )

    counts: dict[str, int] = {}
    token = tokens[frame_index]
    overlay = len(methods) > 1
    for method in methods:
        boxes = boxes_for_frame(method_results[method], token, classes, min_score)
        visible_count = 0
        for box in boxes:
            corners_ego = box_corners_ego(box, frame)
            center_ego = frame.global_to_ego(np.asarray([box["translation"]], dtype=float))[0]
            if abs(center_ego[0]) > view_range or abs(center_ego[1]) > view_range:
                continue
            visible_count += 1
            tracking_id = str(box["tracking_id"])
            name = str(box["tracking_name"])
            color = (
                METHOD_COLORS[method]
                if overlay
                else stable_track_color(tracking_id)
                if color_by == "id"
                else CLASS_COLORS[name]
            )
            linestyle = METHOD_LINESTYLES[method] if overlay else "-"
            linewidth = 1.8 if overlay else 1.5
            corners = ego_to_bev(corners_ego)
            ax.add_patch(
                Polygon(
                    corners,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=0.90,
                )
            )
            heading = ego_to_bev(box_heading_ego(box, frame))
            ax.plot(
                heading[:, 0],
                heading[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.90,
            )
            if history_length > 1:
                history = track_history(
                    method_results[method], tokens, frame_index, tracking_id, history_length, frame
                )
                if len(history) > 1:
                    ax.plot(
                        history[:, 0],
                        history[:, 1],
                        color=color,
                        linewidth=1.1,
                        linestyle=linestyle,
                        alpha=0.65,
                    )
            if labels and not overlay:
                center = ego_to_bev(center_ego.reshape(1, 3))[0]
                suffix = tracking_id.rsplit("_", 1)[-1]
                score = float(box["tracking_score"])
                ax.text(
                    center[0],
                    center[1],
                    f"{name[:3]}:{suffix} {score:.2f}",
                    fontsize=5.5,
                    color=color,
                    clip_on=True,
                )
        counts[method] = visible_count

    ego_width, ego_length = 2.0, 4.6
    ax.add_patch(
        Rectangle(
            (-ego_width / 2, -ego_length / 2),
            ego_width,
            ego_length,
            facecolor="#555555",
            edgecolor="white",
            linewidth=0.8,
            zorder=8,
        )
    )
    ax.arrow(0, 0, 0, 3.5, width=0.13, color="#222222", length_includes_head=True, zorder=9)
    count_text = " | ".join(f"{METHOD_NAMES[m]}: {counts[m]}" for m in methods)
    ax.set_title(f"{title}\n{count_text}", fontsize=10)
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("położenie boczne [m]  (− lewo, + prawo)", fontsize=8)
    ax.set_ylabel("położenie wzdłużne [m]  (przód +)", fontsize=8)
    ax.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.65)
    ax.tick_params(labelsize=7)

    if legend:
        if overlay:
            handles = [
                Line2D(
                    [0], [0], color=METHOD_COLORS[method], linestyle=METHOD_LINESTYLES[method], linewidth=2, label=METHOD_NAMES[method]
                )
                for method in methods
            ]
        elif color_by == "class":
            handles = [
                Line2D([0], [0], color=CLASS_COLORS[name], linewidth=2, label=name)
                for name in TRACKING_CLASSES
                if name in classes
            ]
        else:
            handles = [Line2D([0], [0], color="#555555", linewidth=2, label="kolor = tracking_id")]
        if ground_truth:
            handles.append(Line2D([0], [0], color="#222222", linestyle="--", label="GT"))
        ax.legend(handles=handles, loc="lower right", fontsize=6.5, framealpha=0.90)
    return counts


def class_set(values: Iterable[str]) -> set[str]:
    classes = {value.strip().lower() for value in values if value.strip()}
    invalid = classes - set(TRACKING_CLASSES)
    if invalid:
        raise PipelineError(
            f"Nieobsługiwane klasy: {', '.join(sorted(invalid))}. "
            f"Dozwolone: {', '.join(TRACKING_CLASSES)}"
        )
    if not classes:
        raise PipelineError("Lista klas do wizualizacji jest pusta.")
    return classes
