from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import PipelineError, TRACKING_CLASSES, validation_scene_names


REQUIRED_FIELDS = (
    "translation",
    "size",
    "rotation",
    "detection_name",
    "detection_score",
)


def validation_sample_tokens(nuscenes_root: Path, version: str = "v1.0-trainval") -> dict[str, str]:
    metadata = nuscenes_root / version
    try:
        with (metadata / "scene.json").open("r", encoding="utf-8") as handle:
            scenes = json.load(handle)
        with (metadata / "sample.json").open("r", encoding="utf-8") as handle:
            samples = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Nie można odczytać metadanych nuScenes: {exc}") from exc

    val_names = validation_scene_names()
    scene_by_token = {
        item["token"]: item["name"] for item in scenes if item.get("name") in val_names
    }
    result = {
        item["token"]: scene_by_token[item["scene_token"]]
        for item in samples
        if item.get("scene_token") in scene_by_token
    }
    if not result:
        raise PipelineError("Nie znaleziono sample tokenów splitu val w metadanych nuScenes.")
    return result


def _finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def validate_detection_document(
    document: Any,
    valid_tokens: dict[str, str] | set[str],
    *,
    velocity_policy: str = "require",
    max_errors: int = 100,
) -> dict[str, Any]:
    if velocity_policy not in {"require", "ignore"}:
        raise ValueError("velocity_policy must be 'require' or 'ignore'")
    known = set(valid_tokens)
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(document, dict):
        return {"valid": False, "errors": ["Korzeń JSON musi być obiektem."], "warnings": []}
    results = document.get("results")
    if not isinstance(results, dict):
        return {"valid": False, "errors": ["Pole 'results' musi być obiektem."], "warnings": []}
    if not results:
        return {"valid": False, "errors": ["Pole 'results' jest puste."], "warnings": []}
    if "meta" not in document:
        warnings.append("Brak opcjonalnego pola 'meta'.")

    class_counts: Counter[str] = Counter()
    box_count = 0
    velocity_present = 0
    velocity_missing = 0
    invalid_tokens: list[str] = []

    def error(message: str) -> None:
        if len(errors) < max_errors:
            errors.append(message)

    for token, detections in results.items():
        if not isinstance(token, str) or not token:
            error(f"Niepoprawny sample_token: {token!r}")
            continue
        if token not in known:
            invalid_tokens.append(token)
            error(f"Token nie należy do nuScenes validation: {token}")
        if not isinstance(detections, list):
            error(f"results[{token}] musi być listą.")
            continue
        for index, detection in enumerate(detections):
            box_count += 1
            where = f"results[{token}][{index}]"
            if not isinstance(detection, dict) or not detection:
                error(f"{where} jest pustym lub uszkodzonym rekordem.")
                continue
            for field in REQUIRED_FIELDS:
                if field not in detection:
                    error(f"{where}: brak pola {field!r}.")
            if not _finite_vector(detection.get("translation"), 3):
                error(f"{where}.translation musi zawierać 3 skończone liczby.")
            size = detection.get("size")
            if not _finite_vector(size, 3):
                error(f"{where}.size musi zawierać 3 skończone liczby [w, l, h].")
            elif any(float(item) <= 0 for item in size):
                error(f"{where}.size musi zawierać dodatnie wymiary [w, l, h].")
            rotation = detection.get("rotation")
            if not _finite_vector(rotation, 4):
                error(f"{where}.rotation musi być quaternionem [w, x, y, z].")
            elif math.sqrt(sum(float(item) ** 2 for item in rotation)) < 1e-8:
                error(f"{where}.rotation ma zerową normę.")
            score = detection.get("detection_score")
            if not (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
            ):
                error(f"{where}.detection_score musi być skończoną liczbą.")
            name = detection.get("detection_name")
            if not isinstance(name, str) or not name:
                error(f"{where}.detection_name musi być niepustym napisem.")
            else:
                class_counts[name] += 1
            if "velocity" not in detection:
                velocity_missing += 1
                if velocity_policy == "require":
                    error(f"{where}: brak velocity wymaganej przez politykę 'require'.")
            else:
                velocity_present += 1
                if not _finite_vector(detection["velocity"], 2):
                    error(f"{where}.velocity musi zawierać 2 skończone liczby.")

    non_tracking = sorted(set(class_counts) - set(TRACKING_CLASSES))
    if non_tracking:
        warnings.append(
            "Klasy nieoceniane w benchmarku tracking pozostaną bez mapowania: "
            + ", ".join(non_tracking)
        )
    missing_tokens = known - set(results)
    if missing_tokens:
        warnings.append(
            f"Brak jawnych wpisów dla {len(missing_tokens)} klatek val; zostaną potraktowane jako puste."
        )
    if len(errors) == max_errors:
        warnings.append(f"Raport błędów ograniczono do pierwszych {max_errors} pozycji.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "sample_entries": len(results),
            "expected_validation_samples": len(known),
            "missing_sample_entries": len(missing_tokens),
            "invalid_sample_entries": len(invalid_tokens),
            "detections": box_count,
            "velocity_present": velocity_present,
            "velocity_missing": velocity_missing,
            "velocity_policy": velocity_policy,
            "class_counts": dict(sorted(class_counts.items())),
            "tracking_classes": list(TRACKING_CLASSES),
            "coordinate_contract": "global nuScenes coordinates; no transformation is applied",
            "size_contract": "[width, length, height]; values are not reordered",
            "quaternion_contract": "[w, x, y, z]; values are not reordered or normalized",
        },
    }


def validate_detection_file(
    path: Path,
    nuscenes_root: Path,
    *,
    velocity_policy: str = "require",
    version: str = "v1.0-trainval",
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Nie można odczytać wejściowego JSON {path}: {exc}") from exc
    tokens = validation_sample_tokens(nuscenes_root, version)
    report = validate_detection_document(
        document, tokens, velocity_policy=velocity_policy
    )
    report["input_json"] = str(path.resolve())
    report["nuscenes_root"] = str(nuscenes_root.resolve())
    report["split"] = "val"
    report["mode"] = "2hz_keyframes"
    return report


def subset_detection_file(
    source: Path,
    destination: Path,
    tokens: Iterable[str],
) -> None:
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Nie można utworzyć podzbioru debug: {exc}") from exc
    results = document.get("results", {})
    subset = {token: results.get(token, []) for token in tokens}
    output = {"meta": document.get("meta", {}), "results": subset}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)
