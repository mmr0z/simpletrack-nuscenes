from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMPLETRACK_ROOT = PROJECT_ROOT / "third_party" / "SimpleTrack"
DEFAULT_CONFIG = SIMPLETRACK_ROOT / "configs" / "nu_configs" / "giou.yaml"
TRACKING_CLASSES = (
    "car",
    "bus",
    "trailer",
    "truck",
    "pedestrian",
    "bicycle",
    "motorcycle",
)
METHOD_NAMES = {
    "bevfusion": "BEVFusion",
    "mvp": "MVP",
    "lcf3d": "LCF3D",
}
DATA_COMPONENTS = {
    "token_info": ("token_info", ".json"),
    "ts_info": ("ts_info", ".json"),
    "calib_info": ("calib_info", ".npz"),
    "ego_info": ("ego_info", ".npz"),
    "gt_info": ("gt_info", ".npz"),
    "pc": ("pc/raw_pc", ".npz"),
}


class PipelineError(RuntimeError):
    pass


def canonical_method(value: str) -> str:
    method = value.strip().lower()
    if method not in METHOD_NAMES:
        raise PipelineError(
            f"Nieobsługiwana metoda {value!r}; wybierz: {', '.join(METHOD_NAMES)}"
        )
    return method


def experiment_name(method: str, debug_scene: str | None = None) -> str:
    name = f"{METHOD_NAMES[canonical_method(method)]}_SimpleTrack_2Hz"
    return f"{name}_debug_{debug_scene}" if debug_scene else name


def ensure_simpletrack_checkout() -> None:
    required = (
        SIMPLETRACK_ROOT / "tools" / "main_nuscenes.py",
        SIMPLETRACK_ROOT / "preprocessing" / "nuscenes_data" / "detection.py",
        DEFAULT_CONFIG,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PipelineError("Niekompletny checkout SimpleTrack: " + ", ".join(missing))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Nie można odczytać JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def simpletrack_commit() -> str | None:
    # Vendored distributions have no nested Git repository. Do not report the
    # enclosing pipeline repository's commit as the upstream tracker version.
    marker = SIMPLETRACK_ROOT / "UPSTREAM_COMMIT"
    if not (SIMPLETRACK_ROOT / ".git").exists() and marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SIMPLETRACK_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def format_command(command: Sequence[object]) -> str:
    import shlex

    return shlex.join(str(part) for part in command)


def run_command(
    command: Sequence[object],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> None:
    rendered = [str(part) for part in command]
    print(f"+ {format_command(rendered)}", flush=True)
    if dry_run:
        return
    subprocess.run(
        rendered,
        cwd=str(cwd) if cwd else None,
        check=True,
        env=dict(env) if env else None,
    )


def python_command(script: Path, *args: object) -> list[str]:
    return [sys.executable, str(script), *(str(arg) for arg in args)]


def check_nuscenes_root(root: Path, version: str = "v1.0-trainval") -> None:
    missing = []
    for path in (
        root / version / "scene.json",
        root / version / "sample.json",
        root / "samples" / "LIDAR_TOP",
    ):
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise PipelineError(
            "Ścieżka nuScenes jest niekompletna. Brakuje: " + ", ".join(missing)
        )


def token_files(data_folder: Path) -> dict[str, Path]:
    folder = data_folder / "token_info"
    if not folder.is_dir():
        return {}
    return {path.stem: path for path in sorted(folder.glob("*.json"))}


def validation_scene_names() -> set[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/simpletrack_matplotlib")
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:
        raise PipelineError(
            "Kontrola splitu wymaga nuscenes-devkit w środowisku SimpleTrack."
        ) from exc
    return set(create_splits_scenes()["val"])


def scene_tokens(data_folder: Path, scene: str | None = None) -> dict[str, list[str]]:
    files = token_files(data_folder)
    if scene:
        files = {scene: files[scene]} if scene in files else {}
    result: dict[str, list[str]] = {}
    for scene_name, path in files.items():
        value = read_json(path)
        if not isinstance(value, list) or not all(isinstance(token, str) for token in value):
            raise PipelineError(f"Niepoprawny plik tokenów 2 Hz: {path}")
        result[scene_name] = value
    return result


def assert_preprocessed_data(data_folder: Path, debug_scene: str | None = None) -> list[str]:
    expected_scenes = validation_scene_names()
    if debug_scene and debug_scene not in expected_scenes:
        raise PipelineError(f"Scena {debug_scene!r} nie należy do splitu nuScenes val.")
    tokens = scene_tokens(data_folder, debug_scene)
    if not tokens:
        suffix = f" dla {debug_scene}" if debug_scene else ""
        raise PipelineError(f"Brak token_info{suffix} w {data_folder}")
    scenes = sorted(tokens)
    if debug_scene is None and set(scenes) != expected_scenes:
        missing_scenes = sorted(expected_scenes - set(scenes))
        extra_scenes = sorted(set(scenes) - expected_scenes)
        raise PipelineError(
            "Niezgodny split w token_info: "
            f"brak scen={missing_scenes[:5]} ({len(missing_scenes)}), "
            f"nadmiar={extra_scenes[:5]} ({len(extra_scenes)})."
        )
    missing: list[str] = []
    for _, (relative, extension) in DATA_COMPONENTS.items():
        component = data_folder / relative
        for scene in scenes:
            path = component / f"{scene}{extension}"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        preview = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8})" if len(missing) > 8 else ""
        raise PipelineError(f"Niekompletny preprocessing SimpleTrack: {preview}{more}")
    return scenes


def make_data_view(source: Path, destination: Path, debug_scene: str | None = None) -> None:
    """Create a small symlink view so detector artifacts stay experiment-local."""
    destination.mkdir(parents=True, exist_ok=True)
    for _, (relative, extension) in DATA_COMPONENTS.items():
        src = source / relative
        dst = destination / relative
        if debug_scene:
            dst.mkdir(parents=True, exist_ok=True)
            link = dst / f"{debug_scene}{extension}"
            target = src / f"{debug_scene}{extension}"
            _ensure_symlink(link, target)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _ensure_symlink(dst, src)


def _ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.exists():
        raise PipelineError(f"Brak celu dowiązania: {target}")
    if link.is_symlink():
        if link.resolve() != target:
            raise PipelineError(f"Istniejące dowiązanie wskazuje inny cel: {link}")
        return
    if link.exists():
        raise PipelineError(f"Nie można zastąpić istniejącej ścieżki: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def input_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dependency_snapshot() -> dict[str, str | None]:
    return {
        name: package_version(name)
        for name in (
            "numpy",
            "scipy",
            "filterpy",
            "shapely",
            "numba",
            "pyquaternion",
            "nuscenes-devkit",
            "motmetrics",
        )
    }


def base_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/simpletrack_numba_cache")
    existing = env.get("PYTHONPATH")
    roots = [str(SIMPLETRACK_ROOT), str(PROJECT_ROOT)]
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env
