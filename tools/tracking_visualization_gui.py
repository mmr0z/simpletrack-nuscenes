#!/usr/bin/env python3
"""Okienkowy konfigurator eksportu wizualizacji SimpleTrack."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/simpletrack_matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletrack_pipeline.common import METHOD_NAMES, PROJECT_ROOT


METHODS = ("bevfusion", "mvp", "lcf3d")
VIEWS = (
    "BEV",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class SceneRecord:
    name: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.name}  |  {self.description}" if self.description else self.name


def load_scene_records(
    nuscenes_root: Path, split: str | None = "val"
) -> list[SceneRecord]:
    scene_table = nuscenes_root / "v1.0-trainval" / "scene.json"
    if not scene_table.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono {scene_table}. Wskaż katalog główny nuScenes trainval."
        )
    with scene_table.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Niepoprawna tabela scen: {scene_table}")
    allowed_names: set[str] | None = None
    if split is not None:
        from nuscenes.utils.splits import create_splits_scenes

        splits = create_splits_scenes()
        if split not in splits:
            raise ValueError(f"Nieznany split nuScenes: {split}")
        allowed_names = set(splits[split])
    return sorted(
        [
            SceneRecord(
                name=str(record.get("name", "")),
                description=str(record.get("description", "")).strip(),
            )
            for record in records
            if isinstance(record, dict) and record.get("name")
            and (allowed_names is None or record["name"] in allowed_names)
        ],
        key=lambda record: record.name,
    )


def build_visualization_command(
    *,
    python: Path,
    scenes: Sequence[str],
    methods: Sequence[str],
    views: Sequence[str],
    nuscenes_root: Path,
    results_root: Path,
    output_dir: Path,
    frame_index: int,
    dpi: int,
    save_png: bool,
    save_gif: bool,
    show_gt: bool,
    show_detection_status: bool,
    match_distance: float,
    show_lidar: bool,
    labels: bool,
    color_by: str,
) -> list[str]:
    if not scenes or not methods or not views:
        raise ValueError("Wybierz co najmniej jedną scenę, model i widok.")
    if not save_png and not save_gif:
        raise ValueError("Wybierz co najmniej jeden format zapisu.")
    if frame_index < 0 or dpi <= 0 or match_distance <= 0:
        raise ValueError("Indeks klatki nie może być ujemny, a DPI i próg muszą być dodatnie.")
    command = [
        str(python),
        str(PROJECT_ROOT / "tools" / "visualize_tracking_individual_models.py"),
        "--nuscenes-root",
        str(nuscenes_root),
        "--results-root",
        str(results_root),
        "--output-dir",
        str(output_dir),
        "--frame-index",
        str(frame_index),
        "--dpi",
        str(dpi),
        "--color-by",
        color_by,
    ]
    for scene in scenes:
        command.extend(("--scene", scene))
    for method in methods:
        command.extend(("--method", method))
    for view in views:
        command.extend(("--view", view))
    if save_gif:
        command.append("--gif" if save_png else "--gif-only")
    if show_gt:
        command.append("--show-gt")
    if show_detection_status:
        command.extend(("--show-detection-status", "--match-distance", str(match_distance)))
    if not show_lidar:
        command.append("--no-lidar")
    if labels:
        command.append("--labels")
    return command


def default_nuscenes_root() -> str:
    candidate = Path("/home/workstation/mmdetection3d/data/nuscenes")
    return str(candidate) if candidate.is_dir() else ""


class VisualizationApp:
    def __init__(self, root: object) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("SimpleTrack — eksport wizualizacji nuScenes")
        self.root.geometry("1120x780")
        self.root.minsize(920, 680)
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.worker_running = False
        self.scene_records: list[SceneRecord] = []
        self.filtered_scenes: list[SceneRecord] = []

        self.nuscenes_var = tk.StringVar(value=default_nuscenes_root())
        self.results_var = tk.StringVar(value=str(PROJECT_ROOT / "tracking_results"))
        self.output_var = tk.StringVar(
            value=str(PROJECT_ROOT / "tracking_visualizations" / "gui_exports")
        )
        self.search_var = tk.StringVar()
        self.frame_var = tk.IntVar(value=0)
        self.dpi_var = tk.IntVar(value=120)
        self.png_var = tk.BooleanVar(value=True)
        self.gif_var = tk.BooleanVar(value=False)
        self.gt_var = tk.BooleanVar(value=False)
        self.status_var = tk.BooleanVar(value=True)
        self.match_distance_var = tk.DoubleVar(value=2.0)
        self.lidar_var = tk.BooleanVar(value=True)
        self.labels_var = tk.BooleanVar(value=False)
        self.color_var = tk.StringVar(value="class")
        self.method_vars = {method: tk.BooleanVar(value=method != "lcf3d") for method in METHODS}
        self.view_vars = {view: tk.BooleanVar(value=view in ("BEV", "CAM_FRONT")) for view in VIEWS}

        self._build_layout()
        self.search_var.trace_add("write", lambda *_: self._filter_scenes())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.nuscenes_var.get():
            self._load_scenes(show_error=False)

    def _build_layout(self) -> None:
        from tkinter import scrolledtext

        ttk = self.ttk
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        paths = ttk.LabelFrame(root, text="Ścieżki", padding=8)
        paths.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, "nuScenes:", self.nuscenes_var, self._choose_nuscenes)
        self._path_row(paths, 1, "Wyniki trackingu:", self.results_var, self._choose_results)
        self._path_row(paths, 2, "Katalog zapisu:", self.output_var, self._choose_output)

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        scenes = ttk.LabelFrame(body, text="Sceny nuScenes validation (150)", padding=8)
        scenes.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        scenes.columnconfigure(0, weight=1)
        scenes.rowconfigure(2, weight=1)
        search = ttk.Entry(scenes, textvariable=self.search_var)
        search.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(scenes, text="Wczytaj sceny", command=self._load_scenes).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Label(scenes, text="Szukaj po nazwie lub opisie:").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 2)
        )
        list_frame = ttk.Frame(scenes)
        list_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.scene_list = self.tk.Listbox(
            list_frame, selectmode=self.tk.EXTENDED, exportselection=False
        )
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.scene_list.yview)
        self.scene_list.configure(yscrollcommand=scrollbar.set)
        self.scene_list.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        scene_buttons = ttk.Frame(scenes)
        scene_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(scene_buttons, text="Zaznacz widoczne", command=self._select_all_scenes).pack(side="left")
        ttk.Button(scene_buttons, text="Wyczyść", command=self._clear_scenes).pack(side="left", padx=6)
        self.scene_count_label = ttk.Label(scene_buttons, text="0 scen")
        self.scene_count_label.pack(side="right")

        controls = ttk.Frame(body)
        controls.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        controls.columnconfigure(0, weight=1)

        models = ttk.LabelFrame(controls, text="Modele", padding=8)
        models.grid(row=0, column=0, sticky="ew")
        for column, method in enumerate(METHODS):
            ttk.Checkbutton(
                models, text=METHOD_NAMES[method], variable=self.method_vars[method]
            ).grid(row=0, column=column, sticky="w", padx=(0, 12))

        views = ttk.LabelFrame(controls, text="Widoki — każdy zapisuje się osobno", padding=8)
        views.grid(row=1, column=0, sticky="ew", pady=6)
        for index, view in enumerate(VIEWS):
            ttk.Checkbutton(views, text=view, variable=self.view_vars[view]).grid(
                row=index // 2, column=index % 2, sticky="w", padx=(0, 15), pady=2
            )

        options = ttk.LabelFrame(controls, text="Zapis i wygląd", padding=8)
        options.grid(row=2, column=0, sticky="ew")
        ttk.Label(options, text="Klatka (0–40):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(options, from_=0, to=40, textvariable=self.frame_var, width=7).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(options, text="DPI:").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(options, from_=60, to=300, textvariable=self.dpi_var, width=7).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Label(options, text="Kolory:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.color_var,
            values=("class", "id"),
            state="readonly",
            width=8,
        ).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(options, text="PNG", variable=self.png_var).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(options, text="GIF (cała scena)", variable=self.gif_var).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(options, text="Pokaż GT", variable=self.gt_var).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(options, text="Punkty LiDAR", variable=self.lidar_var).grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(options, text="Etykiety ID", variable=self.labels_var).grid(row=5, column=0, sticky="w")
        ttk.Checkbutton(options, text="Oznacz TP / FP / FN", variable=self.status_var).grid(
            row=5, column=1, sticky="w"
        )
        ttk.Label(options, text="Próg dopasowania [m]:").grid(row=6, column=0, sticky="w")
        ttk.Spinbox(
            options,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.match_distance_var,
            width=7,
        ).grid(row=6, column=1, sticky="w")

        actions = ttk.Frame(controls)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(
            actions, text="Zapisz wybrane widoki", command=self._start_export
        )
        self.run_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(
            actions, text="Przerwij", command=self._stop_export, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=(6, 0))
        ttk.Button(actions, text="Otwórz katalog", command=self._open_output).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        logs = ttk.LabelFrame(root, text="Log renderowania", padding=6)
        logs.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))
        root.rowconfigure(2, weight=1)
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(logs, height=10, state="disabled", wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")

    def _path_row(self, parent: object, row: int, label: str, variable: object, command: object) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        self.ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=2)
        self.ttk.Button(parent, text="Wybierz…", command=command).grid(
            row=row, column=2, padx=(6, 0), pady=2
        )

    def _choose_directory(self, variable: object, title: str, reload_scenes: bool = False) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(title=title, initialdir=variable.get() or str(PROJECT_ROOT))
        if selected:
            variable.set(selected)
            if reload_scenes:
                self._load_scenes()

    def _choose_nuscenes(self) -> None:
        self._choose_directory(self.nuscenes_var, "Wybierz katalog nuScenes", True)

    def _choose_results(self) -> None:
        self._choose_directory(self.results_var, "Wybierz katalog tracking_results")

    def _choose_output(self) -> None:
        self._choose_directory(self.output_var, "Wybierz katalog zapisu")

    def _load_scenes(self, show_error: bool = True) -> None:
        from tkinter import messagebox

        try:
            self.scene_records = load_scene_records(Path(self.nuscenes_var.get()).expanduser())
            self._filter_scenes()
            self._append_log(f"Wczytano {len(self.scene_records)} scen z nuScenes.\n")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.scene_records = []
            self._filter_scenes()
            if show_error:
                messagebox.showerror("Nie można wczytać scen", str(exc))

    def _filter_scenes(self) -> None:
        query = self.search_var.get().strip().lower()
        self.filtered_scenes = [
            record
            for record in self.scene_records
            if not query or query in record.name.lower() or query in record.description.lower()
        ]
        self.scene_list.delete(0, self.tk.END)
        for record in self.filtered_scenes:
            self.scene_list.insert(self.tk.END, record.label)
        self.scene_count_label.configure(text=f"{len(self.filtered_scenes)} scen")

    def _select_all_scenes(self) -> None:
        self.scene_list.selection_set(0, self.tk.END)

    def _clear_scenes(self) -> None:
        self.scene_list.selection_clear(0, self.tk.END)

    def _selected_scenes(self) -> list[str]:
        return [self.filtered_scenes[index].name for index in self.scene_list.curselection()]

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(self.tk.END, text)
        self.log.see(self.tk.END)
        self.log.configure(state="disabled")

    def _start_export(self) -> None:
        from tkinter import messagebox

        try:
            nuscenes_root = Path(self.nuscenes_var.get()).expanduser()
            results_root = Path(self.results_var.get()).expanduser()
            output_dir = Path(self.output_var.get()).expanduser()
            if not nuscenes_root.is_dir():
                raise ValueError("Wskaż istniejący katalog nuScenes.")
            if not results_root.is_dir():
                raise ValueError("Wskaż istniejący katalog z wynikami trackingu.")
            scenes = self._selected_scenes()
            methods = [method for method, variable in self.method_vars.items() if variable.get()]
            views = [view for view, variable in self.view_vars.items() if variable.get()]
            command = build_visualization_command(
                python=Path(sys.executable),
                scenes=scenes,
                methods=methods,
                views=views,
                nuscenes_root=nuscenes_root,
                results_root=results_root,
                output_dir=output_dir,
                frame_index=int(self.frame_var.get()),
                dpi=int(self.dpi_var.get()),
                save_png=self.png_var.get(),
                save_gif=self.gif_var.get(),
                show_gt=self.gt_var.get(),
                show_detection_status=self.status_var.get(),
                match_distance=float(self.match_distance_var.get()),
                show_lidar=self.lidar_var.get(),
                labels=self.labels_var.get(),
                color_by=self.color_var.get(),
            )
        except (OSError, ValueError, self.tk.TclError) as exc:
            messagebox.showerror("Nie można rozpocząć eksportu", str(exc))
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        self._append_log("\nUruchamiam:\n" + " ".join(command) + "\n\n")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(12)
        self.worker_running = True
        threading.Thread(target=self._run_command, args=(command,), daemon=True).start()
        self.root.after(100, self._poll_messages)

    def _run_command(self, command: Sequence[str]) -> None:
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", "/tmp/simpletrack_matplotlib")
        try:
            self.process = subprocess.Popen(
                list(command),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.messages.put(("log", line))
            return_code = self.process.wait()
            self.messages.put(("done", return_code))
        except OSError as exc:
            self.messages.put(("error", str(exc)))
        finally:
            self.process = None

    def _poll_messages(self) -> None:
        from tkinter import messagebox

        finished = False
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                finished = True
                if payload == 0:
                    messagebox.showinfo("Eksport zakończony", f"Pliki zapisano w:\n{self.output_var.get()}")
                else:
                    messagebox.showerror("Eksport nieudany", f"Renderer zakończył się kodem {payload}.")
            elif kind == "error":
                finished = True
                messagebox.showerror("Błąd uruchomienia", str(payload))
        if finished:
            self.worker_running = False
            self.progress.stop()
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
        elif self.worker_running:
            self.root.after(100, self._poll_messages)

    def _stop_export(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._append_log("\nPrzerywanie renderowania…\n")
            self.process.terminate()

    def _open_output(self) -> None:
        from tkinter import messagebox

        output = Path(self.output_var.get()).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(output)])
        except OSError as exc:
            messagebox.showerror("Nie można otworzyć katalogu", str(exc))

    def _on_close(self) -> None:
        from tkinter import messagebox

        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Renderowanie trwa", "Przerwać renderowanie i zamknąć okno?"):
                return
            self.process.terminate()
        self.root.destroy()


def main() -> int:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception as exc:
        print(f"Nie można uruchomić interfejsu Tkinter: {exc}", file=sys.stderr)
        return 2
    VisualizationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
