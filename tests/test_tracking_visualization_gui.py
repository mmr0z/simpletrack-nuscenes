import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.tracking_visualization_gui import (
    build_visualization_command,
    load_scene_records,
)


class TrackingVisualizationGuiTest(unittest.TestCase):
    def test_scene_table_is_loaded_and_sorted(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "v1.0-trainval" / "scene.json"
            table.parent.mkdir(parents=True)
            table.write_text(
                json.dumps(
                    [
                        {"name": "scene-0002", "description": "second"},
                        {"name": "scene-0001", "description": "first"},
                    ]
                ),
                encoding="utf-8",
            )
            records = load_scene_records(root, split=None)
            self.assertEqual([record.name for record in records], ["scene-0001", "scene-0002"])
            self.assertIn("first", records[0].label)

    def test_png_command_contains_only_selected_models_and_views(self):
        command = build_visualization_command(
            python=Path("/venv/python"),
            scenes=["scene-0012"],
            methods=["bevfusion", "mvp"],
            views=["BEV", "CAM_BACK"],
            nuscenes_root=Path("/data/nuscenes"),
            results_root=Path("/results"),
            output_dir=Path("/output"),
            frame_index=4,
            dpi=120,
            save_png=True,
            save_gif=False,
            show_gt=True,
            show_detection_status=True,
            match_distance=2.0,
            show_lidar=True,
            labels=False,
            color_by="class",
        )
        self.assertEqual(command.count("--method"), 2)
        self.assertEqual(command.count("--view"), 2)
        self.assertIn("CAM_BACK", command)
        self.assertNotIn("lcf3d", command)
        self.assertNotIn("--gif", command)
        self.assertNotIn("--gif-only", command)
        self.assertIn("--show-gt", command)
        self.assertIn("--show-detection-status", command)

    def test_gif_only_is_explicit(self):
        command = build_visualization_command(
            python=Path("python"),
            scenes=["scene-0012"],
            methods=["mvp"],
            views=["CAM_FRONT"],
            nuscenes_root=Path("nuscenes"),
            results_root=Path("results"),
            output_dir=Path("output"),
            frame_index=0,
            dpi=90,
            save_png=False,
            save_gif=True,
            show_gt=False,
            show_detection_status=False,
            match_distance=2.0,
            show_lidar=False,
            labels=True,
            color_by="id",
        )
        self.assertIn("--gif-only", command)
        self.assertIn("--no-lidar", command)
        self.assertIn("--labels", command)


if __name__ == "__main__":
    unittest.main()
