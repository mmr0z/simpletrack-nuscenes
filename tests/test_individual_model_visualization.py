import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simpletrack_pipeline.common import PipelineError
from tools.visualize_tracking_individual_models import requested_scenes, selected_views


class IndividualModelVisualizationTest(unittest.TestCase):
    def test_scene_file_and_explicit_scenes_are_deduplicated(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenes.txt"
            path.write_text("scene-0002\n# comment\nscene-0003\n", encoding="utf-8")
            self.assertEqual(
                requested_scenes(["scene-0001", "scene-0002"], path),
                ["scene-0001", "scene-0002", "scene-0003"],
            )

    def test_missing_scene_source_is_rejected(self):
        with self.assertRaises(PipelineError):
            requested_scenes(None, None)

    def test_views_are_independent_and_deduplicated(self):
        self.assertEqual(
            selected_views(["BEV", "CAM_BACK", "BEV"]), ["BEV", "CAM_BACK"]
        )
        self.assertEqual(selected_views(None), ["BEV", "CAM_FRONT"])


if __name__ == "__main__":
    unittest.main()
