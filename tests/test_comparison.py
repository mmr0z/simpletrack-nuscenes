import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from simpletrack_pipeline.common import PipelineError, write_json
from tools.compare_tracking_results import check_methodology
from tools.evaluate_tracking import export_metrics


class ComparisonGuardTest(unittest.TestCase):
    def test_identical_methodology_is_accepted(self):
        common = {
            "config_sha256": "same",
            "split": "val",
            "mode": "2hz_keyframes",
            "velocity_policy": "require",
        }
        check_methodology([common, dict(common), dict(common)], allow=False)

    def test_different_config_is_rejected(self):
        first = {
            "config_sha256": "a",
            "split": "val",
            "mode": "2hz_keyframes",
            "velocity_policy": "require",
        }
        second = dict(first, config_sha256="b")
        with self.assertRaises(PipelineError):
            check_methodology([first, second], allow=False)

    def test_metric_exports_keep_global_and_per_class_values(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            summary = {
                "amota": 0.5,
                "amotp": 1.0,
                "recall": 0.7,
                "motar": 0.6,
                "mota": 0.4,
                "motp": 0.9,
                "ids": 3,
                "fp": 4,
                "fn": 5,
                "label_metrics": {
                    "amota": {"car": 0.5},
                    "ids": {"car": 3},
                },
            }
            summary_path = folder / "metrics_summary.json"
            write_json(summary_path, summary)
            export_metrics(summary_path, folder)
            self.assertTrue((folder / "metrics_readable.json").is_file())
            self.assertTrue((folder / "metrics_global.csv").is_file())
            self.assertTrue((folder / "metrics_per_class.csv").is_file())


if __name__ == "__main__":
    unittest.main()
