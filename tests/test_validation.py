import unittest

from simpletrack_pipeline.common import experiment_name
from simpletrack_pipeline.validation import validate_detection_document


def valid_box():
    return {
        "translation": [1.0, 2.0, 3.0],
        "size": [2.0, 4.0, 1.5],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "velocity": [0.1, -0.2],
        "detection_name": "car",
        "detection_score": 0.9,
    }


class DetectionValidationTest(unittest.TestCase):
    def test_valid_document(self):
        document = {"results": {"token": [valid_box()]}, "meta": {}}
        report = validate_detection_document(document, {"token"})
        self.assertTrue(report["valid"])
        self.assertEqual(report["summary"]["detections"], 1)

    def test_missing_velocity_can_only_be_explicitly_ignored(self):
        box = valid_box()
        del box["velocity"]
        document = {"results": {"token": [box]}, "meta": {}}
        self.assertFalse(
            validate_detection_document(document, {"token"}, velocity_policy="require")[
                "valid"
            ]
        )
        self.assertTrue(
            validate_detection_document(document, {"token"}, velocity_policy="ignore")[
                "valid"
            ]
        )

    def test_nan_and_unknown_token_are_rejected(self):
        box = valid_box()
        box["detection_score"] = float("nan")
        report = validate_detection_document(
            {"results": {"not-val": [box]}}, {"token"}
        )
        self.assertFalse(report["valid"])
        self.assertGreaterEqual(len(report["errors"]), 2)

    def test_method_names_are_stable(self):
        self.assertEqual(experiment_name("bevfusion"), "BEVFusion_SimpleTrack_2Hz")
        self.assertEqual(experiment_name("mvp"), "MVP_SimpleTrack_2Hz")
        self.assertEqual(experiment_name("lcf3d"), "LCF3D_SimpleTrack_2Hz")


if __name__ == "__main__":
    unittest.main()
