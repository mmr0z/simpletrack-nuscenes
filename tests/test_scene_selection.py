import unittest

import numpy as np

from tools.select_tracking_scenes import context_tags, match_frame, select_rows


class SceneSelectionTest(unittest.TestCase):
    def test_context_tags_cover_night_and_rain(self):
        tags = context_tags("Night, rain, difficult lighting, wait at intersection")
        self.assertIn("night", tags)
        self.assertIn("rain", tags)
        self.assertIn("intersection", tags)

    def test_matching_requires_same_class_and_two_meter_gate(self):
        first = {"car": np.array([[0.0, 0.0], [10.0, 0.0]])}
        second = {
            "car": np.array([[0.5, 0.0], [20.0, 0.0]]),
            "pedestrian": np.array([[0.0, 0.0]]),
        }
        denominator, matches, distance_sum, distance_count = match_frame(first, second, 2.0)
        self.assertEqual(denominator, 5)
        self.assertEqual(matches, 1)
        self.assertEqual(distance_count, 1)
        self.assertAlmostEqual(distance_sum, 0.5)

    def test_selection_is_unique(self):
        rows = []
        for index in range(14):
            rows.append(
                {
                    "scene": f"scene-{index:04d}",
                    "location": f"location-{index % 4}",
                    "context_tags": [
                        ("night", "rain", "intersection", "construction", "dense_traffic")[index % 5]
                    ],
                    "diagnostic_score": index / 14,
                    "disagreement_ratio_2m": index / 14,
                    "vulnerable_boxes_per_frame": index,
                    "heavy_boxes_per_frame": 14 - index,
                }
            )
        selected = select_rows(rows, 12)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["scene"] for row in selected}), 12)


if __name__ == "__main__":
    unittest.main()
