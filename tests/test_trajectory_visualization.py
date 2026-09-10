import unittest

import numpy as np

from tools.visualize_tracking_trajectories_3d import (
    collect_tracks,
    split_contiguous,
    trajectory_coordinates,
)


class TrajectoryVisualizationTest(unittest.TestCase):
    def test_gaps_are_not_connected(self):
        observations = [(0, np.zeros(3)), (1, np.zeros(3)), (4, np.zeros(3))]
        segments = split_contiguous(observations)
        self.assertEqual([[item[0] for item in segment] for segment in segments], [[0, 1], [4]])

    def test_time_is_third_coordinate(self):
        observations = [(0, np.array([10.0, 20.0, 1.0])), (1, np.array([12.0, 19.0, 1.0]))]
        _, coordinates = trajectory_coordinates(
            observations,
            np.array([0.0, 0.5]),
            np.array([10.0, 20.0, 1.0]),
            np.eye(3),
            "time",
        )
        np.testing.assert_allclose(coordinates, [[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]])

    def test_short_tracks_are_visual_filter_only(self):
        results = {
            "a": [
                {"tracking_id": "one", "tracking_name": "car", "tracking_score": 1.0, "translation": [0, 0, 0]},
                {"tracking_id": "two", "tracking_name": "car", "tracking_score": 1.0, "translation": [0, 0, 0]},
            ],
            "b": [
                {"tracking_id": "two", "tracking_name": "car", "tracking_score": 1.0, "translation": [1, 0, 0]},
            ],
        }
        tracks = collect_tracks(results, ["a", "b"], {"car"}, 0.0, 2)
        self.assertEqual([track["tracking_id"] for track in tracks], ["two"])


if __name__ == "__main__":
    unittest.main()
