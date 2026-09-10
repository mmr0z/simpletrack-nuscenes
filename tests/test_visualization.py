import unittest

import numpy as np

from simpletrack_pipeline.visualization import (
    CameraFrame,
    EgoFrame,
    box_corners_ego,
    detection_match_text,
    ego_to_bev,
    match_detections_to_ground_truth,
    stable_track_color,
)


class VisualizationGeometryTest(unittest.TestCase):
    def frame(self, *, ego_rotation=None, lidar_rotation=None, lidar_translation=None):
        return EgoFrame(
            "token",
            None,
            np.eye(3) if ego_rotation is None else ego_rotation,
            np.array([10.0, 20.0, 0.0]),
            np.eye(3) if lidar_rotation is None else lidar_rotation,
            np.zeros(3) if lidar_translation is None else lidar_translation,
        )

    def test_global_to_ego_translation_and_rotation(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        frame = self.frame(ego_rotation=rotation)
        global_point = np.array([[10.0, 21.0, 0.0]])
        np.testing.assert_allclose(frame.global_to_ego(global_point), [[1.0, 0.0, 0.0]])

    def test_lidar_points_are_calibrated_into_ego(self):
        rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        frame = self.frame(lidar_rotation=rotation, lidar_translation=np.array([1.0, 2.0, 0.0]))
        np.testing.assert_allclose(frame.lidar_to_ego([[1.0, 0.0, 0.0]]), [[1.0, 1.0, 0.0]])

    def test_nuscenes_width_length_order(self):
        frame = EgoFrame("token", None, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))
        box = {
            "translation": [0.0, 0.0, 0.0],
            "size": [2.0, 4.0, 1.5],
            "rotation": [1.0, 0.0, 0.0, 0.0],
        }
        corners = box_corners_ego(box, frame)
        self.assertEqual(set(corners[:, 0]), {-2.0, 2.0})
        self.assertEqual(set(corners[:, 1]), {-1.0, 1.0})

    def test_bev_orientation_places_forward_at_top(self):
        points = ego_to_bev(np.array([[5.0, 2.0, 0.0]]))
        np.testing.assert_allclose(points, [[-2.0, 5.0]])

    def test_track_color_is_stable(self):
        self.assertEqual(stable_track_color("car_0_1"), stable_track_color("car_0_1"))
        self.assertNotEqual(stable_track_color("car_0_1"), stable_track_color("car_0_2"))

    def test_camera_projection(self):
        frame = CameraFrame(
            "token",
            "CAM_FRONT",
            None,
            (1600, 900),
            np.array([[100.0, 0.0, 800.0], [0.0, 100.0, 450.0], [0.0, 0.0, 1.0]]),
            np.eye(3),
            np.zeros(3),
        )
        np.testing.assert_allclose(frame.project([[2.0, 4.0, 10.0]]), [[820.0, 490.0]])

    def test_detection_matching_is_class_aware(self):
        predictions = [
            {"tracking_name": "car", "translation": [0.0, 0.0, 0.0]},
            {"tracking_name": "pedestrian", "translation": [10.0, 0.0, 0.0]},
        ]
        ground_truth = [
            {"tracking_name": "car", "translation": [1.0, 0.0, 0.0]},
            {"tracking_name": "car", "translation": [10.0, 0.0, 0.0]},
        ]
        match = match_detections_to_ground_truth(predictions, ground_truth, 2.0)
        self.assertEqual((match.tp, match.fp, match.fn), (1, 1, 1))
        self.assertEqual(match.matched_pairs, ((0, 0),))
        self.assertIn("TP 1 | FP 1 | FN 1", detection_match_text(match))

    def test_detection_matching_rejects_distance_over_threshold(self):
        predictions = [{"tracking_name": "car", "translation": [0.0, 0.0, 0.0]}]
        ground_truth = [{"tracking_name": "car", "translation": [2.1, 0.0, 0.0]}]
        match = match_detections_to_ground_truth(predictions, ground_truth, 2.0)
        self.assertEqual((match.tp, match.fp, match.fn), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
