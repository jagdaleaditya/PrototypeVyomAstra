"""
Unit Tests for RackRelativeFeatures and Extended RackRelativePose
================================================================
Verifies:
1. 3D Hand-object distance and relative vector calculation in rack frame.
2. 3D Object frame-to-frame displacement, pickup displacement, and cumulative path.
3. 3D Hand velocities, speeds, and smoothing.
4. 3D Object velocity and speed.
5. Body orientation relative to rack axes.
6. Feature vector export (32 dimensions) for learned HAR.
7. Extended RackRelativePose hand and object transformation methods.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HMR_PATH = os.path.join(PROJECT_ROOT, "hmr")
if HMR_PATH not in sys.path:
    sys.path.insert(0, HMR_PATH)

from rack_relative_features import RackRelativeFeatures
from rack_relative_pose import RackRelativePose, CoordinateConvention


class MockLandmark:
    """Mock MediaPipe landmark."""
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class TestRackRelativeFeatures(unittest.TestCase):
    """
    Unit test suite for RackRelativeFeatures.
    """

    def setUp(self):
        self.features = RackRelativeFeatures(velocity_smoothing_alpha=1.0)  # alpha=1.0 for exact numerical test

    def test_hand_object_distance(self):
        """Test 3D Euclidean distance and closest hand identification."""
        lh_rack = np.array([0.0, 0.0, 0.5])
        rh_rack = np.array([0.5, 0.0, 0.5])
        obj_rack = np.array([0.0, 0.0, 0.0])

        res = self.features.update(
            left_hand_rack=lh_rack,
            right_hand_rack=rh_rack,
            object_rack=obj_rack,
            timestamp=1.0
        )

        hod = res["hand_object_distance"]
        self.assertEqual(hod["closest_hand"], "LEFT")
        self.assertAlmostEqual(hod["left_hand_dist_m"], 0.5, places=3)
        self.assertAlmostEqual(hod["right_hand_dist_m"], np.sqrt(0.5**2 + 0.5**2), places=3)
        self.assertAlmostEqual(hod["min_distance_m"], 0.5, places=3)
        np.testing.assert_allclose(hod["relative_vector_rack"], [0.0, 0.0, -0.5], atol=1e-3)

    def test_object_displacement_and_cumulative(self):
        """Test frame displacement, cumulative distance, and pickup displacement."""
        t0 = 1.0
        p0 = np.array([0.0, 0.0, 0.0])
        self.features.update(object_rack=p0, timestamp=t0)

        # Frame 1: move +0.1 in X
        t1 = 1.1
        p1 = np.array([0.1, 0.0, 0.0])
        res1 = self.features.update(object_rack=p1, timestamp=t1)
        od1 = res1["object_displacement"]
        self.assertAlmostEqual(od1["frame_displacement_mag_m"], 0.1, places=3)
        self.assertAlmostEqual(od1["cumulative_distance_m"], 0.1, places=3)

        # Set pickup reference at p1
        self.features.set_pickup_reference(p1)

        # Frame 2: move +0.2 in Y
        t2 = 1.2
        p2 = np.array([0.1, 0.2, 0.0])
        res2 = self.features.update(object_rack=p2, timestamp=t2)
        od2 = res2["object_displacement"]
        self.assertAlmostEqual(od2["frame_displacement_mag_m"], 0.2, places=3)
        self.assertAlmostEqual(od2["cumulative_distance_m"], 0.3, places=3)
        # Displacement relative to pickup reference p1:
        self.assertAlmostEqual(od2["pickup_displacement_mag_m"], 0.2, places=3)
        np.testing.assert_allclose(od2["pickup_displacement_m"], [0.0, 0.2, 0.0], atol=1e-3)

    def test_velocities_calculation(self):
        """Test numerical differentiation for hand and object velocities."""
        t0 = 0.0
        self.features.update(
            left_hand_rack=[0.0, 0.0, 0.0],
            object_rack=[1.0, 0.0, 0.0],
            timestamp=t0
        )

        # Delta t = 0.1s, hand moves 0.1m along Z, object moves 0.05m along X
        t1 = 0.1
        res = self.features.update(
            left_hand_rack=[0.0, 0.0, 0.1],
            object_rack=[1.05, 0.0, 0.0],
            timestamp=t1
        )

        hv = res["hand_velocity"]
        ov = res["object_velocity"]

        self.assertAlmostEqual(hv["left_hand_speed_mps"], 1.0, places=3)
        np.testing.assert_allclose(hv["left_hand_velocity_mps"], [0.0, 0.0, 1.0], atol=1e-3)

        self.assertAlmostEqual(ov["speed_mps"], 0.5, places=3)
        np.testing.assert_allclose(ov["velocity_mps"], [0.5, 0.0, 0.0], atol=1e-3)

    def test_body_orientation_metrics(self):
        """Test body orientation mapping into feature structure."""
        body_mock = {
            "euler_angles_deg": {"yaw": 15.5, "pitch": -2.1, "roll": 0.5},
            "facing_rack_score": 0.88,
            "facing_vector": [0.0, 0.2, -0.98],
            "torso_center": [0.05, 0.3, 1.25]
        }

        res = self.features.update(body_orientation=body_mock, timestamp=1.0)
        bo = res["body_orientation"]

        self.assertEqual(bo["torso_yaw_deg"], 15.5)
        self.assertEqual(bo["torso_pitch_deg"], -2.1)
        self.assertEqual(bo["torso_roll_deg"], 0.5)
        self.assertEqual(bo["facing_rack_score"], 0.88)
        self.assertAlmostEqual(bo["astronaut_distance_to_rack_m"], 1.25, places=3)

    def test_ml_feature_vector_export(self):
        """Verify export to 1D feature vector for learned HAR."""
        self.features.update(
            left_hand_rack=[0.1, 0.2, 0.3],
            right_hand_rack=[-0.1, 0.2, 0.3],
            object_rack=[0.0, 0.2, 0.2],
            body_orientation={
                "euler_angles_deg": {"yaw": 10.0, "pitch": 0.0, "roll": 0.0},
                "facing_rack_score": 0.95,
                "facing_vector": [0.0, 0.0, -1.0],
                "torso_center": [0.0, 0.4, 1.0]
            },
            timestamp=1.0
        )

        feat_vec = self.features.to_feature_vector()
        feat_names = self.features.get_feature_names()

        self.assertIsInstance(feat_vec, np.ndarray)
        self.assertEqual(feat_vec.shape, (32,))
        self.assertEqual(len(feat_names), 32)
        # Verify no NaN or Inf
        self.assertFalse(np.isnan(feat_vec).any())
        self.assertFalse(np.isinf(feat_vec).any())

    def test_reset(self):
        """Test reset functionality."""
        self.features.update(object_rack=[1.0, 1.0, 1.0], timestamp=1.0)
        self.features.update(object_rack=[2.0, 1.0, 1.0], timestamp=2.0)
        self.assertGreater(self.features.cumulative_obj_dist, 0.0)

        self.features.reset()
        self.assertIsNone(self.features.last_obj_pos)
        self.assertEqual(self.features.cumulative_obj_dist, 0.0)
        self.assertEqual(len(self.features.current_features), 0)


class TestExtendedRackRelativePose(unittest.TestCase):
    """
    Tests hand and object transformations in RackRelativePose.
    """

    def setUp(self):
        self.module = RackRelativePose(
            rack_calibration_file="__nonexistent_rack__.npz",
            camera_calibration_file="__nonexistent_cam__.npz",
            convention=CoordinateConvention.RACK_LOCAL
        )
        # Set synthetic calibration: Camera shifted by [0, 0, 1.0] in front of rack
        R_identity = np.eye(3, dtype=np.float64)
        t_cam = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)  # Rack is at Z=1 in camera frame
        K = np.array([
            [500.0,   0.0, 320.0],
            [  0.0, 500.0, 240.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)
        self.module.set_calibration(R_identity, t_cam, camera_matrix=K)

    def test_transform_hand_camera_coords(self):
        """Test transforming hand from 3D camera coordinates to rack frame."""
        hand_cam = np.array([0.1, 0.2, 1.5])
        # P_rack = P_cam - [0, 0, 1.0] = [0.1, 0.2, 0.5]
        hand_rack = self.module.transform_hand_to_rack(hand_cam)
        np.testing.assert_allclose(hand_rack, [0.1, 0.2, 0.5], atol=1e-6)

    def test_transform_hand_landmark(self):
        """Test transforming hand from MediaPipe Landmark object."""
        lm = MockLandmark(0.05, -0.1, 1.2)
        hand_rack = self.module.transform_hand_to_rack(lm)
        np.testing.assert_allclose(hand_rack, [0.05, -0.1, 0.2], atol=1e-6)

    def test_transform_object_3d(self):
        """Test transforming object 3D point."""
        obj_cam = [0.0, 0.0, 1.0]  # At rack origin in camera coordinates
        obj_rack = self.module.transform_object_to_rack(obj_cam)
        np.testing.assert_allclose(obj_rack, [0.0, 0.0, 0.0], atol=1e-6)

    def test_transform_object_2d_pixel_with_depth(self):
        """Test unprojecting 2D pixel with known depth."""
        # Principal point (320, 240) with depth 1.0m is directly on optical axis (0, 0, 1)
        obj_rack = self.module.transform_object_to_rack((320, 240), depth_m=1.0)
        np.testing.assert_allclose(obj_rack, [0.0, 0.0, 0.0], atol=1e-6)

    def test_transform_object_2d_bounding_box(self):
        """Test unprojecting 2D bounding box [300, 220, 340, 260] (center 320, 240)."""
        box = [300, 220, 340, 260]
        obj_rack = self.module.transform_object_to_rack(box, depth_m=1.0)
        np.testing.assert_allclose(obj_rack, [0.0, 0.0, 0.0], atol=1e-6)

    def test_transform_object_rack_plane_intersection(self):
        """Test ray intersection with rack plane (Z_rack = 0)."""
        # Ray through optical axis (320, 240) hits rack plane at (0, 0, 0)
        obj_rack = self.module.transform_object_to_rack((320, 240), rack_plane_z=0.0)
        np.testing.assert_allclose(obj_rack, [0.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
