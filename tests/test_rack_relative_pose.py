"""
Unit Tests for RackRelativePose Module
======================================
Tests forward and inverse 3D transformations using synthetic SE(3) transformation matrices,
coordinate conventions, body orientation estimations, and calibration loading.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root and hmr are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HMR_PATH = os.path.join(PROJECT_ROOT, "hmr")
if HMR_PATH not in sys.path:
    sys.path.insert(0, HMR_PATH)

from rack_relative_pose import (
    RackRelativePose,
    CoordinateConvention,
    CONVENTION_MATRICES
)


class TestRackRelativePoseSynthetic(unittest.TestCase):
    """
    Test suite using known synthetic rotation matrices and translation vectors.
    """

    def setUp(self):
        # Create uncalibrated instance
        self.module = RackRelativePose(
            rack_calibration_file="__nonexistent_rack__.npz",
            camera_calibration_file="__nonexistent_cam__.npz",
            convention=CoordinateConvention.RACK_LOCAL
        )

    def test_uncalibrated_behavior(self):
        """Verify uncalibrated module raises error on transform and returns None on process."""
        self.assertFalse(self.module.is_calibrated)
        with self.assertRaises(RuntimeError):
            self.module.transform_camera_to_rack([0, 0, 1])
        res = self.module.process_landmarks({"NOSE": [0, 0, 1]})
        self.assertIsNone(res)

    def test_identity_transformation(self):
        """
        With Identity rotation and zero translation, Camera frame == Rack frame.
        """
        R_identity = np.eye(3, dtype=np.float64)
        t_zero = np.zeros((3, 1), dtype=np.float64)
        self.module.set_calibration(R_identity, t_zero)

        self.assertTrue(self.module.is_calibrated)

        test_points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([-0.5, 1.25, -2.5]),
        ]

        for pt_cam in test_points:
            pt_rack = self.module.transform_camera_to_rack(pt_cam)
            np.testing.assert_allclose(pt_rack, pt_cam, atol=1e-12)

            # Test round-trip inverse
            pt_cam_recon = self.module.transform_rack_to_camera(pt_rack)
            np.testing.assert_allclose(pt_cam_recon, pt_cam, atol=1e-12)

    def test_pure_translation(self):
        """
        Test pure translation: P_cam = P_rack + t  =>  P_rack = P_cam - t
        """
        R_identity = np.eye(3, dtype=np.float64)
        t = np.array([[0.5], [-0.3], [1.2]], dtype=np.float64)
        self.module.set_calibration(R_identity, t)

        pt_cam = np.array([1.5, 0.7, 2.2], dtype=np.float64)
        expected_rack = pt_cam - t.flatten()  # [1.0, 1.0, 1.0]

        pt_rack = self.module.transform_camera_to_rack(pt_cam)
        np.testing.assert_allclose(pt_rack, expected_rack, atol=1e-12)

        pt_cam_recon = self.module.transform_rack_to_camera(pt_rack)
        np.testing.assert_allclose(pt_cam_recon, pt_cam, atol=1e-12)

    def test_pure_rotation_90deg_z(self):
        """
        Test 90-degree rotation about Z-axis.
        R_rack_to_cam = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        """
        theta = np.pi / 2.0
        R_z90 = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0]
        ], dtype=np.float64)
        t_zero = np.zeros((3, 1), dtype=np.float64)
        self.module.set_calibration(R_z90, t_zero)

        # Rack point [1, 0, 0] transformed to cam: R @ P_rack = [0, 1, 0]
        # Therefore, cam point [0, 1, 0] transformed to rack must be [1, 0, 0]
        pt_cam = np.array([0.0, 1.0, 0.0])
        pt_rack = self.module.transform_camera_to_rack(pt_cam)
        expected_rack = np.array([1.0, 0.0, 0.0])

        np.testing.assert_allclose(pt_rack, expected_rack, atol=1e-12)

        # Roundtrip
        pt_cam_recon = self.module.transform_rack_to_camera(pt_rack)
        np.testing.assert_allclose(pt_cam_recon, pt_cam, atol=1e-12)

    def test_general_se3_synthetic_transformation(self):
        """
        Test arbitrary synthetic 3D rotation and translation.
        """
        # Euler angles: yaw=30 deg, pitch=45 deg, roll=60 deg
        cy = np.cos(np.radians(30))
        sy = np.sin(np.radians(30))
        cp = np.cos(np.radians(45))
        sp = np.sin(np.radians(45))
        cr = np.cos(np.radians(60))
        sr = np.sin(np.radians(60))

        R_synthetic = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ], dtype=np.float64)

        t_synthetic = np.array([[-0.245], [0.182], [0.891]], dtype=np.float64)

        self.module.set_calibration(R_synthetic, t_synthetic)

        # Batch points
        rng = np.random.default_rng(42)
        random_cam_points = rng.uniform(-5.0, 5.0, size=(100, 3))

        # Test single transformation vs batch transformation
        batch_rack = self.module.transform_points_camera_to_rack(random_cam_points)

        for i, pt_cam in enumerate(random_cam_points):
            single_rack = self.module.transform_camera_to_rack(pt_cam)
            np.testing.assert_allclose(batch_rack[i], single_rack, atol=1e-12)

            # Invert and check roundtrip
            pt_cam_recon = self.module.transform_rack_to_camera(single_rack)
            np.testing.assert_allclose(pt_cam_recon, pt_cam, atol=1e-12)

    def test_configurable_coordinate_conventions(self):
        """
        Verify that switching conventions correctly remaps the output axes.
        """
        R_identity = np.eye(3, dtype=np.float64)
        t_zero = np.zeros((3, 1), dtype=np.float64)
        self.module.set_calibration(R_identity, t_zero)

        pt_cam = np.array([1.0, 2.0, 3.0])  # In RACK_LOCAL: [X=1, Y=2, Z=3]

        # 1. RACK_LOCAL: [1, 2, 3]
        p_local = self.module.transform_camera_to_rack(
            pt_cam, convention=CoordinateConvention.RACK_LOCAL
        )
        np.testing.assert_allclose(p_local, [1.0, 2.0, 3.0])

        # 2. OPENCV: X=X_rack, Y=-Y_rack, Z=Z_rack => [1, -2, 3]
        p_cv = self.module.transform_camera_to_rack(
            pt_cam, convention=CoordinateConvention.OPENCV
        )
        np.testing.assert_allclose(p_cv, [1.0, -2.0, 3.0])

        # 3. ROS (REP-103): X_ros = Z_rack, Y_ros = -X_rack, Z_ros = Y_rack => [3, -1, 2]
        p_ros = self.module.transform_camera_to_rack(
            pt_cam, convention=CoordinateConvention.ROS
        )
        np.testing.assert_allclose(p_ros, [3.0, -1.0, 2.0])

        # 4. NED: X_ned = Z_rack, Y_ned = X_rack, Z_ned = -Y_rack => [3, 1, -2]
        p_ned = self.module.transform_camera_to_rack(
            pt_cam, convention=CoordinateConvention.NED
        )
        np.testing.assert_allclose(p_ned, [3.0, 1.0, -2.0])

        # 5. CUSTOM: user-defined permutation
        custom_mat = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ])
        self.module.set_convention(CoordinateConvention.CUSTOM, custom_matrix=custom_mat)
        p_custom = self.module.transform_camera_to_rack(pt_cam)
        np.testing.assert_allclose(p_custom, [2.0, 3.0, 1.0])

    def test_orientation_euler_and_quaternion(self):
        """
        Verify Euler angles and quaternion extraction from rotation matrix.
        """
        # 90 deg Yaw about Z
        R_yaw90 = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])
        yaw, pitch, roll = RackRelativePose.rotation_matrix_to_euler(R_yaw90)
        self.assertAlmostEqual(yaw, 90.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(roll, 0.0, places=5)

        qw, qx, qy, qz = RackRelativePose.rotation_matrix_to_quaternion(R_yaw90)
        # Expected quaternion for 90 deg Z rotation: [cos(45), 0, 0, sin(45)] = [0.7071, 0, 0, 0.7071]
        self.assertAlmostEqual(qw, np.cos(np.pi/4), places=4)
        self.assertAlmostEqual(qx, 0.0, places=4)
        self.assertAlmostEqual(qy, 0.0, places=4)
        self.assertAlmostEqual(qz, np.sin(np.pi/4), places=4)

    def test_body_frame_and_facing_score(self):
        """
        Verify torso orthonormal frame and facing score when person is facing the rack vs facing away.
        """
        R_identity = np.eye(3, dtype=np.float64)
        t_zero = np.zeros((3, 1), dtype=np.float64)
        self.module.set_calibration(R_identity, t_zero)

        # Person standing in front of rack (at Z = +1.0 meter) facing the rack (-Z direction)
        # Shoulders along X: Left shoulder at (+0.2, 0.3, 1.0), Right shoulder at (-0.2, 0.3, 1.0)
        # Hips at Y = 0.0: Left hip at (+0.15, 0.0, 1.0), Right hip at (-0.15, 0.0, 1.0)
        landmarks_facing_rack = {
            "LEFT_SHOULDER": np.array([0.2, 0.3, 1.0]),
            "RIGHT_SHOULDER": np.array([-0.2, 0.3, 1.0]),
            "LEFT_HIP": np.array([0.15, 0.0, 1.0]),
            "RIGHT_HIP": np.array([-0.15, 0.0, 1.0]),
            "LEFT_WRIST": np.array([0.1, 0.2, 0.5]),
            "RIGHT_WRIST": np.array([-0.1, 0.2, 0.5]),
            "NOSE": np.array([0.0, 0.45, 1.0])
        }

        result = self.module.process_landmarks(landmarks_facing_rack)
        self.assertIsNotNone(result)
        self.assertTrue(result["calibrated"])

        body = result["body_orientation"]
        self.assertIsNotNone(body)
        # Person facing rack directly: chest normal should point in -Z direction
        facing_score = body["facing_rack_score"]
        self.assertGreater(facing_score, 0.95)

        # Person turned 180 degrees away: Left shoulder at -0.2, Right shoulder at +0.2
        landmarks_facing_away = {
            "LEFT_SHOULDER": np.array([-0.2, 0.3, 1.0]),
            "RIGHT_SHOULDER": np.array([0.2, 0.3, 1.0]),
            "LEFT_HIP": np.array([-0.15, 0.0, 1.0]),
            "RIGHT_HIP": np.array([0.15, 0.0, 1.0]),
        }
        res_away = self.module.process_landmarks(landmarks_facing_away)
        body_away = res_away["body_orientation"]
        self.assertLess(body_away["facing_rack_score"], -0.95)


class TestRackRelativePoseRealCalibration(unittest.TestCase):
    """
    Test suite using the actual calibrated files in the repository.
    """

    def test_load_real_npz_files(self):
        """Verify successful loading of real calibration files in data/."""
        module = RackRelativePose()
        self.assertTrue(module.is_calibrated)
        self.assertIsNotNone(module.camera_matrix)
        self.assertIsNotNone(module.distortion)
        self.assertIsNotNone(module.R_rack_to_cam)
        self.assertIsNotNone(module.tvec_rack)

        # Test transformation of rack origin [0, 0, 0]:
        # P_cam_origin = R @ [0,0,0] + t = tvec
        origin_cam = module.transform_rack_to_camera([0.0, 0.0, 0.0])
        np.testing.assert_allclose(origin_cam, module.tvec_rack.flatten(), atol=1e-6)

        # Invert origin_cam: should return [0, 0, 0]
        origin_rack = module.transform_camera_to_rack(origin_cam)
        np.testing.assert_allclose(origin_rack, [0.0, 0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
