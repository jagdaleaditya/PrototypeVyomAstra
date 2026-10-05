"""
DONUTS - Rack Relative 3D Pose Estimation Module
=================================================
Transforms 3D human pose landmarks and spatial points from Camera Coordinate Frame
into Rack Coordinate Frame using calibrated ArUco extrinsic transformations.

Supports:
- Configurable coordinate conventions (RACK_LOCAL, OPENCV, ROS/REP103, NED, CUSTOM).
- Complete 3D position and orientation outputs (Rotation Matrix, Quaternion, Euler angles).
- Dynamic/Synthetic calibration injection for unit testing.
- Visual axis projection for camera-frame and rack-frame validation.
- Graceful fallback when calibration is absent.
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import cv2


class CoordinateConvention(str, Enum):
    """
    Standard coordinate frame conventions.
    """
    # Rack local frame (X: right along rack, Y: up along rack, Z: normal pointing out towards astronaut)
    RACK_LOCAL = "RACK_LOCAL"
    # Standard OpenCV camera frame (X: right, Y: down, Z: forward away from camera)
    OPENCV = "OPENCV"
    # Robot Operating System (ROS / REP-103) frame (X: forward, Y: left, Z: up)
    ROS = "ROS"
    # Aerospace / Navigation frame (North-East-Down: X: forward, Y: right, Z: down)
    NED = "NED"
    # Custom user-defined rotation/permutation
    CUSTOM = "CUSTOM"


# Conversion matrices from RACK_LOCAL (X: right, Y: up, Z: out/forward) to other conventions:
# P_target = CONVENTION_MATRICES[convention] @ P_rack_local
CONVENTION_MATRICES = {
    CoordinateConvention.RACK_LOCAL: np.eye(3, dtype=np.float64),
    CoordinateConvention.OPENCV: np.array([
        [1.0,  0.0,  0.0],   # X_opencv =  X_rack (right)
        [0.0, -1.0,  0.0],   # Y_opencv = -Y_rack (down)
        [0.0,  0.0,  1.0],   # Z_opencv =  Z_rack (forward)
    ], dtype=np.float64),
    CoordinateConvention.ROS: np.array([
        [0.0,  0.0,  1.0],   # X_ros =  Z_rack (forward)
        [-1.0, 0.0,  0.0],   # Y_ros = -X_rack (left)
        [0.0,  1.0,  0.0],   # Z_ros =  Y_rack (up)
    ], dtype=np.float64),
    CoordinateConvention.NED: np.array([
        [0.0,  0.0,  1.0],   # X_ned =  Z_rack (forward)
        [1.0,  0.0,  0.0],   # Y_ned =  X_rack (right)
        [0.0, -1.0,  0.0],   # Z_ned = -Y_rack (down)
    ], dtype=np.float64),
}


class RackRelativePose:
    """
    Transforms human joint coordinates from camera coordinates into rack coordinates.
    Computes 3D joint positions, body orientations, Euler angles, and distance metrics.
    """

    # Selected MediaPipe Pose Landmark Indices
    LANDMARK_MAP = {
        "NOSE": 0,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
    }

    def __init__(
        self,
        rack_calibration_file: Optional[str] = None,
        camera_calibration_file: Optional[str] = None,
        convention: Union[CoordinateConvention, str] = CoordinateConvention.RACK_LOCAL,
        custom_convention_matrix: Optional[np.ndarray] = None,
        visualize: bool = False
    ):
        self.convention = (
            CoordinateConvention(convention)
            if isinstance(convention, str)
            else convention
        )
        self.custom_convention_matrix = custom_convention_matrix
        self.visualize = visualize

        # Calibration parameters
        self.is_calibrated: bool = False
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion: Optional[np.ndarray] = None
        self.rvec_rack: Optional[np.ndarray] = None
        self.tvec_rack: Optional[np.ndarray] = None
        self.R_rack_to_cam: Optional[np.ndarray] = None
        self.R_cam_to_rack: Optional[np.ndarray] = None
        self.t_cam_to_rack: Optional[np.ndarray] = None

        # Resolve paths if not explicitly provided
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if rack_calibration_file is None:
            rack_calibration_file = os.path.join(
                project_root, "data", "rack_calibration.npz"
            )
        if camera_calibration_file is None:
            camera_calibration_file = os.path.join(
                project_root, "data", "camera_calibration.npz"
            )

        self.rack_calib_path = rack_calibration_file
        self.camera_calib_path = camera_calibration_file

        self.load_calibration(
            rack_file=self.rack_calib_path,
            camera_file=self.camera_calib_path
        )

    # =========================================================================
    # CALIBRATION MANAGEMENT
    # =========================================================================

    def load_calibration(
        self,
        rack_file: str,
        camera_file: str
    ) -> bool:
        """
        Load camera intrinsics and ArUco rack extrinsic calibration from .npz files.
        """
        camera_loaded = False
        rack_loaded = False

        if os.path.exists(camera_file):
            try:
                cam_data = np.load(camera_file)
                self.camera_matrix = np.asarray(cam_data["camera_matrix"], dtype=np.float64)
                self.distortion = np.asarray(cam_data["distortion"], dtype=np.float64)
                camera_loaded = True
            except Exception as e:
                print(f"[RackRelativePose] Warning: Failed to load camera calibration: {e}")

        if os.path.exists(rack_file):
            try:
                rack_data = np.load(rack_file)
                rvec = np.asarray(rack_data["rvec"], dtype=np.float64).reshape(3, 1)
                tvec = np.asarray(rack_data["tvec"], dtype=np.float64).reshape(3, 1)
                R_rack_to_cam = cv2.Rodrigues(rvec)[0]

                self.set_calibration(
                    R_rack_to_cam=R_rack_to_cam,
                    t_rack_to_cam=tvec,
                    camera_matrix=self.camera_matrix,
                    distortion=self.distortion
                )
                rack_loaded = True
            except Exception as e:
                print(f"[RackRelativePose] Warning: Failed to load rack calibration: {e}")

        self.is_calibrated = (camera_loaded and rack_loaded)
        return self.is_calibrated

    def set_calibration(
        self,
        R_rack_to_cam: np.ndarray,
        t_rack_to_cam: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None
    ) -> None:
        """
        Set or override calibration matrices (used by live calibration or unit testing).
        
        solvePnP relation:
            P_cam = R_rack_to_cam @ P_rack + t_rack_to_cam
        Inverse (Camera to Rack):
            P_rack = R_rack_to_cam.T @ (P_cam - t_rack_to_cam)
                   = R_cam_to_rack @ P_cam + t_cam_to_rack
        where:
            R_cam_to_rack = R_rack_to_cam.T
            t_cam_to_rack = -R_rack_to_cam.T @ t_rack_to_cam
        """
        self.R_rack_to_cam = np.asarray(R_rack_to_cam, dtype=np.float64).reshape(3, 3)
        self.tvec_rack = np.asarray(t_rack_to_cam, dtype=np.float64).reshape(3, 1)
        self.rvec_rack = cv2.Rodrigues(self.R_rack_to_cam)[0]

        # Invert to obtain camera -> rack transformation
        self.R_cam_to_rack = self.R_rack_to_cam.T
        self.t_cam_to_rack = -self.R_cam_to_rack @ self.tvec_rack

        if camera_matrix is not None:
            self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if distortion is not None:
            self.distortion = np.asarray(distortion, dtype=np.float64)

        self.is_calibrated = True

    # =========================================================================
    # COORDINATE CONVENTION HELPERS
    # =========================================================================

    def get_convention_matrix(
        self,
        convention: Optional[CoordinateConvention] = None
    ) -> np.ndarray:
        """
        Retrieve 3x3 basis transformation matrix for the specified convention.
        """
        conv = convention or self.convention
        if conv == CoordinateConvention.CUSTOM:
            if self.custom_convention_matrix is None:
                return np.eye(3, dtype=np.float64)
            return np.asarray(self.custom_convention_matrix, dtype=np.float64)
        return CONVENTION_MATRICES.get(conv, np.eye(3, dtype=np.float64))

    def set_convention(
        self,
        convention: Union[CoordinateConvention, str],
        custom_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Update the active coordinate convention.
        """
        self.convention = (
            CoordinateConvention(convention)
            if isinstance(convention, str)
            else convention
        )
        if custom_matrix is not None:
            self.custom_convention_matrix = np.asarray(custom_matrix, dtype=np.float64)

    # =========================================================================
    # 3D POINT TRANSFORMATION
    # =========================================================================

    def transform_camera_to_rack(
        self,
        point_cam: Union[np.ndarray, List[float], Tuple[float, float, float]],
        convention: Optional[CoordinateConvention] = None
    ) -> np.ndarray:
        """
        Transform a single 3D point from Camera coordinates into Rack coordinates:
            P_rack = R_cam_to_rack @ P_cam + t_cam_to_rack
        
        Args:
            point_cam: (3,) or (3, 1) vector in camera coordinate frame (meters).
            convention: Optional convention override.

        Returns:
            (3,) vector in rack coordinate frame (meters).
        """
        if not self.is_calibrated:
            raise RuntimeError("RackRelativePose is not calibrated. Load or set calibration first.")

        p_c = np.asarray(point_cam, dtype=np.float64).reshape(3, 1)
        p_rack = self.R_cam_to_rack @ p_c + self.t_cam_to_rack

        # Apply convention mapping
        conv_matrix = self.get_convention_matrix(convention)
        p_out = conv_matrix @ p_rack
        return p_out.flatten()

    def unproject_pixel_to_camera(
        self,
        u: float,
        v: float,
        depth_m: float
    ) -> np.ndarray:
        """
        Unproject a 2D image pixel (u, v) at a given depth Z into 3D Camera coordinates.
        Uses camera intrinsic matrix K.
        """
        if self.camera_matrix is None:
            fx, fy = 600.0, 600.0
            cx, cy = 320.0, 240.0
        else:
            fx = float(self.camera_matrix[0, 0])
            fy = float(self.camera_matrix[1, 1])
            cx = float(self.camera_matrix[0, 2])
            cy = float(self.camera_matrix[1, 2])

        x_cam = ((u - cx) * depth_m) / fx
        y_cam = ((v - cy) * depth_m) / fy
        z_cam = depth_m
        return np.array([x_cam, y_cam, z_cam], dtype=np.float64)

    def unproject_pixel_to_rack_plane(
        self,
        u: float,
        v: float,
        rack_plane_z: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Cast a ray from camera through pixel (u, v) and find its 3D intersection with
        the plane Z_rack = rack_plane_z.
        """
        if not self.is_calibrated or self.camera_matrix is None:
            return None

        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        # Ray direction in camera frame
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64).reshape(3, 1)

        # In rack frame: P_rack = lambda * R_cam_to_rack @ d_cam + t_cam_to_rack
        rd = self.R_cam_to_rack @ d_cam
        rd_z = float(rd[2, 0])
        tz = float(self.t_cam_to_rack[2, 0])

        if abs(rd_z) < 1e-6:
            return None

        lam = (rack_plane_z - tz) / rd_z
        if lam <= 0:
            return None

        p_cam = lam * d_cam
        p_rack = self.R_cam_to_rack @ p_cam + self.t_cam_to_rack
        conv_matrix = self.get_convention_matrix()
        return (conv_matrix @ p_rack).flatten()

    def transform_hand_to_rack(
        self,
        hand_input: Any,
        camera_offset: Optional[np.ndarray] = None,
        convention: Optional[CoordinateConvention] = None
    ) -> Optional[np.ndarray]:
        """
        Transform a detected hand coordinate into the rack coordinate system.
        Supports:
        - (3,) array/tuple in 3D camera coordinates.
        - MediaPipe landmark object with .x, .y, .z.
        """
        if not self.is_calibrated:
            return None

        if hasattr(hand_input, "x") and hasattr(hand_input, "y") and hasattr(hand_input, "z"):
            pt_cam = np.array([hand_input.x, hand_input.y, hand_input.z], dtype=np.float64)
            if camera_offset is not None:
                pt_cam = pt_cam + camera_offset
            return self.transform_camera_to_rack(pt_cam, convention=convention)

        pt_arr = np.asarray(hand_input, dtype=np.float64).flatten()
        if pt_arr.shape[0] >= 3:
            return self.transform_camera_to_rack(pt_arr[:3], convention=convention)

        return None

    def transform_object_to_rack(
        self,
        object_input: Any,
        depth_m: Optional[float] = None,
        reference_hand_cam: Optional[np.ndarray] = None,
        rack_plane_z: float = 0.0,
        convention: Optional[CoordinateConvention] = None
    ) -> Optional[np.ndarray]:
        """
        Transform a tracked object into the rack coordinate system.
        Supports:
        - 3D camera point: (3,) array/list [X_c, Y_c, Z_c].
        - 2D pixel center: (u, v).
        - 2D bounding box: (x1, y1, x2, y2).
        
        If 2D, depth is resolved via:
        1. Explicit depth_m if provided.
        2. Reference hand camera depth if provided.
        3. Ray intersection with rack plane (Z_rack = rack_plane_z).
        4. Nominal distance from rack translation vector.
        """
        if not self.is_calibrated:
            return None

        # Check if 3D camera coordinates were directly passed
        obj_arr = np.asarray(object_input, dtype=np.float64).flatten()
        if obj_arr.shape[0] == 3 and depth_m is None and reference_hand_cam is None:
            return self.transform_camera_to_rack(obj_arr, convention=convention)

        # Otherwise treat as 2D pixel center or bounding box
        if obj_arr.shape[0] == 4:
            # Bounding box: [x1, y1, x2, y2]
            u = (obj_arr[0] + obj_arr[2]) / 2.0
            v = (obj_arr[1] + obj_arr[3]) / 2.0
        elif obj_arr.shape[0] >= 2:
            # Pixel center: [u, v]
            u = obj_arr[0]
            v = obj_arr[1]
        else:
            return None

        # Determine 3D camera point
        if depth_m is not None:
            p_cam = self.unproject_pixel_to_camera(u, v, depth_m)
            return self.transform_camera_to_rack(p_cam, convention=convention)

        if reference_hand_cam is not None:
            ref_z = float(np.asarray(reference_hand_cam, dtype=np.float64).flatten()[2])
            p_cam = self.unproject_pixel_to_camera(u, v, ref_z)
            return self.transform_camera_to_rack(p_cam, convention=convention)

        # Attempt plane intersection with rack surface
        p_plane = self.unproject_pixel_to_rack_plane(u, v, rack_plane_z=rack_plane_z)
        if p_plane is not None:
            if convention is not None:
                return self.get_convention_matrix(convention) @ p_plane
            return p_plane

        # Fallback to calibrated rack distance
        nominal_z = float(self.tvec_rack[2, 0]) if self.tvec_rack is not None else 0.5
        p_cam = self.unproject_pixel_to_camera(u, v, nominal_z)
        return self.transform_camera_to_rack(p_cam, convention=convention)

    def transform_rack_to_camera(
        self,
        point_rack: Union[np.ndarray, List[float], Tuple[float, float, float]],
        from_convention: Optional[CoordinateConvention] = None
    ) -> np.ndarray:
        """
        Transform a single 3D point from Rack coordinates back into Camera coordinates.
        Inverse of transform_camera_to_rack.
        """
        if not self.is_calibrated:
            raise RuntimeError("RackRelativePose is not calibrated.")

        p_in = np.asarray(point_rack, dtype=np.float64).reshape(3, 1)
        conv_matrix = self.get_convention_matrix(from_convention)
        # Invert convention: P_rack_local = conv_matrix.T @ p_in
        p_rack_local = conv_matrix.T @ p_in

        p_cam = self.R_rack_to_cam @ p_rack_local + self.tvec_rack
        return p_cam.flatten()

    def transform_points_camera_to_rack(
        self,
        points_cam: np.ndarray,
        convention: Optional[CoordinateConvention] = None
    ) -> np.ndarray:
        """
        Batch transform an array of (N, 3) points from Camera to Rack coordinates.
        """
        if not self.is_calibrated:
            raise RuntimeError("RackRelativePose is not calibrated.")

        pts = np.asarray(points_cam, dtype=np.float64)
        if pts.ndim == 1 and pts.shape[0] == 3:
            return self.transform_camera_to_rack(pts, convention=convention)

        # pts is (N, 3): (pts - t_cam.T) @ R_rack
        p_rack = (pts - self.tvec_rack.reshape(1, 3)) @ self.R_rack_to_cam
        conv_matrix = self.get_convention_matrix(convention)
        return (p_rack @ conv_matrix.T)

    # =========================================================================
    # ORIENTATION & ANGLE CONVERSIONS
    # =========================================================================

    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract Yaw, Pitch, Roll (ZYX convention) in degrees from a 3x3 rotation matrix.
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0

        return (
            float(np.degrees(yaw)),
            float(np.degrees(pitch)),
            float(np.degrees(roll))
        )

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Convert 3x3 rotation matrix to quaternion (w, x, y, z).
        """
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        # Normalize quaternion
        norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm > 1e-9:
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        return (float(qw), float(qx), float(qy), float(qz))

    # =========================================================================
    # HUMAN POSE PROCESSING & BODY ORIENTATION
    # =========================================================================

    def compute_body_frame(
        self,
        positions_rack: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, Any]]:
        """
        Construct an orthonormal body coordinate frame from torso landmarks:
        - Lateral vector (X_body): Right shoulder -> Left shoulder
        - Vertical vector (Y_body): Pelvis/Hips -> Shoulders midpoint
        - Normal/Facing vector (Z_body): X_body x Y_body (pointing outward from chest)
        """
        required = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        if not all(k in positions_rack for k in required):
            return None

        l_sh = positions_rack["LEFT_SHOULDER"]
        r_sh = positions_rack["RIGHT_SHOULDER"]
        l_hip = positions_rack["LEFT_HIP"]
        r_hip = positions_rack["RIGHT_HIP"]

        # Lateral axis: Left shoulder to Right shoulder (Person's Rightward direction)
        v_right = r_sh - l_sh
        norm_lat = np.linalg.norm(v_right)
        if norm_lat < 1e-4:
            return None
        v_right = v_right / norm_lat

        # Vertical axis: Mid-hip to Mid-shoulder (Person's Upward direction)
        mid_sh = (l_sh + r_sh) / 2.0
        mid_hip = (l_hip + r_hip) / 2.0
        v_up = mid_sh - mid_hip
        norm_up = np.linalg.norm(v_up)
        if norm_up < 1e-4:
            return None
        v_up = v_up / norm_up

        # Normal axis: Forward from chest (Right x Up = Front in right-handed convention)
        v_front = np.cross(v_right, v_up)
        norm_front = np.linalg.norm(v_front)
        if norm_front < 1e-4:
            return None
        v_front = v_front / norm_front

        # Re-orthogonalize: v_up = v_front x v_right
        v_up = np.cross(v_front, v_right)

        # Assemble rotation matrix [v_right, v_up, v_front] with det = +1
        R_body = np.column_stack([v_right, v_up, v_front])

        # Euler angles & Quaternion
        yaw, pitch, roll = self.rotation_matrix_to_euler(R_body)
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R_body)

        # Facing score relative to rack surface (rack normal in RACK_LOCAL is +Z: [0, 0, 1])
        # A person facing the rack directly has chest normal pointing in -Z direction:
        rack_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        conv_matrix = self.get_convention_matrix()
        rack_normal_conv = conv_matrix @ rack_normal
        facing_dot = -float(np.dot(v_front, rack_normal_conv))

        return {
            "rotation_matrix": R_body,
            "quaternion": {"w": qw, "x": qx, "y": qy, "z": qz},
            "euler_angles_deg": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "facing_vector": v_front.tolist(),
            "facing_rack_score": facing_dot,  # ~1.0 when facing rack, ~-1.0 when facing away
            "torso_center": mid_sh.tolist()
        }

    def process_landmarks(
        self,
        landmarks: Any,
        camera_offset: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process MediaPipe pose landmarks (or 3D joint dictionary) into rack coordinates.

        Args:
            landmarks: Either:
                - List of MediaPipe landmarks with .x, .y, .z attributes
                - Dict mapping joint name (e.g. 'LEFT_WRIST') to (x, y, z)
                - Array of joint points (N, 3)
            camera_offset: Optional 3D offset (meters) from human reference point to camera.
                           Defaults to nominal offset if raw MediaPipe hip-centered coords are passed.

        Returns:
            Dictionary with rack-relative joint positions, body orientation, and metrics.
        """
        if not self.is_calibrated:
            return None

        # Extract joints in camera coordinates
        raw_joints_cam: Dict[str, np.ndarray] = {}

        if isinstance(landmarks, dict):
            for name, pt in landmarks.items():
                raw_joints_cam[name] = np.asarray(pt, dtype=np.float64).flatten()[:3]
        elif hasattr(landmarks, "__len__") and len(landmarks) > 0:
            first = landmarks[0]
            if hasattr(first, "x") and hasattr(first, "y") and hasattr(first, "z"):
                for name, idx in self.LANDMARK_MAP.items():
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        pt = np.array([lm.x, lm.y, lm.z], dtype=np.float64)
                        if camera_offset is not None:
                            pt = pt + camera_offset
                        raw_joints_cam[name] = pt
            elif isinstance(first, (list, tuple, np.ndarray)):
                for name, idx in self.LANDMARK_MAP.items():
                    if idx < len(landmarks):
                        raw_joints_cam[name] = np.asarray(landmarks[idx], dtype=np.float64).flatten()[:3]

        if not raw_joints_cam:
            return None

        # Transform all joints to rack coordinates
        rack_positions: Dict[str, np.ndarray] = {}
        rack_positions_serializable: Dict[str, List[float]] = {}

        for name, pt_cam in raw_joints_cam.items():
            p_rack = self.transform_camera_to_rack(pt_cam)
            rack_positions[name] = p_rack
            rack_positions_serializable[name] = [
                round(float(p_rack[0]), 4),
                round(float(p_rack[1]), 4),
                round(float(p_rack[2]), 4)
            ]

        # Calculate torso/body orientation
        body_frame = self.compute_body_frame(rack_positions)

        # Compute reach and distance metrics
        metrics: Dict[str, Any] = {}
        for hand in ["LEFT_WRIST", "RIGHT_WRIST"]:
            if hand in rack_positions:
                p = rack_positions[hand]
                metrics[f"{hand.lower()}_distance_to_rack"] = round(float(p[2]), 4)
                metrics[f"{hand.lower()}_reach_dist"] = round(float(np.linalg.norm(p)), 4)

        if "NOSE" in rack_positions:
            p_nose = rack_positions["NOSE"]
            metrics["head_distance_to_rack"] = round(float(p_nose[2]), 4)

        return {
            "convention": self.convention.value,
            "calibrated": True,
            "positions": rack_positions_serializable,
            "body_orientation": body_frame,
            "metrics": metrics
        }

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def draw_axes_on_frame(
        self,
        frame: np.ndarray,
        axis_length: float = 0.08,
        draw_camera_axes: bool = True,
        draw_rack_axes: bool = True
    ) -> np.ndarray:
        """
        Render 3D coordinate frame axes for both Camera and Rack on the visual frame.
        - Red: X axis
        - Green: Y axis
        - Blue: Z axis
        """
        if not self.is_calibrated or self.camera_matrix is None:
            return frame

        vis_frame = frame

        # 1. Draw Rack Frame Axes at Rack Origin (using calibrated rvec/tvec)
        if draw_rack_axes and self.rvec_rack is not None and self.tvec_rack is not None:
            cv2.drawFrameAxes(
                vis_frame,
                self.camera_matrix,
                self.distortion,
                self.rvec_rack,
                self.tvec_rack,
                axis_length,
                thickness=3
            )

            # Project rack origin to add text label
            proj_origin, _ = cv2.projectPoints(
                np.zeros((1, 3), dtype=np.float64),
                self.rvec_rack,
                self.tvec_rack,
                self.camera_matrix,
                self.distortion
            )
            ox, oy = proj_origin.ravel().astype(int)
            h, w = vis_frame.shape[:2]
            if 0 <= ox < w and 0 <= oy < h:
                cv2.putText(
                    vis_frame,
                    "RACK (0,0,0)",
                    (ox + 10, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

        # 2. Draw Camera Frame Axes (Corner overlay at top-right of image)
        if draw_camera_axes:
            # Render a small camera coordinate icon in top-left or top-right
            corner_x, corner_y = 70, 70
            cv2.circle(vis_frame, (corner_x, corner_y), 5, (255, 255, 255), -1)
            cv2.line(vis_frame, (corner_x, corner_y), (corner_x + 40, corner_y), (0, 0, 255), 2)  # X: Red
            cv2.line(vis_frame, (corner_x, corner_y), (corner_x, corner_y + 40), (0, 255, 0), 2)  # Y: Green
            cv2.putText(vis_frame, "CAM X", (corner_x + 45, corner_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(vis_frame, "CAM Y", (corner_x - 15, corner_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(vis_frame, "CAM Z (INTO SCREEN)", (corner_x - 30, corner_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        return vis_frame

    def draw_rack_relative_pose(
        self,
        frame: np.ndarray,
        pose_data: Optional[Dict[str, Any]],
        y_offset: int = 225
    ) -> np.ndarray:
        """
        Overlay rack-relative 3D pose info (positions, facing status) onto the frame.
        """
        if pose_data is None:
            return frame

        vis_frame = self.draw_axes_on_frame(frame)

        positions = pose_data.get("positions", {})
        body = pose_data.get("body_orientation")

        cv2.putText(
            vis_frame,
            f"RACK-REL POSE ({pose_data.get('convention', 'RACK_LOCAL')}):",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )
        y_offset += 24

        if "LEFT_WRIST" in positions:
            lw = positions["LEFT_WRIST"]
            cv2.putText(
                vis_frame,
                f"L Wrist: X={lw[0]:+.2f} Y={lw[1]:+.2f} Z={lw[2]:+.2f}m",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )
            y_offset += 20

        if "RIGHT_WRIST" in positions:
            rw = positions["RIGHT_WRIST"]
            cv2.putText(
                vis_frame,
                f"R Wrist: X={rw[0]:+.2f} Y={rw[1]:+.2f} Z={rw[2]:+.2f}m",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )
            y_offset += 20

        if body:
            facing = body.get("facing_rack_score", 0.0)
            facing_str = "FACING RACK" if facing > 0.3 else ("FACING AWAY" if facing < -0.3 else "SIDEWAYS")
            color = (0, 255, 0) if facing > 0.3 else (0, 165, 255)
            cv2.putText(
                vis_frame,
                f"Orientation: {facing_str} (score: {facing:+.2f})",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2
            )

        return vis_frame
