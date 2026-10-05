"""
DONUTS - Rack Relative Features Extraction Module
=================================================
Calculates 3D kinematic and spatial features in the physical rack coordinate system:
- 3D Hand-Object Euclidean distance and directional vector.
- 3D Object displacement (frame-to-frame, cumulative, and pickup reference).
- 3D Hand velocities and scalar speeds.
- 3D Object velocity and scalar speed.
- Body orientation relative to rack axes (Euler angles, facing score, facing vector).

Designed with a clean, extensible interface for downstream learned Human Activity
Recognition (HAR) architectures (e.g. LSTM, GRU, Transformers, XGBoost).
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class RackRelativeFeatures:
    """
    Computes and temporal-filters 3D spatial and kinematic features relative to the rack.
    Maintains temporal state for numerical differentiation (velocities and displacements).
    """

    FEATURE_NAMES: List[str] = [
        "min_hand_object_distance",
        "left_hand_object_distance",
        "right_hand_object_distance",
        "closest_hand_rel_x",
        "closest_hand_rel_y",
        "closest_hand_rel_z",
        "object_speed",
        "object_vel_x",
        "object_vel_y",
        "object_vel_z",
        "object_displacement_pickup_mag",
        "object_displacement_pickup_x",
        "object_displacement_pickup_y",
        "object_displacement_pickup_z",
        "object_cumulative_distance",
        "closest_hand_speed",
        "closest_hand_vel_x",
        "closest_hand_vel_y",
        "closest_hand_vel_z",
        "left_hand_speed",
        "right_hand_speed",
        "torso_yaw_deg",
        "torso_pitch_deg",
        "torso_roll_deg",
        "facing_rack_score",
        "facing_vector_x",
        "facing_vector_y",
        "facing_vector_z",
        "astronaut_distance_to_rack",
        "object_pos_x",
        "object_pos_y",
        "object_pos_z",
    ]

    def __init__(self, velocity_smoothing_alpha: float = 0.70):
        """
        Args:
            velocity_smoothing_alpha: Exponential moving average coefficient (0 to 1).
                                       Higher value gives more weight to recent measurements.
        """
        self.alpha = float(np.clip(velocity_smoothing_alpha, 0.0, 1.0))

        # Temporal state tracking
        self.last_timestamp: Optional[float] = None

        self.last_lh_pos: Optional[np.ndarray] = None
        self.last_rh_pos: Optional[np.ndarray] = None
        self.last_obj_pos: Optional[np.ndarray] = None

        self.initial_obj_pos: Optional[np.ndarray] = None
        self.pickup_obj_pos: Optional[np.ndarray] = None
        self.cumulative_obj_dist: float = 0.0

        # Smoothed 3D velocities
        self.lh_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self.rh_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self.obj_vel: np.ndarray = np.zeros(3, dtype=np.float64)

        # Current computed frame features
        self.current_features: Dict[str, Any] = {}

    def reset(self) -> None:
        """
        Reset temporal state for a new experiment cycle.
        """
        self.last_timestamp = None
        self.last_lh_pos = None
        self.last_rh_pos = None
        self.last_obj_pos = None
        self.initial_obj_pos = None
        self.pickup_obj_pos = None
        self.cumulative_obj_dist = 0.0
        self.lh_vel = np.zeros(3, dtype=np.float64)
        self.rh_vel = np.zeros(3, dtype=np.float64)
        self.obj_vel = np.zeros(3, dtype=np.float64)
        self.current_features.clear()

    def set_pickup_reference(self, pos: Optional[np.ndarray] = None) -> None:
        """
        Record the object's 3D position when a PICK event is confirmed.
        Subsequent displacement metrics are measured relative to this reference.
        """
        if pos is not None:
            self.pickup_obj_pos = np.asarray(pos, dtype=np.float64).flatten()[:3]
        elif self.last_obj_pos is not None:
            self.pickup_obj_pos = self.last_obj_pos.copy()

    def update(
        self,
        left_hand_rack: Optional[Union[np.ndarray, List[float]]] = None,
        right_hand_rack: Optional[Union[np.ndarray, List[float]]] = None,
        object_rack: Optional[Union[np.ndarray, List[float]]] = None,
        body_orientation: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute real-time 3D spatial, kinematic, and orientation features.

        Args:
            left_hand_rack: (3,) 3D coordinates of left hand/wrist in rack frame (meters).
            right_hand_rack: (3,) 3D coordinates of right hand/wrist in rack frame (meters).
            object_rack: (3,) 3D coordinates of tracked object in rack frame (meters).
            body_orientation: Dictionary output from RackRelativePose.compute_body_frame().
            timestamp: Timestamp in seconds (defaults to time.perf_counter()).

        Returns:
            Dictionary containing structured features.
        """
        current_time = timestamp if timestamp is not None else time.perf_counter()
        dt = (
            (current_time - self.last_timestamp)
            if self.last_timestamp is not None
            else 0.0
        )
        self.last_timestamp = current_time

        # Sanitize inputs to (3,) numpy float64 arrays
        lh_p = (
            np.asarray(left_hand_rack, dtype=np.float64).flatten()[:3]
            if left_hand_rack is not None
            else None
        )
        rh_p = (
            np.asarray(right_hand_rack, dtype=np.float64).flatten()[:3]
            if right_hand_rack is not None
            else None
        )
        obj_p = (
            np.asarray(object_rack, dtype=np.float64).flatten()[:3]
            if object_rack is not None
            else None
        )

        # -------------------------------------------------------------
        # 1. HAND VELOCITIES
        # -------------------------------------------------------------
        if lh_p is not None and self.last_lh_pos is not None and dt > 1e-4:
            raw_lh_vel = (lh_p - self.last_lh_pos) / dt
            self.lh_vel = self.alpha * raw_lh_vel + (1.0 - self.alpha) * self.lh_vel
        elif lh_p is None:
            self.lh_vel = np.zeros(3, dtype=np.float64)

        if rh_p is not None and self.last_rh_pos is not None and dt > 1e-4:
            raw_rh_vel = (rh_p - self.last_rh_pos) / dt
            self.rh_vel = self.alpha * raw_rh_vel + (1.0 - self.alpha) * self.rh_vel
        elif rh_p is None:
            self.rh_vel = np.zeros(3, dtype=np.float64)

        lh_speed = float(np.linalg.norm(self.lh_vel))
        rh_speed = float(np.linalg.norm(self.rh_vel))

        if lh_p is not None:
            self.last_lh_pos = lh_p.copy()
        if rh_p is not None:
            self.last_rh_pos = rh_p.copy()

        # -------------------------------------------------------------
        # 2. OBJECT VELOCITY & DISPLACEMENT
        # -------------------------------------------------------------
        obj_disp_frame = np.zeros(3, dtype=np.float64)
        obj_disp_frame_mag = 0.0
        obj_disp_pickup = np.zeros(3, dtype=np.float64)
        obj_disp_pickup_mag = 0.0

        if obj_p is not None:
            if self.initial_obj_pos is None:
                self.initial_obj_pos = obj_p.copy()

            if self.last_obj_pos is not None:
                obj_disp_frame = obj_p - self.last_obj_pos
                obj_disp_frame_mag = float(np.linalg.norm(obj_disp_frame))
                self.cumulative_obj_dist += obj_disp_frame_mag

                if dt > 1e-4:
                    raw_obj_vel = obj_disp_frame / dt
                    self.obj_vel = (
                        self.alpha * raw_obj_vel + (1.0 - self.alpha) * self.obj_vel
                    )

            if self.pickup_obj_pos is not None:
                obj_disp_pickup = obj_p - self.pickup_obj_pos
                obj_disp_pickup_mag = float(np.linalg.norm(obj_disp_pickup))

            self.last_obj_pos = obj_p.copy()
        else:
            self.obj_vel = np.zeros(3, dtype=np.float64)

        obj_speed = float(np.linalg.norm(self.obj_vel))

        # -------------------------------------------------------------
        # 3. HAND-OBJECT DISTANCES
        # -------------------------------------------------------------
        lh_dist = 999.0
        rh_dist = 999.0
        closest_hand = "NONE"
        closest_hand_dist = 999.0
        closest_hand_vec = np.zeros(3, dtype=np.float64)
        closest_hand_vel = np.zeros(3, dtype=np.float64)
        closest_hand_speed = 0.0

        if obj_p is not None:
            if lh_p is not None:
                lh_dist = float(np.linalg.norm(lh_p - obj_p))
            if rh_p is not None:
                rh_dist = float(np.linalg.norm(rh_p - obj_p))

            if lh_dist < rh_dist and lh_dist < 999.0:
                closest_hand = "LEFT"
                closest_hand_dist = lh_dist
                closest_hand_vec = obj_p - lh_p
                closest_hand_vel = self.lh_vel.copy()
                closest_hand_speed = lh_speed
            elif rh_dist <= lh_dist and rh_dist < 999.0:
                closest_hand = "RIGHT"
                closest_hand_dist = rh_dist
                closest_hand_vec = obj_p - rh_p
                closest_hand_vel = self.rh_vel.copy()
                closest_hand_speed = rh_speed

        # -------------------------------------------------------------
        # 4. BODY ORIENTATION RELATIVE TO RACK AXES
        # -------------------------------------------------------------
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        facing_score = 0.0
        facing_vec = [0.0, 0.0, 0.0]
        torso_center = [0.0, 0.0, 0.0]
        astro_rack_dist = 0.0

        if body_orientation is not None:
            euler = body_orientation.get("euler_angles_deg", {})
            yaw = float(euler.get("yaw", 0.0))
            pitch = float(euler.get("pitch", 0.0))
            roll = float(euler.get("roll", 0.0))
            facing_score = float(body_orientation.get("facing_rack_score", 0.0))
            facing_vec = list(body_orientation.get("facing_vector", [0.0, 0.0, 0.0]))
            torso_center = list(body_orientation.get("torso_center", [0.0, 0.0, 0.0]))
            if len(torso_center) >= 3:
                astro_rack_dist = float(torso_center[2])

        # -------------------------------------------------------------
        # ASSEMBLE OUTPUT DICTIONARY
        # -------------------------------------------------------------
        features = {
            "hand_object_distance": {
                "min_distance_m": round(closest_hand_dist, 4) if closest_hand_dist < 990 else None,
                "left_hand_dist_m": round(lh_dist, 4) if lh_dist < 990 else None,
                "right_hand_dist_m": round(rh_dist, 4) if rh_dist < 990 else None,
                "closest_hand": closest_hand,
                "relative_vector_rack": [round(float(x), 4) for x in closest_hand_vec],
            },
            "object_displacement": {
                "frame_displacement_m": [round(float(x), 4) for x in obj_disp_frame],
                "frame_displacement_mag_m": round(obj_disp_frame_mag, 4),
                "pickup_displacement_m": [round(float(x), 4) for x in obj_disp_pickup],
                "pickup_displacement_mag_m": round(obj_disp_pickup_mag, 4),
                "cumulative_distance_m": round(self.cumulative_obj_dist, 4),
                "current_position_rack": [round(float(x), 4) for x in obj_p] if obj_p is not None else None,
            },
            "hand_velocity": {
                "left_hand_velocity_mps": [round(float(x), 4) for x in self.lh_vel],
                "left_hand_speed_mps": round(lh_speed, 4),
                "right_hand_velocity_mps": [round(float(x), 4) for x in self.rh_vel],
                "right_hand_speed_mps": round(rh_speed, 4),
                "closest_hand_velocity_mps": [round(float(x), 4) for x in closest_hand_vel],
                "closest_hand_speed_mps": round(closest_hand_speed, 4),
            },
            "object_velocity": {
                "velocity_mps": [round(float(x), 4) for x in self.obj_vel],
                "speed_mps": round(obj_speed, 4),
            },
            "body_orientation": {
                "torso_yaw_deg": round(yaw, 2),
                "torso_pitch_deg": round(pitch, 2),
                "torso_roll_deg": round(roll, 2),
                "facing_rack_score": round(facing_score, 4),
                "facing_vector_rack": [round(float(x), 4) for x in facing_vec],
                "torso_center_rack": [round(float(x), 4) for x in torso_center],
                "astronaut_distance_to_rack_m": round(astro_rack_dist, 4),
            }
        }

        self.current_features = features
        return features

    # =========================================================================
    # ML / LEARNED HAR EXPORT INTERFACE
    # =========================================================================

    def to_feature_vector(self) -> np.ndarray:
        """
        Export a fixed-length 1D numerical NumPy vector (float32) for downstream
        machine learning / deep learning HAR classifiers (e.g. LSTM, SVM, XGBoost).
        Dimensions: (32,)
        """
        cf = self.current_features
        if not cf:
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)

        hod = cf.get("hand_object_distance", {})
        od = cf.get("object_displacement", {})
        hv = cf.get("hand_velocity", {})
        ov = cf.get("object_velocity", {})
        bo = cf.get("body_orientation", {})

        min_dist = hod.get("min_distance_m") or 0.0
        lh_dist = hod.get("left_hand_dist_m") or 0.0
        rh_dist = hod.get("right_hand_dist_m") or 0.0
        ch_vec = hod.get("relative_vector_rack", [0.0, 0.0, 0.0])

        obj_speed = ov.get("speed_mps", 0.0)
        obj_vel = ov.get("velocity_mps", [0.0, 0.0, 0.0])

        disp_pickup_mag = od.get("pickup_displacement_mag_m", 0.0)
        disp_pickup = od.get("pickup_displacement_m", [0.0, 0.0, 0.0])
        cum_dist = od.get("cumulative_distance_m", 0.0)
        obj_pos = od.get("current_position_rack") or [0.0, 0.0, 0.0]

        ch_speed = hv.get("closest_hand_speed_mps", 0.0)
        ch_vel = hv.get("closest_hand_velocity_mps", [0.0, 0.0, 0.0])
        lh_speed = hv.get("left_hand_speed_mps", 0.0)
        rh_speed = hv.get("right_hand_speed_mps", 0.0)

        yaw = bo.get("torso_yaw_deg", 0.0)
        pitch = bo.get("torso_pitch_deg", 0.0)
        roll = bo.get("torso_roll_deg", 0.0)
        facing = bo.get("facing_rack_score", 0.0)
        facing_v = bo.get("facing_vector_rack", [0.0, 0.0, 0.0])
        dist_rack = bo.get("astronaut_distance_to_rack_m", 0.0)

        vec = [
            float(min_dist),
            float(lh_dist),
            float(rh_dist),
            float(ch_vec[0]), float(ch_vec[1]), float(ch_vec[2]),
            float(obj_speed),
            float(obj_vel[0]), float(obj_vel[1]), float(obj_vel[2]),
            float(disp_pickup_mag),
            float(disp_pickup[0]), float(disp_pickup[1]), float(disp_pickup[2]),
            float(cum_dist),
            float(ch_speed),
            float(ch_vel[0]), float(ch_vel[1]), float(ch_vel[2]),
            float(lh_speed),
            float(rh_speed),
            float(yaw),
            float(pitch),
            float(roll),
            float(facing),
            float(facing_v[0]), float(facing_v[1]), float(facing_v[2]),
            float(dist_rack),
            float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]),
        ]

        return np.asarray(vec, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the current features as a dictionary.
        """
        return self.current_features

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        Return the list of feature column names matching to_feature_vector().
        """
        return list(cls.FEATURE_NAMES)
