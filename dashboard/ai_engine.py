import os

import sys

import cv2

import math

import time

IS_CLOUD = os.environ.get("RENDER") == "true"

from ultralytics import YOLO

import mediapipe as mp

# ============================================================

# PROJECT PATH

# ============================================================

PROJECT_ROOT = os.path.dirname(

    os.path.dirname(

        os.path.abspath(__file__)

    )

)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(0, PROJECT_ROOT)

HMR_PATH = os.path.join(

    PROJECT_ROOT,

    "hmr"

)

if HMR_PATH not in sys.path:

    sys.path.insert(0, HMR_PATH)

APP_PATH = os.path.join(PROJECT_ROOT, "app")

if APP_PATH not in sys.path:

    sys.path.insert(0, APP_PATH)

from activity_recognition import ActivityRecognizer

from experiment_controller import ExperimentController

from event_logger import EventLogger

from rack_relative_pose import RackRelativePose

from rack_relative_features import RackRelativeFeatures

# Import AlertManager (graceful fallback if app/ not set up yet)

try:

    from alerts.alert_manager import AlertManager

    _ALERT_MANAGER_AVAILABLE = True

except ImportError:

    _ALERT_MANAGER_AVAILABLE = False

# ============================================================

# DONUTS AI ENGINE

# ============================================================

class DonutsAI:

    def __init__(self):

        print()

        print("==========================================")

        print("DONUTS AI ENGINE")

        print("==========================================")

        # ----------------------------------------------------

        # YOLO

        # ----------------------------------------------------

        model_path = os.path.join(

            PROJECT_ROOT,

            "yolov8n.pt"

        )

        self.model = YOLO(model_path)

        print(">>> YOLO object model loaded.")

        # ----------------------------------------------------

        # MEDIAPIPE

        # ----------------------------------------------------

        # ----------------------------------------------------

        # POSE ENGINE

        # ----------------------------------------------------

        self.landmarker = None

        self.cloud_pose = None

        if IS_CLOUD:

            print(">>> Cloud mode: using YOLO pose model.")

            pose_model_path = os.path.join(

                PROJECT_ROOT,

                "yolov8n-pose.pt"

            )

            self.cloud_pose = YOLO(pose_model_path)

            print(">>> YOLO pose model loaded.")

        else:

            pose_model_path = os.path.join(

                PROJECT_ROOT,

                "models",

                "mediapipe",

                "pose_landmarker_lite.task"

            )

            BaseOptions = mp.tasks.BaseOptions

            PoseLandmarker = mp.tasks.vision.PoseLandmarker

            PoseLandmarkerOptions = (

                mp.tasks.vision.PoseLandmarkerOptions

            )

            VisionRunningMode = (

                mp.tasks.vision.RunningMode

            )

            options = PoseLandmarkerOptions(

                base_options=BaseOptions(

                    model_asset_path=pose_model_path

                ),

                running_mode=VisionRunningMode.IMAGE

            )

            self.landmarker = (

                PoseLandmarker.create_from_options(options)

            )

            print(">>> MediaPipe pose model loaded.")

        # ----------------------------------------------------

        # ACTIVITY RECOGNITION

        # ----------------------------------------------------

        self.activity_recognizer = (

            ActivityRecognizer()

        )

        self.state = "IDLE"

        self.activity_label = (

            self.activity_recognizer.set_state(

                self.state

            )

        )

        self.activity_status = (

            self.activity_recognizer.status

        )

        # ----------------------------------------------------

        # EXPERIMENT CONTROLLER

        # ----------------------------------------------------

        self.experiment_controller = (

            ExperimentController()

        )

        self.controller_status = "READY"

        self.controller_message = (

            "WAITING FOR EXPERIMENT"

        )

        # ----------------------------------------------------

        # EVENT LOGGER

        # ----------------------------------------------------

        self.event_logger = EventLogger(

            self.experiment_controller.experiment_name

        )

        # ----------------------------------------------------

        # ALERT MANAGER (non-blocking voice alerts)

        # ----------------------------------------------------

        if _ALERT_MANAGER_AVAILABLE:

            self.alert_manager = AlertManager(

                voice_enabled=True,

                cooldown_seconds=5.0,

                voice_rate=160,

                voice_volume=1.0,

            )

            # Inject into experiment controller so it uses async alerts

            self.experiment_controller.alert_manager = self.alert_manager

            print(">>> AlertManager initialized (async voice).")

        else:

            self.alert_manager = None

            print(">>> AlertManager not available — using legacy voice_alert.")

        # ----------------------------------------------------

        # FEATURE FLAG: RACK RELATIVE POSE

        # ----------------------------------------------------

        self.enable_rack_relative_pose = (

            os.environ.get("DONUTS_ENABLE_RACK_RELATIVE_POSE", "").lower() in ("1", "true", "yes")

            or self.experiment_controller.config.get("enable_rack_relative_pose", False)

        )

        self.rack_relative_pose = None

        if self.enable_rack_relative_pose:

            try:

                self.rack_relative_pose = RackRelativePose(visualize=True)

                print(">>> [FEATURE FLAG ON] RackRelativePose enabled.")

            except Exception as e:

                print(f">>> [WARNING] Failed to initialize RackRelativePose: {e}")

        else:

            print(">>> [FEATURE FLAG OFF] RackRelativePose disabled (MVP mode).")

        # ----------------------------------------------------

        # STABILITY SETTINGS

        # ----------------------------------------------------

        self.CONFIDENCE_THRESHOLD = 0.35

        self.DETECTION_CONFIRM_FRAMES = 3

        self.MAX_MISSED_FRAMES = 8

        self.SMOOTHING = 0.60

        # ----------------------------------------------------

        # INTERACTION THRESHOLDS

        # ----------------------------------------------------

        self.APPROACH_DISTANCE = 0.20

        self.PICKUP_DISTANCE_PIXELS = 120

        self.RELEASE_DISTANCE = 0.18

        self.MOVEMENT_THRESHOLD = 0.035

        self.PICKUP_CONFIRM_FRAMES = 3

        self.RELEASE_CONFIRM_FRAMES = 3

        # ----------------------------------------------------

        # TRACKING STATE

        # ----------------------------------------------------

        self.detection_count = 0

        self.missed_frames = 0

        self.pickup_frames = 0

        self.release_frames = 0

        self.stable_box = None

        self.smoothed_center = None

        self.pickup_center = None

        # ----------------------------------------------------

        # PERFORMANCE

        # ----------------------------------------------------

        self.fps = 0.0

        self.fps_counter = 0

        self.fps_start = time.perf_counter()

        self.yolo_ms = 0.0

        self.pose_ms = 0.0

        print(">>> DONUTS AI ENGINE READY")

        print("==========================================")

        print()

    # ========================================================

    # HELPER

    # ========================================================

    @staticmethod

    def point_to_box_distance(

        point,

        box,

        width,

        height

    ):

        px = point[0] * width

        py = point[1] * height

        x1, y1, x2, y2 = box

        closest_x = max(

            x1,

            min(px, x2)

        )

        closest_y = max(

            y1,

            min(py, y2)

        )

        dx = px - closest_x

        dy = py - closest_y

        distance_pixels = math.sqrt(

            dx * dx + dy * dy

        )

        diagonal = math.sqrt(

            width * width +

            height * height

        )

        return distance_pixels / diagonal

    # ========================================================

    @staticmethod

    def box_center(box):

        x1, y1, x2, y2 = box

        return (

            (x1 + x2) / 2,

            (y1 + y2) / 2

        )

    # ========================================================

    @staticmethod

    def center_distance(

        c1,

        c2,

        width,

        height

    ):

        dx = (

            c1[0] - c2[0]

        ) / width

        dy = (

            c1[1] - c2[1]

        ) / height

        return math.sqrt(

            dx * dx +

            dy * dy

        )

    # ========================================================

    # STATE CHANGE

    # ========================================================

    def handle_state_change(

        self,

        new_state

    ):

        if new_state == self.state:

            return

        old_state = self.state

        # ----------------------------------------------------

        # IDLE IS A RESET / READY STATE

        # ----------------------------------------------------

        # IDLE is not an experiment step, so never send it to

        # ExperimentController.process_activity().

        #

        # After RELEASED completes an experiment, the next

        # RELEASED -> IDLE transition resets the controller

        # and prepares a fresh experiment cycle.

        # ----------------------------------------------------

        if new_state == "IDLE":

            self.state = "IDLE"

            self.activity_label = (

                self.activity_recognizer.set_state(

                    "IDLE"

                )

            )

            self.activity_status = (

                self.activity_recognizer.status

            )

            if self.experiment_controller.completed:

                self.experiment_controller.reset()

                self.controller_status = "READY"

                self.controller_message = (

                    "WAITING FOR EXPERIMENT"

                )

                # Start a new log for the next experiment.

                self.event_logger = EventLogger(

                    self.experiment_controller.experiment_name

                )

                print()

                print(">>> NEW EXPERIMENT READY")

                print(">>> CONTROLLER RESET")

            else:

                self.controller_status = "READY"

                self.controller_message = (

                    "WAITING FOR EXPERIMENT"

                )

            print(

                f">>> STATE: {old_state} -> IDLE"

            )

            return

        # ----------------------------------------------------

        # NORMAL EXPERIMENT STATE

        # ----------------------------------------------------

        self.state = new_state

        # ----------------------------------------------------

        # Activity Recognition

        # ----------------------------------------------------

        self.activity_label = (

            self.activity_recognizer.set_state(

                new_state

            )

        )

        self.activity_status = (

            self.activity_recognizer.status

        )

        # ----------------------------------------------------

        # Experiment Controller

        # ----------------------------------------------------

        result = (

            self.experiment_controller.process_activity(

                new_state

            )

        )

        self.controller_status = (

            result["status"]

        )

        self.controller_message = (

            result["message"]

        )

        # ----------------------------------------------------

        # Event Logger

        # ----------------------------------------------------

        if result["accepted"]:

            self.event_logger.log_event(

                new_state,

                result["status"],

                result["message"]

            )

            if result["status"] == "COMPLETE":

                self.event_logger.complete()

        else:

            self.event_logger.warning(

                result.get(

                    "detected",

                    new_state

                ),

                result.get(

                    "expected",

                    "UNKNOWN"

                )

            )

        print(

            f">>> STATE: {old_state} -> {new_state}"

        )

    # ========================================================

    # PROCESS ONE FRAME

    # ========================================================

    def process_frame(self, frame):

        height, width = frame.shape[:2]

        rack_pose_data = None

        # ====================================================

        # FPS

        # ====================================================

        self.fps_counter += 1

        current_time = time.perf_counter()

        if (

            current_time -

            self.fps_start

        ) >= 1.0:

            self.fps = (

                self.fps_counter /

                (

                    current_time -

                    self.fps_start

                )

            )

            self.fps_counter = 0

            self.fps_start = current_time

        # ====================================================

        # YOLO

        # ====================================================

        yolo_start = time.perf_counter()

        results = self.model.track(

            frame,

            persist=True,

            verbose=False,

            conf=self.CONFIDENCE_THRESHOLD

        )

        self.yolo_ms = (

            time.perf_counter() -

            yolo_start

        ) * 1000

        # ====================================================

        # FIND BOTTLE

        # ====================================================

        detected_box = None

        detected_confidence = 0.0

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                confidence = float(

                    box.conf[0]

                )

                class_id = int(

                    box.cls[0]

                )

                class_name = (

                    self.model.names[class_id]

                )

                if class_name != "bottle":

                    continue

                if (

                    confidence >

                    detected_confidence

                ):

                    detected_confidence = (

                        confidence

                    )

                    detected_box = tuple(

                        map(

                            int,

                            box.xyxy[0]

                        )

                    )

        # ====================================================

        # STABILIZE DETECTION

        # ====================================================

        if detected_box is not None:

            self.detection_count += 1

            self.missed_frames = 0

        else:

            self.detection_count = 0

            self.missed_frames += 1

        # ----------------------------------------------------

        # Confirm bottle

        # ----------------------------------------------------

        if (

            detected_box is not None

            and

            self.detection_count >=

            self.DETECTION_CONFIRM_FRAMES

        ):

            self.stable_box = detected_box

        # ----------------------------------------------------

        # Keep previous bottle temporarily

        # ----------------------------------------------------

        if detected_box is None:

            if (

                self.stable_box is not None

                and

                self.missed_frames <=

                self.MAX_MISSED_FRAMES

            ):

                detected_box = self.stable_box

            else:

                self.stable_box = None

                self.smoothed_center = None

        # ====================================================

        # PROCESS BOTTLE

        # ====================================================

        if detected_box is not None:

            x1, y1, x2, y2 = detected_box

            # ------------------------------------------------

            # Smooth center

            # ------------------------------------------------

            current_center = (

                self.box_center(

                    detected_box

                )

            )

            if self.smoothed_center is None:

                self.smoothed_center = (

                    current_center

                )

            else:

                self.smoothed_center = (

                    self.SMOOTHING *

                    self.smoothed_center[0]

                    +

                    (1 - self.SMOOTHING) *

                    current_center[0],

                    self.SMOOTHING *

                    self.smoothed_center[1]

                    +

                    (1 - self.SMOOTHING) *

                    current_center[1]

                )

            # ------------------------------------------------

            # Draw bottle

            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 255),

                3

            )

            cv2.putText(

                frame,

                f"BOTTLE {detected_confidence:.2f}",

                (

                    x1,

                    max(25, y1 - 10)

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2

            )

            # ------------------------------------------------

            # Draw center

            # ------------------------------------------------

            center_pixel = (

                int(self.smoothed_center[0]),

                int(self.smoothed_center[1])

            )

            cv2.circle(

                frame,

                center_pixel,

                7,

                (0, 255, 0),

                -1

            )

        # ====================================================
        # POSE ENGINE
        # ====================================================

        pose_start = time.perf_counter()

        pose_result = None
        cloud_keypoints = None

        if IS_CLOUD:

            # ------------------------------------------------
            # CLOUD MODE — YOLO POSE
            # ------------------------------------------------

            pose_results = self.cloud_pose(
                frame,
                verbose=False,
                conf=0.35
            )

            if pose_results and pose_results[0].keypoints is not None:
                keypoints = pose_results[0].keypoints

                if keypoints.xy is not None and len(keypoints.xy) > 0:
                    cloud_keypoints = keypoints.xy[0].cpu().numpy()

            self.pose_ms = (
                time.perf_counter() - pose_start
            ) * 1000

        else:

            # ------------------------------------------------
            # LOCAL MODE — MEDIAPIPE
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            pose_result = self.landmarker.detect(mp_image)

            self.pose_ms = (
                time.perf_counter() - pose_start
            ) * 1000

        # ====================================================
        # RACK RELATIVE POSE — local MediaPipe only
        # ====================================================

        if (
            not IS_CLOUD
            and detected_box is not None
            and self.enable_rack_relative_pose
            and self.rack_relative_pose is not None
            and pose_result is not None
        ):

            if pose_result.pose_world_landmarks:
                rack_pose_data = self.rack_relative_pose.process_landmarks(
                    pose_result.pose_world_landmarks[0]
                )

                if self.rack_relative_pose.visualize:
                    frame = self.rack_relative_pose.draw_rack_relative_pose(
                        frame, rack_pose_data
                    )

        # =================================================
        # HAND POINTS
        # =================================================

        hand_points = []

        if detected_box is not None:

            if IS_CLOUD:

                # YOLO COCO pose keypoints:
                # 9  = LEFT WRIST
                # 10 = RIGHT WRIST

                if cloud_keypoints is not None and len(cloud_keypoints) > 10:

                    left_wrist = cloud_keypoints[9]
                    right_wrist = cloud_keypoints[10]

                    left_point = (
                        float(left_wrist[0] / width),
                        float(left_wrist[1] / height)
                    )

                    right_point = (
                        float(right_wrist[0] / width),
                        float(right_wrist[1] / height)
                    )

                    hand_points.append((
                        "LEFT HAND",
                        left_point
                    ))

                    hand_points.append((
                        "RIGHT HAND",
                        right_point
                    ))

                    cv2.circle(
                        frame,
                        (int(left_wrist[0]), int(left_wrist[1])),
                        8,
                        (255, 0, 0),
                        -1
                    )

                    cv2.circle(
                        frame,
                        (int(right_wrist[0]), int(right_wrist[1])),
                        8,
                        (0, 0, 255),
                        -1
                    )

            else:

                # ------------------------------------------------
                # MEDIAPIPE HAND POINTS
                # ------------------------------------------------

                if pose_result is not None and pose_result.pose_landmarks:

                    landmarks = pose_result.pose_landmarks[0]

                    left_wrist = landmarks[15]
                    right_wrist = landmarks[16]

                    left_point = (
                        left_wrist.x,
                        left_wrist.y
                    )

                    right_point = (
                        right_wrist.x,
                        right_wrist.y
                    )

                    hand_points.append((
                        "LEFT HAND",
                        left_point
                    ))

                    hand_points.append((
                        "RIGHT HAND",
                        right_point
                    ))

                    cv2.circle(
                        frame,
                        (
                            int(left_wrist.x * width),
                            int(left_wrist.y * height)
                        ),
                        8,
                        (255, 0, 0),
                        -1
                    )

                    cv2.circle(
                        frame,
                        (
                            int(right_wrist.x * width),
                            int(right_wrist.y * height)
                        ),
                        8,
                        (0, 0, 255),
                        -1
                    )

            # CLOSEST HAND

            # =================================================

            closest_hand = None

            closest_distance = 999.0

            for hand_name, hand_point in hand_points:

                distance = (

                    self.point_to_box_distance(

                        hand_point,

                        detected_box,

                        width,

                        height

                    )

                )

                if distance < closest_distance:

                    closest_distance = distance

                    closest_hand = hand_name

            # =================================================

            # STATE MACHINE

            # =================================================

            # -------------------------------------------------

            # IDLE

            # -------------------------------------------------

            if self.state == "IDLE":

                self.pickup_frames = 0

                self.release_frames = 0

                if (

                    closest_distance <

                    self.APPROACH_DISTANCE

                ):

                    self.pickup_frames = 0

                    self.handle_state_change(

                        "APPROACHING"

                    )

            # -------------------------------------------------

            # APPROACHING

            # -------------------------------------------------

            elif self.state == "APPROACHING":

                pickup_threshold = (

                    self.PICKUP_DISTANCE_PIXELS /

                    math.sqrt(

                        width * width +

                        height * height

                    )

                )

                if (

                    closest_distance <

                    pickup_threshold

                ):

                    self.pickup_frames += 1

                else:

                    self.pickup_frames = 0

                if (

                    self.pickup_frames >=

                    self.PICKUP_CONFIRM_FRAMES

                ):

                    self.pickup_center = (

                        self.smoothed_center

                    )

                    self.pickup_frames = 0

                    self.handle_state_change(

                        "PICKED"

                    )

            # -------------------------------------------------

            # PICKED

            # -------------------------------------------------

            elif self.state == "PICKED":

                if self.pickup_center is not None:

                    movement = (

                        self.center_distance(

                            self.smoothed_center,

                            self.pickup_center,

                            width,

                            height

                        )

                    )

                    if (

                        movement >

                        self.MOVEMENT_THRESHOLD

                    ):

                        self.handle_state_change(

                            "MOVING"

                        )

            # -------------------------------------------------

            # MOVING

            # -------------------------------------------------

            elif self.state == "MOVING":

                if (

                    closest_distance >

                    self.RELEASE_DISTANCE

                ):

                    self.release_frames += 1

                else:

                    self.release_frames = 0

                if (

                    self.release_frames >=

                    self.RELEASE_CONFIRM_FRAMES

                ):

                    self.release_frames = 0

                    self.handle_state_change(

                        "RELEASED"

                    )

            # -------------------------------------------------

            # RELEASED

            # -------------------------------------------------

            elif self.state == "RELEASED":

                if (

                    closest_distance >

                    self.RELEASE_DISTANCE

                ):

                    self.pickup_center = None

                    self.pickup_frames = 0

                    self.release_frames = 0

                    self.handle_state_change(

                        "IDLE"

                    )

            # =================================================

            # CAMERA OVERLAY

            # =================================================

            cv2.putText(

                frame,

                f"STATE: {self.state}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0, 255, 0),

                3

            )

            if closest_hand is not None:

                cv2.putText(

                    frame,

                    f"HAND: {closest_hand}",

                    (20, 75),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (255, 255, 255),

                    2

                )

                pixel_distance = (

                    closest_distance *

                    math.sqrt(

                        width * width +

                        height * height

                    )

                )

                cv2.putText(

                    frame,

                    (

                        f"HAND-BOTTLE: "

                        f"{pixel_distance:.0f}px"

                    ),

                    (20, 105),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (255, 255, 255),

                    2

                )

            cv2.putText(

                frame,

                f"FPS: {self.fps:.1f}",

                (20, 140),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (255, 255, 255),

                2

            )

            cv2.putText(

                frame,

                f"YOLO: {self.yolo_ms:.0f} ms",

                (20, 168),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2

            )

            cv2.putText(

                frame,

                f"POSE: {self.pose_ms:.0f} ms",

                (20, 195),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2

            )

        else:

            cv2.putText(

                frame,

                "BOTTLE: SEARCHING...",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 0, 255),

                2

            )

        # ====================================================

        # RETURN RESULTS TO DASHBOARD

        # ====================================================

        return {

            "frame": frame,

            "state": self.state,

            "activity": self.activity_label,

            "activity_status": self.activity_status,

            "controller_status": self.controller_status,

            "controller_message": self.controller_message,

            "step_index": (

                self.experiment_controller.step_index

            ),

            "total_steps": len(

                self.experiment_controller.STEP_ORDER

            ),

            "completed": (

                self.experiment_controller.completed

            ),

            "next_step": (

                self.experiment_controller.get_expected_step()

            ),

            "fps": self.fps,

            "rack_relative_pose": rack_pose_data

        }

    # ========================================================

    # CLEANUP

    # ========================================================

    def close(self):

        try:

            if self.landmarker is not None:
                self.landmarker.close()

        except Exception:

            pass
