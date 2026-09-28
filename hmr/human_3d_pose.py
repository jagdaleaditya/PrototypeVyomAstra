import cv2
import mediapipe as mp
from pathlib import Path

print("==========================================")
print("DONUTS - Human 3D Pose Test")
print("==========================================")

# -------------------------------------------------
# Project paths
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "pose_landmarker_lite.task"
)

# -------------------------------------------------
# Check model
# -------------------------------------------------

if not MODEL_PATH.exists():
    print("ERROR: Pose model not found.")
    print(MODEL_PATH)
    exit()

print("Pose model found.")
print(MODEL_PATH)

# -------------------------------------------------
# MediaPipe Tasks API
# -------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------------------------
# Camera
# -------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print()
print("Camera started.")
print("Show a person.")
print("Move your arms and body.")
print("Press Q to quit.")
print("==========================================")

# -------------------------------------------------
# Create Pose Landmarker
# -------------------------------------------------

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera.")
            break

        # -----------------------------------------
        # OpenCV BGR → RGB
        # -----------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # -----------------------------------------
        # Detect pose
        # -----------------------------------------

        result = landmarker.detect(mp_image)

        # -----------------------------------------
        # If human detected
        # -----------------------------------------

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            # -------------------------------------
            # Draw skeleton
            # -------------------------------------

            connections = [
                (11, 12),       # shoulders
                (11, 13),       # left upper arm
                (13, 15),       # left forearm
                (12, 14),       # right upper arm
                (14, 16),       # right forearm
                (11, 23),       # left torso
                (12, 24),       # right torso
                (23, 24),       # hips
                (23, 25),       # left thigh
                (25, 27),       # left shin
                (24, 26),       # right thigh
                (26, 28),       # right shin
                (0, 11),        # nose → left shoulder
                (0, 12),        # nose → right shoulder
            ]

            height, width = frame.shape[:2]

            for start, end in connections:

                p1 = landmarks[start]
                p2 = landmarks[end]

                x1 = int(p1.x * width)
                y1 = int(p1.y * height)

                x2 = int(p2.x * width)
                y2 = int(p2.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            # -------------------------------------
            # Draw landmarks
            # -------------------------------------

            for landmark in landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 255),
                    -1
                )

            # -------------------------------------
            # World 3D landmarks
            # -------------------------------------

            if result.pose_world_landmarks:

                world = (
                    result.pose_world_landmarks[0]
                )

                # Important landmarks
                selected_points = {
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

                y_position = 25

                for name, index in selected_points.items():

                    point = world[index]

                    x3d = point.x
                    y3d = point.y
                    z3d = point.z

                    text = (
                        f"{name}: "
                        f"X={x3d:+.3f} "
                        f"Y={y3d:+.3f} "
                        f"Z={z3d:+.3f}"
                    )

                    cv2.putText(
                        frame,
                        text,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 0),
                        1
                    )

                    y_position += 17

                # ---------------------------------
                # Terminal output
                # ---------------------------------

                nose = world[0]

                print(
                    f"\r3D Nose -> "
                    f"X={nose.x:+.3f} "
                    f"Y={nose.y:+.3f} "
                    f"Z={nose.z:+.3f}",
                    end=""
                )

        else:

            cv2.putText(
                frame,
                "NO HUMAN POSE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # -----------------------------------------
        # Display
        # -----------------------------------------

        cv2.imshow(
            "DONUTS - Human 3D Pose",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()

cv2.destroyAllWindows()

print()
print("==========================================")
print("Human 3D pose stopped.")
print("==========================================")