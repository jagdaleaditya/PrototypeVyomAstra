import cv2
import numpy as np
import mediapipe as mp
import os

print("==========================================")
print("DONUTS - Rack Relative 3D Pose Test")
print("==========================================")

# -------------------------------------------------
# 1. LOAD CAMERA CALIBRATION
# -------------------------------------------------

camera_file = "data/camera_calibration.npz"

if not os.path.exists(camera_file):
    print("ERROR: Camera calibration not found.")
    exit()

camera_data = np.load(camera_file)

camera_matrix = camera_data["camera_matrix"]
dist_coeffs = camera_data["distortion"]

print("Camera calibration loaded.")

# -------------------------------------------------
# 2. LOAD RACK CALIBRATION
# -------------------------------------------------

rack_file = "data/rack_calibration.npz"

if not os.path.exists(rack_file):
    print("ERROR: Rack calibration not found.")
    exit()

rack_data = np.load(rack_file)

rvec = rack_data["rvec"]
tvec = rack_data["tvec"]

print("Rack calibration loaded.")

# Rotation matrix from rack calibration
R_camera_to_rack = cv2.Rodrigues(rvec)[0]

# -------------------------------------------------
# 3. LOAD MEDIAPIPE MODEL
# -------------------------------------------------

model_path = "models/mediapipe/pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print("ERROR: MediaPipe model not found.")
    exit()

print("Pose model found.")
print(model_path)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1
)

landmarker = PoseLandmarker.create_from_options(options)

# -------------------------------------------------
# 4. START CAMERA
# -------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    landmarker.close()
    exit()

print()
print("Camera started.")
print("Show a person.")
print("Move around naturally.")
print("Press Q to quit.")
print("==========================================")

# -------------------------------------------------
# 5. POSE CONNECTIONS
# -------------------------------------------------

connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),

    (9, 10),

    (11, 12),

    (11, 13), (13, 15),
    (12, 14), (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25), (25, 27),
    (24, 26), (26, 28),

    (27, 29), (29, 31),
    (28, 30), (30, 32)
]

# -------------------------------------------------
# 6. MAIN LOOP
# -------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Run MediaPipe
    result = landmarker.detect(mp_image)

    # -------------------------------------------------
    # HUMAN DETECTED
    # -------------------------------------------------

    if result.pose_world_landmarks and result.pose_landmarks:

        # IMPORTANT:
        # pose_landmarks = 2D image coordinates
        # pose_world_landmarks = 3D coordinates

        image_landmarks = result.pose_landmarks[0]
        world_landmarks = result.pose_world_landmarks[0]

        h, w, _ = frame.shape

        # -------------------------------------------------
        # DRAW 2D SKELETON
        # -------------------------------------------------

        for a, b in connections:

            if a >= len(image_landmarks) or b >= len(image_landmarks):
                continue

            pa = image_landmarks[a]
            pb = image_landmarks[b]

            xa = int(pa.x * w)
            ya = int(pa.y * h)

            xb = int(pb.x * w)
            yb = int(pb.y * h)

            cv2.line(
                frame,
                (xa, ya),
                (xb, yb),
                (0, 255, 0),
                2
            )

        # -------------------------------------------------
        # DRAW LANDMARK POINTS
        # -------------------------------------------------

        for landmark in image_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # -------------------------------------------------
        # GET 3D NOSE
        # -------------------------------------------------

        nose = world_landmarks[0]

        x = nose.x
        y = nose.y
        z = nose.z

        cv2.putText(
            frame,
            f"MP 3D Nose: X={x:+.2f} Y={y:+.2f} Z={z:+.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # -------------------------------------------------
        # RACK STATUS
        # -------------------------------------------------

        cv2.putText(
            frame,
            "Rack calibration: ACTIVE",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        # -------------------------------------------------
        # SELECTED 3D JOINTS
        # -------------------------------------------------

        selected = {
            "Nose": 0,
            "Left Shoulder": 11,
            "Right Shoulder": 12,
            "Left Hip": 23,
            "Right Hip": 24
        }

        y_text = 105

        for name, idx in selected.items():

            p = world_landmarks[idx]

            text = (
                f"{name}: "
                f"X={p.x:+.3f} "
                f"Y={p.y:+.3f} "
                f"Z={p.z:+.3f}"
            )

            cv2.putText(
                frame,
                text,
                (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            y_text += 25

    else:

        cv2.putText(
            frame,
            "No person detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # -------------------------------------------------
    # SHOW CAMERA WINDOW
    # -------------------------------------------------

    cv2.imshow(
        "DONUTS - Rack Relative 3D Pose",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# -------------------------------------------------
# CLEANUP
# -------------------------------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Rack relative pose test stopped.")
print("==========================================")