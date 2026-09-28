import cv2
import mediapipe as mp
import numpy as np
import os
import math

print("==========================================")
print("DONUTS - Rack Relative Pose Features")
print("==========================================")

# -------------------------------------------------
# 1. LOAD RACK CALIBRATION
# -------------------------------------------------

rack_file = "data/rack_calibration.npz"

if not os.path.exists(rack_file):
    print("ERROR: Rack calibration not found.")
    exit()

rack_data = np.load(rack_file)

rvec = rack_data["rvec"]
tvec = rack_data["tvec"]

R_rack = cv2.Rodrigues(rvec)[0]

print("Rack calibration loaded.")

# -------------------------------------------------
# 2. LOAD MEDIAPIPE MODEL
# -------------------------------------------------

model_path = "models/mediapipe/pose_landmarker_lite.task"

if not os.path.exists(model_path):
    print("ERROR: MediaPipe model not found.")
    exit()

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.IMAGE,
    num_poses=1
)

landmarker = PoseLandmarker.create_from_options(options)

print("MediaPipe 3D pose model loaded.")

# -------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------

def distance_3d(a, b):
    """
    Calculate Euclidean distance between
    two 3D landmarks.
    """
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def angle_3d(a, b, c):
    """
    Calculate angle ABC in 3D.
    """
    v1 = np.array([
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ])

    v2 = np.array([
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ])

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)

    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


# -------------------------------------------------
# 4. CAMERA
# -------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    landmarker.close()
    exit()

print()
print("Camera started.")
print()
print("Move your:")
print("  - arms")
print("  - shoulders")
print("  - body")
print()
print("Watch the feature values.")
print("Press Q to quit.")
print("==========================================")

# -------------------------------------------------
# 5. MAIN LOOP
# -------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.pose_world_landmarks and result.pose_landmarks:

        world = result.pose_world_landmarks[0]
        image = result.pose_landmarks[0]

        h, w, _ = frame.shape

        # -------------------------------------------------
        # DRAW SKELETON
        # -------------------------------------------------

        connections = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28)
        ]

        for a, b in connections:

            pa = image[a]
            pb = image[b]

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
        # IMPORTANT LANDMARKS
        # -------------------------------------------------

        nose = world[0]

        left_shoulder = world[11]
        right_shoulder = world[12]

        left_elbow = world[13]
        right_elbow = world[14]

        left_wrist = world[15]
        right_wrist = world[16]

        left_hip = world[23]
        right_hip = world[24]

        # -------------------------------------------------
        # BODY RELATIVE FEATURES
        # -------------------------------------------------

        shoulder_width = distance_3d(
            left_shoulder,
            right_shoulder
        )

        hip_width = distance_3d(
            left_hip,
            right_hip
        )

        left_arm_length = (
            distance_3d(left_shoulder, left_elbow)
            +
            distance_3d(left_elbow, left_wrist)
        )

        right_arm_length = (
            distance_3d(right_shoulder, right_elbow)
            +
            distance_3d(right_elbow, right_wrist)
        )

        left_hand_to_hip = distance_3d(
            left_wrist,
            left_hip
        )

        right_hand_to_hip = distance_3d(
            right_wrist,
            right_hip
        )

        left_elbow_angle = angle_3d(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        right_elbow_angle = angle_3d(
            right_shoulder,
            right_elbow,
            right_wrist
        )

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        cv2.putText(
            frame,
            "DONUTS - 3D BODY FEATURES",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        features = [
            f"Shoulder width: {shoulder_width:.3f}",
            f"Hip width:      {hip_width:.3f}",
            f"Left arm:       {left_arm_length:.3f}",
            f"Right arm:      {right_arm_length:.3f}",
            f"L hand->hip:    {left_hand_to_hip:.3f}",
            f"R hand->hip:    {right_hand_to_hip:.3f}",
            f"L elbow angle:  {left_elbow_angle:.1f}",
            f"R elbow angle:  {right_elbow_angle:.1f}"
        ]

        y = 65

        for text in features:

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1
            )

            y += 25

        # -------------------------------------------------
        # PRINT PERIODIC FEATURE DATA
        # -------------------------------------------------

        print(
            f"\rShoulder={shoulder_width:.3f} | "
            f"Hip={hip_width:.3f} | "
            f"LHandHip={left_hand_to_hip:.3f} | "
            f"RHandHip={right_hand_to_hip:.3f}",
            end=""
        )

    else:

        cv2.putText(
            frame,
            "No person detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # -------------------------------------------------
    # SHOW
    # -------------------------------------------------

    cv2.imshow(
        "DONUTS - Rack Relative Pose Features",
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
print()
print("==========================================")
print("Rack relative feature test stopped.")
print("==========================================")