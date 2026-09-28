import cv2
import numpy as np
from pathlib import Path

print("==========================================")
print("DONUTS - Physical Rack Calibration")
print("==========================================")

# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAMERA_CALIBRATION_FILE = (
    PROJECT_ROOT / "data" / "camera_calibration.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "rack_calibration.npz"
)

# -------------------------------------------------
# REAL PHYSICAL MEASUREMENTS
# -------------------------------------------------

MARKER_SIZE = 0.043          # 4.3 cm
CENTER_DISTANCE_X = 0.129    # 12.9 cm
CENTER_DISTANCE_Y = 0.060    # 6.0 cm

HALF_MARKER = MARKER_SIZE / 2

HALF_X = CENTER_DISTANCE_X / 2
HALF_Y = CENTER_DISTANCE_Y / 2

# -------------------------------------------------
# RACK MARKER CENTERS
# -------------------------------------------------

RACK_CENTERS = {
    0: np.array([-HALF_X, +HALF_Y, 0.0], dtype=np.float32),
    1: np.array([+HALF_X, +HALF_Y, 0.0], dtype=np.float32),
    2: np.array([-HALF_X, -HALF_Y, 0.0], dtype=np.float32),
    3: np.array([+HALF_X, -HALF_Y, 0.0], dtype=np.float32),
}

# -------------------------------------------------
# ARUCO SETUP
# -------------------------------------------------

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

PARAMETERS = cv2.aruco.DetectorParameters()

DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    PARAMETERS
)

# -------------------------------------------------
# LOAD CAMERA CALIBRATION
# -------------------------------------------------

if not CAMERA_CALIBRATION_FILE.exists():
    print("ERROR: Camera calibration file not found.")
    print(CAMERA_CALIBRATION_FILE)
    exit()

calibration = np.load(CAMERA_CALIBRATION_FILE)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion"]

print("Camera calibration loaded.")
print()
print("Physical rack dimensions:")
print(f"Marker size: {MARKER_SIZE:.3f} m")
print(f"Horizontal center distance: {CENTER_DISTANCE_X:.3f} m")
print(f"Vertical center distance: {CENTER_DISTANCE_Y:.3f} m")
print()

# -------------------------------------------------
# CREATE 3D POINTS
# -------------------------------------------------

object_points = []
marker_ids_for_points = []

# Marker corner order:
#
# 0 -------- 1
# |          |
# |          |
# 3 -------- 2
#
# This matches OpenCV ArUco corner ordering.

corner_offsets = np.array(
    [
        [-HALF_MARKER, +HALF_MARKER, 0.0],
        [+HALF_MARKER, +HALF_MARKER, 0.0],
        [+HALF_MARKER, -HALF_MARKER, 0.0],
        [-HALF_MARKER, -HALF_MARKER, 0.0],
    ],
    dtype=np.float32
)

for marker_id in [0, 1, 2, 3]:

    center = RACK_CENTERS[marker_id]

    for offset in corner_offsets:
        object_points.append(center + offset)
        marker_ids_for_points.append(marker_id)

object_points = np.asarray(
    object_points,
    dtype=np.float32
)

print(f"3D rack points created: {len(object_points)}")

# -------------------------------------------------
# CAMERA
# -------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print()
print("==========================================")
print("IMPORTANT")
print("==========================================")
print("Place the printed marker board FLAT and STILL.")
print()
print("Move the LAPTOP CAMERA around the board.")
print()
print("Do NOT move the printed board.")
print()
print("Press SPACE to capture calibration.")
print("Press Q to quit.")
print("==========================================")

captured = False

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    corners, ids, rejected = DETECTOR.detectMarkers(frame)

    detected_ids = []

    if ids is not None:

        detected_ids = ids.flatten().tolist()

        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

        for i, marker_id in enumerate(ids.flatten()):

            if marker_id not in [0, 1, 2, 3]:
                continue

            pts = corners[i][0]

            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            cv2.putText(
                frame,
                f"ID {marker_id}",
                (center_x - 30, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    valid_ids = [
        marker_id
        for marker_id in [0, 1, 2, 3]
        if marker_id in detected_ids
    ]

    # Status
    if len(valid_ids) == 4:

        status = "ALL 4 MARKERS DETECTED - READY"

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        status = (
            f"Need all 4 markers "
            f"({len(valid_ids)}/4)"
        )

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        "SPACE = Capture    Q = Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "DONUTS - Physical Rack Calibration",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------------------------
    # CAPTURE
    # -------------------------------------------------

    if key == ord(" ") and len(valid_ids) == 4:

        print()
        print("==========================================")
        print("Capturing rack calibration...")
        print("==========================================")

        image_points = []

        # Rebuild image points in ID order
        id_to_corners = {}

        for i, marker_id in enumerate(ids.flatten()):

            if marker_id in [0, 1, 2, 3]:
                id_to_corners[marker_id] = (
                    corners[i][0].astype(np.float32)
                )

        for marker_id in [0, 1, 2, 3]:

            marker_corners = id_to_corners[marker_id]

            for point in marker_corners:
                image_points.append(point)

        image_points = np.asarray(
            image_points,
            dtype=np.float32
        )

        print(f"3D points: {len(object_points)}")
        print(f"Image points: {len(image_points)}")

        # -------------------------------------------------
        # SOLVE PNP
        # -------------------------------------------------

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE
        )

        if not success:

            print("ERROR: solvePnP failed.")

        else:

            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # -------------------------------------------------
            # REPROJECTION ERROR
            # -------------------------------------------------

            projected_points, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                camera_matrix,
                distortion
            )

            projected_points = (
                projected_points.reshape(-1, 2)
            )

            actual_points = (
                image_points.reshape(-1, 2)
            )

            errors = np.linalg.norm(
                actual_points - projected_points,
                axis=1
            )

            mean_error = float(
                np.mean(errors)
            )

            print()
            print("Calibration successful!")
            print()
            print("Rotation Matrix:")
            print(rotation_matrix)
            print()
            print("Translation Vector:")
            print(tvec)
            print()
            print(
                f"Mean reprojection error: "
                f"{mean_error:.2f} pixels"
            )

            # -------------------------------------------------
            # SAVE
            # -------------------------------------------------

            np.savez(
                OUTPUT_FILE,
                rvec=rvec,
                tvec=tvec,
                rotation_matrix=rotation_matrix,
                marker_size=MARKER_SIZE,
                center_distance_x=CENTER_DISTANCE_X,
                center_distance_y=CENTER_DISTANCE_Y,
                reprojection_error=mean_error
            )

            print()
            print("==========================================")
            print("RACK CALIBRATION SAVED")
            print("==========================================")
            print(OUTPUT_FILE)
            print()
            print("Physical rack calibration complete.")
            print("==========================================")

            captured = True

            # Draw the calibrated rack axes
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                distortion,
                rvec,
                tvec,
                0.10
            )

            cv2.imshow(
                "DONUTS - Physical Rack Calibration",
                frame
            )

            cv2.waitKey(1500)

            break

    elif key == ord("q"):
        break

# -------------------------------------------------
# CLEANUP
# -------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Rack calibration stopped.")
print("==========================================")