import cv2
import numpy as np
from pathlib import Path

# ==========================================
# DONUTS - ArUco 3D Pose Estimation
# OpenCV 5 compatible
# ==========================================

print("==========================================")
print("DONUTS ArUco 3D Pose Estimation")
print("==========================================")

# ------------------------------------------
# Paths
# ------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CALIBRATION_FILE = (
    PROJECT_ROOT /
    "data" /
    "camera_calibration.npz"
)

# ------------------------------------------
# ArUco configuration
# ------------------------------------------

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

PARAMETERS = cv2.aruco.DetectorParameters()

DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    PARAMETERS
)

# ------------------------------------------
# Marker physical size
# ------------------------------------------
#
# Prototype assumption:
# 5 cm x 5 cm
#
# IMPORTANT:
# Later replace this with the ACTUAL
# physical marker size on the rack.
#

MARKER_SIZE = 0.05

HALF_SIZE = MARKER_SIZE / 2.0


# ------------------------------------------
# Load camera calibration
# ------------------------------------------

if not CALIBRATION_FILE.exists():

    print("ERROR: Camera calibration file not found.")
    print(CALIBRATION_FILE)
    exit()

calibration = np.load(
    CALIBRATION_FILE
)

camera_matrix = calibration[
    "camera_matrix"
]

distortion = calibration[
    "distortion"
]

print("Camera calibration loaded.")

print()
print("Camera Matrix:")
print(camera_matrix)

print()
print("Distortion:")
print(distortion.ravel())


# ------------------------------------------
# Marker 3D coordinates
# ------------------------------------------
#
# Marker coordinate system:
#
#       Y
#       ↑
#       |
#       |
#       ●──────→ X
#
# Z points out of the marker toward camera.
#
# ------------------------------------------

object_points = np.array(
    [
        [-HALF_SIZE,  HALF_SIZE, 0],
        [ HALF_SIZE,  HALF_SIZE, 0],
        [ HALF_SIZE, -HALF_SIZE, 0],
        [-HALF_SIZE, -HALF_SIZE, 0]
    ],
    dtype=np.float32
)


# ------------------------------------------
# Open camera
# ------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()

print()
print("Camera started.")
print()
print("Show the 4 ArUco markers.")
print("Press Q to quit.")
print("==========================================")


# ------------------------------------------
# Main loop
# ------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera.")
        break

    # --------------------------------------
    # Detect markers
    # --------------------------------------

    corners, ids, rejected = (
        DETECTOR.detectMarkers(frame)
    )

    if ids is not None:

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

        # ----------------------------------
        # Process every marker
        # ----------------------------------

        for i, marker_id in enumerate(
            ids.flatten()
        ):

            # --------------------------------
            # Image coordinates
            # --------------------------------

            image_points = corners[i][0].astype(
                np.float32
            )

            # --------------------------------
            # Estimate pose with solvePnP
            # --------------------------------

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            # --------------------------------
            # Draw coordinate axes
            # --------------------------------

            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                distortion,
                rvec,
                tvec,
                MARKER_SIZE * 0.7
            )

            # --------------------------------
            # Position
            # --------------------------------

            x = float(tvec[0][0])
            y = float(tvec[1][0])
            z = float(tvec[2][0])

            # --------------------------------
            # Rotation matrix
            # --------------------------------

            rotation_matrix, _ = cv2.Rodrigues(
                rvec
            )

            # --------------------------------
            # Marker center
            # --------------------------------

            center_x = int(
                image_points[:, 0].mean()
            )

            center_y = int(
                image_points[:, 1].mean()
            )

            # --------------------------------
            # Display position
            # --------------------------------

            text1 = (
                f"ID {marker_id}"
            )

            text2 = (
                f"X:{x:.3f} "
                f"Y:{y:.3f} "
                f"Z:{z:.3f}m"
            )

            cv2.putText(
                frame,
                text1,
                (
                    center_x - 60,
                    center_y - 35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                text2,
                (
                    center_x - 100,
                    center_y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # --------------------------------
            # Terminal output
            # --------------------------------

            print(
                f"ID {marker_id}: "
                f"X={x:.3f} m, "
                f"Y={y:.3f} m, "
                f"Z={z:.3f} m"
            )

            # Uncomment this if you want
            # to inspect the rotation matrix.
            #
            # print("Rotation matrix:")
            # print(rotation_matrix)


    else:

        cv2.putText(
            frame,
            "No ArUco markers detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )


    # --------------------------------------
    # Display
    # --------------------------------------

    cv2.imshow(
        "DONUTS - ArUco 3D Pose",
        frame
    )


    # --------------------------------------
    # Quit
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ------------------------------------------
# Cleanup
# ------------------------------------------

cap.release()

cv2.destroyAllWindows()

print()
print("==========================================")
print("ArUco pose estimation stopped.")
print("==========================================")