import cv2
import numpy as np
from pathlib import Path

# ==========================================
# DONUTS - Rack Coordinate System
# ==========================================

print("==========================================")
print("DONUTS Rack Coordinate System")
print("==========================================")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CALIBRATION_FILE = (
    PROJECT_ROOT /
    "data" /
    "camera_calibration.npz"
)

# ==========================================
# SETTINGS
# ==========================================

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

PARAMETERS = cv2.aruco.DetectorParameters()

DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    PARAMETERS
)

# Prototype marker size
# IMPORTANT:
# Replace this with the REAL physical marker
# side length for the final rack.
MARKER_SIZE = 0.05  # 5 cm

HALF_SIZE = MARKER_SIZE / 2.0

# ==========================================
# RACK GEOMETRY
# ==========================================
#
# Rack coordinate system:
#
#          Y
#          ↑
#          |
#    ID 0 -------- ID 1
#     |              |
#     |              |
#    ID 2 -------- ID 3
#                    → X
#
# Z points outward from the rack.
#
# IMPORTANT:
# These distances are PROTOTYPE values.
# Measure the real rack later.
#

MARKER_CENTER_DISTANCE_X = 0.075
MARKER_CENTER_DISTANCE_Y = 0.035

# Rack origin = center of the four markers

HALF_X = MARKER_CENTER_DISTANCE_X / 2.0
HALF_Y = MARKER_CENTER_DISTANCE_Y / 2.0

# Marker center positions in rack coordinate system
RACK_POINTS = {
    0: np.array(
        [-HALF_X, HALF_Y, 0.0],
        dtype=np.float32
    ),

    1: np.array(
        [HALF_X, HALF_Y, 0.0],
        dtype=np.float32
    ),

    2: np.array(
        [-HALF_X, -HALF_Y, 0.0],
        dtype=np.float32
    ),

    3: np.array(
        [HALF_X, -HALF_Y, 0.0],
        dtype=np.float32
    )
}

# ==========================================
# LOAD CAMERA CALIBRATION
# ==========================================

if not CALIBRATION_FILE.exists():

    print("ERROR: Camera calibration file not found.")
    print(CALIBRATION_FILE)
    exit()

calibration = np.load(CALIBRATION_FILE)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion"]

print("Camera calibration loaded.")
print()

# ==========================================
# MARKER 3D POINTS
# ==========================================

marker_object_points = np.array(
    [
        [-HALF_SIZE, HALF_SIZE, 0],
        [HALF_SIZE, HALF_SIZE, 0],
        [HALF_SIZE, -HALF_SIZE, 0],
        [-HALF_SIZE, -HALF_SIZE, 0]
    ],
    dtype=np.float32
)

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print()
print("Show the four ArUco markers.")
print("IDs 0, 1, 2 and 3 are used.")
print("Other IDs will be ignored.")
print()
print("Press Q to quit.")
print("==========================================")

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera.")
        break

    corners, ids, rejected = (
        DETECTOR.detectMarkers(frame)
    )

    detected_marker_poses = {}

    if ids is not None:

        ids_flat = ids.flatten()

        for i, marker_id in enumerate(ids_flat):

            marker_id = int(marker_id)

            # ----------------------------------
            # Ignore markers other than 0-3
            # ----------------------------------

            if marker_id not in RACK_POINTS:

                continue

            image_points = corners[i][0].astype(
                np.float32
            )

            success, rvec, tvec = cv2.solvePnP(
                marker_object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:

                continue

            detected_marker_poses[marker_id] = (
                rvec,
                tvec,
                image_points
            )

            # Draw marker
            cv2.aruco.drawDetectedMarkers(
                frame,
                [corners[i]],
                np.array([[marker_id]])
            )

            # Draw marker coordinate axes
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                distortion,
                rvec,
                tvec,
                MARKER_SIZE * 0.7
            )

    # ==========================================
    # CHECK FOUR MARKERS
    # ==========================================

    if len(detected_marker_poses) == 4:

        # --------------------------------------
        # Collect marker centers in camera frame
        # --------------------------------------

        camera_points = []

        rack_points = []

        for marker_id in [0, 1, 2, 3]:

            rvec, tvec, image_points = (
                detected_marker_poses[marker_id]
            )

            camera_points.append(
                tvec.reshape(3)
            )

            rack_points.append(
                RACK_POINTS[marker_id]
            )

        camera_points = np.asarray(
            camera_points,
            dtype=np.float32
        )

        rack_points = np.asarray(
            rack_points,
            dtype=np.float32
        )

        # ======================================
        # FIND CAMERA -> RACK TRANSFORMATION
        # ======================================

        #
        # We want:
        #
        # Camera point
        #       ↓
        # Rack point
        #
        # P_rack = R * P_camera + T
        #

        camera_center = camera_points.mean(
            axis=0
        )

        rack_center = rack_points.mean(
            axis=0
        )

        camera_centered = (
            camera_points -
            camera_center
        )

        rack_centered = (
            rack_points -
            rack_center
        )

        H = (
            camera_centered.T
            @ rack_centered
        )

        U, S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        # Prevent reflection
        if np.linalg.det(R) < 0:

            Vt[-1, :] *= -1

            R = Vt.T @ U.T

        T = (
            rack_center -
            R @ camera_center
        )

        # ======================================
        # DISPLAY TRANSFORMATION
        # ======================================

        print()
        print("==========================================")
        print("RACK COORDINATE SYSTEM DETECTED")
        print("==========================================")

        print()
        print("Rotation Matrix:")
        print(R)

        print()
        print("Translation:")
        print(T)

        # ======================================
        # TRANSFORMATION QUALITY
        # ======================================

        transformed_points = (
            (R @ camera_points.T).T +
            T
        )

        errors = np.linalg.norm(
            transformed_points -
            rack_points,
            axis=1
        )

        mean_error = errors.mean()

        print()
        print(
            f"Mean transformation error: "
            f"{mean_error:.5f} m"
        )

        print("==========================================")

        # ======================================
        # DISPLAY ON CAMERA
        # ======================================

        cv2.putText(
            frame,
            "RACK FRAME: ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Markers: 4/4",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Error: {mean_error:.4f} m",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ======================================
        # DRAW RACK AXES
        # ======================================

        rack_origin_camera = camera_center

        axis_length = 0.10

        # Since R maps camera -> rack,
        # R.T maps rack -> camera.

        rack_to_camera_R = R.T

        rack_x_camera = (
            rack_to_camera_R
            @ np.array(
                [axis_length, 0, 0]
            )
        )

        rack_y_camera = (
            rack_to_camera_R
            @ np.array(
                [0, axis_length, 0]
            )
        )

        rack_z_camera = (
            rack_to_camera_R
            @ np.array(
                [0, 0, axis_length]
            )
        )

        def project_point(point):

            point = np.asarray(
                point,
                dtype=np.float32
            ).reshape(3, 1)

            rvec_zero = np.zeros(
                (3, 1),
                dtype=np.float32
            )

            tvec_point = point

            projected, _ = cv2.projectPoints(
                point.reshape(1, 3),
                rvec_zero,
                tvec_point,
                camera_matrix,
                distortion
            )

            return tuple(
                projected[0][0].astype(int)
            )

        origin_pixel = project_point(
            rack_origin_camera
        )

        x_pixel = project_point(
            rack_origin_camera +
            rack_x_camera
        )

        y_pixel = project_point(
            rack_origin_camera +
            rack_y_camera
        )

        z_pixel = project_point(
            rack_origin_camera +
            rack_z_camera
        )

        # Draw axes

        cv2.line(
            frame,
            origin_pixel,
            x_pixel,
            (255, 0, 0),
            3
        )

        cv2.line(
            frame,
            origin_pixel,
            y_pixel,
            (0, 255, 0),
            3
        )

        cv2.line(
            frame,
            origin_pixel,
            z_pixel,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "X",
            x_pixel,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Y",
            y_pixel,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Z",
            z_pixel,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        # ======================================
        # NOT ALL FOUR MARKERS FOUND
        # ======================================

        detected_ids = sorted(
            detected_marker_poses.keys()
        )

        cv2.putText(
            frame,
            "Waiting for rack markers...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Detected: {detected_ids}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # ==========================================
    # SHOW CAMERA
    # ==========================================

    cv2.imshow(
        "DONUTS - Rack Coordinate System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Rack coordinate system stopped.")

print("==========================================")