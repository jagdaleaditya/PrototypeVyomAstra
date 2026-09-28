import cv2
import numpy as np
from pathlib import Path

print("==========================================")
print("DONUTS - Rack Frame Test")
print("==========================================")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAMERA_CALIBRATION = (
    PROJECT_ROOT / "data" / "camera_calibration.npz"
)

RACK_CALIBRATION = (
    PROJECT_ROOT / "data" / "rack_calibration.npz"
)

# --------------------------------------------------
# Check files
# --------------------------------------------------

if not CAMERA_CALIBRATION.exists():
    print("ERROR: Camera calibration not found.")
    print(CAMERA_CALIBRATION)
    exit()

if not RACK_CALIBRATION.exists():
    print("ERROR: Rack calibration not found.")
    print(RACK_CALIBRATION)
    exit()

# --------------------------------------------------
# Load camera calibration
# --------------------------------------------------

camera_data = np.load(CAMERA_CALIBRATION)

camera_matrix = camera_data["camera_matrix"]
distortion = camera_data["distortion"]

# --------------------------------------------------
# Load rack calibration
# --------------------------------------------------

rack_data = np.load(RACK_CALIBRATION)

rvec = rack_data["rvec"]
tvec = rack_data["tvec"]

rotation_matrix = rack_data["rotation_matrix"]

print("Camera calibration loaded.")
print("Rack calibration loaded.")

print()
print("Saved Rotation Matrix:")
print(rotation_matrix)

print()
print("Saved Translation:")
print(tvec)

# --------------------------------------------------
# ArUco detector
# --------------------------------------------------

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

PARAMETERS = cv2.aruco.DetectorParameters()

DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    PARAMETERS
)

# --------------------------------------------------
# Marker size
# --------------------------------------------------

MARKER_SIZE = float(
    rack_data["marker_size"]
)

HALF_SIZE = MARKER_SIZE / 2.0

# Marker corners
marker_object_points = np.array(
    [
        [-HALF_SIZE,  HALF_SIZE, 0],
        [ HALF_SIZE,  HALF_SIZE, 0],
        [ HALF_SIZE, -HALF_SIZE, 0],
        [-HALF_SIZE, -HALF_SIZE, 0]
    ],
    dtype=np.float32
)

# --------------------------------------------------
# Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print()
print("Camera started.")
print()
print("Show the 4 rack markers.")
print()
print("Move the CAMERA around.")
print("The rack coordinate system should remain fixed.")
print()
print("Press Q to quit.")
print("==========================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    corners, ids, rejected = DETECTOR.detectMarkers(frame)

    valid_markers = []

    if ids is not None:

        for i, marker_id in enumerate(ids.flatten()):

            marker_id = int(marker_id)

            # Only our rack markers
            if marker_id not in [0, 1, 2, 3]:
                continue

            image_points = corners[i][0].astype(
                np.float32
            )

            # Estimate marker pose in CAMERA frame
            success, marker_rvec, marker_tvec = cv2.solvePnP(
                marker_object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            valid_markers.append(
                (
                    marker_id,
                    marker_tvec,
                    image_points
                )
            )

            # Draw marker
            cv2.polylines(
                frame,
                [image_points.astype(np.int32)],
                True,
                (0, 255, 0),
                2
            )

            # Marker center
            center_x = int(
                np.mean(image_points[:, 0])
            )

            center_y = int(
                np.mean(image_points[:, 1])
            )

            cv2.putText(
                frame,
                f"ID {marker_id}",
                (center_x - 30, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # --------------------------------------------------
    # Display rack frame status
    # --------------------------------------------------

    if len(valid_markers) >= 3:

        cv2.putText(
            frame,
            "RACK FRAME: ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Markers: {len(valid_markers)}/4",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "RACK FRAME: WAITING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Markers: {len(valid_markers)}/4",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    # --------------------------------------------------
    # Draw saved rack coordinate axes
    #
    # The saved rvec/tvec describes:
    #
    # Rack → Camera
    #
    # Therefore drawFrameAxes shows the rack
    # coordinate system inside the camera image.
    # --------------------------------------------------

    cv2.drawFrameAxes(
        frame,
        camera_matrix,
        distortion,
        rvec,
        tvec,
        0.10
    )

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    cv2.putText(
        frame,
        "X = RED",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Y = GREEN",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Z = BLUE",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )

    cv2.imshow(
        "DONUTS - Rack Coordinate Frame",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Rack frame test stopped.")
print("==========================================")
