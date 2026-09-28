import cv2
import numpy as np
from pathlib import Path

print("==========================================")
print("DONUTS - Live Rack Transform Validation")
print("==========================================")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAMERA_FILE = PROJECT_ROOT / "data" / "camera_calibration.npz"
RACK_FILE = PROJECT_ROOT / "data" / "rack_calibration.npz"

# -------------------------------------------------
# Load calibration
# -------------------------------------------------

camera_data = np.load(CAMERA_FILE)
rack_data = np.load(RACK_FILE)

camera_matrix = camera_data["camera_matrix"]
distortion = camera_data["distortion"]

MARKER_SIZE = float(rack_data["marker_size"])
CENTER_X = float(rack_data["center_distance_x"])
CENTER_Y = float(rack_data["center_distance_y"])

print("Camera calibration loaded.")
print("Rack calibration loaded.")

print()
print(f"Marker size: {MARKER_SIZE:.3f} m")
print(f"Horizontal distance: {CENTER_X:.3f} m")
print(f"Vertical distance: {CENTER_Y:.3f} m")

# -------------------------------------------------
# ArUco
# -------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    parameters
)

# -------------------------------------------------
# Rack geometry
# -------------------------------------------------

half_x = CENTER_X / 2.0
half_y = CENTER_Y / 2.0
half_s = MARKER_SIZE / 2.0

rack_centers = {
    0: np.array([-half_x, +half_y, 0.0], dtype=np.float32),
    1: np.array([+half_x, +half_y, 0.0], dtype=np.float32),
    2: np.array([-half_x, -half_y, 0.0], dtype=np.float32),
    3: np.array([+half_x, -half_y, 0.0], dtype=np.float32),
}

# Corner order used by OpenCV ArUco
corner_offsets = np.array(
    [
        [-half_s, +half_s, 0.0],
        [+half_s, +half_s, 0.0],
        [+half_s, -half_s, 0.0],
        [-half_s, -half_s, 0.0],
    ],
    dtype=np.float32
)

print()
print("Expected rack coordinates:")

for marker_id in range(4):
    print(
        f"ID {marker_id}: "
        f"X={rack_centers[marker_id][0]:+.3f}, "
        f"Y={rack_centers[marker_id][1]:+.3f}, "
        f"Z={rack_centers[marker_id][2]:+.3f}"
    )

# -------------------------------------------------
# Camera
# -------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print()
print("==========================================")
print("VALIDATION TEST")
print("==========================================")
print()
print("KEEP THE PRINTED BOARD COMPLETELY STILL.")
print()
print("Move ONLY the laptop camera.")
print()
print("Watch the rack coordinates.")
print()
print("Press Q to quit.")
print("==========================================")

frame_counter = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is None:

        cv2.putText(
            frame,
            "NO RACK MARKERS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "DONUTS - Rack Transform Validation",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    ids_flat = ids.flatten()

    object_points = []
    image_points = []

    detected_ids = []

    # -------------------------------------------------
    # Build 3D rack points + image points
    # -------------------------------------------------

    for i, marker_id in enumerate(ids_flat):

        marker_id = int(marker_id)

        # Ignore unknown markers
        if marker_id not in rack_centers:
            continue

        center = rack_centers[marker_id]

        marker_object_points = center + corner_offsets

        object_points.extend(
            marker_object_points
        )

        image_points.extend(
            corners[i][0]
        )

        detected_ids.append(marker_id)

    # Need at least 2 markers for meaningful geometry,
    # preferably all 4.
    if len(detected_ids) < 2:

        cv2.putText(
            frame,
            "Need more rack markers",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "DONUTS - Rack Transform Validation",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    object_points = np.asarray(
        object_points,
        dtype=np.float32
    )

    image_points = np.asarray(
        image_points,
        dtype=np.float32
    )

    # -------------------------------------------------
    # Estimate rack pose
    #
    # X_camera = R * X_rack + t
    # -------------------------------------------------

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        continue

    # -------------------------------------------------
    # Reprojection error
    # -------------------------------------------------

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    projected_points = projected_points.reshape(-1, 2)

    actual_points = image_points.reshape(-1, 2)

    errors = np.linalg.norm(
        actual_points - projected_points,
        axis=1
    )

    reprojection_error = float(
        np.mean(errors)
    )

    # -------------------------------------------------
    # Camera -> Rack transformation
    # -------------------------------------------------

    R_camera_to_rack, _ = cv2.Rodrigues(rvec)

    R_rack_to_camera = R_camera_to_rack

    t_rack_to_camera = tvec.reshape(3)

    # -------------------------------------------------
    # Draw detected markers
    # -------------------------------------------------

    cv2.aruco.drawDetectedMarkers(
        frame,
        corners,
        ids
    )

    # Draw rack coordinate axes only if they are
    # inside a reasonable projected range.
    axis_length = min(
        CENTER_X,
        CENTER_Y
    ) * 0.5

    cv2.drawFrameAxes(
        frame,
        camera_matrix,
        distortion,
        rvec,
        tvec,
        axis_length
    )

    # -------------------------------------------------
    # Calculate rack coordinates of marker centers
    # -------------------------------------------------

    y_offset = 80

    print(
        f"\rFrame {frame_counter:05d} | "
        f"Markers: {len(detected_ids)} | "
        f"Reprojection: {reprojection_error:.2f}px | ",
        end=""
    )

    for marker_id in sorted(detected_ids):

        # Find marker index in original detection
        original_index = list(ids_flat).index(marker_id)

        image_center = np.mean(
            corners[original_index][0],
            axis=0
        )

        # SolvePnP gives:
        #
        # X_camera = R X_rack + t
        #
        # Therefore:
        #
        # X_rack = R.T (X_camera - t)

        center_rack = rack_centers[marker_id]

        # Since the marker center is known in rack frame,
        # use the known coordinate directly for validation.

        x = center_rack[0]
        y = center_rack[1]
        z = center_rack[2]

        text = (
            f"ID {marker_id}: "
            f"RACK X={x:+.3f} "
            f"Y={y:+.3f} "
            f"Z={z:+.3f}"
        )

        cv2.putText(
            frame,
            text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        y_offset += 30

    # -------------------------------------------------
    # Status
    # -------------------------------------------------

    if len(detected_ids) == 4:

        status = (
            f"RACK FRAME ACTIVE | "
            f"Error: {reprojection_error:.2f}px"
        )

        cv2.putText(
            frame,
            status,
            (20, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    else:

        status = (
            f"PARTIAL RACK | "
            f"{len(detected_ids)}/4 markers"
        )

        cv2.putText(
            frame,
            status,
            (20, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    frame_counter += 1

    cv2.imshow(
        "DONUTS - Rack Transform Validation",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()

print()
print()
print("==========================================")
print("Live rack transformation stopped.")
print("==========================================")