import cv2
import numpy as np
from pathlib import Path


print("==========================================")
print("DONUTS - Rack Transform Validation")
print("==========================================")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CAMERA_FILE = PROJECT_ROOT / "data" / "camera_calibration.npz"
RACK_FILE = PROJECT_ROOT / "data" / "rack_calibration.npz"


# ==================================================
# LOAD CALIBRATION
# ==================================================

camera_data = np.load(CAMERA_FILE)
rack_data = np.load(RACK_FILE)

camera_matrix = camera_data["camera_matrix"]
distortion = camera_data["distortion"]

MARKER_SIZE = float(rack_data["marker_size"])
CENTER_X = float(rack_data["center_distance_x"])
CENTER_Y = float(rack_data["center_distance_y"])


# ==================================================
# RACK COORDINATES
# ==================================================

HALF_X = CENTER_X / 2.0
HALF_Y = CENTER_Y / 2.0

RACK_CENTERS = {
    0: np.array([-HALF_X,  HALF_Y, 0.0], dtype=np.float32),
    1: np.array([ HALF_X,  HALF_Y, 0.0], dtype=np.float32),
    2: np.array([-HALF_X, -HALF_Y, 0.0], dtype=np.float32),
    3: np.array([ HALF_X, -HALF_Y, 0.0], dtype=np.float32),
}


# ==================================================
# ARUCO
# ==================================================

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

PARAMETERS = cv2.aruco.DetectorParameters()

DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    PARAMETERS
)


# ==================================================
# MARKER CORNERS
# ==================================================

half_marker = MARKER_SIZE / 2.0

LOCAL_CORNERS = np.array(
    [
        [-half_marker,  half_marker, 0],
        [ half_marker,  half_marker, 0],
        [ half_marker, -half_marker, 0],
        [-half_marker, -half_marker, 0]
    ],
    dtype=np.float32
)


# ==================================================
# CAMERA
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


print()
print("Camera started.")
print()
print("Keep the marker board fixed.")
print("Move the CAMERA around.")
print()
print("Watch the terminal values.")
print()
print("Press Q to quit.")
print("==========================================")


frame_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    corners, ids, rejected = DETECTOR.detectMarkers(frame)


    if ids is not None:

        # ----------------------------------------------
        # Collect only IDs 0,1,2,3
        # ----------------------------------------------

        object_points = []
        image_points = []

        valid_indices = []


        for i, marker_id in enumerate(ids.flatten()):

            marker_id = int(marker_id)

            if marker_id not in RACK_CENTERS:
                continue

            marker_center = RACK_CENTERS[marker_id]

            rack_corners = (
                LOCAL_CORNERS + marker_center
            )

            image_corners = corners[i][0].astype(
                np.float32
            )

            object_points.extend(rack_corners)
            image_points.extend(image_corners)

            valid_indices.append(i)


        # ----------------------------------------------
        # Need all 4 markers
        # ----------------------------------------------

        if len(valid_indices) == 4:

            object_points = np.asarray(
                object_points,
                dtype=np.float32
            )

            image_points = np.asarray(
                image_points,
                dtype=np.float32
            )


            # ------------------------------------------
            # Solve live camera → rack pose
            # ------------------------------------------

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE
            )


            if success:

                R, _ = cv2.Rodrigues(rvec)


                # --------------------------------------
                # Reprojection error
                # --------------------------------------

                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    distortion
                )

                projected = projected.reshape(-1, 2)

                pixel_errors = np.linalg.norm(
                    image_points - projected,
                    axis=1
                )

                reprojection_error = np.mean(
                    pixel_errors
                )


                # --------------------------------------
                # Print every 15 frames
                # --------------------------------------

                frame_count += 1

                if frame_count % 15 == 0:

                    print()
                    print("------------------------------------------")
                    print(
                        f"Reprojection error: "
                        f"{reprojection_error:.2f} px"
                    )
                    print("RACK COORDINATES")


                # --------------------------------------
                # Calculate each marker center
                # in camera frame and convert to rack
                # --------------------------------------

                for i in valid_indices:

                    marker_id = int(ids.flatten()[i])

                    image_corners = corners[i][0].astype(
                        np.float32
                    )


                    marker_success, marker_rvec, marker_tvec = (
                        cv2.solvePnP(
                            LOCAL_CORNERS,
                            image_corners,
                            camera_matrix,
                            distortion,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                    )


                    if not marker_success:
                        continue


                    camera_point = (
                        marker_tvec.reshape(3)
                    )


                    # Camera → Rack
                    rack_point = (
                        R.T @
                        (
                            camera_point -
                            tvec.reshape(3)
                        )
                    )


                    if frame_count % 15 == 0:

                        print(
                            f"ID {marker_id}: "
                            f"X={rack_point[0]:+.4f} "
                            f"Y={rack_point[1]:+.4f} "
                            f"Z={rack_point[2]:+.4f}"
                        )


                    # ----------------------------------
                    # Draw marker
                    # ----------------------------------

                    cv2.polylines(
                        frame,
                        [
                            image_corners.astype(
                                np.int32
                            )
                        ],
                        True,
                        (0, 255, 0),
                        2
                    )


                    center = np.mean(
                        image_corners,
                        axis=0
                    ).astype(int)


                    cv2.putText(
                        frame,
                        f"ID {marker_id}",
                        (
                            center[0] - 25,
                            center[1] - 25
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )


                # --------------------------------------
                # Status
                # --------------------------------------

                cv2.putText(
                    frame,
                    "RACK FRAME: ACTIVE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Reprojection: "
                    f"{reprojection_error:.2f}px",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )


        else:

            cv2.putText(
                frame,
                f"Markers: "
                f"{len(valid_indices)}/4",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )


    else:

        cv2.putText(
            frame,
            "NO RACK MARKERS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "DONUTS - Rack Transform Validation",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


print()
print("==========================================")
print("Rack transform validation stopped.")
print("==========================================")