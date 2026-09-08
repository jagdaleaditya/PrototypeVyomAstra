import cv2

# ==========================================
# DONUTS - 4 ArUco Marker Detection
# ==========================================

print("Starting DONUTS Rack Marker Detection...")

# ------------------------------------------
# ArUco dictionary
# ------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# ------------------------------------------
# Detector parameters
# ------------------------------------------

parameters = cv2.aruco.DetectorParameters()

# ------------------------------------------
# Create detector
# ------------------------------------------

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    parameters
)

# ------------------------------------------
# Open webcam
# ------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Show the 4-marker board to the camera.")
print("Press Q to quit.")
print("----------------------------------------")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera.")
        break


    # --------------------------------------
    # Detect markers
    # --------------------------------------

    corners, ids, rejected = detector.detectMarkers(
        frame
    )


    # --------------------------------------
    # If markers detected
    # --------------------------------------

    if ids is not None:

        # Draw marker boxes
        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

        detected_ids = ids.flatten().tolist()

        # Print detected IDs
        print(
            "Detected markers:",
            detected_ids
        )


        # ----------------------------------
        # Process each marker
        # ----------------------------------

        for i, marker_id in enumerate(
            ids.flatten()
        ):

            marker_corners = corners[i][0]


            # Calculate center
            center_x = int(
                marker_corners[:, 0].mean()
            )

            center_y = int(
                marker_corners[:, 1].mean()
            )


            # Draw center
            cv2.circle(
                frame,
                (center_x, center_y),
                7,
                (255, 0, 0),
                -1
            )


            # Display ID + position
            text = (
                f"ID {marker_id} "
                f"({center_x},{center_y})"
            )


            cv2.putText(
                frame,
                text,
                (
                    center_x + 10,
                    center_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


        # ----------------------------------
        # Number of markers
        # ----------------------------------

        cv2.putText(
            frame,
            f"Markers detected: {len(ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    # --------------------------------------
    # No markers
    # --------------------------------------

    else:

        cv2.putText(
            frame,
            "No ArUco markers detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # --------------------------------------
    # Show camera
    # --------------------------------------

    cv2.imshow(
        "DONUTS - Rack Calibration Test",
        frame
    )


    # --------------------------------------
    # Quit
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("----------------------------------------")
print("Rack marker test stopped.")