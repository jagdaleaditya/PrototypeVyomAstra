import cv2

# ==========================================
# DONUTS - ArUco Marker Detection Test
# ==========================================

print("Starting DONUTS ArUco detection...")

# ------------------------------------------
# 1. Select ArUco dictionary
# ------------------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# ------------------------------------------
# 2. Create detector parameters
# ------------------------------------------
parameters = cv2.aruco.DetectorParameters()

# ------------------------------------------
# 3. Create ArUco detector
# ------------------------------------------
detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    parameters
)

# ------------------------------------------
# 4. Open laptop webcam
# ------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Show the ORIGINAL ArUco marker to the camera.")
print("Press Q to quit.")
print("------------------------------------------")

# ------------------------------------------
# 5. Main camera loop
# ------------------------------------------
while True:

    # Read camera frame
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # --------------------------------------
    # Detect ArUco markers
    # --------------------------------------
    corners, ids, rejected = detector.detectMarkers(frame)

    # --------------------------------------
    # If marker detected
    # --------------------------------------
    if ids is not None:

        # Draw box around detected markers
        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

        # Get marker IDs
        detected_ids = ids.flatten().tolist()

        # Print IDs in terminal
        print("Detected marker ID:", detected_ids)

        # Display detected IDs on camera
        cv2.putText(
            frame,
            f"Detected: {detected_ids}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # Calculate center of each marker
        # ----------------------------------
        for i, marker_id in enumerate(ids.flatten()):

            # Four corners of marker
            marker_corners = corners[i][0]

            # Calculate center X
            center_x = int(
                marker_corners[:, 0].mean()
            )

            # Calculate center Y
            center_y = int(
                marker_corners[:, 1].mean()
            )

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                6,
                (255, 0, 0),
                -1
            )

            # Display ID and position
            text = f"ID {marker_id}: ({center_x}, {center_y})"

            cv2.putText(
                frame,
                text,
                (center_x + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    # --------------------------------------
    # If NO marker detected
    # --------------------------------------
    else:

        cv2.putText(
            frame,
            "No ArUco marker detected",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # --------------------------------------
    # Show camera
    # --------------------------------------
    cv2.imshow(
        "DONUTS - ArUco Detection",
        frame
    )

    # --------------------------------------
    # Press Q to quit
    # --------------------------------------
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------------------------------
# 6. Release camera
# ------------------------------------------
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()

print("------------------------------------------")
print("DONUTS ArUco test stopped.")