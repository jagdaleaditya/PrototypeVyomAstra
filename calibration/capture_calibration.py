import cv2
from pathlib import Path

# ==========================================
# DONUTS - Robust Camera Calibration Capture
# ==========================================

CHESSBOARD_SIZE = (9, 6)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "data" / "calibration_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("==========================================")
print("DONUTS Camera Calibration Capture")
print("==========================================")
print("Show the chessboard to the camera.")
print()
print("Controls:")
print("  SPACE = capture detected chessboard")
print("  Q     = quit")
print("==========================================")

image_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------
    # Robust chessboard detection
    # --------------------------------------

    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):

        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_EXHAUSTIVE |
                  cv2.CALIB_CB_ACCURACY
        )

    # Fallback to normal detector
    if not found:

        found, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SIZE,
            None
        )

        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

    display = frame.copy()

    # --------------------------------------
    # Display detection result
    # --------------------------------------

    if found:

        cv2.drawChessboardCorners(
            display,
            CHESSBOARD_SIZE,
            corners,
            found
        )

        cv2.putText(
            display,
            "CHESSBOARD DETECTED - PRESS SPACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            display,
            "Chessboard NOT detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

    cv2.putText(
        display,
        f"Images captured: {image_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        "SPACE = Capture | Q = Quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "DONUTS - Calibration Capture",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------
    # Capture
    # --------------------------------------

    if key == 32:

        if found:

            filename = (
                SAVE_DIR /
                f"calibration_{image_count:02d}.jpg"
            )

            cv2.imwrite(
                str(filename),
                frame
            )

            image_count += 1

            print(
                f"Captured image {image_count}: "
                f"{filename}"
            )

        else:

            print(
                "Chessboard not detected. "
                "Move the board into view."
            )

    # --------------------------------------
    # Quit
    # --------------------------------------

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Calibration capture finished.")
print(f"Images captured: {image_count}")
print(f"Saved in: {SAVE_DIR}")
print("==========================================")