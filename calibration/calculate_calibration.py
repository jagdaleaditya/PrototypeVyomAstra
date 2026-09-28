import cv2
import numpy as np
from pathlib import Path

# ==========================================
# DONUTS - Camera Calibration Calculator
# ==========================================

CHESSBOARD_SIZE = (9, 6)

# Physical square size.
# Since our chessboard is displayed on a screen,
# we use 1.0 as an arbitrary unit.
SQUARE_SIZE = 1.0

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Calibration images
IMAGE_DIR = PROJECT_ROOT / "data" / "calibration_images"

# Output file
OUTPUT_FILE = PROJECT_ROOT / "data" / "camera_calibration.npz"


print("==========================================")
print("DONUTS Camera Calibration")
print("==========================================")

# ------------------------------------------
# Create 3D reference points
# ------------------------------------------

object_points_template = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

object_points_template[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

object_points_template *= SQUARE_SIZE


# ------------------------------------------
# Storage
# ------------------------------------------

object_points = []
image_points = []

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)

print(f"Found {len(images)} calibration images.")

if len(images) == 0:
    print("ERROR: No calibration images found.")
    exit()


image_size = None
successful = 0


# ------------------------------------------
# Detect chessboard in every image
# ------------------------------------------

for image_path in images:

    print(
        f"Processing: {image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print("  ERROR: Could not read image.")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image_size = (
        gray.shape[1],
        gray.shape[0]
    )

    found = False
    corners = None

    # Robust detector
    if hasattr(cv2, "findChessboardCornersSB"):

        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_EXHAUSTIVE |
                  cv2.CALIB_CB_ACCURACY
        )

    # Fallback
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

    if found:

        object_points.append(
            object_points_template.copy()
        )

        image_points.append(
            corners
        )

        successful += 1

        print("  Chessboard detected.")

    else:

        print("  Chessboard NOT detected.")


# ------------------------------------------
# Check results
# ------------------------------------------

print()
print("==========================================")
print("Detection Summary")
print("==========================================")
print(f"Images provided: {len(images)}")
print(f"Successful detections: {successful}")


if successful < 10:

    print()
    print("ERROR: Too few successful detections.")
    print("Need at least 10 good images.")
    exit()


# ------------------------------------------
# Perform camera calibration
# ------------------------------------------

print()
print("Calculating camera parameters...")

ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)


# ------------------------------------------
# Calculate reprojection error
# ------------------------------------------

total_error = 0

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion
    )
# Convert both point arrays to the same shape/type
actual_points = np.asarray(
    image_points[i],
    dtype=np.float32
).reshape(-1, 2)

predicted_points = np.asarray(
    projected_points,
    dtype=np.float32
).reshape(-1, 2)

error = np.linalg.norm(
    actual_points - predicted_points
) / len(actual_points)

total_error += error

mean_error = total_error / len(object_points)


# ------------------------------------------
# Print results
# ------------------------------------------

print()
print("==========================================")
print("CAMERA CALIBRATION RESULTS")
print("==========================================")

print()
print("Camera Matrix:")
print(camera_matrix)

print()
print("Distortion Coefficients:")
print(distortion.ravel())

print()
print(f"Reprojection Error: {mean_error:.4f}")

print()
print("RMS Calibration Error:")
print(f"{ret:.4f}")


# ------------------------------------------
# Save calibration
# ------------------------------------------

np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    distortion=distortion,
    rvecs=np.array(rvecs, dtype=object),
    tvecs=np.array(tvecs, dtype=object),
    image_width=image_size[0],
    image_height=image_size[1],
    reprojection_error=mean_error,
    rms_error=ret
)

print()
print("==========================================")
print("Calibration saved successfully!")
print("==========================================")
print(f"File: {OUTPUT_FILE}")
print("==========================================")