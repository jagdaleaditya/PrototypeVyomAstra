import cv2
import numpy as np
import os

# ==========================================
# DONUTS - Camera Calibration Chessboard
# ==========================================

# Number of INNER corners
# 9 x 6 means the actual chessboard has
# 10 x 7 squares.
INNER_CORNERS_X = 9
INNER_CORNERS_Y = 6

# Size of each square in pixels
SQUARE_SIZE = 100

# Number of squares
SQUARES_X = INNER_CORNERS_X + 1
SQUARES_Y = INNER_CORNERS_Y + 1

width = SQUARES_X * SQUARE_SIZE
height = SQUARES_Y * SQUARE_SIZE

# Create white image
board = np.ones(
    (height, width),
    dtype=np.uint8
) * 255

# Draw chessboard
for y in range(SQUARES_Y):
    for x in range(SQUARES_X):

        if (x + y) % 2 == 0:

            x1 = x * SQUARE_SIZE
            y1 = y * SQUARE_SIZE

            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE

            cv2.rectangle(
                board,
                (x1, y1),
                (x2, y2),
                0,
                -1
            )

# Create calibration folder
os.makedirs("calibration", exist_ok=True)

filename = "calibration/calibration_chessboard.png"

cv2.imwrite(
    filename,
    board
)

print("==========================================")
print("DONUTS Calibration Chessboard Created")
print("==========================================")
print(f"Inner corners: {INNER_CORNERS_X} x {INNER_CORNERS_Y}")
print(f"Board squares: {SQUARES_X} x {SQUARES_Y}")
print(f"Saved to: {filename}")
print("==========================================")