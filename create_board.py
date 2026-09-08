import cv2
import numpy as np

# ==========================================
# DONUTS - Create 4 Marker Rack Board
# ==========================================

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# Board size
width = 1200
height = 800

# Create white background
board = np.ones((height, width), dtype=np.uint8) * 255

# Marker size
marker_size = 250

# Marker positions
positions = {
    0: (100, 100),
    1: (850, 100),
    2: (100, 450),
    3: (850, 450)
}

# Generate markers
for marker_id, (x, y) in positions.items():

    marker = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        marker_size
    )

    board[
        y:y + marker_size,
        x:x + marker_size
    ] = marker

    # Add ID label
    cv2.putText(
        board,
        f"ID {marker_id}",
        (x, y + marker_size + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        0,
        2
    )

# Save board
cv2.imwrite(
    "markers/donuts_rack_board.png",
    board
)

print("4-marker board created!")
print("Saved as:")
print("markers/donuts_rack_board.png")