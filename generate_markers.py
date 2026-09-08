import cv2
import os

# Create output folder
os.makedirs("markers", exist_ok=True)

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# Generate markers 0, 1, 2, 3
for marker_id in range(4):

    marker = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        500
    )

    filename = f"markers/aruco_{marker_id}.png"
    cv2.imwrite(filename, marker)

    print(f"Generated: {filename}")

print("\nDone! Check the 'markers' folder.")
