import cv2
from ultralytics import YOLO

print("==========================================")
print("DONUTS - Human 2D Pose Test")
print("==========================================")

# YOLO pose model
model = YOLO("yolov8n-pose.pt")

# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Show a person.")
print("Move your arms and body.")
print("Press Q to quit.")
print("==========================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # Pose estimation
    results = model(
        frame,
        verbose=False
    )

    for result in results:

        # Draw skeleton + keypoints
        annotated_frame = result.plot()

        # Get keypoints
        if result.keypoints is not None:

            keypoints = result.keypoints.xy

            if len(keypoints) > 0:

                person_points = keypoints[0]

                print(
                    "\rDetected joints: "
                    f"{len(person_points)}",
                    end=""
                )

        frame = annotated_frame

    cv2.imshow(
        "DONUTS - Human 2D Pose",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Human 2D pose stopped.")
print("==========================================")