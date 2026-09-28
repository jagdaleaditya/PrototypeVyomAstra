import cv2
from ultralytics import YOLO

print("==========================================")
print("DONUTS - Human Detection Test")
print("==========================================")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Show a person in front of the camera.")
print("Press Q to quit.")
print("==========================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # Run YOLO
    results = model(
        frame,
        verbose=False
    )

    person_count = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            # COCO class 0 = person
            if class_id != 0:
                continue

            if confidence < 0.50:
                continue

            person_count += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"PERSON "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display count
    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "DONUTS - Human Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("Human detection stopped.")
print("==========================================")