from ultralytics import YOLO
import cv2

print("==========================================")
print("DONUTS - Object Detection Test")
print("==========================================")

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

print("YOLO model loaded.")
print()
print("Detectable objects will appear on screen.")
print("Press Q to quit.")
print("==========================================")

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Run YOLO
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Get detected objects
    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            print(
                f"\rDetected: {class_name:<15} "
                f"Confidence: {confidence:.2f}",
                end=""
            )

    # Show result
    cv2.imshow(
        "DONUTS - Object Detection",
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print()
print()
print("==========================================")
print("Object detection stopped.")
print("==========================================")