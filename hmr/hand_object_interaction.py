import cv2
import math
import time
from ultralytics import YOLO
import mediapipe as mp
from activity_recognition import ActivityRecognizer
from experiment_controller import ExperimentController
from event_logger import EventLogger


# ============================================================
# DONUTS - Stable Hand Object Interaction
# ============================================================

print("==========================================")
print("DONUTS - Stable Hand Object Interaction")
print("==========================================")


# ============================================================
# LOAD YOLO
# ============================================================

model = YOLO("yolov8n.pt")

print("YOLO object model loaded.")


# ============================================================
# LOAD MEDIAPIPE
# ============================================================

MODEL_PATH = "models/mediapipe/pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = PoseLandmarker.create_from_options(options)

print("MediaPipe pose model loaded.")


# ============================================================
# INSTRUCTIONS
# ============================================================

print()
print("==========================================")
print("TEST INSTRUCTIONS")
print("==========================================")
print("1. Put the bottle clearly in view.")
print("2. Keep your hand away from the bottle.")
print("3. Slowly move your hand toward it.")
print("4. Pick up the bottle.")
print("5. Move the bottle.")
print("6. Release the bottle.")
print()
print("Press Q to quit.")
print("==========================================")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()


# ============================================================
# CAMERA BUFFER OPTIMIZATION
# ============================================================
# Keeps the camera from building up old frames.
# This helps reduce visible delay/lag.

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ============================================================
# STABILITY SETTINGS
# ============================================================

# YOLO confidence threshold
# KEEPING YOUR WORKING VALUE
CONFIDENCE_THRESHOLD = 0.35


# Number of frames required before accepting bottle
DETECTION_CONFIRM_FRAMES = 3


# Keep previous bottle for this many frames
# if YOLO temporarily misses it.
MAX_MISSED_FRAMES = 8


# Position smoothing
# Lower value = faster response
SMOOTHING = 0.60


# ============================================================
# INTERACTION THRESHOLDS
# ============================================================

# Distance from wrist to bottle bounding box
APPROACH_DISTANCE = 0.20
PICKUP_DISTANCE_PIXELS = 120
RELEASE_DISTANCE = 0.18


# Movement of bottle
MOVEMENT_THRESHOLD = 0.035


# Number of consecutive close frames
# required before confirming pickup
PICKUP_CONFIRM_FRAMES = 3


# Number of consecutive far frames
# required before confirming release
RELEASE_CONFIRM_FRAMES = 3


# ============================================================
# STATE
# ============================================================
state = "IDLE"

# Activity recognition
activity_recognizer = ActivityRecognizer()
activity_label = activity_recognizer.set_state(state)
activity_status = activity_recognizer.status

# Experiment controller
experiment_controller = ExperimentController()

# Current controller information
controller_status = "READY"
controller_message = "WAITING FOR EXPERIMENT"

event_logger = EventLogger(
    experiment_controller.experiment_name
)

def handle_state_change(new_state):
    global state
    global activity_label
    global activity_status
    global controller_status
    global controller_message

    # IDLE is handled separately
    if new_state == "IDLE":
        state = new_state

        activity_label = activity_recognizer.set_state(new_state)
        activity_status = activity_recognizer.status

        # Reset controller after a completed experiment
        if experiment_controller.completed:
            experiment_controller.reset()
            controller_status = "READY"
            controller_message = "WAITING FOR EXPERIMENT"

        return

    # Ignore duplicate state
    if new_state == state:
        return

    # Update main state
    state = new_state

    # Activity Recognition
    activity_label = activity_recognizer.set_state(new_state)
    activity_status = activity_recognizer.status

    # Experiment Controller
    result = experiment_controller.process_activity(new_state)

    controller_status = result["status"]
    controller_message = result["message"]

    if result["accepted"]:
        event_logger.log_event(
            new_state,
            result["status"],
            result["message"]
        )

        if result["status"] == "COMPLETE":
            event_logger.complete()

    else:
        event_logger.warning(
            result["detected"],
            result["expected"]
        )


# ------------------------------------------------------------
# Bottle detection counter
# ONLY used for bottle detection stability
# ------------------------------------------------------------

detection_count = 0
missed_frames = 0


# ------------------------------------------------------------
# Pickup counter
# ONLY used for pickup confirmation
# ------------------------------------------------------------

pickup_frames = 0


# ------------------------------------------------------------
# Release counter
# ONLY used for release confirmation
# ------------------------------------------------------------

release_frames = 0


# ------------------------------------------------------------
# Bottle tracking
# ------------------------------------------------------------

stable_box = None
smoothed_center = None
pickup_center = None


# ============================================================
# PERFORMANCE MONITORING
# ============================================================

show_performance = True

fps = 0.0
fps_counter = 0
fps_start = time.perf_counter()

yolo_ms = 0.0
pose_ms = 0.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def point_to_box_distance(point, box, width, height):

    """
    Calculate normalized distance from a point
    to the nearest point of a bounding box.
    """

    px = point[0] * width
    py = point[1] * height

    x1, y1, x2, y2 = box

    closest_x = max(x1, min(px, x2))
    closest_y = max(y1, min(py, y2))

    dx = px - closest_x
    dy = py - closest_y

    distance_pixels = math.sqrt(
        dx * dx +
        dy * dy
    )

    diagonal = math.sqrt(
        width * width +
        height * height
    )

    return distance_pixels / diagonal


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def center_distance(c1, c2, width, height):

    dx = (c1[0] - c2[0]) / width
    dy = (c1[1] - c2[1]) / height

    return math.sqrt(
        dx * dx +
        dy * dy
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera frame.")
        break


    height, width = frame.shape[:2]


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    fps_counter += 1

    current_time = time.perf_counter()

    if current_time - fps_start >= 1.0:

        fps = fps_counter / (current_time - fps_start)

        fps_counter = 0
        fps_start = current_time


    # ========================================================
    # YOLO OBJECT TRACKING
    # ========================================================

    yolo_start = time.perf_counter()

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=CONFIDENCE_THRESHOLD
    )

    yolo_ms = (
        time.perf_counter() -
        yolo_start
    ) * 1000


    # ========================================================
    # FIND BOTTLE
    # ========================================================

    detected_box = None
    detected_confidence = 0.0

    boxes = results[0].boxes


    if boxes is not None:

        for box in boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[class_id]


            # ONLY BOTTLE
            if class_name != "bottle":

                continue


            # Keep highest confidence bottle
            if confidence > detected_confidence:

                detected_confidence = confidence

                detected_box = tuple(
                    map(
                        int,
                        box.xyxy[0]
                    )
                )


    # ========================================================
    # STABILIZE BOTTLE DETECTION
    # ========================================================

    if detected_box is not None:

        detection_count += 1
        missed_frames = 0

    else:

        detection_count = 0
        missed_frames += 1


    # --------------------------------------------------------
    # Confirm bottle after several frames
    # --------------------------------------------------------

    if (
        detected_box is not None
        and
        detection_count >= DETECTION_CONFIRM_FRAMES
    ):

        stable_box = detected_box


    # --------------------------------------------------------
    # Temporarily keep previous bottle
    # if YOLO misses it
    # --------------------------------------------------------

    if detected_box is None:

        if (
            stable_box is not None
            and
            missed_frames <= MAX_MISSED_FRAMES
        ):

            detected_box = stable_box

        else:

            stable_box = None
            smoothed_center = None


    # ========================================================
    # PROCESS BOTTLE
    # ========================================================

    if detected_box is not None:

        x1, y1, x2, y2 = detected_box


        # ====================================================
        # SMOOTH BOTTLE CENTER
        # ====================================================

        current_center = box_center(
            detected_box
        )


        if smoothed_center is None:

            smoothed_center = current_center

        else:

            smoothed_center = (

                SMOOTHING *
                smoothed_center[0]
                +
                (1 - SMOOTHING) *
                current_center[0],

                SMOOTHING *
                smoothed_center[1]
                +
                (1 - SMOOTHING) *
                current_center[1]
            )


        # ====================================================
        # DRAW BOTTLE
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3
        )


        cv2.putText(
            frame,
            f"BOTTLE {detected_confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        # ====================================================
        # SMOOTHED CENTER
        # ====================================================

        smooth_pixel = (
            int(smoothed_center[0]),
            int(smoothed_center[1])
        )


        cv2.circle(
            frame,
            smooth_pixel,
            7,
            (0, 255, 0),
            -1
        )


        # ====================================================
        # MEDIAPIPE POSE
        # ====================================================

        pose_start = time.perf_counter()


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        pose_result = landmarker.detect(
            mp_image
        )


        pose_ms = (
            time.perf_counter() -
            pose_start
        ) * 1000


        # ====================================================
        # HAND POINTS
        # ====================================================

        hand_points = []


        if pose_result.pose_landmarks:

            landmarks = pose_result.pose_landmarks[0]


            # MediaPipe:
            # 15 = left wrist
            # 16 = right wrist

            left_wrist = landmarks[15]
            right_wrist = landmarks[16]


            left_point = (
                left_wrist.x,
                left_wrist.y
            )


            right_point = (
                right_wrist.x,
                right_wrist.y
            )


            hand_points.append(
                (
                    "LEFT HAND",
                    left_point
                )
            )


            hand_points.append(
                (
                    "RIGHT HAND",
                    right_point
                )
            )


            # =================================================
            # DRAW LEFT WRIST
            # =================================================

            cv2.circle(
                frame,
                (
                    int(left_wrist.x * width),
                    int(left_wrist.y * height)
                ),
                8,
                (255, 0, 0),
                -1
            )


            # =================================================
            # DRAW RIGHT WRIST
            # =================================================

            cv2.circle(
                frame,
                (
                    int(right_wrist.x * width),
                    int(right_wrist.y * height)
                ),
                8,
                (0, 0, 255),
                -1
            )


        # ====================================================
        # FIND CLOSEST HAND
        # ====================================================

        closest_hand = None
        closest_distance = 999.0


        for hand_name, hand_point in hand_points:

            d = point_to_box_distance(
                hand_point,
                detected_box,
                width,
                height
            )


            if d < closest_distance:

                closest_distance = d
                closest_hand = hand_name


        # ====================================================
        # STATE MACHINE
        # ====================================================

        # ----------------------------------------------------
        # IDLE
        # ----------------------------------------------------

        if state == "IDLE":

            pickup_frames = 0
            release_frames = 0


            if closest_distance < APPROACH_DISTANCE:

                handle_state_change("APPROACHING")
               

                pickup_frames = 0

                print(
                    "\n>>> HAND APPROACHING BOTTLE"
                )


        # ----------------------------------------------------
        # APPROACHING
        # ----------------------------------------------------

        elif state == "APPROACHING":

            # Hand is close enough to bottle
            if closest_distance < (PICKUP_DISTANCE_PIXELS / math.sqrt(width * width + height * height)):

                pickup_frames += 1

            else:

                # Reset ONLY pickup counter
                pickup_frames = 0


            # Confirm pickup after consecutive frames
            if pickup_frames >= PICKUP_CONFIRM_FRAMES:

                handle_state_change("PICKED")
               

                pickup_center = smoothed_center

                pickup_frames = 0

                print(
                    "\n>>> BOTTLE PICKED"
                )


        # ----------------------------------------------------
        # PICKED
        # ----------------------------------------------------

        elif state == "PICKED":

            if pickup_center is not None:

                movement = center_distance(
                    smoothed_center,
                    pickup_center,
                    width,
                    height
                )


                if movement > MOVEMENT_THRESHOLD:

                    handle_state_change("MOVING")
                    

                    print(
                        "\n>>> BOTTLE MOVING"
                    )


        # ----------------------------------------------------
        # MOVING
        # ----------------------------------------------------

        elif state == "MOVING":

            if closest_distance > RELEASE_DISTANCE:

                release_frames += 1

            else:

                release_frames = 0


            # Confirm release after consecutive frames
            if release_frames >= RELEASE_CONFIRM_FRAMES:

                handle_state_change("RELEASED")
                

                release_frames = 0

                print(
                    "\n>>> BOTTLE RELEASED"
                )


        # ----------------------------------------------------
        # RELEASED
        # ----------------------------------------------------

        elif state == "RELEASED":

            if closest_distance > RELEASE_DISTANCE:

                handle_state_change("IDLE")

                pickup_center = None

                release_frames = 0

                pickup_frames = 0

                print(
                    "\n>>> READY FOR NEXT INTERACTION"
                )


        # ====================================================
        # ACTIVITY RECOGNITION
        # Count only state changes, not every frame.
        # ====================================================

        

        # ====================================================
        # DISPLAY STATE
        # ====================================================

        cv2.putText(
            frame,
            f"State: {state}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

        # ====================================================
        # ACTIVITY RECOGNITION
        # ====================================================

        cv2.putText(
            frame,
            f"Activity: {activity_label}",
            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Activity status: {activity_status}",
            (20, 282),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Activity: {activity_recognizer.message}",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Activity Status: {activity_recognizer.status}",
            (20, 295),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ====================================================
        # DISPLAY HAND
        # ====================================================

        if closest_hand is not None:

            cv2.putText(
                frame,
                f"Hand: {closest_hand}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Hand-Bottle: {closest_distance:.3f} ({closest_distance * math.sqrt(width * width + height * height):.0f}px)",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # PERFORMANCE DISPLAY
        # ====================================================

        if show_performance:

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"YOLO: {yolo_ms:.0f} ms",
                (20, 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Pose: {pose_ms:.0f} ms",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Pickup frames: {pickup_frames}/{PICKUP_CONFIRM_FRAMES}",
                (20, 222),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


    else:

        # ====================================================
        # NO BOTTLE
        # ====================================================

        cv2.putText(
            frame,
            "Bottle: SEARCHING...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "DONUTS - Stable Hand Object Interaction",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print()
print("==========================================")
print("Stable hand-object interaction stopped.")
print("==========================================")