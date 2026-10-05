"""
DONUTS - Rack Relative 3D Pose Live Visualizer
=============================================
Interactive standalone visualization tool demonstrating:
- Live camera-frame axes (Red=X, Green=Y, Blue=Z).
- Calibrated 3D Rack-frame coordinate axes.
- Real-time 3D human pose transformed into Rack coordinates.
- Astronaut body orientation and facing detection.
- Dynamic switching between coordinate conventions (RACK_LOCAL, OPENCV, ROS, NED).

Keyboard Controls:
  c - Cycle coordinate conventions (RACK_LOCAL -> OPENCV -> ROS -> NED)
  a - Toggle axis visualization
  q - Quit visualizer
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# Add project root and hmr to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HMR_PATH = os.path.join(PROJECT_ROOT, "hmr")
if HMR_PATH not in sys.path:
    sys.path.insert(0, HMR_PATH)

from rack_relative_pose import RackRelativePose, CoordinateConvention


def main():
    print("==================================================")
    print("DONUTS - Rack Relative 3D Pose Live Visualizer")
    print("==================================================")

    # 1. Initialize RackRelativePose module
    rack_pose = RackRelativePose(visualize=True)
    if not rack_pose.is_calibrated:
        print("[WARNING] Rack calibration not loaded. Please verify data/rack_calibration.npz exists.")
    else:
        print(">>> ArUco Rack Calibration loaded successfully.")

    # 2. Initialize MediaPipe PoseLandmarker
    model_path = os.path.join(PROJECT_ROOT, "models", "mediapipe", "pose_landmarker_lite.task")
    if not os.path.exists(model_path):
        print(f"[ERROR] Pose model not found at {model_path}")
        return

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # 3. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    conventions = [
        CoordinateConvention.RACK_LOCAL,
        CoordinateConvention.OPENCV,
        CoordinateConvention.ROS,
        CoordinateConvention.NED
    ]
    conv_idx = 0
    show_axes = True

    print("\nControls:")
    print("  'c' -> Cycle coordinate conventions")
    print("  'a' -> Toggle 3D axes")
    print("  'q' -> Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = landmarker.detect(mp_image)

        pose_data = None
        if pose_result.pose_world_landmarks:
            world_landmarks = pose_result.pose_world_landmarks[0]
            pose_data = rack_pose.process_landmarks(world_landmarks)

        # Draw visualization
        if show_axes and rack_pose.is_calibrated:
            frame = rack_pose.draw_axes_on_frame(frame, axis_length=0.10)

        if pose_data:
            frame = rack_pose.draw_rack_relative_pose(frame, pose_data)

        # Convention & control overlay
        active_conv = conventions[conv_idx].value
        cv2.putText(
            frame,
            f"Convention: {active_conv} [Press 'C' to switch]",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.imshow("DONUTS - Rack Relative Pose Visualizer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            conv_idx = (conv_idx + 1) % len(conventions)
            rack_pose.set_convention(conventions[conv_idx])
            print(f">>> Active Convention switched to: {conventions[conv_idx].value}")
        elif key == ord("a"):
            show_axes = not show_axes
            print(f">>> Show Axes: {show_axes}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Visualizer closed.")


if __name__ == "__main__":
    main()
