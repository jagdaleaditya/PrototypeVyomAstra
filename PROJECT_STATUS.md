# DONUTS — PROJECT STATUS AUDIT
**Date:** 2026-10-01  
**Audited By:** Antigravity AI Engine  
**Phase:** PHASE 0 — AUDIT COMPLETE

---

## 1. PROJECT OVERVIEW

**Project Name:** DONUTS (Dynamic Observation Network for Unified Tracking System)  
**Entry Point:** `dashboard/dashboard.py` (launched via `run.bat`)  
**Language:** Python 3.13  
**Framework:** PySide6 (GUI), OpenCV, YOLO (Ultralytics), MediaPipe, PyTorch  
**Architecture:** Monolithic dashboard with AI engine backend  

---

## 2. MODULE AUDIT

### MODULE: Video Input
```
STATUS:     PARTIAL
FILES:      dashboard/dashboard.py (lines 43–53), hmr/hand_object_interaction.py (lines 76–90)
NOTES:
  - Webcam-only, hardcoded to index 0
  - No VideoSource abstraction exists
  - No support for video file, RTSP, or IP camera
  - Camera buffer optimization present in hand_object_interaction.py (not in dashboard)
  - No FPS monitoring exposed to dashboard
  - Camera is opened directly in dashboard __init__ — no graceful reconnection
MISSING:    VideoSource class, configurable source, file/RTSP support
```

### MODULE: Object Detection (YOLO)
```
STATUS:     WORKING (but hardcoded to "bottle" class only)
FILES:      dashboard/ai_engine.py (lines 57–66, 481–540), hmr/object_detection.py
NOTES:
  - Uses YOLOv8n (yolov8n.pt) with YOLO.track() for per-frame tracking
  - Confidence threshold: 0.35 (configurable via self.CONFIDENCE_THRESHOLD)
  - Stability filters: DETECTION_CONFIRM_FRAMES=3, MAX_MISSED_FRAMES=8
  - Detection is filtered to class_name == "bottle" ONLY (hardcoded)
  - object_detection.py is a standalone test script, NOT a reusable class
  - No structured detection output dict; bbox tuples used inline
MISSING:    Configurable class filter, structured detection output, pluggable model interface
```

### MODULE: Human Detection
```
STATUS:     PARTIAL (test script only, not used in main pipeline)
FILES:      hmr/human_detection.py
NOTES:
  - Standalone test script, not a reusable class
  - COCO class 0 = person detection working
  - Not integrated into dashboard/ai_engine.py pipeline
  - No person_id tracking across frames
MISSING:    HumanDetector class, integration into pipeline, person tracking
```

### MODULE: Human Pose Estimation
```
STATUS:     WORKING (MediaPipe-based, integrated in ai_engine.py)
FILES:      dashboard/ai_engine.py (lines 70–97, 677–790), hmr/human_pose.py, hmr/human_3d_pose.py
NOTES:
  - MediaPipe PoseLandmarker (Tasks API) loaded correctly
  - 2D landmarks (pose_landmarks) used for wrist detection
  - 3D world landmarks (pose_world_landmarks) passed to RackRelativePose
  - YOLOv8n-pose.pt available but NOT used in main pipeline (MediaPipe used instead)
  - Pose only runs when bottle is detected (inside detected_box block) — problem
  - human_pose.py and human_3d_pose.py are standalone test scripts
MISSING:    PoseEstimator class, runs independently of object detection
```

### MODULE: Camera Calibration
```
STATUS:     WORKING
FILES:      data/camera_calibration.npz
NOTES:
  - Camera calibration data exists and loads correctly
  - 38 calibration images in data/calibration_images/
  - Used by RackRelativePose for intrinsic parameters
  - No CalibrationManager class; loaded ad-hoc
MISSING:    CalibrationManager class with validate(), status exposure to dashboard
```

### MODULE: Rack Calibration (ArUco)
```
STATUS:     WORKING (data exists, module is a script)
FILES:      calibration/rack_calibration.py, data/rack_calibration.npz
NOTES:
  - 4 ArUco markers (IDs 0–3, DICT_4X4_50) in 2×2 grid on rack
  - Physical dimensions: 4.3cm markers, 12.9cm×6.0cm center-to-center
  - solvePnP (SOLVEPNP_IPPE) with reprojection error calculation
  - rack_calibration.npz (rvec, tvec, rotation_matrix) exists and valid
  - rack_calibration.py is a standalone script, NOT a class
  - No CalibrationManager abstraction
MISSING:    CalibrationManager class, runtime loading API
```

### MODULE: Rack Coordinate Transformation
```
STATUS:     WORKING (well-implemented class)
FILES:      hmr/rack_relative_pose.py
NOTES:
  - RackRelativePose class with full SE(3) transformation
  - Supports RACK_LOCAL, OPENCV, ROS, NED, CUSTOM conventions
  - transform_camera_to_rack(), transform_rack_to_camera()
  - Batch transform: transform_points_camera_to_rack()
  - Body frame computation: compute_body_frame() with facing score
  - Visualization: draw_axes_on_frame(), draw_rack_relative_pose()
  - Feature flag controlled (DONUTS_ENABLE_RACK_RELATIVE_POSE env var)
  - 21/21 unit tests PASS
GAPS:       Not always active (feature flag off by default)
```

### MODULE: Rack Relative Features
```
STATUS:     WORKING (well-implemented class)
FILES:      hmr/rack_relative_features.py
NOTES:
  - RackRelativeFeatures class with temporal state tracking
  - 31 features: hand-object distances, velocities, body orientation
  - ML-ready: to_feature_vector() → np.ndarray (float32, shape 32)
  - Temporal smoothing with exponential moving average
  - 6/6 unit tests PASS
GAPS:       Not connected to dashboard display
```

### MODULE: Object Tracking
```
STATUS:     PARTIAL
FILES:      dashboard/ai_engine.py (lines 192–249, 541–630)
NOTES:
  - YOLO .track(persist=True) provides built-in tracking IDs
  - Position smoothing with exponential moving average
  - Stable box: keeps last detection for MAX_MISSED_FRAMES=8 frames
  - Single object only (bottle)
  - No velocity tracking
  - No trajectory history
  - No object_id exposed to state machine
MISSING:    Multi-object tracking, tracker class, velocity/trajectory storage
```

### MODULE: Hand-Object Interaction
```
STATUS:     WORKING (functional but monolithic script)
FILES:      hmr/hand_object_interaction.py, dashboard/ai_engine.py (lines 793–954)
NOTES:
  - Wrist-to-bounding-box distance (normalized by frame diagonal)
  - APPROACH: distance < 0.20 (normalized)
  - PICKUP: PICKUP_DISTANCE_PIXELS=120, confirmed over 3 frames
  - MOVING: bottle center displacement > 0.035 (normalized)
  - RELEASE: distance > 0.18, confirmed over 3 frames
  - Both backup files exist (hand_object_interaction_backup.py, _voice_backup.py)
  - hand_object_interaction.py is a standalone script (not a class)
  - Logic duplicated between hand_object_interaction.py and ai_engine.py
MISSING:    HandObjectInteraction class, interaction confidence score, multi-object support
```

### MODULE: Activity Recognition
```
STATUS:     WORKING (minimal/deterministic)
FILES:      hmr/activity_recognition.py
NOTES:
  - ActivityRecognizer class exists
  - Hardcoded 4-step sequence: APPROACHING → PICKED → MOVING → RELEASED
  - State-based, not temporal window-based
  - No temporal evidence accumulation
  - No confidence score
  - Sequence is hardcoded (not loaded from config)
MISSING:    Temporal window, confidence, configurable from experiment.json
```

### MODULE: Experiment State Machine
```
STATUS:     WORKING (functional, config-driven)
FILES:      hmr/experiment_controller.py, config/experiment.json
NOTES:
  - ExperimentController loads steps from config/experiment.json
  - Steps: APPROACHING → PICKED → MOVING → RELEASED
  - Detects wrong step order (returns WARNING)
  - Calls voice alert on wrong step
  - process_activity() returns accepted/rejected + status + message
  - No STEP_SKIPPED detection (only WRONG_ORDER detection)
  - No timeout detection
  - No error states (CALIBRATION_ERROR, PERSON_NOT_FOUND, etc.)
  - step_index and completed are exposed
MISSING:    STEP_SKIPPED detection, timeout, formal state enum, next_step in config
```

### MODULE: Sequence Validation
```
STATUS:     PARTIAL
FILES:      hmr/experiment_controller.py
NOTES:
  - Correct step: accepted, step_index advances
  - Wrong step: rejected, voice alert fired
  - No multi-evidence validation (only activity state)
  - No confidence-based validation
MISSING:    Evidence-based validation, confidence scoring, skipped step detection
```

### MODULE: Voice Alert System
```
STATUS:     WORKING (but blocking)
FILES:      hmr/voice_alert.py, hmr/experiment_controller.py
NOTES:
  - pyttsx3 engine initialized at module level (blocks main thread)
  - speak() function works
  - Called on wrong step detection
  - No cooldown mechanism
  - No alert categories/priorities
  - Blocking: engine.runAndWait() freezes frame processing
  - No async/non-blocking implementation
MISSING:    AlertManager class, cooldown, non-blocking async, configurable voice
```

### MODULE: Event Logger
```
STATUS:     WORKING
FILES:      hmr/event_logger.py, data/experiment_logs/
NOTES:
  - EventLogger class creates JSON log files
  - Timestamped filenames: experiment_YYYYMMDD_HHMMSS.json
  - Logs: activity, status, message per event
  - warning() logs incorrect steps
  - complete() marks end time and status
  - 19 existing log files in data/experiment_logs/
  - JSON format (not JSONL — single object per file)
  - Saves on every event (potential performance issue at high FPS)
MISSING:    Skipped step logging, calibration events, camera error events, system events
```

### MODULE: Video Recording
```
STATUS:     MISSING
FILES:      None
NOTES:      No video recording implemented anywhere
MISSING:    Raw recording, annotated recording, timestamped filenames
```

### MODULE: Dashboard/GUI
```
STATUS:     WORKING
FILES:      dashboard/dashboard.py, dashboard/ai_engine.py
NOTES:
  - PySide6 QMainWindow with ~30ms timer loop
  - Shows: live camera, experiment name, current activity, step progress, next step
  - Shows: controller status, sequence indicator (○●✓)
  - Event log panel (QListWidget) — NOT connected to EventLogger (static init messages only)
  - Camera opened directly in dashboard (hardcoded index 0)
  - No FPS display in dashboard UI
  - No detection/pose/calibration status indicators
  - No color-coded status (green/yellow/red)
  - Sequence names hardcoded in dashboard (APPROACH, PICK, MOVE, RELEASE)
  - Not loaded from experiment config
MISSING:    FPS display, calibration status, dynamic sequence from config, event log feed,
            color-coded status, video recording controls
```

### MODULE: Configuration System
```
STATUS:     PARTIAL
FILES:      config/experiment.json
NOTES:
  - experiment.json exists with experiment_name, steps, enable_rack_relative_pose
  - No global config.yaml (camera, detection thresholds hardcoded in ai_engine.py)
  - Thresholds (CONFIDENCE_THRESHOLD=0.35, etc.) hardcoded as class attributes
MISSING:    config.yaml, camera config, detection config, recording config, alert config
```

### MODULE: Multi-Camera Architecture
```
STATUS:     MISSING
FILES:      None
NOTES:      Architecture is single-camera only, no CameraManager
MISSING:    CameraManager, synchronization layer
```

### MODULE: Performance Monitoring
```
STATUS:     PARTIAL
FILES:      dashboard/ai_engine.py (lines 210–219, 455–492)
NOTES:
  - FPS calculated every 1 second
  - YOLO inference time (yolo_ms) tracked
  - MediaPipe inference time (pose_ms) tracked
  - Values shown in camera overlay text only
  - Not exposed to dashboard UI panels
MISSING:    CPU/GPU/RAM monitoring, queue latency, UI performance panel
```

---

## 3. WORKING MODULES (CONFIRMED)

| Module | Status | Evidence |
|--------|--------|---------|
| YOLO Object Detection (bottle) | ✅ WORKING | Used in ai_engine.py, app runs |
| MediaPipe Pose Estimation | ✅ WORKING | Used in ai_engine.py, app runs |
| Rack Coordinate Transformation | ✅ WORKING | 9 unit tests pass |
| Rack Relative Features | ✅ WORKING | 6 unit tests pass |
| Camera Calibration (data) | ✅ WORKING | data/camera_calibration.npz exists |
| Rack Calibration (data) | ✅ WORKING | data/rack_calibration.npz exists |
| Experiment Controller | ✅ WORKING | Loads from JSON, handles correct/wrong |
| Activity Recognizer | ✅ WORKING | State machine works |
| Event Logger | ✅ WORKING | JSON logs in data/experiment_logs/ |
| Voice Alert (pyttsx3) | ✅ WORKING | pyttsx3 installed, speak() works |
| Dashboard GUI | ✅ WORKING | PySide6 starts, displays camera |
| run.bat | ✅ WORKING | Activates venv, starts dashboard |

---

## 4. PARTIALLY WORKING MODULES

| Module | Problem |
|--------|---------|
| Video Input | Hardcoded webcam index 0, no abstraction |
| Hand-Object Interaction | Script, not class; duplicated in ai_engine.py |
| Object Tracking | Single bottle only, no tracker class |
| Pose Estimation | Only runs when bottle detected |
| Activity Recognition | Hardcoded steps, no temporal evidence |
| Dashboard Event Log | Not connected to EventLogger |
| Performance Monitoring | Console/overlay only, not in dashboard UI |
| Configuration | Only experiment.json, no global config |

---

## 5. BROKEN / MISSING MODULES

| Module | Status |
|--------|--------|
| VideoSource class | ❌ MISSING |
| HumanDetector class | ❌ MISSING |
| ObjectTracker class | ❌ MISSING |
| HandObjectInteraction class | ❌ MISSING |
| CalibrationManager | ❌ MISSING |
| Video Recording | ❌ MISSING |
| Skipped Step Detection | ❌ MISSING |
| AlertManager (async) | ❌ MISSING |
| config.yaml | ❌ MISSING |
| setup.bat | ❌ MISSING |
| ARCHITECTURE.md | ❌ MISSING |
| INSTALLATION.md | ❌ MISSING |
| RUNNING.md | ❌ MISSING |
| TESTING.md | ❌ MISSING |
| Multi-camera support | ❌ MISSING |

---

## 6. CURRENT DEPENDENCIES

All installed in `.venv`:

| Package | Version | Role |
|---------|---------|------|
| ultralytics | 8.4.164 | YOLOv8 |
| mediapipe | 1.0.1 | Pose estimation |
| opencv-contrib-python | 5.0.0.93 | Vision + ArUco |
| PySide6 | 6.9.2 | GUI framework |
| pyttsx3 | 2.99 | Voice alerts (offline TTS) |
| numpy | 2.5.3 | Numerical computing |
| torch | 2.14.0 | PyTorch (YOLO backend) |
| psutil | 7.2.2 | System performance |
| pytest | 9.1.1 | Testing |
| PyYAML | 6.0.3 | Config file parsing |

---

## 7. CURRENT RUN COMMANDS

```batch
# Start the application:
run.bat   (→ calls .venv\Scripts\activate.bat, then python dashboard\dashboard.py)

# Run tests:
.venv\Scripts\python.exe -m pytest tests\ -v

# Individual standalone scripts (development/test):
.venv\Scripts\python.exe hmr\object_detection.py
.venv\Scripts\python.exe hmr\human_detection.py
.venv\Scripts\python.exe hmr\human_pose.py
.venv\Scripts\python.exe calibration\rack_calibration.py
```

---

## 8. CURRENT DASHBOARD ARCHITECTURE

```
dashboard/dashboard.py (PySide6 QMainWindow)
    │
    ├── cv2.VideoCapture(0)  ← hardcoded webcam
    │
    ├── DonutsAI (dashboard/ai_engine.py)
    │     ├── YOLO model (yolov8n.pt)
    │     ├── MediaPipe PoseLandmarker
    │     ├── ActivityRecognizer (hmr/activity_recognition.py)
    │     ├── ExperimentController (hmr/experiment_controller.py)
    │     │     └── loads config/experiment.json
    │     ├── EventLogger (hmr/event_logger.py)
    │     └── RackRelativePose (hmr/rack_relative_pose.py) [feature-flag gated]
    │
    ├── QTimer (30ms → update_system())
    │     ├── camera.read()
    │     ├── ai.process_frame(frame)
    │     │     ├── YOLO tracking
    │     │     ├── Bottle detection + smoothing
    │     │     ├── MediaPipe pose (only if bottle found)
    │     │     ├── Wrist-to-bottle distance
    │     │     └── State machine: IDLE→APPROACHING→PICKED→MOVING→RELEASED
    │     └── UI updates (activity, step, next_step, sequence indicators)
    │
    └── UI Panels:
          ├── LIVE AI CAMERA (700×430 label)
          ├── Experiment/Activity/Step/NextStep/Controller info
          ├── EXPERIMENT SEQUENCE (4 fixed steps)
          └── EVENT LOG (static startup messages)
```

---

## 9. TEST RESULTS (as of audit)

```
tests/test_rack_relative_pose.py    9/9  PASS
tests/test_rack_relative_features.py  12/12  PASS  (6 features + 6 extended pose)
Total: 21/21 PASS

tests/test_pipeline_integration.py: NOT run yet (requires GUI/YOLO init - slow)
```

---

## 10. RECOMMENDED IMPLEMENTATION ORDER

### PHASE 1 — FOUNDATION (Priority: CRITICAL)
1. Create `config/config.yaml` — centralized configuration
2. Create `app/video/video_source.py` — VideoSource abstraction
3. Fix pose estimation to run independently of bottle detection
4. Connect EventLogger output to dashboard event log panel
5. Add FPS display to dashboard UI
6. Create `setup.bat`
7. Update `requirements.txt` to include pytest, pyyaml

### PHASE 2 — TRACKING + INTERACTION (Priority: HIGH)
1. Refactor `hand_object_interaction.py` → `HandObjectInteractionDetector` class
2. Create `app/vision/tracker.py` — ObjectTracker class
3. Add velocity tracking for objects
4. Implement interaction confidence scores

### PHASE 3 — ACTIVITY RECOGNITION (Priority: HIGH)
1. Make ActivityRecognizer load steps from experiment config (not hardcoded)
2. Add temporal evidence window (avoid single-frame triggers)
3. Add activity confidence scores

### PHASE 4 — EXPERIMENT STATE MACHINE (Priority: HIGH)
1. Add STEP_SKIPPED detection to ExperimentController
2. Add timeout per step
3. Add formal error states
4. Connect next_step to experiment config steps

### PHASE 5 — ALERT + LOGGING (Priority: MEDIUM)
1. Create AlertManager with cooldown + async pyttsx3
2. Extend EventLogger to JSONL format
3. Add skipped/timeout/calibration/camera event logging

### PHASE 6 — DASHBOARD INTEGRATION (Priority: MEDIUM)
1. Connect EventLogger to dashboard event log panel (real-time feed)
2. Add color-coded status indicators
3. Add calibration status panel
4. Load sequence names from experiment config (not hardcoded)
5. Add performance metrics panel

### PHASE 7 — VIDEO RECORDING (Priority: MEDIUM)
1. Implement raw + annotated video recording
2. Timestamped filenames linked to experiment session

### PHASE 8 — PERFORMANCE + FINAL (Priority: LOW)
1. Async processing pipeline (frame queue)
2. CPU/GPU/RAM monitoring panel
3. Documentation (ARCHITECTURE.md, INSTALLATION.md, RUNNING.md, TESTING.md)
4. Final end-to-end test

---

## 11. CRITICAL ISSUES TO FIX FIRST

1. **Pose runs only inside bottle detection block** — pose should run every frame
2. **Event log panel is static** — not connected to EventLogger
3. **Voice alert is blocking** — freezes frame processing during speech
4. **Sequence hardcoded in dashboard** — must load from config
5. **No STEP_SKIPPED detection** — only wrong-order detection exists
6. **No config.yaml** — camera index, thresholds are all hardcoded
7. **main.py is empty** — entry point file exists but is empty

---

## 12. WHAT TO PRESERVE UNCHANGED

- `hmr/rack_relative_pose.py` — well-tested, production-quality
- `hmr/rack_relative_features.py` — well-tested, production-quality  
- `hmr/experiment_controller.py` — working, just needs extensions
- `hmr/event_logger.py` — working, just needs extensions
- `hmr/activity_recognition.py` — working, just needs config-driven steps
- `data/camera_calibration.npz` — valid calibration data
- `data/rack_calibration.npz` — valid calibration data
- `config/experiment.json` — valid config (will extend to YAML)
- `dashboard/dashboard.py` — working UI (will extend)
- `dashboard/ai_engine.py` — working AI engine (will extend)
- All 21 existing unit tests (must stay passing)
