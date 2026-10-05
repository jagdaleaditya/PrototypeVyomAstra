import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# DONUTS AI
# ==========================================

from dashboard.ai_engine import DonutsAI
from event_logger import EventLogger


print()
print("==========================================")
print("DONUTS WEB AI ENGINE")
print("==========================================")


web_ai = DonutsAI()


print(">>> WEB AI ENGINE READY")
print("==========================================")
print()


# ==========================================
# WEBSOCKET HANDLER
# ==========================================

async def process_websocket(websocket: WebSocket):

    print(">>> WebSocket request received")

    # Accept browser connection
    await websocket.accept()

    print(">>> WEB CLIENT CONNECTED")

    # Session-level lock prevents the completed experiment from
    # immediately restarting while the astronaut remains near the bottle.
    session_locked = False
    completed_response = None

    try:

        while True:

            # ==================================
            # RECEIVE FRAME FROM BROWSER
            # ==================================

            try:

                message = await websocket.receive()

            except WebSocketDisconnect:

                print()
                print(">>> WEB CLIENT DISCONNECTED")
                break

            except Exception as error:

                print()
                print(">>> WEBSOCKET RECEIVE ERROR:")
                print(error)
                break


            # ==================================
            # TEXT COMMANDS FROM BROWSER
            # ==================================

            if "text" in message:

                try:
                    command = json.loads(message["text"])
                except Exception:
                    print(">>> WARNING: Invalid WebSocket command")
                    continue

                if command.get("type") == "reset_experiment":

                    print()
                    print("==========================================")
                    print(">>> RESET COMMAND RECEIVED")
                    print("==========================================")

                    # Reset controller and activity recognizer.
                    web_ai.experiment_controller.reset()
                    web_ai.activity_recognizer.reset_cycle()

                    # Reset AI state.
                    web_ai.state = "IDLE"
                    web_ai.activity_label = "WAITING"
                    web_ai.activity_status = "WAITING"
                    web_ai.controller_status = "READY"
                    web_ai.controller_message = "WAITING FOR EXPERIMENT"

                    # Reset detection/tracking state so the next
                    # experiment starts cleanly.
                    web_ai.detection_count = 0
                    web_ai.missed_frames = 0
                    web_ai.pickup_frames = 0
                    web_ai.release_frames = 0
                    web_ai.stable_box = None
                    web_ai.smoothed_center = None
                    web_ai.pickup_center = None

                    # Start a fresh event log.
                    web_ai.event_logger = EventLogger(
                        web_ai.experiment_controller.experiment_name
                    )

                    # Unlock this browser session.
                    session_locked = False
                    completed_response = None

                    print(">>> EXPERIMENT UNLOCKED")
                    print(">>> READY FOR NEW EXPERIMENT")
                    print("==========================================")

                continue


            # ==================================
            # BINARY CAMERA FRAME
            # ==================================

            if "bytes" not in message:
                continue

            data = message["bytes"]

            print(
                f">>> Received frame: {len(data)} bytes",
                end="\r"
            )


            # ==================================
            # HOLD COMPLETED EXPERIMENT
            # ==================================

            if session_locked and completed_response is not None:

                # Keep displaying the final completed frame/status, but
                # do NOT call DonutsAI.process_frame() again. This prevents
                # RELEASED -> IDLE from automatically starting a new cycle.
                response = completed_response

                try:
                    await websocket.send_text(
                        json.dumps(response)
                    )
                except WebSocketDisconnect:
                    print()
                    print(">>> WEB CLIENT DISCONNECTED")
                    break
                except Exception as error:
                    print()
                    print(">>> WEBSOCKET SEND ERROR:")
                    print(error)
                    break

                continue


            # ==================================
            # JPEG → NUMPY
            # ==================================

            np_data = np.frombuffer(
                data,
                dtype=np.uint8
            )


            # ==================================
            # NUMPY → OPENCV
            # ==================================

            frame = cv2.imdecode(
                np_data,
                cv2.IMREAD_COLOR
            )


            if frame is None:

                print()
                print(
                    ">>> WARNING: Could not decode frame"
                )

                continue


            # ==================================
            # RUN DONUTS AI
            # ==================================

            try:

                result = web_ai.process_frame(
                    frame
                )

            except Exception as error:

                print()
                print(">>> DONUTS AI ERROR:")
                print(error)

                continue


            processed_frame = result["frame"]


            # ==================================
            # OPENCV → JPEG
            # ==================================

            success, encoded = cv2.imencode(
                ".jpg",
                processed_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    70
                ]
            )


            if not success:

                print()
                print(
                    ">>> WARNING: JPEG encoding failed"
                )

                continue


            # ==================================
            # JPEG → BASE64
            # ==================================

            frame_base64 = base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8")


            # ==================================
            # BUILD RESPONSE
            # ==================================

            response = {

                "frame":
                    frame_base64,

                "state":
                    result.get(
                        "state",
                        "IDLE"
                    ),

                "activity":
                    result.get(
                        "activity",
                        "WAITING"
                    ),

                "activity_status":
                    result.get(
                        "activity_status",
                        "WAITING"
                    ),

                "controller_status":
                    result.get(
                        "controller_status",
                        "READY"
                    ),

                "controller_message":
                    result.get(
                        "controller_message",
                        ""
                    ),

                "step_index":
                    result.get(
                        "step_index",
                        0
                    ),

                "total_steps":
                    result.get(
                        "total_steps",
                        4
                    ),

                "completed":
                    result.get(
                        "completed",
                        False
                    ),

                "next_step":
                    result.get(
                        "next_step",
                        "APPROACHING"
                    ),

                "fps":
                    result.get(
                        "fps",
                        0
                    )
            }


            # ==================================
            # LOCK AFTER EXPERIMENT COMPLETION
            # ==================================

            if result.get("completed", False):

                session_locked = True
                completed_response = response

                print()
                print(">>> WEB SESSION LOCKED AFTER COMPLETION")
                print(">>> WAITING FOR RESET COMMAND")


            # ==================================
            # SEND RESULT TO BROWSER
            # ==================================

            try:

                await websocket.send_text(
                    json.dumps(response)
                )

            except WebSocketDisconnect:

                print()
                print(">>> WEB CLIENT DISCONNECTED")
                break

            except Exception as error:

                print()
                print(">>> WEBSOCKET SEND ERROR:")
                print(error)
                break


    except WebSocketDisconnect:

        print()
        print(">>> WEB CLIENT DISCONNECTED")


    except Exception as error:

        print()
        print("==========================================")
        print(">>> WEB AI ERROR")
        print("==========================================")
        print(error)
        print("==========================================")


    finally:

        print()
        print(">>> WEB AI SESSION ENDED")