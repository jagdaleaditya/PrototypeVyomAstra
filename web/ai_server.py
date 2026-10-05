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

    try:

        while True:

            # ==================================
            # RECEIVE FRAME FROM BROWSER
            # ==================================

            try:

                data = await websocket.receive_bytes()

            except WebSocketDisconnect:

                print()
                print(">>> WEB CLIENT DISCONNECTED")
                break

            except Exception as error:

                print()
                print(">>> WEBSOCKET RECEIVE ERROR:")
                print(error)
                break


            print(
                f">>> Received frame: {len(data)} bytes",
                end="\r"
            )


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