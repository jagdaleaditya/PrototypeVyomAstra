from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.ai_server import process_websocket


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="DONUTS Live Demo"
)


# ==========================================
# WEB DIRECTORY
# ==========================================

WEB_DIR = Path(__file__).resolve().parent


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/static",
    StaticFiles(
        directory=WEB_DIR
    ),
    name="static"
)


# ==========================================
# HOME
# ==========================================

@app.get("/")
async def home():

    return FileResponse(
        WEB_DIR / "index.html"
    )


# ==========================================
# HEALTH
# ==========================================

@app.get("/health")
async def health():

    return {
        "project": "DONUTS",
        "status": "online"
    }


# ==========================================
# WEBSOCKET
# ==========================================

@app.websocket("/ws/ai")
async def websocket_ai(
    websocket: WebSocket
):

    print()
    print(">>> /ws/ai ROUTE HIT")

    await process_websocket(
        websocket
    )