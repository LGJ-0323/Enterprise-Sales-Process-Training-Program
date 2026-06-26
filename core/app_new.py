import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse, RedirectResponse

try:
    from .fastrtc_test import LAST_STATUS, stream
except ImportError:
    from fastrtc_test import LAST_STATUS, stream


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
stream.mount(app)
app = gr.mount_gradio_app(app, stream.ui, path="/ui")


@app.get("/")
async def index():
    return RedirectResponse(url="/ui")


@app.get("/custom")
async def custom_index():
    html_content = (BASE_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@app.get("/debug/status")
async def debug_status():
    return LAST_STATUS


@app.post("/debug/client-log")
async def client_log(request: Request):
    payload = await request.json()
    print(f"CLIENT: {payload}", flush=True)
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8510"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
