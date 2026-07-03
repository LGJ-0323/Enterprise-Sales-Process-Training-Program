from __future__ import annotations

import base64
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

try:
    from .voice_ws import handle_audio_probe, handle_voice_ws
except ImportError:
    from voice_ws import handle_audio_probe, handle_voice_ws

try:
    from .conversation_store import get_session, get_session_turns
    from .fastrtc_new_wechat import (
        LAST_STATUS,
        pcm_to_wav_bytes,
        run_customer_turn,
        transcribe_uploaded_audio,
    )
    from .training_config import difficulty_choices, stage_choices, voice_choices
except ImportError:
    from conversation_store import get_session, get_session_turns
    from fastrtc_new_wechat import (
        LAST_STATUS,
        pcm_to_wav_bytes,
        run_customer_turn,
        transcribe_uploaded_audio,
    )
    from training_config import difficulty_choices, stage_choices, voice_choices


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Mobile H5 Training Prototype")


MOBILE_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>雄达物流客户陪练</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9dee8;
      --brand: #007a78;
      --brand-dark: #065f5d;
      --warn: #b42318;
      --soft: #eef7f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .app {
      max-width: 640px;
      margin: 0 auto;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 14px;
      gap: 12px;
    }
    header {
      padding: 8px 2px 2px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.25;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 5px;
      font-size: 13px;
      color: var(--muted);
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .controls {
      display: grid;
      gap: 10px;
    }
    label {
      display: grid;
      gap: 5px;
      font-size: 13px;
      font-weight: 650;
      color: #344054;
    }
    select {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 10px;
      font-size: 15px;
      background: #fff;
      color: var(--ink);
    }
    .recorder {
      display: grid;
      gap: 12px;
      text-align: center;
    }
    .record-btn {
      width: 132px;
      height: 132px;
      margin: 6px auto 0;
      border: 0;
      border-radius: 50%;
      background: var(--brand);
      color: #fff;
      font-size: 17px;
      font-weight: 750;
      box-shadow: 0 10px 24px rgba(0, 122, 120, 0.28);
      touch-action: manipulation;
    }
    .record-btn:disabled {
      opacity: .62;
      box-shadow: none;
    }
    .record-btn.recording {
      background: var(--warn);
      box-shadow: 0 10px 24px rgba(180, 35, 24, 0.25);
    }
    .play-btn {
      display: none;
      width: min(100%, 260px);
      min-height: 46px;
      margin: 0 auto;
      border: 0;
      border-radius: 8px;
      background: #172033;
      color: #fff;
      font-size: 15px;
      font-weight: 720;
    }
    .play-btn.show {
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .status {
      min-height: 22px;
      font-size: 14px;
      color: var(--muted);
    }
    .timer {
      font-size: 28px;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
    }
    .conversation {
      display: grid;
      gap: 10px;
      flex: 1;
    }
    .bubble {
      padding: 10px 12px;
      border-radius: 8px;
      line-height: 1.55;
      font-size: 15px;
      border: 1px solid var(--line);
      background: #fff;
      text-align: left;
    }
    .bubble.user {
      background: #f9fafb;
    }
    .bubble.customer {
      background: var(--soft);
      border-color: #b8ddda;
    }
    .bubble-title {
      margin-bottom: 4px;
      font-size: 12px;
      font-weight: 760;
      color: var(--brand-dark);
    }
    audio {
      width: 100%;
      margin-top: 2px;
    }
    .error {
      color: var(--warn);
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <main class="app">
    <header>
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <h1>雄达物流客户陪练</h1>
          <div class="sub">点击录音，停止后自动识别并播放客户回复</div>
        </div>
        <a href="/realtime" style="font-size:13px;color:var(--brand);text-decoration:none;white-space:nowrap;margin-top:4px">实时模式 →</a>
      </div>
    </header>

    <section class="panel controls">
      <label>
        阶段
        <select id="stage"></select>
      </label>
      <label>
        难度
        <select id="difficulty"></select>
      </label>
      <label>
        音色
        <select id="voice"></select>
      </label>
    </section>

    <section class="panel recorder">
      <div class="timer" id="timer">00:00</div>
      <button class="record-btn" id="recordBtn" type="button">开始录音</button>
      <button class="play-btn" id="manualPlayBtn" type="button">播放客户语音</button>
      <audio id="replyAudio" preload="auto" playsinline webkit-playsinline></audio>
      <div class="status" id="status">准备就绪</div>
    </section>

    <section class="conversation" id="conversation">
      <div class="bubble customer">
        <div class="bubble-title">客户</div>
        你好，我这边可以听你说一句，尽量直接一点。
      </div>
    </section>
  </main>

  <script>
    const els = {
      stage: document.getElementById("stage"),
      difficulty: document.getElementById("difficulty"),
      voice: document.getElementById("voice"),
      recordBtn: document.getElementById("recordBtn"),
      manualPlayBtn: document.getElementById("manualPlayBtn"),
      replyAudio: document.getElementById("replyAudio"),
      status: document.getElementById("status"),
      timer: document.getElementById("timer"),
      conversation: document.getElementById("conversation"),
    };

    let recorder = null;
    let recording = false;
    let startedAt = 0;
    let timerId = null;
    let unlockedAudioContext = null;
    const sessionId = localStorage.getItem("mobile_training_session_id") || `h5-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem("mobile_training_session_id", sessionId);

    function setStatus(text, isError = false) {
      els.status.textContent = text;
      els.status.className = isError ? "status error" : "status";
    }

    function fillSelect(select, options, value) {
      select.innerHTML = "";
      options.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.value;
        option.textContent = item.label;
        select.appendChild(option);
      });
      if (value) select.value = value;
    }

    function addBubble(type, title, text) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${type}`;
      const heading = document.createElement("div");
      heading.className = "bubble-title";
      heading.textContent = title;
      const body = document.createElement("div");
      body.textContent = text || "";
      bubble.appendChild(heading);
      bubble.appendChild(body);
      els.conversation.appendChild(bubble);
      bubble.scrollIntoView({ behavior: "smooth", block: "end" });
      return bubble;
    }

    function addManualPlayer(src) {
      els.replyAudio.src = src;
      els.replyAudio.controls = true;
      els.manualPlayBtn.classList.add("show");
      addBubble("customer", "播放", "自动播放被浏览器拦截，请点击上方“播放客户语音”。");
    }

    function updateTimer() {
      const seconds = Math.floor((Date.now() - startedAt) / 1000);
      const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
      const ss = String(seconds % 60).padStart(2, "0");
      els.timer.textContent = `${mm}:${ss}`;
    }

    async function loadConfig() {
      const res = await fetch("/api/training/config");
      if (!res.ok) throw new Error("配置加载失败");
      const data = await res.json();
      fillSelect(els.stage, data.stages, data.defaults.stage_id);
      fillSelect(els.difficulty, data.difficulties, data.defaults.difficulty_id);
      fillSelect(els.voice, data.voices, data.defaults.voice_id);
    }

    async function unlockAudioPlayback() {
      try {
        els.replyAudio.muted = true;
        els.replyAudio.src = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=";
        await els.replyAudio.play().catch(() => {});
        els.replyAudio.pause();
        els.replyAudio.currentTime = 0;
        els.replyAudio.muted = false;

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
          unlockedAudioContext = unlockedAudioContext || new AudioContext();
          if (unlockedAudioContext.state === "suspended") {
            await unlockedAudioContext.resume();
          }
          const buffer = unlockedAudioContext.createBuffer(1, 1, 22050);
          const source = unlockedAudioContext.createBufferSource();
          source.buffer = buffer;
          source.connect(unlockedAudioContext.destination);
          source.start(0);
        }
      } catch (err) {
        console.warn("AudioContext unlock failed", err);
      }
    }

    async function playAudioData(audioBase64, audioMime) {
      const src = `data:${audioMime};base64,${audioBase64}`;
      els.manualPlayBtn.classList.remove("show");
      els.replyAudio.controls = false;
      els.replyAudio.src = src;
      els.replyAudio.load();
      try {
        await els.replyAudio.play();
        return true;
      } catch (err) {
        console.warn("HTMLAudioElement autoplay failed", err);
        try {
          if (!unlockedAudioContext) throw err;
          const binary = atob(audioBase64);
          const bytes = new Uint8Array(binary.length);
          for (let index = 0; index < binary.length; index += 1) {
            bytes[index] = binary.charCodeAt(index);
          }
          const decoded = await unlockedAudioContext.decodeAudioData(bytes.buffer);
          const source = unlockedAudioContext.createBufferSource();
          source.buffer = decoded;
          source.connect(unlockedAudioContext.destination);
          source.start(0);
          return true;
        } catch (contextErr) {
          console.warn("AudioContext playback failed", contextErr);
          addManualPlayer(src);
          return false;
        }
      }
    }

    class NativeRecorderAdapter {
      constructor() {
        this.mediaRecorder = null;
        this.stream = null;
        this.chunks = [];
      }

      isSupported() {
        return Boolean(
          navigator.mediaDevices &&
          navigator.mediaDevices.getUserMedia &&
          window.MediaRecorder
        );
      }

      async start() {
        if (!this.isSupported()) {
          throw new Error(
            "当前浏览器不支持原生录音。手机端请使用 HTTPS 地址；后续可切换企业微信 JS-SDK 录音。"
          );
        }
        this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const supportedTypes = [
          "audio/webm;codecs=opus",
          "audio/webm",
          "audio/mp4",
          "audio/aac",
        ];
        const mimeType = supportedTypes.find((type) => MediaRecorder.isTypeSupported(type)) || "";
        this.mediaRecorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
        this.chunks = [];
        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) this.chunks.push(event.data);
        };
        this.mediaRecorder.start();
      }

      async stop() {
        if (!this.mediaRecorder || this.mediaRecorder.state === "inactive") {
          throw new Error("录音尚未开始。");
        }
        const stopped = new Promise((resolve) => {
          this.mediaRecorder.onstop = () => {
            if (this.stream) {
              this.stream.getTracks().forEach((track) => track.stop());
            }
            const mimeType = this.chunks[0]?.type || this.mediaRecorder.mimeType || "audio/webm";
            resolve(new Blob(this.chunks, { type: mimeType }));
          };
        });
        this.mediaRecorder.stop();
        return stopped;
      }
    }

    function createRecorderAdapter() {
      return new NativeRecorderAdapter();
    }

    async function startRecording() {
      await unlockAudioPlayback();
      recorder = createRecorderAdapter();
      await recorder.start();
      recording = true;
      startedAt = Date.now();
      timerId = setInterval(updateTimer, 250);
      updateTimer();
      els.recordBtn.textContent = "停止录音";
      els.recordBtn.classList.add("recording");
      setStatus("正在录音");
    }

    async function stopRecording() {
      if (!recorder) return;
      recording = false;
      clearInterval(timerId);
      els.recordBtn.disabled = true;
      els.recordBtn.textContent = "处理中";
      els.recordBtn.classList.remove("recording");
      setStatus("正在上传并生成客户回复");
      const blob = await recorder.stop();
      await uploadRecording(blob);
    }

    async function uploadRecording(blob) {
      try {
        if (blob.size < 800) throw new Error("录音太短，请重新录一段。");
        const form = new FormData();
        const extension = blob.type.includes("mp4") || blob.type.includes("aac") ? "mp4" : "webm";
        form.append("audio", blob, `recording.${extension}`);
        form.append("stage_id", els.stage.value);
        form.append("difficulty_id", els.difficulty.value);
        form.append("voice_id", els.voice.value);
        form.append("session_id", sessionId);

        const res = await fetch("/api/training/voice-turn", {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          throw new Error(data.detail || data.error || "请求失败");
        }

        addBubble("user", "我", data.prompt);
        addBubble("customer", "客户", data.response_text);
        if (data.audio_base64) {
          const played = await playAudioData(data.audio_base64, data.audio_mime);
          setStatus(played ? "回复已自动播放" : "自动播放被拦截，请手动播放");
        } else {
          setStatus("已生成文字回复，但没有返回音频", true);
        }
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        els.recordBtn.disabled = false;
        els.recordBtn.textContent = "开始录音";
        els.timer.textContent = "00:00";
      }
    }

    els.recordBtn.addEventListener("click", async () => {
      try {
        if (recording) {
          await stopRecording();
        } else {
          await startRecording();
        }
      } catch (err) {
        els.recordBtn.disabled = false;
        els.recordBtn.textContent = "开始录音";
        els.recordBtn.classList.remove("recording");
        setStatus(err.message || String(err), true);
      }
    });

    els.manualPlayBtn.addEventListener("click", async () => {
      try {
        await unlockAudioPlayback();
        els.replyAudio.muted = false;
        els.replyAudio.controls = true;
        await els.replyAudio.play();
        els.manualPlayBtn.classList.remove("show");
        setStatus("客户语音正在播放");
      } catch (err) {
        setStatus("仍然无法播放，请使用页面里的播放器控件。", true);
      }
    });

    loadConfig().catch((err) => setStatus(err.message || String(err), true));
  </script>
</body>
</html>
"""


def choice_payload(choices):
    return [{"label": label, "value": value} for label, value in choices]


@app.get("/")
async def index():
    return RedirectResponse(url="/mobile")


@app.get("/mobile")
async def mobile_page():
    return HTMLResponse(MOBILE_HTML)


@app.get("/api/training/config")
async def training_config():
    return {
        "stages": choice_payload(stage_choices()),
        "difficulties": choice_payload(difficulty_choices()),
        "voices": choice_payload(voice_choices()),
        "defaults": {
            "stage_id": os.getenv("TRAINING_STAGE_ID", "cold_call"),
            "difficulty_id": os.getenv("TRAINING_DIFFICULTY_ID", "easy"),
            "voice_id": os.getenv("TRAINING_VOICE_ID", "longsanshu_v3"),
        },
    }


@app.post("/api/training/voice-turn")
async def voice_turn(
    audio: UploadFile = File(...),
    stage_id: str = Form("cold_call"),
    difficulty_id: str = Form("easy"),
    voice_id: str = Form("longsanshu_v3"),
    session_id: str = Form("h5-local"),
):
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="没有收到录音。")
        prompt = transcribe_uploaded_audio(audio_bytes, audio.filename or "recording.webm")
        turn = run_customer_turn(
            prompt=prompt,
            stage_id=stage_id,
            difficulty_id=difficulty_id,
            voice_id=voice_id,
            session_id=session_id,
        )
        wav_bytes = pcm_to_wav_bytes(turn["audio_bytes"], int(turn.get("sample_rate") or 24000))
        return {
            "ok": True,
            "prompt": turn["prompt"],
            "response_text": turn["response_text"],
            "training": turn["training"],
            "turn_index": turn["turn_index"],
            "guardrail": turn["guardrail"],
            "audio_mime": "audio/wav",
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/debug/status")
async def debug_status():
    return LAST_STATUS


@app.get("/debug/session-turns")
async def debug_session_turns(session_id: str):
    session = get_session(session_id)
    turns = get_session_turns(session_id)
    return {
        "session_id": session_id,
        "session": session,
        "turn_count": len(turns),
        "turns": turns,
    }


@app.post("/debug/client-log")
async def client_log(request: Request):
    payload = await request.json()
    print(f"CLIENT: {payload}", flush=True)
    return {"ok": True}


# ── WebSocket 端点 ─────────────────────────────────────────

@app.websocket("/ws/audio-probe")
async def ws_audio_probe(ws: WebSocket):
    """音频探针：验证 WebSocket + PCM 收发是否正常。"""
    await handle_audio_probe(ws)


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    """实时语音管道：VAD → ASR → LLM → TTS → 流式下发。"""
    await handle_voice_ws(ws)


# ── 实时模式 H5 页面 ─────────────────────────────────────

REALTIME_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>实时语音 · 雄达物流客户陪练</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f6f7f9; --panel: #ffffff; --ink: #172033; --muted: #667085;
    --line: #d9dee8; --brand: #007a78; --brand-dark: #065f5d;
    --warn: #b42318; --soft: #eef7f6; --ok: #027a48;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--ink);
  }
  .app { max-width: 640px; margin: 0 auto; min-height: 100vh; display: flex; flex-direction: column; padding: 14px; gap: 12px; }
  header { padding: 8px 2px 2px; display: flex; justify-content: space-between; align-items: flex-start; }
  h1 { margin: 0; font-size: 22px; line-height: 1.25; }
  .sub { margin-top: 5px; font-size: 13px; color: var(--muted); }
  .mode-badge { display: inline-block; padding: 2px 10px; border-radius: 100px; font-size: 12px; font-weight: 700; background: var(--ok); color: #fff; }
  .mode-badge.fallback { background: var(--muted); }
  a.nav { font-size: 13px; color: var(--brand); text-decoration: none; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
  .controls { display: grid; gap: 10px; }
  label { display: grid; gap: 5px; font-size: 13px; font-weight: 650; color: #344054; }
  select { width: 100%; min-height: 42px; border: 1px solid var(--line); border-radius: 8px; padding: 0 10px; font-size: 15px; background: #fff; color: var(--ink); }
  .status-bar { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 8px; font-size: 15px; font-weight: 650; }
  .status-bar.listening { background: #ecfdf3; color: #027a48; border: 1px solid #a6f4c5; }
  .status-bar.processing { background: #fffaeb; color: #b54708; border: 1px solid #fedf89; }
  .status-bar.speaking { background: #eef4ff; color: #3538cd; border: 1px solid #c7d7fe; }
  .status-bar.error { background: #fef3f2; color: #b42318; border: 1px solid #fecdca; }
  .status-bar.idle { background: #f9fafb; color: #667085; border: 1px solid #e5e7eb; }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .dot.listening { background: #12b76a; animation: pulse 1.5s infinite; }
  .dot.processing { background: #f79009; animation: pulse 0.6s infinite; }
  .dot.speaking { background: #6172f3; }
  .dot.idle { background: #98a2b3; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .action-btn {
    width: min(100%, 280px); min-height: 52px; margin: 6px auto; border: 0; border-radius: 10px;
    background: var(--brand); color: #fff; font-size: 17px; font-weight: 750;
    box-shadow: 0 8px 20px rgba(0,122,120,0.25); touch-action: manipulation;
  }
  .action-btn:disabled { opacity: .55; box-shadow: none; }
  .action-btn.stop { background: var(--warn); box-shadow: 0 8px 20px rgba(180,35,24,0.22); }
  .conversation { display: grid; gap: 10px; flex: 1; }
  .bubble { padding: 10px 12px; border-radius: 8px; line-height: 1.55; font-size: 15px; border: 1px solid var(--line); background: #fff; text-align: left; }
  .bubble.user { background: #f9fafb; }
  .bubble.customer { background: var(--soft); border-color: #b8ddda; }
  .bubble-title { margin-bottom: 4px; font-size: 12px; font-weight: 760; color: var(--brand-dark); }
  .ws-indicator { font-size: 11px; color: var(--muted); text-align: center; margin-top: 4px; }
  .interim { font-size: 13px; color: var(--muted); font-style: italic; padding: 4px 0; min-height: 20px; }
  .error { color: var(--warn); white-space: pre-wrap; font-size: 13px; }
  .tip-box { background: #fffaeb; border: 1px solid #fedf89; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #b54708; }
</style>
</head>
<body>
<main class="app">
  <header>
    <div>
      <h1>雄达物流客户陪练</h1>
      <div class="sub">实时对话模式 · 直接说话，无需按键</div>
    </div>
    <span class="mode-badge">实时</span>
  </header>

  <a class="nav" href="/mobile">← 切换到对讲机模式（兼容性更好）</a>

  <!-- 配置 -->
  <section class="panel controls" id="configSection" style="display:block;">
    <label>阶段<select id="stage"></select></label>
    <label>难度<select id="difficulty"></select></label>
    <label>音色<select id="voice"></select></label>
  </section>

  <!-- 状态栏 -->
  <div class="status-bar idle" id="statusBar">
    <span class="dot idle" id="statusDot"></span>
    <span id="statusText">准备就绪</span>
  </div>

  <!-- ASR 中间文本 -->
  <div class="interim" id="interim"></div>

  <!-- 操作按钮 -->
  <button class="action-btn" id="actionBtn" type="button">开始对话</button>
  <div class="ws-indicator" id="wsIndicator"></div>

  <!-- 对话气泡 -->
  <section class="conversation" id="conversation">
    <div class="bubble customer">
      <div class="bubble-title">客户</div>
      你好，我这边可以听你说一句，尽量直接一点。
    </div>
  </section>

  <div class="tip-box" id="compatNote">
    💡 <strong>iOS 企微用户</strong>：如无法授权麦克风或音频卡顿，请切换到
    <a href="/mobile">对讲机模式</a>。
  </div>
</main>

<script>
// ═══════════════════════════════════════════════════
//  DOM refs
// ═══════════════════════════════════════════════════
const $ = (id) => document.getElementById(id);
const els = {
  stage: $("stage"), difficulty: $("difficulty"), voice: $("voice"),
  statusBar: $("statusBar"), statusDot: $("statusDot"), statusText: $("statusText"),
  interim: $("interim"), actionBtn: $("actionBtn"), wsIndicator: $("wsIndicator"),
  conversation: $("conversation"), configSection: $("configSection"),
};

// ═══════════════════════════════════════════════════
//  状态
// ═══════════════════════════════════════════════════
const SAMPLE_RATE = 16000;
const CHUNK_MS = 100;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_MS / 1000);
const SCRIPT_BUFFER_SIZE = 2048;

let ws = null;
let audioCtx = null;
let stream = null;
let active = false;
let currentStage = "idle";
let lastLevelUpdateAt = 0;
let sessionId = localStorage.getItem("ws_voice_session_id") || `ws-${Date.now()}-${Math.random().toString(16).slice(2)}`;
localStorage.setItem("ws_voice_session_id", sessionId);

// ═══════════════════════════════════════════════════
//  UI helpers
// ═══════════════════════════════════════════════════
function setStatus(stage, text) {
  currentStage = stage || "idle";
  const bar = els.statusBar, dot = els.statusDot, txt = els.statusText;
  bar.className = `status-bar ${currentStage}`;
  dot.className = `dot ${currentStage}`;
  txt.textContent = text;
}

function addBubble(type, title, text) {
  const b = document.createElement("div");
  b.className = `bubble ${type}`;
  const h = document.createElement("div");
  h.className = "bubble-title"; h.textContent = title;
  const body = document.createElement("div");
  body.textContent = text || "";
  b.appendChild(h); b.appendChild(body);
  els.conversation.appendChild(b);
  b.scrollIntoView({ behavior: "smooth", block: "end" });
}

function fillSelect(sel, opts, val) {
  sel.innerHTML = "";
  opts.forEach(o => { const opt = document.createElement("option"); opt.value = o.value; opt.textContent = o.label; sel.appendChild(opt); });
  if (val) sel.value = val;
}

// ═══════════════════════════════════════════════════
//  音频播放队列
// ═══════════════════════════════════════════════════
const playQueue = [];
let playing = false;

function enqueueAudio(pcm16Buffer) {
  playQueue.push(pcm16Buffer);
  if (!playing) drainQueue();
}

function drainQueue() {
  if (playQueue.length === 0) { playing = false; return; }
  playing = true;
  const buf = playQueue.shift();
  try {
    const float32 = new Float32Array(buf.length);
    for (let i = 0; i < buf.length; i++) float32[i] = buf[i] / 32768;
    const ab = audioCtx.createBuffer(1, float32.length, 24000);
    ab.getChannelData(0).set(float32);
    const src = audioCtx.createBufferSource();
    src.buffer = ab;
    src.connect(audioCtx.destination);
    src.onended = drainQueue;
    src.start();
  } catch (e) {
    console.warn("playback error", e);
    drainQueue();
  }
}

// ═══════════════════════════════════════════════════
//  AudioContext 解锁（iOS）
// ═══════════════════════════════════════════════════
async function unlockAudio() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC();
  }
  if (audioCtx.state === "suspended") {
    await audioCtx.resume();
  }
  // 播放静音解锁
  const buf = audioCtx.createBuffer(1, 1, 22050);
  const src = audioCtx.createBufferSource();
  src.buffer = buf; src.connect(audioCtx.destination); src.start(0);
}

// ═══════════════════════════════════════════════════
//  PCM 采集（ScriptProcessor → WebSocket）
// ═══════════════════════════════════════════════════
let scriptProcessor = null;

async function startMic() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true }
  });

  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC({ sampleRate: SAMPLE_RATE });
  }

  const source = audioCtx.createMediaStreamSource(stream);
  scriptProcessor = audioCtx.createScriptProcessor(SCRIPT_BUFFER_SIZE, 1, 1);

  scriptProcessor.onaudioprocess = (e) => {
    if (!active || !ws || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(input.length);
    let sumSq = 0;
    for (let i = 0; i < input.length; i++) {
      sumSq += input[i] * input[i];
      int16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
    }
    const now = Date.now();
    if (currentStage === "listening" && now - lastLevelUpdateAt > 250) {
      const rms = Math.sqrt(sumSq / input.length);
      const bars = Math.min(10, Math.round(rms * 180));
      els.interim.textContent = bars > 0
        ? `麦克风输入 ${"▮".repeat(bars)}${"▯".repeat(10 - bars)}`
        : "正在听你说话...";
      lastLevelUpdateAt = now;
    }
    ws.send(int16.buffer);
  };

  source.connect(scriptProcessor);
  scriptProcessor.connect(audioCtx.destination); // 不回显但需要连接
}

function stopMic() {
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
}

// ═══════════════════════════════════════════════════
//  WebSocket
// ═══════════════════════════════════════════════════
function connectWS() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${location.host}/ws/voice`;

  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    els.wsIndicator.textContent = "🔗 WebSocket 已连接";
    ws.send(JSON.stringify({
      type: "control", action: "start",
      session_id: sessionId,
      config: {
        stage_id: els.stage.value,
        difficulty_id: els.difficulty.value,
        voice_id: els.voice.value,
      }
    }));
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      // 二进制音频 → 播放队列
      const int16 = new Int16Array(event.data);
      enqueueAudio(int16);
    } else {
      // JSON 消息
      try {
        const msg = JSON.parse(event.data);
        handleWSMessage(msg);
      } catch (e) { console.warn("bad json", e); }
    }
  };

  ws.onclose = () => {
    els.wsIndicator.textContent = "⚠️ WebSocket 断开";
    if (active) {
      setStatus("idle", "连接断开，尝试重连...");
      setTimeout(() => { if (active) connectWS(); }, 2000);
    }
  };

  ws.onerror = () => {
    els.wsIndicator.textContent = "❌ WebSocket 错误";
  };
}

function handleWSMessage(msg) {
  switch (msg.type) {
    case "ready":
      els.wsIndicator.textContent = `✅ 已就绪 (SR: ${msg.sample_rate_in}Hz → ${msg.sample_rate_out}Hz)`;
      break;
    case "status":
      setStatus(msg.stage || "idle", msg.detail || msg.stage || "");
      break;
    case "asr_final":
      els.interim.textContent = "";
      addBubble("user", "我", msg.text);
      break;
    case "response_text":
      addBubble("customer", "客户", msg.text);
      break;
    case "tts_done":
      setStatus("listening", "继续说话");
      break;
    case "session_end":
      setStatus("idle", `会话结束，共 ${msg.turn_count || 0} 轮`);
      active = false;
      els.actionBtn.textContent = "开始对话";
      els.actionBtn.classList.remove("stop");
      stopMic();
      break;
    case "error":
      setStatus("error", msg.message);
      break;
  }
}

// ═══════════════════════════════════════════════════
//  按钮
// ═══════════════════════════════════════════════════
els.actionBtn.addEventListener("click", async () => {
  if (active) {
    // 停止
    active = false;
    if (ws) ws.send(JSON.stringify({ type: "control", action: "end" }));
    stopMic();
    setStatus("idle", "已停止");
    els.actionBtn.textContent = "开始对话";
    els.actionBtn.classList.remove("stop");
    els.configSection.style.display = "block";
  } else {
    // 开始
    try {
      await unlockAudio();
      await startMic();
      connectWS();
      active = true;
      setStatus("listening", "正在听你说话...");
      els.actionBtn.textContent = "结束对话";
      els.actionBtn.classList.add("stop");
      els.configSection.style.display = "none";
    } catch (err) {
      setStatus("error", err.message || String(err));
      active = false;
    }
  }
});

// ═══════════════════════════════════════════════════
//  初始化
// ═══════════════════════════════════════════════════
async function init() {
  try {
    const res = await fetch("/api/training/config");
    const data = await res.json();
    fillSelect(els.stage, data.stages, data.defaults.stage_id);
    fillSelect(els.difficulty, data.difficulties, data.defaults.difficulty_id);
    fillSelect(els.voice, data.voices, data.defaults.voice_id);
  } catch (e) {
    console.warn("config load failed", e);
  }

  // 检测是否支持
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("error", "当前浏览器不支持麦克风。请使用 HTTPS 并在企微中打开。");
    els.actionBtn.disabled = true;
  }
}

init();
</script>
</body>
</html>
"""


@app.get("/realtime")
async def realtime_page():
    return HTMLResponse(REALTIME_HTML)


if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8511"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
