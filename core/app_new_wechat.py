"""
app_new_wechat.py — 移动 H5 训练入口

FastAPI 应用，提供移动 H5 页面和后端接口：
- /mobile:          H5 Omni 实时语音页面（浏览器 PCM → Qwen-Omni → 文本 + 音频）
- /mobile/upload:   H5 录音上传兼容页面（浏览器录音 → 上传 → ASR → LLM → TTS → 播放）
- /realtime:        经典实时 WebSocket 语音模式（VAD → ASR → LLM → TTS）
- /api/training/config:   获取训练配置（阶段、难度、音色选项）
- /api/training/voice-turn: 处理录音上传，返回客户回复文本和音频
- /ws/audio-probe:  WebSocket 音频探针（最小验证）
- /ws/voice:        WebSocket 完整实时语音管道

与桌面版（app_new_web）的区别：
- 桌面版使用 WebRTC 实时流，适合 PC 浏览器
- 移动版优先使用 Omni WebSocket 代理，保留录音上传兼容入口
"""

from __future__ import annotations
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import uvicorn

import base64
import os
import random
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


try:
    from .voice_ws import handle_audio_probe, handle_voice_ws
    from .omni_voice_ws import handle_mobile_omni_ws
except ImportError:
    from voice_ws import handle_audio_probe, handle_voice_ws
    from omni_voice_ws import handle_mobile_omni_ws

try:
    from .conversation_store import get_session, get_session_turns
    from .fastrtc_new_web import (
        LAST_STATUS,
        pcm_to_wav_bytes,
        run_customer_turn,
        transcribe_uploaded_audio,
    )
    from .fastrtc_new_web import (
        DEFAULT_STAGE_ID,
        DEFAULT_DIFFICULTY_ID,
        DEFAULT_VOICE_ID,
        resolve_voice,
    )
    from .training_config import difficulty_choices, stage_choices, voice_choices
    from .training_config import resolve_training, _label
    from .case_loader import get_case, get_case_summary, find_cases, case_count as get_case_count, all_business_lines, rank_cases_best
    from .training_data_context import summarize_case_assets
    from .conversation_store import recent_completed_sessions
except ImportError:
    from conversation_store import get_session, get_session_turns
    from fastrtc_new_web import (
        LAST_STATUS,
        pcm_to_wav_bytes,
        run_customer_turn,
        transcribe_uploaded_audio,
    )
    from fastrtc_new_web import (
        DEFAULT_STAGE_ID,
        DEFAULT_DIFFICULTY_ID,
        DEFAULT_VOICE_ID,
        resolve_voice,
    )
    from training_config import difficulty_choices, stage_choices, voice_choices
    from training_config import resolve_training, _label
    from case_loader import get_case, get_case_summary, find_cases, case_count as get_case_count, all_business_lines, rank_cases_best
    from training_data_context import summarize_case_assets
    from conversation_store import recent_completed_sessions


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
INDUSTRY_CHOICES = [
    {"label": "🌐 不限", "value": ""},
    {"label": "🛒 跨境电商", "value": "cross_border_ecommerce"},
    {"label": "🏭 制造工厂", "value": "manufacturer"},
    {"label": "📦 贸易公司", "value": "trading_company"},
    {"label": "🧾 传统外贸", "value": "traditional_trade"},
    {"label": "🏬 海外仓/FBA", "value": "overseas_warehouse_fba"},
]
SUB_INDUSTRY_CHOICES = [
    {"label": "不限", "value": ""},
    {"label": "机械设备", "value": "machinery_equipment"},
    {"label": "食品饮料", "value": "food_beverage"},
    {"label": "玻璃设备", "value": "glass_equipment"},
    {"label": "家居建材", "value": "home_building_materials"},
    {"label": "电子电器", "value": "electronics"},
    {"label": "汽配零件", "value": "auto_parts"},
    {"label": "服装纺织", "value": "apparel_textile"},
    {"label": "医疗器械", "value": "medical_devices"},
    {"label": "化工原料", "value": "chemical_materials"},
    {"label": "户外用品", "value": "outdoor_goods"},
    {"label": "美妆个护", "value": "beauty_personal_care"},
]


def _render_template(name: str) -> HTMLResponse:
    """读取 core/templates/ 下的 HTML 模板并返回。"""
    path = TEMPLATES_DIR / f"{name}.html"
    if not path.exists():
        raise HTTPException(404, f"模板 {name} 不存在")
    return HTMLResponse(path.read_text(encoding="utf-8"))


# ── 配置白名单（防前端注入） ──────────────────────────
def _sanitize_config(raw: dict) -> dict:
    """只保留白名单内的字段，过滤前端传入的任意值。"""
    stages = {s[1] for s in stage_choices()}
    diffs = {d[1] for d in difficulty_choices()}
    voices = {v[1] for v in voice_choices()}
    blines = set(all_business_lines())
    industries = {item["value"] for item in INDUSTRY_CHOICES if item["value"]}
    sub_industries = {item["value"]
                      for item in SUB_INDUSTRY_CHOICES if item["value"]}
    clean = {}
    for key, valid_set in [
        ("stage_id", stages), ("difficulty_id", diffs), ("voice_id", voices),
            ("industry", industries), ("sub_industry", sub_industries),
            ("business_line", blines), ("case_id", None)]:
        val = str(raw.get(key, "")).strip()
        if val and (valid_set is None or val in valid_set):
            clean[key] = val
    return clean


app = FastAPI(title="Mobile H5 Training Prototype")


# 注：MOBILE_HTML / REALTIME_HTML 已迁移至 core/templates/ 目录
# 下方为历史遗留的内嵌 HTML 字符串，不再使用但保留以避免大范围行号偏移
MOBILE_HTML = """

    : root {
      color-scheme: light;
      --bg:  # f6f7f9;
      --panel:  # ffffff;
      --ink:  # 172033;
      --muted:  # 667085;
      --line:  # d9dee8;
      --brand:  # 007a78;
      --brand-dark:  # 065f5d;
      --warn:  # b42318;
      --soft:  # eef7f6;
    }
    * {box-sizing: border-box; }
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
      color:  # 344054;
    }
    select {
      width: 100 %;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 10px;
      font-size: 15px;
      background:  # fff;
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
      border-radius: 50 %;
      background: var(--brand);
      color:  # fff;
      font-size: 17px;
      font-weight: 750;
      box-shadow: 0 10px 24px rgba(0, 122, 120, 0.28);
      touch-action: manipulation;
    }
    .record-btn: disabled {
      opacity: .62;
      box-shadow: none;
    }
    .record-btn.recording {
      background: var(--warn);
      box-shadow: 0 10px 24px rgba(180, 35, 24, 0.25);
    }
    .play-btn {
      display: none;
      width: min(100 %, 260px);
      min-height: 46px;
      margin: 0 auto;
      border: 0;
      border-radius: 8px;
      background:  # 172033;
      color:  # fff;
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
      background:  # fff;
      text-align: left;
    }
    .bubble.user {
      background:  # f9fafb;
    }
    .bubble.customer {
      background: var(--soft);
      border-color:  # b8ddda;
    }
    .bubble-title {
      margin-bottom: 4px;
      font-size: 12px;
      font-weight: 760;
      color: var(--brand-dark);
    }
    audio {
      width: 100 %;
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
    const sessionId = `h5-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
    return _render_template("realtime")


@app.get("/mobile/upload")
async def mobile_upload_page():
    return _render_template("mobile")


@app.get("/mobile/compat")
async def mobile_compat_page():
    return _render_template("mobile_compat")


@app.get("/api/training/config")
async def training_config():
    return {
        "stages": choice_payload(stage_choices()),
        "difficulties": choice_payload(difficulty_choices()),
        "voices": choice_payload(voice_choices()),
        "industries": INDUSTRY_CHOICES,
        "sub_industries": SUB_INDUSTRY_CHOICES,
        "business_lines": all_business_lines(),
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
        prompt = transcribe_uploaded_audio(
            audio_bytes, audio.filename or "recording.webm")
        turn = run_customer_turn(
            prompt=prompt,
            stage_id=stage_id,
            difficulty_id=difficulty_id,
            voice_id=voice_id,
            session_id=session_id,
        )
        wav_bytes = pcm_to_wav_bytes(
            turn["audio_bytes"], int(turn.get("sample_rate") or 24000))
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
        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


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


# ── 训练资料 API ──────────────────────────────────────────

def _resolve_case_for_wechat(
    stage_id: str,
    difficulty_id: str,
    case_id: str | None = None,
    business_line: str | None = None,
):
    """解析训练配置并匹配案例（wechat 版轻量实现）。"""
    stage, customer, difficulty = resolve_training(
        stage_id, None, difficulty_id)
    case = get_case(case_id) if case_id else None
    if not case:
        candidates = find_cases(
            _label(stage), _label(difficulty), business_line)
        case = random.choice(candidates) if candidates else None
    return stage, customer, difficulty, case


@app.get("/api/persona")
async def api_persona(
    stage_id: str = "cold_call",
    difficulty_id: str = "easy",
    voice_id: str = "longsanshu_v3",
    business_line: str = "",
    case_id: str = "",
):
    try:
        stage, customer, difficulty, case = _resolve_case_for_wechat(
            stage_id,
            difficulty_id,
            case_id=case_id or None,
            business_line=business_line or None,
        )
        voice = resolve_voice(voice_id)
        assets = summarize_case_assets(case)
        return {
            "name": assets.get("customer_name") or customer.get("name", "?"),
            "role": assets.get("customer_role") or customer.get("role", ""),
            "style": assets.get("communication_style") or customer.get("attitude", {}).get("label", ""),
            "desc": assets.get("scene") or customer.get("attitude", {}).get("style", ""),
            "traits": assets.get("main_concerns") or [],
            "goal": stage.get("training_goal", ""),
            "voice_label": voice.get("label", voice_id),
            "stage": _label(stage),
            "difficulty": _label(difficulty),
            "case_fewshot": assets.get("few_shot_count", 0),
            "case_total": get_case_count(),
            "raw_call_id": assets.get("raw_call_id"),
            "rubric_id": assets.get("rubric_id"),
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/training-assets")
async def api_training_assets(
    stage_id: str = "cold_call",
    difficulty_id: str = "easy",
    business_line: str = "",
    case_id: str = "",
):
    try:
        _, _, _, case = _resolve_case_for_wechat(
            stage_id,
            difficulty_id,
            case_id=case_id or None,
            business_line=business_line or None,
        )
        return summarize_case_assets(case)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── WebSocket 端点 ─────────────────────────────────────────

@app.websocket("/ws/audio-probe")
async def ws_audio_probe(ws: WebSocket):
    """音频探针：验证 WebSocket + PCM 收发是否正常。"""
    await handle_audio_probe(ws)


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    """实时语音管道：VAD → ASR → LLM → TTS → 流式下发。"""
    await handle_voice_ws(ws)


@app.websocket("/ws/mobile-omni")
async def ws_mobile_omni(ws: WebSocket):
    """Mobile Omni realtime pipeline: browser PCM ↔ Qwen-Omni ↔ browser PCM."""
    await handle_mobile_omni_ws(ws)


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
  .bubble.customer.guardrail { border-color: #f59e0b; border-width: 2px; }
  .guardrail-note { font-size: 11px; color: #b45309; margin-top: 4px; font-weight: 650; }
  .bubble-title { margin-bottom: 4px; font-size: 12px; font-weight: 760; color: var(--brand-dark); }
  .ws-indicator { font-size: 11px; color: var(--muted); text-align: center; margin-top: 4px; }
  .interim { font-size: 13px; color: var(--muted); font-style: italic; padding: 4px 0; min-height: 20px; }
  .tts-indicator { font-size: 13px; color: #3538cd; font-weight: 650; padding: 2px 0; min-height: 20px; text-align: center; }
  .error { color: var(--warn); white-space: pre-wrap; font-size: 13px; }
  .tip-box { background: #fffaeb; border: 1px solid #fedf89; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #b54708; }
  /* ── 客户画像 + 训练目标卡片 ── */
  .persona-goal-card { display: none; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .persona-goal-card.show { display: block; }
  .card-toggle { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; cursor: pointer; user-select: none; border-bottom: 1px solid var(--line); }
  .card-toggle .label { font-size: 14px; font-weight: 750; color: var(--ink); }
  .card-toggle .arrow { font-size: 12px; color: var(--muted); transition: transform 0.2s; }
  .card-toggle.open .arrow { transform: rotate(180deg); }
  .card-body { display: none; padding: 12px 14px; }
  .card-body.show { display: block; }
  .persona-row { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 10px; }
  .persona-avatar { width: 44px; height: 44px; border-radius: 50%; background: var(--soft); border: 1px solid #b8ddda; display: grid; place-items: center; font-size: 18px; flex-shrink: 0; color: var(--brand-dark); }
  .persona-info { flex: 1; min-width: 0; }
  .persona-name { font-size: 16px; font-weight: 800; margin-bottom: 2px; }
  .persona-meta { font-size: 12px; color: var(--muted); line-height: 1.4; }
  .tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
  .tag-sm { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; background: #f1f5f9; color: #475569; }
  .goal-row { margin-top: 8px; padding-top: 10px; border-top: 1px solid var(--line); }
  .goal-text { font-size: 13px; color: var(--ink); line-height: 1.5; margin-bottom: 6px; }
  .must-list { font-size: 12px; color: var(--muted); line-height: 1.5; }
  .must-list span { display: inline-block; margin-right: 8px; }
  .must-list .do { color: #027a48; }
  .must-list .dont { color: #b42318; }
  /* ── 评分面板 ── */
  .scoring-panel { display: none; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
  .scoring-panel.show { display: block; }
  .score-header { display: flex; align-items: flex-end; gap: 10px; padding: 14px; }
  .score-big { font-size: 46px; font-weight: 900; line-height: 1; color: var(--brand-dark); }
  .score-label { font-size: 13px; color: var(--muted); padding-bottom: 6px; }
  .dim-bars { padding: 0 14px 14px; display: grid; gap: 8px; }
  .dim-row { display: grid; grid-template-columns: 80px 1fr 36px; gap: 8px; align-items: center; font-size: 12px; }
  .dim-name { color: #344054; font-weight: 650; }
  .dim-track { height: 6px; border-radius: 3px; background: #e8eef6; overflow: hidden; }
  .dim-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--brand), #2563eb); transition: width 0.4s; }
  .dim-pct { text-align: right; color: var(--muted); font-weight: 700; }
  .feedback-section { border-top: 1px solid var(--line); padding: 12px 14px; display: grid; gap: 8px; }
  .fb-item { display: flex; gap: 8px; font-size: 13px; line-height: 1.45; padding: 8px 10px; border-radius: 7px; }
  .fb-item.good { background: #ecfdf3; }
  .fb-item.warn { background: #fffaeb; }
  .fb-item.bad { background: #fef3f2; }
  .fb-icon { font-weight: 900; flex-shrink: 0; width: 18px; text-align: center; }
  /* ── State 变更 Toast ── */
  .state-toast { position: fixed; top: 18px; left: 50%; transform: translateX(-50%); padding: 8px 18px; border-radius: 20px; background: #172033; color: #fff; font-size: 13px; font-weight: 700; z-index: 999; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
  .state-toast.show { opacity: 1; }
  /* ── 会话信息栏 ── */
  .session-info { display: none; font-size: 11px; color: var(--muted); text-align: center; padding: 4px 0; }
  .session-info.show { display: block; }
  /* ── 训练记录 ── */
  .history-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin-top: 4px; }
  .history-panel .panel-header { padding: 10px 14px; border-bottom: 1px solid var(--line); font-size: 14px; font-weight: 750; color: var(--ink); display: flex; justify-content: space-between; }
  .history-panel .panel-header .reload-btn { font-size: 12px; color: var(--brand); cursor: pointer; background: none; border: none; }
  .history-item { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-bottom: 1px solid #f0f0f0; cursor: pointer; font-size: 13px; }
  .history-item:last-child { border-bottom: 0; }
  .history-item:hover { background: #f8fafb; }
  .history-item .hi-score { width: 36px; height: 36px; border-radius: 8px; display: grid; place-items: center; font-weight: 800; font-size: 14px; flex-shrink: 0; }
  .history-item .hi-score.good { background: #ecfdf3; color: #027a48; }
  .history-item .hi-score.mid { background: #fffaeb; color: #b54708; }
  .history-item .hi-score.low { background: #fef3f2; color: #b42318; }
  .history-item .hi-info { flex: 1; min-width: 0; }
  .history-item .hi-info .hi-title { font-weight: 700; margin-bottom: 2px; }
  .history-item .hi-info .hi-meta { font-size: 11px; color: var(--muted); }
  .history-empty { padding: 20px 14px; text-align: center; color: var(--muted); font-size: 13px; }
  /* ── 对话详情弹层 ── */
  .detail-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 1000; justify-content: center; align-items: flex-end; }
  .detail-overlay.show { display: flex; }
  .detail-sheet { width: 100%; max-width: 640px; max-height: 75vh; background: #fff; border-radius: 14px 14px 0 0; overflow-y: auto; padding: 16px; }
  .detail-sheet .detail-close { float: right; width: 28px; height: 28px; border-radius: 50%; border: 1px solid var(--line); background: #fff; font-size: 16px; cursor: pointer; line-height: 26px; text-align: center; }
  .detail-turn { padding: 8px 0; border-bottom: 1px solid #f0f0f0; font-size: 13px; line-height: 1.5; }
  .detail-turn .dt-role { font-weight: 750; color: var(--brand-dark); }
  .detail-turn .dt-text { margin-top: 2px; }</style>
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

  <!-- 客户画像 + 训练目标（服务端下发后展示） -->
  <section class="persona-goal-card" id="personaGoalCard">
    <div class="card-toggle" id="personaToggle" onclick="toggleCard('personaGoalCard','personaToggle','personaBody')">
      <span class="label">客户画像 & 训练目标</span>
      <span class="arrow">▼</span>
    </div>
    <div class="card-body show" id="personaBody">
      <div class="persona-row">
        <div class="persona-avatar" id="personaAvatar">🧑</div>
        <div class="persona-info">
          <div class="persona-name" id="personaName">等待中...</div>
          <div class="persona-meta" id="personaMeta"></div>
        </div>
      </div>
      <div class="tags" id="personaTags"></div>
      <div class="goal-row">
        <div class="goal-text" id="goalText"></div>
        <div class="must-list">
          <span class="do" id="goalMust"></span>
          <span class="dont" id="goalCritical"></span>
        </div>
      </div>
    </div>
  </section>

  <!-- ASR 中间文本 -->
  <div class="interim" id="interim"></div>
  <div class="tts-indicator" id="ttsIndicator"></div>

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

  <!-- 评分面板（每轮/终局） -->
  <section class="scoring-panel" id="scoringPanel">
    <div class="card-toggle" id="scoreToggle" onclick="toggleCard('scoringPanel','scoreToggle','scoreBody')">
      <span class="label">实时评分</span>
      <span class="arrow">▼</span>
    </div>
    <div class="card-body show" id="scoreBody">
      <div class="score-header">
        <div class="score-big" id="scoreBig">--</div>
        <div class="score-label" id="scoreLabel">等待训练</div>
      </div>
      <div class="dim-bars" id="dimBars"></div>
      <div class="feedback-section" id="feedbackSection"></div>
    </div>
  </section>

  <!-- 训练记录 -->
  <section class="history-panel" id="historyPanel">
    <div class="panel-header">
      <span>训练记录</span>
      <button class="reload-btn" onclick="loadHistory()">刷新</button>
    </div>
    <div id="historyList"><div class="history-empty">加载中...</div></div>
  </section>

  <!-- 对话详情弹层 -->
  <div class="detail-overlay" id="detailOverlay" onclick="if(event.target===this)closeDetail()">
    <div class="detail-sheet" id="detailSheet"></div>
  </div>

  <!-- 会话信息栏 -->
  <div class="session-info" id="sessionInfo"></div>

  <div class="tip-box" id="compatNote">
    💡 <strong>iOS 企微用户</strong>：如无法授权麦克风或音频卡顿，请切换到
    <a href="/mobile">对讲机模式</a>。
  </div>

  <!-- State 变更浮动提示 -->
  <div class="state-toast" id="stateToast"></div>
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
  // 新增 UI 元素
  personaGoalCard: $("personaGoalCard"), personaToggle: $("personaToggle"), personaBody: $("personaBody"),
  personaName: $("personaName"), personaMeta: $("personaMeta"), personaTags: $("personaTags"),
  goalText: $("goalText"), goalMust: $("goalMust"), goalCritical: $("goalCritical"),
  scoringPanel: $("scoringPanel"), scoreBig: $("scoreBig"), scoreLabel: $("scoreLabel"),
  dimBars: $("dimBars"), feedbackSection: $("feedbackSection"),
  stateToast: $("stateToast"), sessionInfo: $("sessionInfo"),
  ttsIndicator: $("ttsIndicator"),
};

// ═══════════════════════════════════════════════════
//  状态
// ═══════════════════════════════════════════════════
const REQUESTED_SAMPLE_RATE = 16000;
const CHUNK_MS = 100;
const CHUNK_SAMPLES = Math.floor(REQUESTED_SAMPLE_RATE * CHUNK_MS / 1000);
const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
const IS_WECOM = /wxwork/i.test(navigator.userAgent);
const USE_TTS_WHOLE_UTTERANCE = false;
const SCRIPT_BUFFER_SIZE = IS_IOS ? 1024 : 2048;
const WS_BUFFER_HIGH_WATERMARK = IS_IOS ? 262144 : 131072;
const TTS_SAMPLE_RATE = 24000;
const TTS_SCHEDULE_LEAD_S = IS_IOS ? 0.06 : 0.03;
const TTS_PREBUFFER_CHUNKS = (IS_IOS || IS_WECOM) ? 5 : 1;
const TTS_PREBUFFER_S = (IS_IOS || IS_WECOM) ? 0.75 : 0.0;
const CLIENT_RMS_GATE = IS_IOS ? 0.0045 : 0.0025;
const CLIENT_SPEECH_GATE = IS_IOS ? 0.009 : 0.0045;
const CLIENT_SILENCE_FLUSH_MS = IS_IOS ? 560 : 550;
const CLIENT_MIN_SPEECH_MS = IS_IOS ? 420 : 320;

let ws = null;
let audioCtx = null;
let stream = null;
let active = false;
let currentStage = "idle";
let lastLevelUpdateAt = 0;
let wsReconnectCount = 0;
const WS_MAX_RECONNECT = 5;
let errorRecoveryTimer = null;
let sessionId = `ws-${Date.now()}-${Math.random().toString(16).slice(2)}`;
let clientSpeechActive = false;
let clientFlushSent = false;
let clientSpeechStartedAt = 0;
let clientLastVoiceAt = 0;

function getInputSampleRate() {
  return Math.round((audioCtx && audioCtx.sampleRate) || REQUESTED_SAMPLE_RATE);
}

function resetClientSpeechState() {
  clientSpeechActive = false;
  clientFlushSent = false;
  clientSpeechStartedAt = 0;
  clientLastVoiceAt = 0;
}

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

function addBubble(type, title, text, extraClass) {
  const b = document.createElement("div");
  b.className = `bubble ${type}${extraClass ? " " + extraClass : ""}`;
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
//  面板折叠
// ═══════════════════════════════════════════════════
function toggleCard(cardId, toggleId, bodyId) {
  const toggle = document.getElementById(toggleId);
  const body = document.getElementById(bodyId);
  if (!toggle || !body) return;
  const open = body.classList.contains("show");
  if (open) { body.classList.remove("show"); toggle.classList.remove("open"); }
  else { body.classList.add("show"); toggle.classList.add("open"); }
}

// ═══════════════════════════════════════════════════
//  消息渲染函数
// ═══════════════════════════════════════════════════
function renderPersona(msg) {
  els.personaName.textContent = msg.name || "客户";
  const meta = [msg.role, msg.location, msg.style].filter(Boolean).join(" · ");
  els.personaMeta.textContent = meta || msg.stage_label || "";
  els.personaTags.innerHTML = (msg.concerns || []).map(c => `<span class="tag-sm">${c}</span>`).join("");
  els.personaGoalCard.classList.add("show");
}

function renderGoal(msg) {
  els.goalText.textContent = msg.goal || "";
  const must = (msg.must_do || []).slice(0, 3).map(s => `✅ ${s}`).join(" ");
  const critical = (msg.critical || []).slice(0, 2).map(s => `🚫 ${s}`).join(" ");
  els.goalMust.innerHTML = must;
  els.goalCritical.innerHTML = critical;
  // 场景化开场白：替换初始客户气泡
  if (msg.opening) {
    const firstBubble = els.conversation.querySelector(".bubble.customer");
    if (firstBubble) {
      const body = firstBubble.querySelector(".bubble-title");
      if (body && body.nextSibling) body.nextSibling.textContent = msg.opening;
    }
  }
}

function showStateChangeToast(msg) {
  const toast = els.stateToast;
  toast.textContent = `客户状态：${msg.from || "?"} → ${msg.to || "?"}`;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2500);
}

function renderEvaluation(msg) {
  const score = msg.score != null ? msg.score : "--";
  els.scoreBig.textContent = score;
  els.scoreLabel.textContent = msg.terminal ? (msg.is_success ? "会话成功" : "会话结束") : "本轮综合得分";
  // 维度进度条
  const dims = msg.dimensions || [];
  if (dims.length > 0) {
    els.dimBars.innerHTML = dims.map(d => {
      const pct = d.max ? Math.round(d.score * 100 / d.max) : d.score;
      return `<div class="dim-row"><span class="dim-name">${d.name}</span><div class="dim-track"><div class="dim-fill" style="width:${pct}%"></div></div><span class="dim-pct">${pct}%</span></div>`;
    }).join("");
  } else {
    els.dimBars.innerHTML = "";
  }
  // 文字反馈（LLM score_notes + triggered_events）
  const notes = msg.score_notes || {};
  const events = msg.triggered_events || [];
  const lines = [];
  Object.entries(notes).forEach(([k, v]) => { if (v) lines.push(`<strong>${k}</strong>：${v}`); });
  if (events.length > 0) lines.push(`触发：${events.join("、")}`);
  if (lines.length > 0 && dims.length === 0) {
    els.feedbackSection.innerHTML = lines.map(l => `<div class="fb-item note"><span>${l}</span></div>`).join("");
  }
  els.scoringPanel.classList.add("show");
}

function renderFinalEvaluation(msg) {
  renderEvaluation(msg);
  els.scoreLabel.textContent = msg.source === "llm" ? "AI 复盘得分 / 100" : "综合评分 / 100";
  // 教练反馈
  const fb = [];
  if (msg.strengths && msg.strengths.length) fb.push({ kind: "good", icon: "✓", text: msg.strengths[0] });
  if (msg.improvements && msg.improvements.length) fb.push({ kind: "warn", icon: "!", text: msg.improvements[0] });
  if (msg.summary) fb.push({ kind: "bad", icon: "✕", text: msg.summary });
  els.feedbackSection.innerHTML = fb.map(f =>
    `<div class="fb-item ${f.kind}"><span class="fb-icon">${f.icon}</span><span>${f.text}</span></div>`
  ).join("");
}

function updateSessionInfo(msg) {
  const parts = [];
  if (msg.turn_count != null) parts.push(`${msg.turn_count} 轮`);
  if (msg.current_state) parts.push(`状态: ${msg.current_state}`);
  if (msg.total_asr_s != null) parts.push(`ASR ${msg.total_asr_s.toFixed(1)}s`);
  if (msg.total_llm_s != null) parts.push(`LLM ${msg.total_llm_s.toFixed(1)}s`);
  els.sessionInfo.textContent = parts.join(" · ");
  els.sessionInfo.classList.add("show");
}

function resetPanels() {
  resetTtsPlayback(true);
  resetClientSpeechState();
  els.personaGoalCard.classList.remove("show");
  els.personaName.textContent = "等待中...";
  els.personaMeta.textContent = "";
  els.personaTags.innerHTML = "";
  els.goalText.textContent = "";
  els.goalMust.innerHTML = "";
  els.goalCritical.innerHTML = "";
  els.scoringPanel.classList.remove("show");
  els.scoreBig.textContent = "--";
  els.scoreLabel.textContent = "等待训练";
  els.dimBars.innerHTML = "";
  els.feedbackSection.innerHTML = "";
  els.sessionInfo.classList.remove("show");
  els.sessionInfo.textContent = "";
}

// ═══════════════════════════════════════════════════
//  音频播放队列
// ═══════════════════════════════════════════════════
const playQueue = [];
let playing = false;
let nextTtsStartAt = 0;
const activeTtsSources = new Set();
let ttsStreamOpen = false;
let ttsStreamDone = false;
const ttsWholeChunks = [];

function enqueueAudio(pcm16Buffer) {
  if (USE_TTS_WHOLE_UTTERANCE) {
    ttsWholeChunks.push(pcm16Buffer);
    updateTtsIndicator();
    return;
  }
  playQueue.push(pcm16Buffer);
  drainQueue();
}

function updateTtsIndicator() {
  if (playing) {
    els.ttsIndicator.textContent = "🔊 客户语音播放中";
    return;
  }
  if (ttsStreamOpen && !ttsStreamDone && (playQueue.length > 0 || ttsWholeChunks.length > 0)) {
    els.ttsIndicator.textContent = "🔊 客户语音缓冲中";
    return;
  }
  els.ttsIndicator.textContent = "";
}

let ttsGainNode = null;

function ensureTtsGainNode() {
  if (!audioCtx) return false;
  if (!ttsGainNode) {
    ttsGainNode = audioCtx.createGain();
    ttsGainNode.gain.value = 0.9;
    ttsGainNode.connect(audioCtx.destination);
  }
  return true;
}

function queuedTtsDurationS() {
  return playQueue.reduce((sum, buf) => sum + (buf.length / TTS_SAMPLE_RATE), 0);
}

function mergePcmChunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Int16Array(totalLength);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function playScheduledBuffer(int16Buffer) {
  if (!audioCtx || !ensureTtsGainNode() || int16Buffer.length === 0) return false;
  if (audioCtx.state === "suspended") {
    audioCtx.resume().catch((e) => console.warn("resume audioCtx failed", e));
  }
  const float32 = new Float32Array(int16Buffer.length);
  for (let i = 0; i < int16Buffer.length; i++) float32[i] = int16Buffer[i] / 32768;
  const audioBuffer = audioCtx.createBuffer(1, float32.length, TTS_SAMPLE_RATE);
  audioBuffer.getChannelData(0).set(float32);
  const src = audioCtx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(ttsGainNode);
  const startAt = Math.max(audioCtx.currentTime + TTS_SCHEDULE_LEAD_S, nextTtsStartAt || 0);
  nextTtsStartAt = startAt + audioBuffer.duration;
  playing = true;
  updateTtsIndicator();
  activeTtsSources.add(src);
  src.onended = () => {
    activeTtsSources.delete(src);
    if (activeTtsSources.size === 0) {
      playing = false;
      nextTtsStartAt = 0;
      if (ttsStreamDone) {
        ttsStreamOpen = false;
        ttsStreamDone = false;
        if (active) setStatus("listening", "继续说话");
      }
      updateTtsIndicator();
    }
  };
  src.start(startAt);
  return true;
}

function beginTtsStream() {
  resetTtsPlayback(true);
  ttsStreamOpen = true;
  ttsStreamDone = false;
  updateTtsIndicator();
}

function finishTtsStream() {
  ttsStreamDone = true;
  if (USE_TTS_WHOLE_UTTERANCE) {
    if (ttsWholeChunks.length > 0) {
      const merged = mergePcmChunks(ttsWholeChunks);
      ttsWholeChunks.length = 0;
      playScheduledBuffer(merged);
    } else if (!playing) {
      ttsStreamOpen = false;
      ttsStreamDone = false;
      if (active) setStatus("listening", "继续说话");
      updateTtsIndicator();
    }
    return;
  }
  drainQueue();
  if (!playing && playQueue.length === 0 && activeTtsSources.size === 0) {
    ttsStreamOpen = false;
    setStatus("listening", "继续说话");
    updateTtsIndicator();
  }
}

function resetTtsPlayback(hardStop = false) {
  if (hardStop) {
    activeTtsSources.forEach((src) => {
      try {
        src.onended = null;
        src.stop(0);
      } catch (e) {
        console.warn("stop playback error", e);
      }
    });
    activeTtsSources.clear();
  }
  playQueue.length = 0;
  ttsWholeChunks.length = 0;
  playing = false;
  nextTtsStartAt = 0;
  ttsStreamOpen = false;
  ttsStreamDone = false;
  updateTtsIndicator();
}

function drainQueue() {
  try {
    if (USE_TTS_WHOLE_UTTERANCE) {
      updateTtsIndicator();
      return;
    }
    if (!audioCtx || !ensureTtsGainNode()) return;
    if (audioCtx.state === "suspended") {
      audioCtx.resume().catch((e) => console.warn("resume audioCtx failed", e));
    }
    if (playQueue.length === 0) {
      if (activeTtsSources.size === 0) {
        playing = false;
        nextTtsStartAt = 0;
        if (ttsStreamDone) {
          ttsStreamOpen = false;
          ttsStreamDone = false;
          if (active) setStatus("listening", "继续说话");
        }
        updateTtsIndicator();
      }
      return;
    }
    if (!playing && !ttsStreamDone) {
      if (playQueue.length < TTS_PREBUFFER_CHUNKS && queuedTtsDurationS() < TTS_PREBUFFER_S) {
        updateTtsIndicator();
        return;
      }
    }
    const now = audioCtx.currentTime;
    if (!nextTtsStartAt || nextTtsStartAt < now + TTS_SCHEDULE_LEAD_S) {
      nextTtsStartAt = now + TTS_SCHEDULE_LEAD_S;
    }
    playing = true;
    updateTtsIndicator();
    while (playQueue.length > 0) {
      const buf = playQueue.shift();
      const float32 = new Float32Array(buf.length);
      for (let i = 0; i < buf.length; i++) float32[i] = buf[i] / 32768;
      const audioBuffer = audioCtx.createBuffer(1, float32.length, TTS_SAMPLE_RATE);
      audioBuffer.getChannelData(0).set(float32);
      const src = audioCtx.createBufferSource();
      src.buffer = audioBuffer;
      src.connect(ttsGainNode);
      const scheduledStartAt = nextTtsStartAt;
      nextTtsStartAt = scheduledStartAt + audioBuffer.duration;
      activeTtsSources.add(src);
      src.onended = () => {
        activeTtsSources.delete(src);
        if (activeTtsSources.size === 0 && playQueue.length === 0) {
          playing = false;
          nextTtsStartAt = 0;
          if (ttsStreamDone) {
            ttsStreamOpen = false;
            ttsStreamDone = false;
            if (active) setStatus("listening", "继续说话");
          }
          updateTtsIndicator();
        }
      };
      src.start(scheduledStartAt);
    }
  } catch (e) {
    console.warn("playback error", e);
    resetTtsPlayback(true);
  }
}

// ═══════════════════════════════════════════════════
//  AudioContext 解锁（iOS）
// ═══════════════════════════════════════════════════
async function unlockAudio() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC({ latencyHint: "interactive" });
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
let silentGainNode = null;

async function startMic() {
  // 先清理旧节点
  if (silentGainNode) { silentGainNode.disconnect(); silentGainNode = null; }

  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: REQUESTED_SAMPLE_RATE, channelCount: 1,
      echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    }
  });

  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC({ latencyHint: "interactive", sampleRate: REQUESTED_SAMPLE_RATE });
  }

  const source = audioCtx.createMediaStreamSource(stream);
  scriptProcessor = audioCtx.createScriptProcessor(SCRIPT_BUFFER_SIZE, 1, 1);

  scriptProcessor.onaudioprocess = (e) => {
    if (!active || !ws || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    // 计算 RMS + 噪声门限
    let sumSq = 0;
    for (let i = 0; i < input.length; i++) sumSq += input[i] * input[i];
    const rms = Math.sqrt(sumSq / input.length);
    const now = Date.now();
    if (rms >= CLIENT_RMS_GATE) {
      if (!clientSpeechActive) {
        clientSpeechActive = true;
        clientFlushSent = false;
        clientSpeechStartedAt = now;
      }
      if (rms >= CLIENT_SPEECH_GATE) {
        clientLastVoiceAt = now;
      }
    } else if (
      clientSpeechActive &&
      !clientFlushSent &&
      currentStage === "listening" &&
      now - clientLastVoiceAt >= CLIENT_SILENCE_FLUSH_MS &&
      now - clientSpeechStartedAt >= CLIENT_MIN_SPEECH_MS
    ) {
      clientFlushSent = true;
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "control", action: "flush" }));
      }
    }
    // 环境底噪跳过，减少服务端无效处理
    if (rms < CLIENT_RMS_GATE) return;
    // PCM 转换（稍降增益，减少削波）
    const gain = 0.85;
    const int16 = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      int16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768 * gain));
    }
    if (currentStage === "listening" && now - lastLevelUpdateAt > 250) {
      const bars = Math.min(10, Math.round(rms * 180));
      els.interim.textContent = bars > 0
        ? `麦克风输入 ${"▮".repeat(bars)}${"▯".repeat(10 - bars)}`
        : "正在听你说话...";
      lastLevelUpdateAt = now;
    }
    if (ws.bufferedAmount < WS_BUFFER_HIGH_WATERMARK) {
      ws.send(int16.buffer);
    }
  };

  source.connect(scriptProcessor);
  // 接到静音节点，保持 onaudioprocess 触发但不回放麦克风
  silentGainNode = audioCtx.createGain();
  silentGainNode.gain.value = 0;
  scriptProcessor.connect(silentGainNode);
  silentGainNode.connect(audioCtx.destination);
}

function stopMic() {
  resetClientSpeechState();
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  if (silentGainNode) {
    silentGainNode.disconnect();
    silentGainNode = null;
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
    wsReconnectCount = 0;
    resetClientSpeechState();
    ws.send(JSON.stringify({
      type: "control", action: "start",
      session_id: sessionId,
      config: {
        stage_id: els.stage.value,
        difficulty_id: els.difficulty.value,
        voice_id: els.voice.value,
        sample_rate: getInputSampleRate(),
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
    wsReconnectCount++;
    els.wsIndicator.textContent = `⚠️ WebSocket 断开 (${wsReconnectCount}/${WS_MAX_RECONNECT})`;
    if (active && wsReconnectCount <= WS_MAX_RECONNECT) {
      setStatus("idle", `连接断开，${2 * wsReconnectCount}s 后重连...`);
      setTimeout(() => { if (active) connectWS(); }, Math.min(2000 * wsReconnectCount, 10000));
    } else if (active) {
      setStatus("error", "连接失败，请点击重试");
      els.wsIndicator.innerHTML = '<button onclick="if(active){wsReconnectCount=0;connectWS();}" style="font-size:12px;padding:2px 8px;border:1px solid var(--line);border-radius:4px;background:#fff;cursor:pointer;">重新连接</button>';
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
      if (msg.stage === "processing" || msg.stage === "speaking" || msg.stage === "idle") {
        resetClientSpeechState();
      }
      if (msg.stage === "speaking") {
        beginTtsStream();
      }
      setStatus(msg.stage || "idle", msg.detail || msg.stage || "");
      break;
    case "persona":
      renderPersona(msg);
      break;
    case "goal":
      renderGoal(msg);
      break;
    case "asr_final":
      resetClientSpeechState();
      els.interim.textContent = "";
      addBubble("user", "我", msg.text);
      break;
    case "response_text":
      addBubble("customer", "客户", msg.text, msg.guardrail ? "guardrail" : null);
      if (msg.guardrail) {
        const note = document.createElement("div");
        note.className = "guardrail-note";
        note.textContent = "⚠ 系统已纠偏";
        els.conversation.lastChild.appendChild(note);
      }
      break;
    case "tts_done":
      finishTtsStream();
      break;
    case "evaluation":
      renderEvaluation(msg);
      break;
    case "final_evaluation":
      renderFinalEvaluation(msg);
      break;
    case "state_change":
      showStateChangeToast(msg);
      break;
    case "session_info":
      updateSessionInfo(msg);
      break;
    case "session_end":
      setStatus("idle", `会话结束，共 ${msg.turn_count || 0} 轮`);
      active = false;
      els.actionBtn.textContent = "开始对话";
      els.actionBtn.classList.remove("stop");
      stopMic();
      loadHistory();
      break;
    case "error":
      setStatus("error", msg.message);
      if (errorRecoveryTimer) clearTimeout(errorRecoveryTimer);
      errorRecoveryTimer = setTimeout(() => {
        if (active) setStatus("listening", "已恢复，可继续说");
      }, 3000);
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
    resetTtsPlayback(true);
    setStatus("idle", "已停止");
    els.actionBtn.textContent = "开始对话";
    els.actionBtn.classList.remove("stop");
    els.configSection.style.display = "block";
  } else {
    // 开始
    try {
      await unlockAudio();
      await startMic();
      resetPanels();
      sessionId = `ws-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
//  训练记录
// ═══════════════════════════════════════════════════
els.historyList = $("historyList");
els.detailOverlay = $("detailOverlay");
els.detailSheet = $("detailSheet");

function scoreClass(score) {
  if (score == null) return "low";
  if (score >= 80) return "good";
  if (score >= 60) return "mid";
  return "low";
}

async function loadHistory() {
  try {
    const res = await fetch("/api/training/history?limit=10");
    const data = await res.json();
    if (!data.length) {
      els.historyList.innerHTML = '<div class="history-empty">暂无训练记录，完成一次陪练后会自动出现在这里</div>';
      return;
    }
    els.historyList.innerHTML = data.map(s => {
      const sc = s.score != null ? s.score : "?";
      const cls = scoreClass(s.score);
      const time = (s.completed_at || "").slice(0, 10) || "—";
      const result = s.is_success ? "✓ 成功" : s.final_state || "已结束";
      return `<div class="history-item" onclick="showSessionDetail('${s.session_id}')">
        <div class="hi-score ${cls}">${sc}</div>
        <div class="hi-info">
          <div class="hi-title">${s.stage || "训练"} · ${s.difficulty || ""}</div>
          <div class="hi-meta">${time} · ${s.turn_count || 0} 轮 · ${result}</div>
        </div>
      </div>`;
    }).join("");
  } catch (e) {
    els.historyList.innerHTML = '<div class="history-empty">加载失败，请刷新</div>';
  }
}

async function showSessionDetail(sessionId) {
  els.detailSheet.textContent = "";
  const loading = document.createElement("div");
  loading.className = "history-empty";
  loading.textContent = "加载中...";
  els.detailSheet.appendChild(loading);
  els.detailOverlay.classList.add("show");
  try {
    const res = await fetch(`/api/training/session-turns?session_id=${encodeURIComponent(sessionId)}`);
    const turns = await res.json();
    els.detailSheet.textContent = "";
    // 关闭按钮
    const closeBtn = document.createElement("button");
    closeBtn.className = "detail-close";
    closeBtn.textContent = "✕";
    closeBtn.onclick = closeDetail;
    els.detailSheet.appendChild(closeBtn);
    // 标题
    const title = document.createElement("h3");
    title.style.cssText = "margin:0 0 10px;font-size:16px";
    title.textContent = "对话详情";
    els.detailSheet.appendChild(title);
    // 逐轮渲染（textContent 防 XSS）
    turns.forEach(t => {
      const turnDiv = document.createElement("div");
      turnDiv.className = "detail-turn";
      const roleDiv = document.createElement("div");
      roleDiv.className = "dt-role";
      roleDiv.textContent = `第${t.turn_index || "?"}轮`;
      turnDiv.appendChild(roleDiv);
      if (t.user_text) {
        const userDiv = document.createElement("div");
        userDiv.className = "dt-text";
        userDiv.textContent = `🗣 我：${t.user_text}`;
        turnDiv.appendChild(userDiv);
      }
      if (t.assistant_text) {
        const asstDiv = document.createElement("div");
        asstDiv.className = "dt-text";
        asstDiv.textContent = `🤖 客户：${t.assistant_text}`;
        turnDiv.appendChild(asstDiv);
      }
      els.detailSheet.appendChild(turnDiv);
    });
    if (!turns.length) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.textContent = "暂无对话记录";
      els.detailSheet.appendChild(empty);
    }
  } catch (e) {
    els.detailSheet.textContent = "";
    const closeBtn = document.createElement("button");
    closeBtn.className = "detail-close";
    closeBtn.textContent = "✕";
    closeBtn.onclick = closeDetail;
    els.detailSheet.appendChild(closeBtn);
    const err = document.createElement("div");
    err.className = "history-empty";
    err.textContent = "加载失败";
    els.detailSheet.appendChild(err);
  }
}

function closeDetail() {
  els.detailOverlay.classList.remove("show");
}

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

  // 加载训练记录
  loadHistory();
}

init();
</script>
</body>
</html>
"""


@app.get("/realtime")
async def realtime_page():
    return _render_template("realtime")


@app.get("/api/training/history")
async def training_history(limit: int = 10):
    sessions = recent_completed_sessions(limit)
    return [
        {
            "session_id": s.get("session_id"),
            "stage": s.get("stage_id", ""),
            "difficulty": s.get("difficulty_id", ""),
            "turn_count": s.get("turn_count", 0),
            "score": (s.get("evaluation") or {}).get("total_score"),
            "final_state": s.get("final_state", ""),
            "is_success": bool(s.get("is_success")),
            "completed_at": s.get("completed_at", ""),
        }
        for s in sessions
    ]


@app.get("/api/training/session-turns")
async def session_turns_api(session_id: str):
    turns = get_session_turns(session_id)
    return [
        {
            "turn_index": t.get("turn_index"),
            "user_text": t.get("user_text"),
            "assistant_text": t.get("assistant_text"),
            "created_at": t.get("created_at"),
        }
        for t in turns
    ]


@app.get("/api/training/resolve-case")
async def resolve_case(stage_id: str = "", difficulty_id: str = "", business_line: str = "", industry: str = "", sub_industry: str = ""):
    cfg = _sanitize_config(
        {"stage_id": stage_id, "difficulty_id": difficulty_id, "business_line": business_line,
         "industry": industry, "sub_industry": sub_industry})
    stage_id = cfg.get("stage_id") or os.getenv(
        "TRAINING_STAGE_ID", "cold_call")
    difficulty_id = cfg.get("difficulty_id") or os.getenv(
        "TRAINING_DIFFICULTY_ID", "easy")
    business_line = cfg.get("business_line", "")
    industry = cfg.get("industry", "")
    sub_industry = cfg.get("sub_industry", "")
    result = rank_cases_best(stage_id, difficulty_id,
                             business_line, industry, sub_industry)
    return result or {"case_id": None, "candidate_count": 0, "score": 0, "score_breakdown": {}, "top_candidates": [], "persona": {}}


@app.get("/api/training/session-summary")
async def session_summary(session_id: str = ""):
    if not session_id:
        raise HTTPException(400, "缺少 session_id")
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在")
    turns = get_session_turns(session_id)
    case = get_case(sess.get("case_id")) if sess.get("case_id") else None
    evaluation = sess.get("evaluation") or {}
    dim_scores = evaluation.get("dimension_scores", []) if isinstance(
        evaluation, dict) else []
    return {
        "session": {"session_id": sess.get("session_id"), "turn_count": sess.get("turn_count"),
                    "completed_at": sess.get("completed_at"), "final_state": sess.get("final_state"),
                    "is_success": bool(sess.get("is_success"))},
        "case": {"case_id": (case or {}).get("case_id"), "scene": (case or {}).get("scene"),
                 "customer_name": ((case or {}).get("customer_role_card") or {}).get("name", ""),
                 "training_type": sess.get("stage_id", ""), "difficulty": sess.get("difficulty_id", "")},
        "evaluation": {"total_score": evaluation.get("total_score") if isinstance(evaluation, dict) else None,
                       "dimension_scores": dim_scores,
                       "strengths": evaluation.get("strengths", []) if isinstance(evaluation, dict) else [],
                       "improvements": evaluation.get("improvements", []) if isinstance(evaluation, dict) else []},
        "turns": [{"turn_index": t.get("turn_index"), "user_text": t.get("user_text"),
                   "assistant_text": t.get("assistant_text"), "created_at": t.get("created_at")} for t in turns],
    }


# ── 三页面架构路由 ──────────────────────────────────
@app.get("/wechat/config")
async def wechat_config_page():
    return _render_template("wechat_config")


@app.get("/wechat/chat")
async def wechat_chat_page():
    return _render_template("wechat_chat")


@app.get("/wechat/score")
async def wechat_score_page():
    return _render_template("wechat_score")


if __name__ == "__main__":
    host = os.getenv("WECHAT_APP_HOST", "127.0.0.1")
    port = int(os.getenv("WECHAT_APP_PORT", "8511"))
    print(f"\n  http://{host}:{port}/wechat/config   |  三页面 — 训练配置")
    print(f"  http://{host}:{port}/wechat/chat     |  三页面 — 对话训练")
    print(f"  http://{host}:{port}/wechat/score    |  三页面 — 训练评分")
    print(f"  http://{host}:{port}/mobile          |  H5 Omni 实时语音")
    print(f"  http://{host}:{port}/mobile/upload   |  H5 录音上传兼容模式")
    print(f"  http://{host}:{port}/realtime        |  经典 ASR/LLM/TTS 实时语音")
    print(f"  http://{host}:{port}/debug/status    |  调试状态\n")
    uvicorn.run(app, host=host, port=port, reload=False)
