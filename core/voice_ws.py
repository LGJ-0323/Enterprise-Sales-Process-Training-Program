"""
voice_ws.py — WebSocket 实时语音处理模块

职责:
  1. /ws/audio-probe  — 最小验证探针（PCM 收发 + 回声测试）
  2. /ws/voice        — 完整流式语音管道（VAD → ASR → LLM → TTS → 流式下发）

依赖:
  · fastrtc_new_web 的 guardrails / prompt 构建 / ASR callback / TTS
  · 复用现有 YAML + JSONL 训练配置（不做重复实现）

VAD:
  · 第一版使用能量检测（RMS threshold），零额外依赖
  · 后续可升级 Silero VAD（桌面版已有参考）
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any

try:
    from .conversation_store import get_recent_memory, save_turn
    from .fastrtc_new_web import (
        _build_session_training_prompt,
        build_customer_guardrail_reply,
        build_role_guard_prompt,
        clean_customer_reply,
        is_hostile_or_confused_user_text,
        is_role_reversed_sales_reply,
        load_customer_profile,
        parse_customer_payload,
        sanitize_recent_memory,
        set_status,
        synthesize_with_retry,
    )
except ImportError:
    from conversation_store import get_recent_memory, save_turn
    from fastrtc_new_web import (
        _build_session_training_prompt,
        build_customer_guardrail_reply,
        build_role_guard_prompt,
        clean_customer_reply,
        is_hostile_or_confused_user_text,
        is_role_reversed_sales_reply,
        load_customer_profile,
        parse_customer_payload,
        sanitize_recent_memory,
        set_status,
        synthesize_with_retry,
    )

import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ── 常量 ────────────────────────────────────────────
SAMPLE_RATE_IN = int(os.getenv("WS_VOICE_SAMPLE_RATE_IN", "16000"))
SAMPLE_RATE_OUT = 24000  # TTS 输出采样率（cosyvoice PCM）
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
BYTES_PER_FRAME = SAMPLE_WIDTH * CHANNELS

# ── VAD 参数（能量检测） ─────────────────────────────
VAD_THRESHOLD = float(os.getenv("WS_VAD_THRESHOLD", "0.01"))       # RMS 阈值
VAD_MIN_SPEECH_MS = int(os.getenv("WS_VAD_MIN_SPEECH_MS", "200"))  # 最短语音
VAD_MIN_SILENCE_MS = int(os.getenv("WS_VAD_MIN_SILENCE_MS", "800"))# 最短静音=句结束
VAD_CHUNK_MS = int(os.getenv("WS_VAD_CHUNK_MS", "100"))            # 检测粒度

# ── ASR 参数 ─────────────────────────────────────────
ASR_TRAILING_SILENCE_MS = int(os.getenv("WS_ASR_TRAILING_SILENCE_MS", "600"))
ASR_CALLBACK_FRAME_BYTES = int(os.getenv("WS_ASR_CALLBACK_FRAME_BYTES", "6400"))
ASR_CALLBACK_FIRST_TIMEOUT_S = float(os.getenv("WS_ASR_CALLBACK_FIRST_TIMEOUT_S", "5.0"))
ASR_CALLBACK_GRACE_S = float(os.getenv("WS_ASR_CALLBACK_GRACE_S", "0.6"))


# ═══════════════════════════════════════════════════════
#  Simple VAD（能量检测）
# ═══════════════════════════════════════════════════════

class SimpleVAD:
    """基于 RMS 能量的轻量 VAD。

    无外部依赖，适合企微 H5 场景快速验证。
    后续可替换为 Silero VAD（桌面版已有参考实现）。
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        chunk_ms: int = VAD_CHUNK_MS,
        sample_rate: int = SAMPLE_RATE_IN,
    ):
        self.threshold = threshold
        self.min_speech_frames = max(1, min_speech_ms // chunk_ms)
        self.min_silence_frames = max(1, min_silence_ms // chunk_ms)
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

        self._buffer = bytearray()
        self._candidate_buffer = bytearray()
        self._utterance_buffer = bytearray()
        self._total_samples = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    # ── 内部 ──

    @staticmethod
    def _rms(pcm_bytes: bytes) -> float:
        """计算 16-bit PCM 的 RMS 能量。"""
        if len(pcm_bytes) < 2:
            return 0.0
        count = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])
        if not samples:
            return 0.0
        sum_sq = sum(float(s) * float(s) for s in samples)
        return math.sqrt(sum_sq / count) / 32768.0

    # ── 对外 ──

    def add_chunk(self, pcm_chunk: bytes) -> None:
        """累积 PCM 数据块。"""
        self._buffer.extend(pcm_chunk)
        self._total_samples += len(pcm_chunk) // BYTES_PER_FRAME

    def detect_events(self) -> list[str]:
        """扫描缓冲区，返回事件列表: ['speech_start', 'speech_end', ...]。

        一次性消费所有完整 chunk，未消费的数据留在 buffer 中。
        """
        events: list[str] = []
        chunk_bytes = self.chunk_samples * BYTES_PER_FRAME

        while len(self._buffer) >= chunk_bytes:
            chunk = bytes(self._buffer[:chunk_bytes])
            del self._buffer[:chunk_bytes]
            rms_val = self._rms(chunk)
            is_speech = rms_val > self.threshold

            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0
                if self._in_speech:
                    self._utterance_buffer.extend(chunk)
                else:
                    self._candidate_buffer.extend(chunk)
                    if self._speech_frames >= self.min_speech_frames:
                        self._in_speech = True
                        self._utterance_buffer.extend(self._candidate_buffer)
                        self._candidate_buffer.clear()
                        events.append("speech_start")
            else:
                if self._in_speech:
                    self._utterance_buffer.extend(chunk)
                    self._silence_frames += 1
                    if self._silence_frames >= self.min_silence_frames:
                        self._in_speech = False
                        self._speech_frames = 0
                        self._silence_frames = 0
                        events.append("speech_end")
                else:
                    self._speech_frames = 0
                    self._silence_frames = 0
                    self._candidate_buffer.clear()

        return events

    def extract_speech(self) -> bytes:
        """提取完整一句话的 PCM 数据（从 speech_start 到 speech_end），重置缓冲区。"""
        pcm = bytes(self._utterance_buffer)
        self.reset()
        return pcm

    def reset(self) -> None:
        """完全重置 VAD 状态和缓冲区。"""
        self._buffer.clear()
        self._candidate_buffer.clear()
        self._utterance_buffer.clear()
        self._total_samples = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    @property
    def is_speaking(self) -> bool:
        return self._in_speech


# ═══════════════════════════════════════════════════════
#  工具：PCM 生成 / 转换
# ═══════════════════════════════════════════════════════

def generate_sine_wave(
    freq: float = 440.0,
    duration_s: float = 0.5,
    sample_rate: int = SAMPLE_RATE_IN,
    amplitude: float = 0.3,
) -> bytes:
    """生成正弦波测试音。"""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        val = int(amplitude * 32767 * math.sin(2 * math.pi * freq * t))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(samples)}h", *samples)


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE_OUT) -> bytes:
    """PCM → WAV（用于调试本地保存，WebSocket 直接发 PCM）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def split_pcm_chunks(pcm_bytes: bytes, chunk_duration_ms: int = 100, sample_rate: int = SAMPLE_RATE_OUT) -> list[bytes]:
    """将 PCM 切分为固定时长的块（用于流式下发 TTS 音频）。"""
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunk_bytes = chunk_samples * BYTES_PER_FRAME
    chunks = []
    for i in range(0, len(pcm_bytes), chunk_bytes):
        chunks.append(pcm_bytes[i : i + chunk_bytes])
    return chunks


# ═══════════════════════════════════════════════════════
#  ASR 流式识别（Callback 模式）
# ═══════════════════════════════════════════════════════

class _StreamAsrCallback(RecognitionCallback):
    """DashScope ASR callback，用于流式 PCM → 实时文本。"""

    def __init__(self):
        self.first_final_event = asyncio.Event()
        self.complete_event = asyncio.Event()
        self.error_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.sentences: list[dict] = []
        self.latest_text = ""

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if not isinstance(sentence, dict):
            return
        text = (sentence.get("text") or "").strip()
        if text:
            self.latest_text = text
        if RecognitionResult.is_sentence_end(sentence):
            sid = sentence.get("sentence_id")
            if sid is None or all(s.get("sentence_id") != sid for s in self.sentences):
                self.sentences.append(sentence)
            self.first_final_event.set()

    def on_complete(self) -> None:
        self.complete_event.set()

    def on_error(self, result: RecognitionResult) -> None:
        self.error_event.set()

    def prompt(self) -> str:
        if self.sentences:
            ordered = sorted(self.sentences, key=lambda s: s.get("sentence_id") or 0)
            return " ".join(s.get("text", "") for s in ordered).strip()
        return self.latest_text.strip()


async def _recognize_pcm_streaming(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE_IN) -> str:
    """将 PCM 字节流传入 DashScope ASR callback，返回识别文本。"""
    loop = asyncio.get_running_loop()

    callback = _StreamAsrCallback()
    recognition = Recognition(
        model=os.getenv("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v2"),
        callback=callback,
        format="pcm",
        sample_rate=sample_rate,
    )

    # start() 是同步的，在 executor 中运行
    await loop.run_in_executor(
        None,
        lambda: recognition.start(
            max_sentence_silence=800,
            semantic_punctuation_enabled=False,
            multi_threshold_mode_enabled=True,
        ),
    )

    # 分帧发送音频数据
    def _send_frames():
        for i in range(0, len(pcm_bytes), ASR_CALLBACK_FRAME_BYTES):
            frame = pcm_bytes[i : i + ASR_CALLBACK_FRAME_BYTES]
            recognition.send_audio_frame(frame)

    await loop.run_in_executor(None, _send_frames)

    # 追加尾静音（帮助 ASR 判定句子结束）
    trailing_bytes = b"\x00" * (int(SAMPLE_RATE_IN * ASR_TRAILING_SILENCE_MS / 1000) * BYTES_PER_FRAME)
    recognition.send_audio_frame(trailing_bytes)

    # 停止识别
    await loop.run_in_executor(None, recognition.stop)

    # 等待第一个 final 句子
    try:
        await asyncio.wait_for(callback.first_final_event.wait(), ASR_CALLBACK_FIRST_TIMEOUT_S)
    except asyncio.TimeoutError:
        return callback.prompt()

    if callback.error_event.is_set():
        return callback.prompt()

    # Grace 等待更多句子
    try:
        await asyncio.wait_for(callback.complete_event.wait(), ASR_CALLBACK_GRACE_S)
    except asyncio.TimeoutError:
        pass

    return callback.prompt()


# ═══════════════════════════════════════════════════════
#  LLM 流式生成
# ═══════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    """按中文标点切句，保留标点在句尾。"""
    import re

    parts = re.split(r"(?<=[。！？；\n])", text)
    result = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result


async def _llm_stream(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
) -> str:
    """流式调用 Qwen，收集完整回复文本。"""
    loop = asyncio.get_running_loop()

    def _call():
        resp = Generation.call(
            model=model or os.getenv("DASHSCOPE_LLM_MODEL", "qwen-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format="message",
            stream=True,
            incremental_output=True,
        )
        result_text = ""
        for chunk in resp:
            if chunk.status_code == 200:
                try:
                    delta = chunk.output.choices[0].message.content
                    if delta:
                        result_text += delta
                except (AttributeError, IndexError):
                    pass
        return result_text

    return await loop.run_in_executor(None, _call)


# ═══════════════════════════════════════════════════════
#  语音会话管理
# ═══════════════════════════════════════════════════════

class VoiceSession:
    """管理一个 WebSocket 语音会话的完整状态。"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.config: dict[str, str] = {}
        self.vad = SimpleVAD()
        self.history: list[dict] = []
        self.tts_pending: asyncio.Queue[bytes] = asyncio.Queue()
        self._interrupted = False
        self._tts_task: asyncio.Task | None = None

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def interrupt(self) -> None:
        """打断当前 TTS 播放。"""
        self._interrupted = True
        # 清空播放队列
        while not self.tts_pending.empty():
            try:
                self.tts_pending.get_nowait()
            except asyncio.QueueEmpty:
                break

    def clear_interrupt(self) -> None:
        self._interrupted = False


# ═══════════════════════════════════════════════════════
#  主线：处理一句话的完整流水线
# ═══════════════════════════════════════════════════════

async def _process_utterance(
    session: VoiceSession,
    pcm_bytes: bytes,
    ws_send_json,
    ws_send_bytes,
) -> None:
    """ASR → LLM → TTS → WebSocket 流式下发。"""
    t0 = time.perf_counter()
    stage_id = session.config.get("stage_id", "cold_call")
    difficulty_id = session.config.get("difficulty_id", "easy")
    voice_id = session.config.get("voice_id", "longsanshu_v3")

    # ── Step 1: ASR ──
    user_text = await _recognize_pcm_streaming(pcm_bytes)
    asr_time = time.perf_counter() - t0

    if not user_text:
        await ws_send_json({
            "type": "status",
            "stage": "listening",
            "detail": "未识别到语音",
        })
        return

    await ws_send_json({"type": "asr_final", "text": user_text})
    await ws_send_json({"type": "status", "stage": "processing"})

    # ── Step 2: 构建 Prompt ──
    try:
        training_prompt, training_summary, session_context = _build_session_training_prompt(
            session.session_id, stage_id, difficulty_id
        )
    except Exception as exc:
        set_status("prompt_build_error", error=str(exc))
        training_prompt = load_customer_profile()
        training_summary = {"customer": "客户", "stage": stage_id, "difficulty": difficulty_id}

    voice_cfg = {}
    try:
        from .fastrtc_new_web import resolve_voice, resolve_avatar_for_customer
    except ImportError:
        from fastrtc_new_web import resolve_voice, resolve_avatar_for_customer

    voice_cfg = resolve_voice(voice_id)
    avatar_cfg = resolve_avatar_for_customer(training_summary.get("customer_id"), "auto")

    st = {
        **training_summary,
        "voice_id": voice_id,
        "voice": voice_cfg.get("label", voice_id),
        "session_id": session.session_id,
    }

    # 记忆
    raw_mem = get_recent_memory(session.session_id, limit=10)
    mem, _ = sanitize_recent_memory(raw_mem)
    mem_prompt = f"【会话记忆】\n{mem}" if mem else "【会话记忆】\n暂无历史对话。"
    guard_prompt = build_role_guard_prompt(training_summary.get("customer"))

    # ── Step 3: LLM ──
    system_content = (
        f"{load_customer_profile()}\n\n"
        f"{training_prompt}\n\n"
        f"{mem_prompt}\n\n"
        f"{guard_prompt}"
    )

    llm_t0 = time.perf_counter()
    raw_response = await _llm_stream(system_content, user_text)
    llm_time = time.perf_counter() - llm_t0

    # Guardrails
    response_text = clean_customer_reply(raw_response, training_summary.get("customer"))
    if not response_text:
        response_text = raw_response.strip() or "嗯，您先说说具体想怎么合作？"

    guardrail = ""
    if is_role_reversed_sales_reply(response_text):
        response_text = build_customer_guardrail_reply(user_text, st)
        guardrail = "role_reversed"
    elif is_hostile_or_confused_user_text(user_text):
        response_text = build_customer_guardrail_reply(user_text, st)
        guardrail = "hostile"

    await ws_send_json({
        "type": "response_text",
        "text": response_text,
        "raw_text": raw_response,
        "guardrail": guardrail or None,
    })

    # ── Step 4: TTS ──
    tts_t0 = time.perf_counter()
    try:
        audio_bytes = await asyncio.get_running_loop().run_in_executor(
            None, synthesize_with_retry, response_text, voice_cfg, 3
        )
    except Exception as exc:
        set_status("tts_error", error=str(exc))
        await ws_send_json({"type": "error", "message": f"TTS 失败: {exc}"})
        return

    tts_time = time.perf_counter() - tts_t0

    # 分块下发 TTS 音频（100ms 每块）
    audio_chunks = split_pcm_chunks(audio_bytes, chunk_duration_ms=100, sample_rate=SAMPLE_RATE_OUT)
    for chunk in audio_chunks:
        if session.interrupted:
            session.clear_interrupt()
            break
        await ws_send_bytes(chunk)
        await asyncio.sleep(0.02)  # 小块发送间隔，防拥塞

    await ws_send_json({"type": "tts_done"})

    # ── Step 5: 保存本轮 ──
    try:
        save_turn(
            session_id=session.session_id,
            user_text=user_text,
            assistant_text=response_text,
            training=st,
        )
    except Exception as exc:
        set_status("save_error", error=str(exc))

    session.history.append({"user_text": user_text, "assistant_text": response_text})

    total_time = time.perf_counter() - t0
    set_status(
        "turn_done",
        prompt=user_text,
        response_text=response_text,
        total_s=round(total_time, 2),
        asr_s=round(asr_time, 2),
        llm_s=round(llm_time, 2),
        tts_s=round(tts_time, 2),
    )

    await ws_send_json({
        "type": "status",
        "stage": "listening",
        "detail": "继续说话",
        "timing": {"total": round(total_time, 2), "asr": round(asr_time, 2), "llm": round(llm_time, 2), "tts": round(tts_time, 2)},
    })


# ═══════════════════════════════════════════════════════
#  WebSocket 端点
# ═══════════════════════════════════════════════════════

async def _iter_ws_messages(ws):
    """Yield mixed WebSocket text/binary messages for Starlette/FastAPI."""
    while True:
        raw = await ws.receive()
        if raw.get("type") == "websocket.disconnect":
            break
        if raw.get("bytes") is not None:
            yield raw["bytes"]
        elif raw.get("text") is not None:
            yield raw["text"]


async def handle_audio_probe(ws):
    """ /ws/audio-probe — 最小验证探针。

    功能:
      1. 接收 PCM chunks → 回传 440Hz 正弦波确认
      2. 支持 JSON 控制: {"action":"ping"} → {"type":"pong"}
      3. 验证 iOS/Android 企微 WebSocket + getUserMedia + AudioContext
    """
    await ws.accept()
    await ws.send_json({
        "type": "ready",
        "message": "音频探针已连接",
        "sample_rate": SAMPLE_RATE_IN,
        "channels": CHANNELS,
        "sample_width": SAMPLE_WIDTH,
    })

    pcm_total = 0
    try:
        async for msg in _iter_ws_messages(ws):
            if isinstance(msg, bytes):
                pcm_total += len(msg)
                # 每收到 800ms 的 PCM，回一个确认音
                threshold_bytes = int(SAMPLE_RATE_IN * 0.8) * BYTES_PER_FRAME
                if pcm_total >= threshold_bytes:
                    tone = generate_sine_wave(freq=440, duration_s=0.15, sample_rate=SAMPLE_RATE_IN, amplitude=0.15)
                    await ws.send_bytes(tone)
                    pcm_total = 0
                    await ws.send_json({"type": "ack", "pcm_received_bytes": pcm_total})

            elif isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                action = data.get("action", "")
                if action == "ping":
                    await ws.send_json({"type": "pong", "echo": data.get("payload", "")})
                elif action == "echo_test":
                    # 回显模式：接下来的 PCM 原样返回
                    await ws.send_json({"type": "echo_mode", "status": "on"})
                    echo_mode = True
                    while echo_mode:
                        try:
                            raw = await asyncio.wait_for(ws.receive(), timeout=10.0)
                        except asyncio.TimeoutError:
                            break
                        if raw.get("type") == "websocket.disconnect":
                            break
                        if raw.get("bytes") is not None:
                            await ws.send_bytes(raw["bytes"])
                        elif raw.get("text") is not None:
                            try:
                                ctrl = json.loads(raw["text"])
                                if ctrl.get("action") == "echo_off":
                                    echo_mode = False
                                    await ws.send_json({"type": "echo_mode", "status": "off"})
                            except json.JSONDecodeError:
                                pass
    except Exception as exc:
        print(f"[audio-probe] disconnected: {exc}")


async def handle_voice_ws(ws):
    """ /ws/voice — 完整流式语音管道。

    协议:
      客户端 → 服务端:
        · Binary: PCM 16-bit 16kHz mono
        · JSON:   {"type":"control","action":"start|pause|resume|end","config":{...}}

      服务端 → 客户端:
        · Binary: PCM 24kHz mono (TTS 音频)
        · JSON:   {"type":"status|asr_final|response_text|tts_done|error", ...}
    """
    await ws.accept()

    # 辅助发送函数（捕获断开异常）
    async def _send_json(data: dict):
        try:
            await ws.send_json(data)
        except Exception:
            pass

    async def _send_bytes(data: bytes):
        try:
            await ws.send_bytes(data)
        except Exception:
            pass

    await _send_json({
        "type": "ready",
        "sample_rate_in": SAMPLE_RATE_IN,
        "sample_rate_out": SAMPLE_RATE_OUT,
    })

    session: VoiceSession | None = None

    try:
        async for msg in _iter_ws_messages(ws):
            if isinstance(msg, bytes):
                if session is None:
                    continue  # 还没 start，忽略音频

                # 如果正在播放 TTS 且用户说话 → 打断
                if session.tts_pending.qsize() > 0 or session.interrupted:
                    # 但不要立即打断，先看是不是真的语音
                    pass

                session.vad.add_chunk(msg)
                events = session.vad.detect_events()

                if "speech_start" in events:
                    await _send_json({"type": "status", "stage": "listening", "detail": "检测到语音"})

                if session.interrupted and "speech_start" in events:
                    # 确认用户在说话，执行打断
                    session.interrupt()
                    await _send_json({"type": "status", "stage": "listening", "detail": "已打断，请继续"})

                if "speech_end" in events and not session.interrupted:
                    pcm = session.vad.extract_speech()
                    if len(pcm) > SAMPLE_RATE_IN * 0.3 * BYTES_PER_FRAME:  # > 300ms
                        await _send_json({"type": "status", "stage": "processing", "detail": "正在识别"})
                        asyncio.create_task(
                            _process_utterance(session, pcm, _send_json, _send_bytes)
                        )

            elif isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                action = data.get("action", "")

                if msg_type == "control":
                    if action == "start":
                        sid = data.get("session_id") or f"ws-{int(time.time() * 1000)}"
                        session = VoiceSession(sid)
                        session.config = data.get("config", {})
                        await _send_json({
                            "type": "status",
                            "stage": "listening",
                            "session_id": session.session_id,
                        })

                    elif action == "pause":
                        if session:
                            session.vad.reset()
                        await _send_json({"type": "status", "stage": "idle"})

                    elif action == "resume":
                        await _send_json({"type": "status", "stage": "listening"})

                    elif action == "end":
                        if session:
                            await _send_json({
                                "type": "session_end",
                                "turn_count": len(session.history),
                            })
                        await ws.close()
                        return

                elif msg_type == "interrupt" and session:
                    session.interrupt()
                    await _send_json({"type": "status", "stage": "listening", "detail": "已打断"})

    except Exception as exc:
        print(f"[voice-ws] disconnected: {exc}")
    finally:
        if session:
            session.vad.reset()
