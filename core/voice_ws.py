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
import audioop
import io
import json
import math
import os
import re
import struct
import time
import wave
from pathlib import Path
from typing import Any

try:
    from .fastrtc_new_web import LAST_STATUS as WEB_LAST_STATUS, run_customer_turn, set_status
    from .fastrtc_new_web import _build_session_training_prompt, _choice_label
    from .case_loader import get_case
    from .training_data_context import summarize_case_assets, find_rubric
    from .training_config import load_stages, stage_choices, difficulty_choices
    from .conversation_store import get_session_turns, end_session, save_session_snapshot
    from .training_evaluator import _heuristic_evaluation
except ImportError:
    from fastrtc_new_web import LAST_STATUS as WEB_LAST_STATUS, run_customer_turn, set_status
    from fastrtc_new_web import _build_session_training_prompt, _choice_label
    from case_loader import get_case
    from training_data_context import summarize_case_assets, find_rubric
    from training_config import load_stages, stage_choices, difficulty_choices
    from conversation_store import get_session_turns, end_session, save_session_snapshot
    from training_evaluator import _heuristic_evaluation

import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ── 常量 ────────────────────────────────────────────
SAMPLE_RATE_IN = int(os.getenv("WS_VOICE_SAMPLE_RATE_IN", "16000"))
ASR_SAMPLE_RATE = int(os.getenv("WS_ASR_SAMPLE_RATE", "16000"))
SAMPLE_RATE_OUT = 24000  # TTS 输出采样率（cosyvoice PCM）
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
BYTES_PER_FRAME = SAMPLE_WIDTH * CHANNELS

# ── VAD 参数（能量检测） ─────────────────────────────
VAD_THRESHOLD = float(os.getenv("WS_VAD_THRESHOLD", "0.024"))       # RMS 阈值
VAD_MIN_SPEECH_MS = int(os.getenv("WS_VAD_MIN_SPEECH_MS", "320"))  # 最短语音
VAD_MIN_SILENCE_MS = int(
    os.getenv("WS_VAD_MIN_SILENCE_MS", "850"))  # 最短静音=句结束
VAD_CHUNK_MS = int(os.getenv("WS_VAD_CHUNK_MS", "100"))            # 检测粒度
VAD_MAX_UTTERANCE_MS = int(
    os.getenv("WS_VAD_MAX_UTTERANCE_MS", "5200"))  # 最长一句，超时强制切句
MIN_UTTERANCE_MS = int(os.getenv("WS_MIN_UTTERANCE_MS", "650"))
MIN_AUDIO_RMS = float(os.getenv("WS_MIN_AUDIO_RMS", "0.010"))

# ── ASR 参数 ─────────────────────────────────────────
ASR_TRAILING_SILENCE_MS = int(os.getenv("WS_ASR_TRAILING_SILENCE_MS", "700"))
ASR_MAX_SENTENCE_SILENCE_MS = int(
    os.getenv("WS_ASR_MAX_SENTENCE_SILENCE_MS", "900"))
ASR_CALLBACK_FRAME_BYTES = int(
    os.getenv("WS_ASR_CALLBACK_FRAME_BYTES", "3200"))
ASR_CALLBACK_FIRST_TIMEOUT_S = float(
    os.getenv("WS_ASR_CALLBACK_FIRST_TIMEOUT_S", "3.5"))
ASR_CALLBACK_GRACE_S = float(os.getenv("WS_ASR_CALLBACK_GRACE_S", "0.18"))
TTS_STREAM_CHUNK_MS = int(os.getenv("WS_TTS_STREAM_CHUNK_MS", "240"))
TTS_STREAM_CHUNK_PAUSE_MS = float(
    os.getenv("WS_TTS_STREAM_CHUNK_PAUSE_MS", "0.005"))


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
        max_utterance_ms: int = VAD_MAX_UTTERANCE_MS,
        chunk_ms: int = VAD_CHUNK_MS,
        sample_rate: int = SAMPLE_RATE_IN,
    ):
        self.threshold = threshold
        self.min_speech_frames = max(1, min_speech_ms // chunk_ms)
        self.min_silence_frames = max(1, min_silence_ms // chunk_ms)
        self.max_utterance_frames = max(1, max_utterance_ms // chunk_ms)
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

        self._buffer = bytearray()
        self._candidate_buffer = bytearray()
        self._utterance_buffer = bytearray()
        self._total_samples = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._utterance_frames = 0
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
                        self._utterance_frames = max(
                            1, len(self._utterance_buffer) // chunk_bytes)
                        self._candidate_buffer.clear()
                        events.append("speech_start")
            else:
                if self._in_speech:
                    self._utterance_buffer.extend(chunk)
                    self._utterance_frames += 1
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

            if self._in_speech and is_speech:
                self._utterance_frames += 1
                if self._utterance_frames >= self.max_utterance_frames:
                    self._in_speech = False
                    self._speech_frames = 0
                    self._silence_frames = 0
                    events.append("speech_end_forced")

        return events

    def extract_speech(self) -> bytes:
        """提取完整一句话的 PCM 数据（从 speech_start 到 speech_end），重置缓冲区。"""
        pcm = bytes(self._utterance_buffer)
        self.reset()
        return pcm

    @property
    def has_pending_utterance(self) -> bool:
        return bool(self._utterance_buffer)

    def reset(self) -> None:
        """完全重置 VAD 状态和缓冲区。"""
        self._buffer.clear()
        self._candidate_buffer.clear()
        self._utterance_buffer.clear()
        self._total_samples = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._utterance_frames = 0
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
        chunks.append(pcm_bytes[i: i + chunk_bytes])
    return chunks


def _coerce_sample_rate(value: Any, default: int = SAMPLE_RATE_IN) -> int:
    try:
        sample_rate = int(float(value))
    except (TypeError, ValueError):
        return default
    return sample_rate if 8000 <= sample_rate <= 96000 else default


def _pcm_duration_ms(pcm_bytes: bytes, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return len(pcm_bytes) / BYTES_PER_FRAME / sample_rate * 1000


def _pcm_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < BYTES_PER_FRAME:
        return 0.0
    try:
        return audioop.rms(pcm_bytes, SAMPLE_WIDTH) / 32768.0
    except audioop.error:
        return 0.0


def _resample_pcm(pcm_bytes: bytes, from_rate: int, to_rate: int = ASR_SAMPLE_RATE) -> bytes:
    if not pcm_bytes or from_rate == to_rate:
        return pcm_bytes
    converted, _ = audioop.ratecv(
        pcm_bytes, SAMPLE_WIDTH, CHANNELS, from_rate, to_rate, None)
    return converted


def _is_probable_asr_hallucination(text: str, duration_ms: float, rms: float) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return True
    compact = re.sub(r"[\s。！？.!?,，、…]+", "", cleaned).lower()
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    latin_words = re.findall(r"[a-zA-Z]+", cleaned)
    known_noise = {
        "merci",
        "oh",
        "wow",
        "the",
        "huh",
        "um",
        "uh",
        "嗯",
        "啊",
    }
    if compact in known_noise and duration_ms < 1800:
        return True
    if "sous titrage" in cleaned.lower():
        return True
    if cjk_count == 0 and len(latin_words) <= 3 and duration_ms < 2500:
        return True
    if duration_ms < MIN_UTTERANCE_MS or rms < MIN_AUDIO_RMS:
        return True
    return False


# ═══════════════════════════════════════════════════════
#  ASR 流式识别（Callback 模式）
# ═══════════════════════════════════════════════════════

class _StreamAsrCallback(RecognitionCallback):
    """DashScope ASR callback，用于流式 PCM → 实时文本。"""

    def __init__(self, loop: asyncio.AbstractEventLoop, on_partial=None):
        self.first_final_event = asyncio.Event()
        self.complete_event = asyncio.Event()
        self.error_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.sentences: list[dict] = []
        self.latest_text = ""
        self.last_partial_text = ""
        self.loop = loop
        self.on_partial = on_partial

    def _emit_partial(self, text: str) -> None:
        if not text or text == self.last_partial_text or self.on_partial is None:
            return
        self.last_partial_text = text

        def _schedule_partial() -> None:
            try:
                result = self.on_partial(text)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                pass

        self.loop.call_soon_threadsafe(_schedule_partial)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if not isinstance(sentence, dict):
            return
        text = (sentence.get("text") or "").strip()
        if text:
            self.latest_text = text
            self._emit_partial(text)
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
            ordered = sorted(
                self.sentences, key=lambda s: s.get("sentence_id") or 0)
            return " ".join(s.get("text", "") for s in ordered).strip()
        return self.latest_text.strip()


async def _recognize_pcm_streaming(
    pcm_bytes: bytes,
    sample_rate: int = ASR_SAMPLE_RATE,
    on_partial=None,
) -> str:
    """将 PCM 字节流传入 DashScope ASR callback，返回识别文本。"""
    loop = asyncio.get_running_loop()

    callback = _StreamAsrCallback(loop, on_partial=on_partial)
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
            max_sentence_silence=ASR_MAX_SENTENCE_SILENCE_MS,
            semantic_punctuation_enabled=False,
            multi_threshold_mode_enabled=True,
        ),
    )

    # 分帧发送音频数据
    def _send_frames():
        for i in range(0, len(pcm_bytes), ASR_CALLBACK_FRAME_BYTES):
            frame = pcm_bytes[i: i + ASR_CALLBACK_FRAME_BYTES]
            recognition.send_audio_frame(frame)

    await loop.run_in_executor(None, _send_frames)

    # 追加尾静音（帮助 ASR 判定句子结束）
    trailing_bytes = b"\x00" * \
        (int(sample_rate * ASR_TRAILING_SILENCE_MS / 1000) * BYTES_PER_FRAME)
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
#  语音会话管理
# ═══════════════════════════════════════════════════════

class VoiceSession:
    """管理一个 WebSocket 语音会话的完整状态。"""

    def __init__(self, session_id: str, sample_rate: int = SAMPLE_RATE_IN):
        self.session_id = session_id
        self.sample_rate = _coerce_sample_rate(sample_rate)
        self.config: dict[str, str] = {}
        self.vad = SimpleVAD(sample_rate=self.sample_rate)
        self.history: list[dict] = []
        self._interrupted = False
        self.processing_lock = asyncio.Lock()

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def interrupt(self) -> None:
        """打断当前 TTS 播放。"""
        self._interrupted = True

    def clear_interrupt(self) -> None:
        self._interrupted = False


# ═══════════════════════════════════════════════════════
#  核心：处理一句话的完整流水线（委托给 run_customer_turn）
# ═══════════════════════════════════════════════════════

def _trainer_info_from_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trainer_user_id",
        "trainer_external_user_id",
        "trainer_name",
        "trainer_department",
        "trainer_source",
        "trainer_avatar_url",
        "user_id",
        "external_user_id",
        "wecom_userid",
        "wecom_user_id",
        "display_name",
        "user_name",
        "department",
        "department_name",
        "avatar_url",
        "source",
    )
    info = {key: config.get(key) for key in keys if config.get(key)}
    trainer = config.get("trainer")
    if isinstance(trainer, dict):
        info["trainer"] = trainer
    return info


async def _process_utterance_safe(
    session: VoiceSession,
    pcm_bytes: bytes,
    ws_send_json,
    ws_send_bytes,
) -> None:
    """带 processing lock 的 utterance 处理入口，防止并发推进 state。"""
    async with session.processing_lock:
        await _process_utterance(session, pcm_bytes, ws_send_json, ws_send_bytes)


async def _process_utterance(
    session: VoiceSession,
    pcm_bytes: bytes,
    ws_send_json,
    ws_send_bytes,
) -> None:
    """ASR → run_customer_turn（完整训练语义）→ WebSocket 流式下发。

    run_customer_turn 包含:
      · _build_session_training_prompt → case + state machine 注入
      · build_customer_quality_prompt  → 防复读 + 客户智能要求
      · qwen3.6-plus + hyperparameters
      · clean_customer_reply + refine_customer_reply
      · parse_customer_payload → advance_session_state
      · save_turn + _maybe_evaluate_completed_session
      · TTS (synthesize_with_retry)

    本函数只负责:
      · ASR 识别 PCM → 文本
      · 把文本丢给 run_customer_turn（线程池中同步执行）
      · 把返回的 TTS 音频分块下发 WebSocket
      · 转发 status/response_text/evaluation 等消息
    """
    t0 = time.perf_counter()

    # ── Step 1: ASR ──
    input_sample_rate = session.sample_rate
    audio_duration_ms = _pcm_duration_ms(pcm_bytes, input_sample_rate)
    audio_rms = _pcm_rms(pcm_bytes)
    if audio_duration_ms < MIN_UTTERANCE_MS or audio_rms < MIN_AUDIO_RMS:
        set_status(
            "ws_audio_ignored",
            audio_duration_ms=round(audio_duration_ms, 1),
            audio_rms=round(audio_rms, 5),
            sample_rate=input_sample_rate,
        )
        await ws_send_json({"type": "status", "stage": "listening", "detail": "未识别到有效语音"})
        return

    try:
        asr_pcm = _resample_pcm(pcm_bytes, input_sample_rate, ASR_SAMPLE_RATE)
    except audioop.error as exc:
        set_status("ws_resample_error", error=str(
            exc), sample_rate=input_sample_rate)
        await ws_send_json({"type": "status", "stage": "listening", "detail": "音频格式异常，请再说一遍"})
        return

    user_text = await _recognize_pcm_streaming(
        asr_pcm,
        sample_rate=ASR_SAMPLE_RATE,
        on_partial=lambda text: ws_send_json(
            {"type": "asr_partial", "text": text}),
    )
    asr_time = time.perf_counter() - t0

    if not user_text or _is_probable_asr_hallucination(user_text, audio_duration_ms, audio_rms):
        set_status(
            "ws_asr_ignored",
            prompt=user_text,
            audio_duration_ms=round(audio_duration_ms, 1),
            audio_rms=round(audio_rms, 5),
            input_sample_rate=input_sample_rate,
            asr_sample_rate=ASR_SAMPLE_RATE,
        )
        await ws_send_json({"type": "status", "stage": "listening", "detail": "未识别到语音"})
        return

    await ws_send_json({"type": "asr_final", "text": user_text})
    await ws_send_json({"type": "status", "stage": "processing", "detail": "正在生成客户回复"})

    # ── Step 2: 完整训练管道（线程池中同步执行） ──
    stage_id = session.config.get("stage_id", "cold_call")
    difficulty_id = session.config.get("difficulty_id", "easy")
    voice_id = session.config.get("voice_id", "longsanshu_v3")
    avatar_id = session.config.get("avatar_id", "auto")
    business_line = session.config.get("business_line")
    preferred_case_id = session.config.get("case_id")
    trainer_info = _trainer_info_from_config(session.config)

    loop = asyncio.get_running_loop()
    try:
        turn = await loop.run_in_executor(
            None,
            lambda: run_customer_turn(
                prompt=user_text,
                stage_id=stage_id,
                difficulty_id=difficulty_id,
                voice_id=voice_id,
                avatar_id=avatar_id,
                session_id=session.session_id,
                business_line=business_line,
                preferred_case_id=preferred_case_id,
                trainer_info=trainer_info,
            ),
        )
    except Exception as exc:
        set_status("pipeline_error", error=str(exc))
        await ws_send_json({"type": "error", "message": f"训练管道异常: {exc}"})
        return

    pipeline_time = time.perf_counter() - t0
    response_text = turn.get("response_text", "")
    guardrail = turn.get("guardrail", "")
    training = turn.get("training", {})
    next_state = turn.get("next_state", "")

    # ── Step 3: 下发文本 + TTS 播放状态 ──
    await ws_send_json({"type": "status", "stage": "speaking", "detail": "正在播放客户语音"})
    await ws_send_json({
        "type": "response_text",
        "text": response_text,
        "guardrail": guardrail or None,
        "next_state": next_state,
    })

    # State 变更通知
    before_state = training.get("previous_state", "")
    current_state = training.get("current_state", "")
    if before_state and current_state and before_state != current_state:
        await ws_send_json({
            "type": "state_change",
            "from": before_state,
            "to": current_state,
        })

    # ── Step 4: TTS 音频分块下发 ──
    audio_bytes = turn.get("audio_bytes", b"")
    if audio_bytes:
        audio_chunks = split_pcm_chunks(
            audio_bytes, chunk_duration_ms=TTS_STREAM_CHUNK_MS, sample_rate=SAMPLE_RATE_OUT)
        for chunk in audio_chunks:
            if session.interrupted:
                session.clear_interrupt()
                break
            await ws_send_bytes(chunk)
            if TTS_STREAM_CHUNK_PAUSE_MS > 0:
                await asyncio.sleep(TTS_STREAM_CHUNK_PAUSE_MS)

    await ws_send_json({"type": "tts_done"})

    # ── Step 5: 启发式数值评分（线程池，纯字符串匹配 <10ms） ──
    is_terminal = training.get(
        "training_complete") or training.get("final_state")
    is_success = training.get("is_success")
    is_failure = training.get("is_failure")

    heuristic_score = None
    heuristic_dims = []
    case_id = training.get("case_id")
    if case_id:
        try:
            turns, case = await asyncio.gather(
                loop.run_in_executor(
                    None, get_session_turns, session.session_id),
                loop.run_in_executor(None, get_case, case_id),
            )
            rubric = find_rubric(case_id=case_id)
            if turns and rubric:
                eval_result = await loop.run_in_executor(
                    None, _heuristic_evaluation,
                    session.session_id, case, rubric, turns[-8:],
                )
                heuristic_score = eval_result.get("total_score")
                heuristic_dims = [
                    {"name": d["dimension"], "score": d["score"],
                        "max": d["max_score"]}
                    for d in eval_result.get("dimension_scores", [])
                ]
        except Exception:
            pass

    await ws_send_json({
        "type": "evaluation",
        "turn_index": training.get("turn_count", len(session.history) + 1),
        "terminal": bool(is_terminal),
        "is_success": bool(is_success),
        "is_failure": bool(is_failure),
        "current_state": current_state or training.get("current_state", ""),
        "final_state": training.get("final_state", ""),
        "score": heuristic_score,
        "dimensions": heuristic_dims,
        "score_notes": turn.get("score_notes", {}),
        "triggered_events": training.get("last_triggered_events", []),
    })

    # 记录本轮
    session.history.append(
        {"user_text": user_text, "assistant_text": response_text})

    total_time = time.perf_counter() - t0
    set_status(
        "turn_done",
        prompt=user_text,
        response_text=response_text,
        total_s=round(total_time, 2),
        asr_s=round(asr_time, 2),
        pipeline_s=round(pipeline_time - asr_time, 2),
        next_state=next_state,
        terminal=bool(is_terminal),
    )

    # ── 下发 session_info（在 history.append 之后，轮次准确） ──
    _timings = {}
    try:
        _st = dict(WEB_LAST_STATUS)
        _timings = _st.get("timings", {}) if isinstance(
            _st.get("timings"), dict) else {}
    except Exception:
        pass
    await ws_send_json({
        "type": "session_info",
        "turn_count": len(session.history),
        "current_state": current_state or training.get("current_state", ""),
        "total_asr_s": round(asr_time, 1),
        "total_llm_s": round(_timings.get("qwen_s", 0), 1),
    })

    # ── Step 6: 获取评分详情（仅终局，轮询后台评估线程） ──
    if is_terminal:
        eval_data = None
        for _ in range(50):  # 最多等 5 秒（评估线程与 TTS 并行，通常已就绪）
            last = dict(WEB_LAST_STATUS)
            if last.get("stage") == "evaluation_done" and isinstance(last.get("evaluation"), dict):
                eval_data = last["evaluation"]
                break
            await asyncio.sleep(0.1)
        if eval_data:
            await ws_send_json({
                "type": "final_evaluation",
                "score": eval_data.get("total_score"),
                "dimensions": [
                    {"name": d.get("dimension"), "score": d.get(
                        "score"), "max": d.get("max_score")}
                    for d in eval_data.get("dimension_scores", [])
                ],
                "strengths": eval_data.get("strengths", []),
                "improvements": eval_data.get("improvements", []),
                "summary": eval_data.get("summary", ""),
                "source": eval_data.get("source", "heuristic"),
            })

    # 如果到达终局，通知客户端
    if is_terminal:
        result = "成功" if is_success else "失败" if is_failure else "已结束"
        await ws_send_json({
            "type": "session_end",
            "turn_count": len(session.history),
            "result": result,
            "final_state": training.get("final_state", ""),
        })

    await ws_send_json({
        "type": "status",
        "stage": "idle" if is_terminal else "listening",
        "detail": "会话已结束" if is_terminal else "继续说话",
        "timing": {
            "total": round(total_time, 2),
            "asr": round(asr_time, 2),
            "pipeline": round(pipeline_time - asr_time, 2),
        },
    })


# ═══════════════════════════════════════════════════════
#  Session 启动：下发 persona + goal
# ═══════════════════════════════════════════════════════

_OPENING_BY_STATUS = {
    "已报价-等待客户确认": "报价我看了，你说说具体怎么操作？",
    "已报价-客户有异议": "你们价格比别人高，怎么解释？",
    "已报价-客户在比价": "我还在对比几家，你先说重点。",
    "已报价-已确认试单": "试单确认了，托书什么时候发？",
    "初次接触-了解需求": "你好，直接说你们能做什么。",
    "初次接触-客户主动询价": "我最近有票货要发，你们能走什么价？",
    "已合作-定期回访": "最近忙什么呢？好久没联系了。",
    "已合作-服务升级沟通": "我们最近量在涨，你们价格能不能再低点？",
    "运输中-问题处理": "我这边有票货被查了，你们能帮忙处理吗？",
    "投诉已处理-回访确认": "上次的问题处理得还行，你打电话是？",
    "问题已解决-回访确认": "货收到了，你打电话有事？",
    "曾合作-已流失3个月": "好久没联系了，你找我有事？",
    "潜在客户-旺季提醒": "听说旺季舱位紧张，你们有什么方案？",
    "已报价-等待反馈": "报价我看了，有几个地方不太明白。",
}


def _derive_opening(current_status: str) -> str:
    """根据客户当前状态推断场景化开场白。"""
    if not current_status:
        return ""
    opening = _OPENING_BY_STATUS.get(current_status.strip())
    if opening:
        return opening
    # 模糊匹配：状态关键词包含"已报价"等
    if "已报价" in current_status:
        return "报价我看了，你说说具体细节。"
    if "初次接触" in current_status or "了解需求" in current_status:
        return "你好，直接说重点。"
    if "已合作" in current_status or "老客户" in current_status:
        return "最近忙什么呢？好久没联系了。"
    if "回访" in current_status:
        return "上次的事怎么样了？"
    return ""


async def _send_persona_and_goal(session: VoiceSession, ws_send_json) -> None:
    """在 session 启动后，下发客户画像和训练目标到 H5 客户端。"""
    stage_id = session.config.get("stage_id", "cold_call")
    difficulty_id = session.config.get("difficulty_id", "easy")
    business_line = session.config.get("business_line")
    preferred_case_id = session.config.get("case_id")
    try:
        _, summary, context = _build_session_training_prompt(
            session.session_id,
            stage_id,
            difficulty_id,
            preferred_case_id=preferred_case_id,
            business_line=business_line,
        )
        case = get_case(context.get("case_id")) if context.get(
            "case_id") else None
        assets = summarize_case_assets(case) if case else {}
        stages = load_stages()
        stage = stages.get(stage_id, {}) if stages else {}
    except Exception as exc:
        print(f"[voice-ws] persona/goal build failed: {exc}")
        return

    await ws_send_json({
        "type": "persona",
        "name": assets.get("customer_name") or summary.get("customer", "客户"),
        "role": assets.get("customer_role", ""),
        "location": assets.get("customer_location", ""),
        "personality": assets.get("personality", ""),
        "style": assets.get("communication_style", ""),
        "concerns": assets.get("main_concerns", []),
        "price_sensitivity": assets.get("price_sensitivity"),
        "trust_start": assets.get("trust_start"),
        "scene": assets.get("scene") or "",
        "stage_label": summary.get("stage", ""),
        "difficulty_label": summary.get("difficulty", ""),
        "case_count": context.get("candidate_count", 0),
    })

    await ws_send_json({
        "type": "goal",
        "goal": stage.get("training_goal", ""),
        "must_do": assets.get("must_do", []),
        "critical": assets.get("critical_mistakes", []),
        "current_status": assets.get("current_status") or "",
        "opening": _derive_opening(assets.get("current_status") or ""),
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
                    tone = generate_sine_wave(
                        freq=440, duration_s=0.15, sample_rate=SAMPLE_RATE_IN, amplitude=0.15)
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
                if session.interrupted:
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

                speech_finished = "speech_end" in events or "speech_end_forced" in events
                if speech_finished and not session.interrupted:
                    pcm = session.vad.extract_speech()
                    min_bytes = int(session.sample_rate *
                                    MIN_UTTERANCE_MS / 1000) * BYTES_PER_FRAME
                    if len(pcm) >= min_bytes:
                        detail = "正在识别"
                        if "speech_end_forced" in events:
                            detail = "检测到长句，开始识别"
                            set_status(
                                "ws_vad_forced_segment",
                                audio_duration_ms=round(
                                    _pcm_duration_ms(pcm, session.sample_rate), 1),
                                sample_rate=session.sample_rate,
                            )
                        await _send_json({"type": "status", "stage": "processing", "detail": detail})
                        asyncio.create_task(
                            _process_utterance_safe(
                                session, pcm, _send_json, _send_bytes)
                        )
                    else:
                        set_status(
                            "ws_audio_too_short",
                            audio_duration_ms=round(
                                _pcm_duration_ms(pcm, session.sample_rate), 1),
                            sample_rate=session.sample_rate,
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
                        sid = data.get(
                            "session_id") or f"ws-{int(time.time() * 1000)}"
                        config = data.get("config", {}) if isinstance(
                            data.get("config"), dict) else {}
                        sample_rate = _coerce_sample_rate(config.get(
                            "sample_rate") or data.get("sample_rate"))
                        session = VoiceSession(sid, sample_rate=sample_rate)
                        session.config = config
                        try:
                            save_session_snapshot(sid, {
                                "session_id": sid,
                                "pipeline": "voice_ws",
                                "selected_config": config,
                                "base_case_id": config.get("case_id"),
                                "candidate_count": config.get("candidate_count"),
                                "selected_labels": {
                                    "industry": config.get("industry_label"),
                                    "sub_industry": config.get("sub_industry_label"),
                                    "business_line": config.get("business_line_label"),
                                },
                            })
                        except Exception as exc:
                            set_status("snapshot_save_error", error=str(exc))
                        await _send_json({
                            "type": "status",
                            "stage": "listening",
                            "session_id": session.session_id,
                            "sample_rate": session.sample_rate,
                        })
                        # 后台下发客户画像 + 训练目标
                        asyncio.create_task(
                            _send_persona_and_goal(session, _send_json))

                    elif action == "pause":
                        if session:
                            session.vad.reset()
                        await _send_json({"type": "status", "stage": "idle"})

                    elif action == "resume":
                        await _send_json({"type": "status", "stage": "listening"})

                    elif action == "flush":
                        if not session or session.processing_lock.locked():
                            continue
                        if not session.vad.has_pending_utterance:
                            continue
                        pcm = session.vad.extract_speech()
                        min_bytes = int(session.sample_rate *
                                        MIN_UTTERANCE_MS / 1000) * BYTES_PER_FRAME
                        if len(pcm) < min_bytes:
                            set_status(
                                "ws_client_flush_too_short",
                                audio_duration_ms=round(
                                    _pcm_duration_ms(pcm, session.sample_rate), 1),
                                sample_rate=session.sample_rate,
                            )
                            await _send_json({"type": "status", "stage": "listening", "detail": "语音太短，请再说完整一点"})
                            continue
                        set_status(
                            "ws_client_flush",
                            audio_duration_ms=round(
                                _pcm_duration_ms(pcm, session.sample_rate), 1),
                            sample_rate=session.sample_rate,
                        )
                        await _send_json({"type": "status", "stage": "processing", "detail": "检测到停顿，开始识别"})
                        asyncio.create_task(
                            _process_utterance_safe(
                                session, pcm, _send_json, _send_bytes)
                        )

                    elif action == "end":
                        if session:
                            end_session(session.session_id,
                                        end_reason="manual")
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
