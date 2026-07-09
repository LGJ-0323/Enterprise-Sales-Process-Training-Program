"""
omni_voice_ws.py - Mobile WebSocket proxy for Qwen-Omni realtime voice.

This module keeps the browser talking only to our FastAPI app. The server
holds the DashScope API key, injects the roleplay prompt, and forwards PCM
audio between the H5 client and Qwen-Omni Realtime.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import dashscope

try:
    from dashscope.audio.qwen_omni import (
        AudioFormat,
        MultiModality,
        OmniRealtimeCallback,
        OmniRealtimeConversation,
    )
except Exception:  # pragma: no cover - handled at runtime for clearer UI errors
    AudioFormat = None
    MultiModality = None
    OmniRealtimeCallback = object
    OmniRealtimeConversation = None

try:
    from .conversation_store import end_session, save_session_snapshot, save_turn, update_turn_audio_bytes
    from .fastrtc_new_web import (
        _build_session_training_prompt,
        _merge_state_context,
        _maybe_evaluate_completed_session,
        _turn_metadata,
        set_status,
    )
    from .training_session import advance_session_state
    from .voice_ws import (
        CHANNELS,
        SAMPLE_WIDTH,
        _coerce_sample_rate,
        _send_persona_and_goal,
    )
except ImportError:
    from conversation_store import end_session, save_session_snapshot, save_turn, update_turn_audio_bytes
    from fastrtc_new_web import (
        _build_session_training_prompt,
        _merge_state_context,
        _maybe_evaluate_completed_session,
        _turn_metadata,
        set_status,
    )
    from training_session import advance_session_state
    from voice_ws import (
        CHANNELS,
        SAMPLE_WIDTH,
        _coerce_sample_rate,
        _send_persona_and_goal,
    )


OMNI_INPUT_SAMPLE_RATE = int(
    os.getenv("DASHSCOPE_OMNI_INPUT_SAMPLE_RATE", "16000"))
OMNI_OUTPUT_SAMPLE_RATE = int(
    os.getenv("DASHSCOPE_OMNI_OUTPUT_SAMPLE_RATE", "24000"))
OMNI_MODEL = os.getenv("DASHSCOPE_OMNI_REALTIME_MODEL",
                       "qwen3.5-omni-flash-realtime")
OMNI_VOICE = os.getenv("DASHSCOPE_OMNI_VOICE", "Tina")
OMNI_TRANSCRIPTION_MODEL = os.getenv(
    "DASHSCOPE_OMNI_TRANSCRIPTION_MODEL",
    "qwen3-asr-flash-realtime",
) or None
OMNI_TURN_DETECTION = os.getenv(
    "DASHSCOPE_OMNI_TURN_DETECTION",
    "server_vad",
).strip().lower()
OMNI_USE_SERVER_VAD = OMNI_TURN_DETECTION not in {
    "manual",
    "none",
    "off",
    "false",
    "0",
}
OMNI_TURN_DETECTION_TYPE = (
    OMNI_TURN_DETECTION if OMNI_USE_SERVER_VAD else "server_vad"
)
OMNI_VAD_THRESHOLD = float(os.getenv("DASHSCOPE_OMNI_VAD_THRESHOLD", "0.2"))
OMNI_VAD_PREFIX_PADDING_MS = int(
    os.getenv("DASHSCOPE_OMNI_VAD_PREFIX_PADDING_MS", "300"))
OMNI_VAD_SILENCE_DURATION_MS = int(
    os.getenv("DASHSCOPE_OMNI_VAD_SILENCE_DURATION_MS", "800"))


def _iter_ws_messages(ws):
    async def generator():
        while True:
            raw = await ws.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            if raw.get("bytes") is not None:
                yield raw["bytes"]
            elif raw.get("text") is not None:
                yield raw["text"]

    return generator()


def _resample_pcm(pcm_bytes: bytes, from_rate: int, to_rate: int, state):
    if not pcm_bytes or from_rate == to_rate:
        return pcm_bytes, state
    converted, state = audioop.ratecv(
        pcm_bytes,
        SAMPLE_WIDTH,
        CHANNELS,
        from_rate,
        to_rate,
        state,
    )
    return converted, state


def _clean_omni_instructions(prompt: str) -> str:
    """Reuse the case prompt, but remove JSON-only output rules for speech."""
    cleaned = prompt or ""
    for marker in ("## 输出要求", "## 杈撳嚭瑕佹眰"):
        idx = cleaned.find(marker)
        if idx >= 0:
            cleaned = cleaned[:idx].rstrip()
            break
    cleaned = re.sub(r"只输出以下\s*JSON.*", "", cleaned, flags=re.I | re.S).strip()
    return (
        cleaned
        + "\n\n## 实时语音输出要求\n"
        "- 你现在通过语音扮演企业客户，用户的声音就是雄达物流销售说的话。\n"
        "- 只说客户会自然说出口的话，不要朗读 JSON、状态名、评分、括号说明或说话人前缀。\n"
        "- 每次回复 1 到 3 句话，口语化、可直接播放，允许根据销售表现自然表现犹豫、追问、认可或拒绝。\n"
        "- 不要扮演销售，不要替销售总结，不要跳出角色解释规则。"
    )


def _event_text(event: dict[str, Any]) -> str:
    for key in ("transcript", "text", "delta"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_text_from_response_done(event: dict[str, Any]) -> str:
    response = event.get("response") if isinstance(
        event.get("response"), dict) else {}
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for key in ("transcript", "text"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
    return "".join(chunks).strip()


def _safe_b64decode(value: str) -> bytes:
    try:
        return base64.b64decode(value)
    except Exception:
        return b""


@dataclass
class OmniRuntimeState:
    session_id: str
    sample_rate: int
    config: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    instructions: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    pending_input_bytes: int = 0
    response_text_parts: list[str] = field(default_factory=list)
    user_text_parts: list[str] = field(default_factory=list)
    audio_bytes: int = 0
    current_response_started: bool = False
    response_text_sent: bool = False
    ratecv_state: Any = None
    closing: bool = False
    last_omni_event_type: str = ""
    last_omni_error: str = ""
    recent_omni_events: list[str] = field(default_factory=list)

    @property
    def user_text(self) -> str:
        return "".join(self.user_text_parts).strip()

    @property
    def response_text(self) -> str:
        return "".join(self.response_text_parts).strip()

    def reset_turn_buffers(self) -> None:
        self.pending_input_bytes = 0
        self.response_text_parts.clear()
        self.user_text_parts.clear()
        self.audio_bytes = 0
        self.current_response_started = False
        self.response_text_sent = False


class _OmniCallback(OmniRealtimeCallback):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue

    def _put(self, kind: str, payload: Any) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (kind, payload))

    def on_open(self) -> None:
        self._put("open", None)

    def on_close(self, close_status_code, close_msg) -> None:
        self._put("close", {"code": close_status_code, "message": close_msg})

    def on_event(self, message) -> None:
        self._put("event", message)


async def _queue_json(queue: asyncio.Queue, payload: dict[str, Any]) -> None:
    await queue.put(("send_json", payload))


async def _queue_bytes(queue: asyncio.Queue, payload: bytes) -> None:
    await queue.put(("send_bytes", payload))


async def _handle_omni_event(event: dict[str, Any], state: OmniRuntimeState, outbound: asyncio.Queue) -> None:
    event_type = str(event.get("type") or "")
    state.last_omni_event_type = event_type
    state.recent_omni_events.append(event_type)
    if len(state.recent_omni_events) > 20:
        del state.recent_omni_events[:-20]

    if event_type == "error":
        error = event.get("error") or {}
        message = error.get("message") if isinstance(
            error, dict) else str(error)
        state.last_omni_error = message or "Omni 实时服务异常"
        await _queue_json(outbound, {"type": "error", "message": message or "Omni 实时服务异常"})
        return

    if event_type in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_start"}:
        await _queue_json(outbound, {"type": "status", "stage": "listening", "detail": "Omni 已听到语音"})
        return

    if event_type in {"input_audio_buffer.speech_stopped", "input_audio_buffer.speech_stop"}:
        await _queue_json(outbound, {"type": "status", "stage": "processing", "detail": "Omni 正在判断本轮话语"})
        return

    if event_type in {"input_audio_buffer.committed", "input_audio_buffer.commit"}:
        await _queue_json(outbound, {"type": "status", "stage": "processing", "detail": "Omni 正在理解并回复"})
        return

    if event_type in {"response.created", "response.output_item.added"}:
        if not state.current_response_started:
            state.current_response_started = True
            await _queue_json(outbound, {"type": "status", "stage": "speaking", "detail": "Omni 正在生成客户回复"})
        return

    if event_type in {
        "conversation.item.input_audio_transcription.delta",
        "input_audio_transcription.delta",
    }:
        text = _event_text(event)
        if text:
            state.user_text_parts.append(text)
            await _queue_json(outbound, {"type": "asr_partial", "text": state.user_text})
        return

    if event_type in {
        "conversation.item.input_audio_transcription.completed",
        "input_audio_transcription.completed",
    }:
        text = _event_text(event)
        if text:
            state.user_text_parts[:] = [text]
            await _queue_json(outbound, {"type": "asr_final", "text": text})
        return

    if event_type == "response.audio_transcript.delta":
        text = _event_text(event)
        if text:
            state.response_text_parts.append(text)
        return

    if event_type == "response.audio_transcript.done":
        text = _event_text(event)
        if text:
            state.response_text_parts[:] = [text]
        if state.response_text and not state.response_text_sent:
            state.response_text_sent = True
            await _queue_json(outbound, {"type": "response_text", "text": state.response_text, "source": "omni"})
        return

    if event_type in {"response.text.delta", "response.output_text.delta"}:
        text = _event_text(event)
        if text:
            state.response_text_parts.append(text)
        return

    if event_type == "response.audio.delta":
        if not state.current_response_started:
            state.current_response_started = True
            await _queue_json(outbound, {"type": "status", "stage": "speaking", "detail": "Omni 正在生成客户回复"})
        audio = _safe_b64decode(str(event.get("delta") or ""))
        if audio:
            state.audio_bytes += len(audio)
            await _queue_bytes(outbound, audio)
        return

    if event_type == "response.done":
        if not state.response_text:
            done_text = _extract_text_from_response_done(event)
            if done_text:
                state.response_text_parts[:] = [done_text]
        if state.response_text and not state.response_text_sent:
            state.response_text_sent = True
            await _queue_json(outbound, {"type": "response_text", "text": state.response_text, "source": "omni"})

        user_text = state.user_text or "（Omni 未返回用户转写）"
        response_text = state.response_text or "（Omni 未返回客户文本）"
        before_state = state.training.get("current_state")
        next_context = advance_session_state(
            state.session_id, before_state, [])
        _merge_state_context(state.training, next_context)
        current_state = state.training.get("current_state", "")
        try:
            turn_id, turn_index = save_turn(
                session_id=state.session_id,
                user_text=user_text,
                assistant_text=response_text,
                training=state.training,
                metadata={
                    "pipeline": "omni_realtime",
                    **_turn_metadata(state.training, before_state, current_state or "", [], {}),
                },
            )
            if state.audio_bytes:
                update_turn_audio_bytes(turn_id, state.audio_bytes)
            _maybe_evaluate_completed_session(state.session_id, state.training)
        except Exception as exc:
            turn_index = len(state.history) + 1
            set_status("omni_save_error", error=f"{type(exc).__name__}: {exc}")

        state.history.append(
            {"user_text": user_text, "assistant_text": response_text})
        await _queue_json(outbound, {"type": "tts_done"})
        await _queue_json(outbound, {
            "type": "evaluation",
            "turn_index": turn_index,
            "terminal": bool(state.training.get("training_complete")),
            "is_success": bool(state.training.get("is_success")),
            "is_failure": bool(state.training.get("is_failure")),
            "current_state": state.training.get("current_state", ""),
            "final_state": state.training.get("final_state", ""),
            "score": None,
            "dimensions": [],
            "score_notes": {"pipeline": "Omni 实时语音链路，本轮未单独调用评分模型。"},
            "triggered_events": [],
        })
        await _queue_json(outbound, {
            "type": "session_info",
            "turn_count": len(state.history),
            "current_state": state.training.get("current_state", ""),
            "total_asr_s": 0,
            "total_llm_s": 0,
        })
        await _queue_json(outbound, {"type": "status", "stage": "listening", "detail": "继续说话"})
        set_status(
            "omni_turn_done",
            prompt=user_text,
            response_text=response_text,
            audio_bytes=state.audio_bytes,
            input_bytes=state.pending_input_bytes,
            recent_events=state.recent_omni_events[-10:],
            training=state.training,
        )
        state.reset_turn_buffers()


async def _pump_outbound(ws, outbound: asyncio.Queue) -> None:
    while True:
        kind, payload = await outbound.get()
        try:
            if kind == "stop":
                return
            if kind == "send_json":
                await ws.send_json(payload)
            elif kind == "send_bytes":
                await ws.send_bytes(payload)
        finally:
            outbound.task_done()


async def _pump_omni_events(callback_queue: asyncio.Queue, state: OmniRuntimeState, outbound: asyncio.Queue) -> None:
    while True:
        kind, payload = await callback_queue.get()
        if kind == "stop":
            return
        if kind == "open":
            await _queue_json(outbound, {"type": "ready", "message": "Omni 实时语音已连接"})
        elif kind == "close":
            code = payload.get("code") if isinstance(payload, dict) else None
            message = payload.get("message") if isinstance(
                payload, dict) else None
            set_status(
                "omni_closed",
                session_id=state.session_id,
                code=code,
                message=message,
                last_event_type=state.last_omni_event_type,
                last_error=state.last_omni_error,
                recent_events=state.recent_omni_events[-10:],
                closing=state.closing,
            )
            if state.closing:
                continue
            detail = "Omni 连接已断开"
            if code or message:
                detail += f"（{code or ''} {message or ''}）"
            elif state.last_omni_error:
                detail += f"：{state.last_omni_error}"
            else:
                detail += "，请重新开始对话"
            await _queue_json(outbound, {"type": "error", "message": detail})
        elif kind == "event" and isinstance(payload, dict):
            await _handle_omni_event(payload, state, outbound)


def _connect_omni(callback: _OmniCallback, instructions: str) -> Any:
    if OmniRealtimeConversation is None:
        raise RuntimeError(
            "当前 dashscope SDK 不包含 qwen_omni realtime 支持，请升级 dashscope。")
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    conv = OmniRealtimeConversation(
        model=OMNI_MODEL,
        callback=callback,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    conv.connect()
    conv.update_session(
        output_modalities=[MultiModality.TEXT, MultiModality.AUDIO],
        voice=OMNI_VOICE,
        input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
        output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        enable_input_audio_transcription=True,
        input_audio_transcription_model=OMNI_TRANSCRIPTION_MODEL,
        enable_turn_detection=OMNI_USE_SERVER_VAD,
        turn_detection_type=OMNI_TURN_DETECTION_TYPE,
        prefix_padding_ms=OMNI_VAD_PREFIX_PADDING_MS,
        turn_detection_threshold=OMNI_VAD_THRESHOLD,
        turn_detection_silence_duration_ms=OMNI_VAD_SILENCE_DURATION_MS,
        instructions=instructions,
    )
    return conv


def _prepare_training(session_id: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    stage_id = config.get("stage_id", "cold_call")
    difficulty_id = config.get("difficulty_id", "easy")
    business_line = config.get("business_line")
    preferred_case_id = config.get("case_id")
    prompt, training, context = _build_session_training_prompt(
        session_id,
        stage_id,
        difficulty_id,
        preferred_case_id=preferred_case_id,
        business_line=business_line,
    )
    _merge_state_context(training, context)
    instructions = _clean_omni_instructions(prompt)
    return instructions, training


async def handle_mobile_omni_ws(ws) -> None:
    """Handle /ws/mobile-omni for the H5 Omni realtime page."""
    await ws.accept()
    outbound: asyncio.Queue = asyncio.Queue()
    callback_queue: asyncio.Queue = asyncio.Queue()
    send_task = asyncio.create_task(_pump_outbound(ws, outbound))

    state: OmniRuntimeState | None = None
    conv = None
    event_task: asyncio.Task | None = None
    connect_lock = threading.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        await outbound.put(("send_json", payload))

    try:
        await send_json({
            "type": "ready",
            "message": "Mobile Omni WebSocket ready",
            "sample_rate": OMNI_INPUT_SAMPLE_RATE,
            "output_sample_rate": OMNI_OUTPUT_SAMPLE_RATE,
            "model": OMNI_MODEL,
            "turn_detection": OMNI_TURN_DETECTION if OMNI_USE_SERVER_VAD else "manual",
        })

        async for msg in _iter_ws_messages(ws):
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    continue

                if data.get("type") != "control":
                    continue
                action = data.get("action")

                if action == "start":
                    sid = data.get(
                        "session_id") or f"omni-{int(time.time() * 1000)}"
                    config = data.get("config", {}) if isinstance(
                        data.get("config"), dict) else {}
                    sample_rate = _coerce_sample_rate(
                        config.get("sample_rate"), OMNI_INPUT_SAMPLE_RATE)
                    state = OmniRuntimeState(
                        sid, sample_rate=sample_rate, config=config)
                    state.instructions, state.training = _prepare_training(
                        sid, config)
                    # 保存会话训练快照
                    try:
                        save_session_snapshot(sid, {
                            "session_id": sid,
                            "pipeline": "omni_realtime",
                            "selected_config": config,
                            "training_prompt_snippet": state.instructions[:500],
                            "training_summary": {k: v for k, v in state.training.items() if isinstance(v, (str, int, bool, float, type(None)))},
                            "created_at": state.training.get("current_state", ""),
                        })
                    except Exception:
                        pass

                    await send_json({
                        "type": "status",
                        "stage": "processing",
                        "detail": "正在连接 Omni 实时语音",
                        "session_id": sid,
                        "sample_rate": sample_rate,
                    })
                    asyncio.create_task(
                        _send_persona_and_goal(state, send_json))

                    loop = asyncio.get_running_loop()
                    callback = _OmniCallback(loop, callback_queue)
                    try:
                        with connect_lock:
                            conv = await loop.run_in_executor(None, _connect_omni, callback, state.instructions)
                    except Exception as exc:
                        await send_json({"type": "error", "message": f"Omni 连接失败: {exc}"})
                        continue

                    event_task = asyncio.create_task(
                        _pump_omni_events(callback_queue, state, outbound))
                    await send_json({"type": "status", "stage": "listening", "detail": "Omni 已就绪，请直接说话"})
                    set_status(
                        "omni_ready",
                        session_id=sid,
                        model=OMNI_MODEL,
                        voice=OMNI_VOICE,
                        turn_detection=OMNI_TURN_DETECTION if OMNI_USE_SERVER_VAD else "manual",
                        instructions_len=len(state.instructions),
                        training=state.training,
                    )

                elif action == "flush":
                    if OMNI_USE_SERVER_VAD:
                        continue
                    if not state or not conv or state.pending_input_bytes <= 0:
                        continue
                    await send_json({"type": "status", "stage": "processing", "detail": "Omni 正在理解并回复"})
                    try:
                        conv.commit()
                        conv.create_response(
                            instructions=state.instructions,
                            output_modalities=[
                                MultiModality.TEXT, MultiModality.AUDIO],
                        )
                    except Exception as exc:
                        await send_json({"type": "error", "message": f"Omni 请求失败: {exc}"})

                elif action == "end":
                    if state:
                        state.closing = True
                        end_session(state.session_id, end_reason="manual")
                        await send_json({"type": "session_end", "turn_count": len(state.history)})
                        await outbound.join()
                    if conv:
                        try:
                            conv.end_session_async()
                            conv.close()
                        except Exception:
                            pass
                    await outbound.put(("stop", None))
                    await ws.close()
                    return

            elif isinstance(msg, bytes):
                if not state or not conv:
                    continue
                try:
                    pcm, state.ratecv_state = _resample_pcm(
                        msg,
                        state.sample_rate,
                        OMNI_INPUT_SAMPLE_RATE,
                        state.ratecv_state,
                    )
                except audioop.error as exc:
                    await send_json({"type": "error", "message": f"音频重采样失败: {exc}"})
                    continue
                if not pcm:
                    continue
                state.pending_input_bytes += len(pcm)
                audio_b64 = base64.b64encode(pcm).decode("ascii")
                try:
                    conv.append_audio(audio_b64)
                except Exception as exc:
                    await send_json({"type": "error", "message": f"Omni 音频发送失败: {exc}"})

    except Exception as exc:
        set_status("omni_ws_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if state:
            state.closing = True
        if conv:
            try:
                conv.close()
            except Exception:
                pass
        if event_task:
            await callback_queue.put(("stop", None))
            event_task.cancel()
        await outbound.put(("stop", None))
        send_task.cancel()
