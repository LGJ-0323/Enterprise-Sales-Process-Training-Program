"""
fastrtc_new_web.py — 融合版实时语音管道

融合 fastrtc_new（guardrails + 会话记忆 + 角色校验） +
     fastrtc_test（WebRTC 流式 + Silero VAD + ReplyOnPause）

职责:
  · response(): WebRTC 流式 handler → ASR → LLM(v2) → guardrails → TTS → yield
  · run_customer_turn(): 同步 API 版（供 wechat 等 HTTP 端使用）
  · stream: FastRTC Stream 对象（供 app_new_web Gradio mount）

P0: build_training_prompt_v2 + lru_cache
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv:
    load_dotenv(PROJECT_DIR / ".env")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", r"D:\tools\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin")
if not shutil.which("ffmpeg") and os.path.exists(os.path.join(FFMPEG_BIN, "ffmpeg.exe")):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

import dashscope
import gradio as gr
import numpy as np
from dashscope import Generation
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
from fastrtc import ReplyOnPause, Stream, audio_to_bytes
from fastrtc.pause_detection.silero import SileroVadOptions
from fastrtc.reply_on_pause import AlgoOptions

# ── 训练模块 ────────────────────────────────────────────
try:
    from .conversation_store import get_recent_memory, save_turn, update_turn_audio_bytes
    from .training_config import (
        build_training_prompt_v2,
        difficulty_choices,
        load_avatars,
        load_difficulties,
        load_stages,
        load_voices,
        resolve_avatar_for_customer,
        resolve_voice,
        stage_choices,
        voice_choices,
    )
except ImportError:
    from conversation_store import get_recent_memory, save_turn, update_turn_audio_bytes
    from training_config import (
        build_training_prompt_v2,
        difficulty_choices,
        load_avatars,
        load_difficulties,
        load_stages,
        load_voices,
        resolve_avatar_for_customer,
        resolve_voice,
        stage_choices,
        voice_choices,
    )

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ── 默认值 ─────────────────────────────────────────────
DEFAULT_PROFILE_PATH = BASE_DIR / "prompts" / "customer_profile.md"
DEFAULT_STAGE_ID = os.getenv("TRAINING_STAGE_ID", "cold_call")
DEFAULT_CUSTOMER_ID = os.getenv("TRAINING_CUSTOMER_ID", "auto")
DEFAULT_DIFFICULTY_ID = os.getenv("TRAINING_DIFFICULTY_ID", "easy")
DEFAULT_VOICE_ID = os.getenv("TRAINING_VOICE_ID", "longsanshu_v3")
DEFAULT_AVATAR_ID = os.getenv("TRAINING_AVATAR_ID", "auto")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ASR_SAMPLE_RATE = _env_int("WEBRTC_ASR_SAMPLE_RATE", 16000)
ASR_FORMAT = os.getenv("WEBRTC_ASR_FORMAT", "pcm").strip().lower()
ASR_TRAILING_SILENCE_MS = _env_int("WEBRTC_ASR_TRAILING_SILENCE_MS", 900)
ASR_MAX_SENTENCE_SILENCE_MS = _env_int("WEBRTC_ASR_MAX_SENTENCE_SILENCE_MS", 800)
ASR_SEMANTIC_PUNCTUATION_ENABLED = _env_bool("WEBRTC_ASR_SEMANTIC_PUNCTUATION_ENABLED", False)
ASR_MULTI_THRESHOLD_MODE_ENABLED = _env_bool("WEBRTC_ASR_MULTI_THRESHOLD_MODE_ENABLED", True)
ASR_USE_CALLBACK = _env_bool("WEBRTC_ASR_USE_CALLBACK", True)
ASR_CALLBACK_FIRST_FINAL_TIMEOUT_S = _env_float("WEBRTC_ASR_CALLBACK_FIRST_FINAL_TIMEOUT_S", 6.0)
ASR_CALLBACK_GRACE_S = _env_float("WEBRTC_ASR_CALLBACK_GRACE_S", 0.9)
ASR_CALLBACK_FRAME_BYTES = _env_int("WEBRTC_ASR_CALLBACK_FRAME_BYTES", 6400)
ASR_SYNC_FALLBACK = _env_bool("WEBRTC_ASR_SYNC_FALLBACK", False)
VAD_AUDIO_CHUNK_DURATION = _env_float("WEBRTC_AUDIO_CHUNK_DURATION", 0.3)
VAD_STARTED_TALKING_THRESHOLD = _env_float("WEBRTC_STARTED_TALKING_THRESHOLD", 0.08)
VAD_SPEECH_THRESHOLD = _env_float("WEBRTC_SPEECH_THRESHOLD", 0.05)
VAD_MAX_CONTINUOUS_SPEECH_S = _env_float("WEBRTC_MAX_CONTINUOUS_SPEECH_S", 8.0)
VAD_THRESHOLD = _env_float("WEBRTC_VAD_THRESHOLD", 0.42)
VAD_MIN_SPEECH_DURATION_MS = _env_int("WEBRTC_MIN_SPEECH_DURATION_MS", 120)
VAD_MIN_SILENCE_DURATION_MS = _env_int("WEBRTC_MIN_SILENCE_DURATION_MS", 1000)
VAD_SPEECH_PAD_MS = _env_int("WEBRTC_SPEECH_PAD_MS", 300)

STAGE_IDS = set(load_stages())
DIFFICULTY_IDS = set(load_difficulties())
VOICE_IDS = set(load_voices())
AVATAR_IDS = set(load_avatars()) | {"auto"}

# ── 全局状态 ───────────────────────────────────────────
LAST_STATUS = {
    "time": None, "stage": "idle", "prompt": "", "response_text": "",
    "audio_bytes": 0, "error": "", "training": {}, "timings": {},
}

def set_status(stage: str, **kwargs):
    LAST_STATUS.update({"time": datetime.now().isoformat(timespec="seconds"), "stage": stage, "error": "", **kwargs})
    print(f"[{LAST_STATUS['time']}] {stage}: {kwargs}", flush=True)


# ═══════════════════════════════════════════════════════
#   Guardrails（从 fastrtc_new 完整复刻）
# ═══════════════════════════════════════════════════════

def load_customer_profile() -> str:
    try: return DEFAULT_PROFILE_PATH.read_text(encoding="utf-8").strip()
    except OSError: return "你是一个中文语音模拟客户。每次 1-3 句话。"

def strip_spoken_identity(text: str, customer_name: str | None = None) -> str:
    cleaned = (text or "").strip()
    if not cleaned: return ""
    escaped_name = re.escape(customer_name.strip()) if customer_name else ""
    labels = ["客户","模拟客户","企业客户","企业人员","外贸主管","物流经理","供应链负责人","采购负责人","总经理","AI","助手"]
    if escaped_name: labels.insert(0, escaped_name)
    prefix = re.compile(rf"^\s*[（(【\[]?\s*(?:{'|'.join(labels)}|[\\u4e00-\\u9fff]{{1,4}}(?:女士|先生|经理|主管|总监|主任|负责人|总))\s*[）)】\]]?\s*[:：]\s*", re.IGNORECASE)
    prev = None
    while prev != cleaned: prev = cleaned; cleaned = prefix.sub("", cleaned).strip()
    return cleaned


def _extract_json_object_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start:end + 1]
    return ""


def _json_string_value(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text or "", re.DOTALL)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1).replace(r"\"", '"').replace(r"\n", "\n")


def clean_customer_reply(raw_text: str, customer_name: str | None = None) -> str:
    """Return only the spoken customer sentence, even if the model outputs JSON."""
    cleaned = (raw_text or "").strip()
    if not cleaned:
        return ""

    json_text = _extract_json_object_text(cleaned)
    if json_text:
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("customer_reply", "reply", "response_text", "response", "text", "content"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return strip_spoken_identity(value, customer_name)
        fallback = _json_string_value(json_text, "customer_reply")
        if fallback:
            return strip_spoken_identity(fallback, customer_name)

    cleaned = strip_spoken_identity(cleaned, customer_name)
    json_prefix = _json_string_value(cleaned, "customer_reply")
    if json_prefix:
        return strip_spoken_identity(json_prefix, customer_name)
    return cleaned


def is_role_reversed_sales_reply(text: str) -> bool:
    c = re.sub(r"\s+", "", text or "")
    if not c: return False
    phrases = ("我们是雄达物流","我是雄达物流","我们雄达物流","雄达物流这边","雄达物流的销售")
    if any(p in c for p in phrases): return True
    return any(re.search(p, c) for p in (
        r"(想|想了解).{0,12}(你们|贵司).{0,24}(需求|发货|出货)",
        r"(你们|贵司).{0,12}(有没有|是否有).{0,24}(发货|出货|物流需求)",
    ))

def is_hostile_or_confused_user_text(text: str) -> bool:
    c = re.sub(r"\s+", "", text or "")
    return any(t in c for t in ("他妈","妈的","在说啥","什么鬼","搞什么","你在说什么","听不懂"))

def build_customer_guardrail_reply(user_text: str, training: dict) -> str:
    t = user_text or ""
    if is_hostile_or_confused_user_text(t): return "我这边听着有点乱。你是雄达物流的销售吧？有事就直接说重点。"
    if any(w in t for w in ("价格","低","便宜")): return "价格低我会关注，但我更关心费用是不是透明。"
    if "雄达物流" in t: return "嗯，你好。你们主要做哪条线？先说重点吧。"
    return "我这边时间不多。你先说说你们能解决什么具体问题？"

def sanitize_recent_memory(memory_text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for line in (memory_text or "").splitlines():
        if "客户：" in line and is_role_reversed_sales_reply(line.split("客户：", 1)[1]): removed += 1; continue
        kept.append(line)
    return "\n".join(kept), removed

def build_role_guard_prompt(customer_name: str | None) -> str:
    return (f"【最高优先级角色校验】\n- 你下一句必须是{customer_name or '当前客户'}作为企业客户的自然回应。\n- role=user 全部来自雄达物流销售；role=assistant 只能来自企业客户。\n- 禁止输出\"我们是雄达物流\"等销售话术。")

def synthesize_with_retry(response_text: str, voice_config: dict, attempts: int = 3) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            s = SpeechSynthesizer(model=voice_config.get("model") or os.getenv("DASHSCOPE_TTS_MODEL","cosyvoice-v1"),
                voice=voice_config.get("voice") or os.getenv("DASHSCOPE_TTS_VOICE","longxiaochun"),
                format=AudioFormat.PCM_24000HZ_MONO_16BIT)
            ab = s.call(response_text)
            if ab: return ab
        except Exception as exc:
            set_status("tts_retry", error=f"{attempt}/{attempts}: {exc}"); time.sleep(0.4)
    raise RuntimeError(f"TTS failed after {attempts} attempts")


def _round_time(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def _audio_duration_s(audio: tuple[int, np.ndarray]) -> float | None:
    try:
        samples = np.asarray(audio[1]).shape[-1]
        return round(float(samples) / float(audio[0]), 3)
    except Exception:
        return None


def _audio_array_to_int16_mono(audio_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio_array)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.dtype == np.int16:
        return arr.astype(np.int16, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        return (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    return arr.astype(np.int16)


def _audio_level_metrics(audio: tuple[int, np.ndarray]) -> dict[str, object]:
    try:
        pcm = _audio_array_to_int16_mono(audio[1]).astype(np.float32)
        if pcm.size == 0:
            return {"audio_peak": 0, "audio_rms": 0.0, "audio_dbfs": None}
        abs_pcm = np.abs(pcm)
        peak = int(np.max(abs_pcm))
        rms = float(np.sqrt(np.mean(np.square(pcm / 32768.0))))
        dbfs = None if rms <= 0 else float(round(20 * np.log10(max(rms, 1e-9)), 1))
        return {"audio_peak": peak, "audio_rms": round(rms, 5), "audio_dbfs": dbfs}
    except Exception as exc:
        return {"audio_level_error": f"{type(exc).__name__}: {exc}"}


def _prepare_webrtc_asr_pcm(audio: tuple[int, np.ndarray]) -> tuple[str, list[str], dict]:
    """Normalize WebRTC audio to raw 16 kHz mono PCM for low-latency ASR."""
    temp_paths: list[str] = []
    metrics: dict[str, object] = {}

    pcm_start = time.perf_counter()
    source_pcm = _audio_array_to_int16_mono(audio[1])
    metrics["audio_pcm_bytes_raw"] = int(source_pcm.nbytes)
    metrics["audio_pcm_prepare_s"] = _round_time(pcm_start)

    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as input_file:
        input_file.write(source_pcm.tobytes())
        input_path = input_file.name
    temp_paths.append(input_path)

    metrics["asr_format"] = "pcm"
    trailing_samples = max(0, int(ASR_SAMPLE_RATE * ASR_TRAILING_SILENCE_MS / 1000))
    if not shutil.which("ffmpeg"):
        if int(audio[0]) == ASR_SAMPLE_RATE:
            if trailing_samples:
                with open(input_path, "ab") as input_file:
                    input_file.write(np.zeros(trailing_samples, dtype=np.int16).tobytes())
                metrics["audio_trailing_silence_ms"] = ASR_TRAILING_SILENCE_MS
            metrics["audio_resample_skipped"] = "ffmpeg_not_found_input_rate_matches"
            metrics["asr_sample_rate"] = ASR_SAMPLE_RATE
            return input_path, temp_paths, metrics
        metrics["audio_resample_skipped"] = "ffmpeg_not_found_fallback_to_mp3"
        return _prepare_webrtc_asr_mp3(audio, metrics, temp_paths)

    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as output_file:
        output_path = output_file.name
    temp_paths.append(output_path)

    resample_start = time.perf_counter()
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(int(audio[0])),
                "-ac",
                "1",
                "-i",
                input_path,
                "-vn",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                str(ASR_SAMPLE_RATE),
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if trailing_samples:
            with open(output_path, "ab") as output_file:
                output_file.write(np.zeros(trailing_samples, dtype=np.int16).tobytes())
            metrics["audio_trailing_silence_ms"] = ASR_TRAILING_SILENCE_MS
        metrics["audio_resample_s"] = _round_time(resample_start)
        metrics["audio_pcm_bytes_16k"] = os.path.getsize(output_path)
        metrics["asr_sample_rate"] = ASR_SAMPLE_RATE
        return output_path, temp_paths, metrics
    except Exception as exc:
        metrics["audio_resample_s"] = _round_time(resample_start)
        metrics["audio_resample_error"] = f"{type(exc).__name__}: {exc}"
        return _prepare_webrtc_asr_mp3(audio, metrics, temp_paths)


def _prepare_webrtc_asr_mp3(
    audio: tuple[int, np.ndarray],
    metrics: dict | None = None,
    temp_paths: list[str] | None = None,
) -> tuple[str, list[str], dict]:
    """Encode WebRTC audio, then normalize it to 16 kHz mono MP3 for ASR."""
    temp_paths = temp_paths or []
    metrics = metrics or {}

    encode_start = time.perf_counter()
    audio_data = audio_to_bytes(audio)
    metrics["audio_mp3_bytes_raw"] = len(audio_data)
    metrics["audio_encode_s"] = _round_time(encode_start)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as input_file:
        input_file.write(audio_data)
        input_path = input_file.name
    temp_paths.append(input_path)

    metrics["asr_format"] = "mp3"
    trailing_silence_s = max(0.0, ASR_TRAILING_SILENCE_MS / 1000)
    if not shutil.which("ffmpeg"):
        metrics["audio_transcode_skipped"] = "ffmpeg_not_found"
        metrics["asr_sample_rate"] = audio[0]
        return input_path, temp_paths, metrics

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output_file:
        output_path = output_file.name
    temp_paths.append(output_path)

    transcode_start = time.perf_counter()
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-af",
                f"apad=pad_dur={trailing_silence_s}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(ASR_SAMPLE_RATE),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "64k",
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if trailing_silence_s:
            metrics["audio_trailing_silence_ms"] = ASR_TRAILING_SILENCE_MS
        metrics["audio_transcode_s"] = _round_time(transcode_start)
        metrics["audio_mp3_bytes_16k"] = os.path.getsize(output_path)
        metrics["asr_sample_rate"] = ASR_SAMPLE_RATE
        return output_path, temp_paths, metrics
    except Exception as exc:
        metrics["audio_transcode_s"] = _round_time(transcode_start)
        metrics["audio_transcode_error"] = f"{type(exc).__name__}: {exc}"
        metrics["asr_sample_rate"] = audio[0]
        return input_path, temp_paths, metrics


def _prepare_webrtc_asr_audio(audio: tuple[int, np.ndarray]) -> tuple[str, list[str], dict]:
    if ASR_FORMAT == "mp3":
        return _prepare_webrtc_asr_mp3(audio)
    return _prepare_webrtc_asr_pcm(audio)


def _cleanup_temp_paths(paths: list[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _parse_asr_sentences(sentences) -> str:
    if isinstance(sentences, list):
        return " ".join(x.get("text", "") for x in sentences).strip()
    if isinstance(sentences, dict):
        return sentences.get("text", "").strip()
    return ""


class _FastAsrCallback(RecognitionCallback):
    def __init__(self):
        self.first_final_event = threading.Event()
        self.complete_event = threading.Event()
        self.error_event = threading.Event()
        self.lock = threading.Lock()
        self.sentences: list[dict] = []
        self.latest_text = ""
        self.error_message = ""
        self.first_final_s: float | None = None
        self.started_at = time.perf_counter()

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if not isinstance(sentence, dict):
            return
        text = (sentence.get("text") or "").strip()
        with self.lock:
            if text:
                self.latest_text = text
            if RecognitionResult.is_sentence_end(sentence):
                sentence_id = sentence.get("sentence_id")
                if sentence_id is None or all(s.get("sentence_id") != sentence_id for s in self.sentences):
                    self.sentences.append(sentence)
                if self.first_final_s is None:
                    self.first_final_s = _round_time(self.started_at)
                self.first_final_event.set()

    def on_complete(self) -> None:
        self.complete_event.set()

    def on_error(self, result: RecognitionResult) -> None:
        self.error_message = getattr(result, "message", "") or "ASR callback error"
        self.error_event.set()
        self.first_final_event.set()

    def prompt(self) -> str:
        with self.lock:
            if self.sentences:
                ordered = sorted(self.sentences, key=lambda s: s.get("sentence_id") or 0)
                return _parse_asr_sentences(ordered)
            return self.latest_text.strip()


def _stop_recognition_async(recognition: Recognition) -> None:
    try:
        recognition.stop()
    except Exception:
        pass


def _recognize_audio_sync(
    audio_path: str,
    audio_format: str,
    sample_rate: int,
    asr_options: dict,
    timings: dict,
) -> tuple[str, str]:
    timings["asr_mode"] = "sync_call"
    rec = Recognition(
        model=os.getenv("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v2"),
        callback=None,
        format=audio_format,
        sample_rate=sample_rate,
    )
    asr_resp = rec.call(audio_path, **asr_options)
    if asr_resp.status_code != 200:
        return "", asr_resp.message
    return _parse_asr_sentences(asr_resp.get_sentence()), ""


def _recognize_audio_callback(
    audio_path: str,
    audio_format: str,
    sample_rate: int,
    asr_options: dict,
    timings: dict,
) -> tuple[str, str]:
    timings["asr_mode"] = "callback"
    callback = _FastAsrCallback()
    rec = Recognition(
        model=os.getenv("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v2"),
        callback=callback,
        format=audio_format,
        sample_rate=sample_rate,
    )
    rec.start(**asr_options)
    with open(audio_path, "rb") as audio_file:
        while True:
            frame = audio_file.read(ASR_CALLBACK_FRAME_BYTES)
            if not frame:
                break
            rec.send_audio_frame(frame)

    threading.Thread(target=_stop_recognition_async, args=(rec,), daemon=True).start()
    if not callback.first_final_event.wait(ASR_CALLBACK_FIRST_FINAL_TIMEOUT_S):
        timings["asr_callback_timeout_s"] = ASR_CALLBACK_FIRST_FINAL_TIMEOUT_S
        prompt = callback.prompt()
        return prompt, "" if prompt else "ASR callback timeout before first final sentence"

    if callback.error_event.is_set():
        return "", callback.error_message

    end_at = time.perf_counter() + ASR_CALLBACK_GRACE_S
    last_prompt = callback.prompt()
    while time.perf_counter() < end_at and not callback.complete_event.is_set():
        time.sleep(0.03)
        prompt = callback.prompt()
        if prompt != last_prompt:
            last_prompt = prompt
            end_at = time.perf_counter() + ASR_CALLBACK_GRACE_S

    timings["asr_callback_first_final_s"] = callback.first_final_s
    timings["asr_callback_grace_s"] = ASR_CALLBACK_GRACE_S
    return callback.prompt(), ""


def _recognize_audio_fast(
    audio_path: str,
    audio_format: str,
    sample_rate: int,
    asr_options: dict,
    timings: dict,
) -> tuple[str, str]:
    if not ASR_USE_CALLBACK:
        return _recognize_audio_sync(audio_path, audio_format, sample_rate, asr_options, timings)
    try:
        return _recognize_audio_callback(audio_path, audio_format, sample_rate, asr_options, timings)
    except Exception as exc:
        timings["asr_callback_error"] = f"{type(exc).__name__}: {exc}"
        if ASR_SYNC_FALLBACK:
            return _recognize_audio_sync(audio_path, audio_format, sample_rate, asr_options, timings)
        return "", f"{type(exc).__name__}: {exc}"


# ═══════════════════════════════════════════════════════
#   run_customer_turn — 同步 API 版（wechat 兼容）
# ═══════════════════════════════════════════════════════

def resolve_runtime_selection(*values):
    s, d, v, a = DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID, DEFAULT_VOICE_ID, DEFAULT_AVATAR_ID
    for val in values:
        if not isinstance(val, str): continue
        if val in STAGE_IDS: s = val
        elif val in DIFFICULTY_IDS: d = val
        elif val in VOICE_IDS: v = val
        elif val in AVATAR_IDS: a = val
    return s, d, v, a

def run_customer_turn(
    prompt: str, stage_id=DEFAULT_STAGE_ID, difficulty_id=DEFAULT_DIFFICULTY_ID,
    voice_id=DEFAULT_VOICE_ID, avatar_id=DEFAULT_AVATAR_ID, session_id="local",
) -> dict:
    """同步版: ASR 后的 LLM + guardrails + TTS → dict"""
    s, d, v, a = resolve_runtime_selection(stage_id, difficulty_id, voice_id, avatar_id)
    training_prompt, training_summary = build_training_prompt_v2(s, DEFAULT_CUSTOMER_ID, d)
    voice_cfg = resolve_voice(v)
    avatar_cfg = resolve_avatar_for_customer(training_summary.get("customer_id"), a)
    st = {**training_summary, "voice_id": v, "voice": voice_cfg.get("label",v),
          "avatar_id": avatar_cfg.get("id",a), "avatar": avatar_cfg.get("label",a)}
    raw_mem = get_recent_memory(session_id, limit=10)
    mem, removed = sanitize_recent_memory(raw_mem)
    mem_prompt = f"【当前会话记忆】\n{mem}" if mem else "【当前会话记忆】\n暂无历史对话。"
    avatar_prompt = f"【客户人物形象】\n形象：{avatar_cfg.get('label',a)}\n角色：{avatar_cfg.get('role','')}\n性格：{avatar_cfg.get('temperament','')}"
    guard_prompt = build_role_guard_prompt(training_summary.get("customer"))
    set_status("loaded", prompt=prompt, training={**st, "session_id": session_id})
    resp = Generation.call(model=os.getenv("DASHSCOPE_LLM_MODEL","qwen-turbo"),
        messages=[{"role":"system","content":f"{load_customer_profile()}\n\n{training_prompt}\n\n{avatar_prompt}\n\n{mem_prompt}\n\n{guard_prompt}"},
                  {"role":"user","content":prompt}], result_format="message")
    if resp.status_code == 200:
        raw = resp.output.choices[0].message.content; rt = clean_customer_reply(raw, training_summary.get("customer"))
        if not rt: rt = raw.strip() or "嗯，您先说说具体想怎么合作？"
        gr = "role_reversed" if is_role_reversed_sales_reply(rt) else ("hostile" if is_hostile_or_confused_user_text(prompt) else "")
        if gr: rt = build_customer_guardrail_reply(prompt, st)
    else: raw = rt = "抱歉，系统开小差了。"; gr = ""
    set_status("qwen_done", response_text=rt, guardrail=gr or None)
    tid, tix = None, None
    try: tid, tix = save_turn(session_id=session_id, user_text=prompt, assistant_text=rt, training=st,
        metadata={"stage_id":s,"difficulty_id":d,"voice_id":v,"avatar_id":a})
    except Exception as e: set_status("save_error", error=str(e))
    ab = synthesize_with_retry(rt, voice_cfg)
    if tid is not None: update_turn_audio_bytes(tid, len(ab))
    return {"prompt":prompt,"response_text":rt,"raw_response_text":raw,"guardrail":gr,"training":st,"turn_index":tix,"audio_bytes":ab,"sample_rate":24000}


# ═══════════════════════════════════════════════════════
#   response — WebRTC 流式版（FastRTC ReplyOnPause）
# ═══════════════════════════════════════════════════════

def response(audio: tuple[int, np.ndarray], *args):
    """FastRTC ReplyOnPause stream handler: ASR → LLM(v2) → guardrails → TTS → yield

    融合 fastrtc_new 的全部 guardrails + 会话记忆 +
        fastrtc_test 的 WebRTC 流式传输。

    *args: FastRTC 传 4 或 5 个参数 (audio, [webrtc_id,] stage, difficulty, voice)
    """
    try:
        overall_start = time.perf_counter()
        timings: dict[str, object] = {}
        if len(args) >= 3: stage_id, diff_id, voice_id = args[-3], args[-2], args[-1]
        elif len(args) == 2: stage_id, diff_id, voice_id = args[0], args[1], DEFAULT_VOICE_ID
        elif len(args) == 1: stage_id, diff_id, voice_id = args[0], DEFAULT_DIFFICULTY_ID, DEFAULT_VOICE_ID
        else: stage_id, diff_id, voice_id = DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID, DEFAULT_VOICE_ID
        s, d, v = stage_id or DEFAULT_STAGE_ID, diff_id or DEFAULT_DIFFICULTY_ID, voice_id or DEFAULT_VOICE_ID

        audio_duration_s = _audio_duration_s(audio)
        timings.update(_audio_level_metrics(audio))
        set_status(
            "received_audio",
            prompt="",
            response_text="",
            audio_bytes=0,
            audio_duration_s=audio_duration_s,
            audio_sample_rate=audio[0],
            training={"stage_id": s, "difficulty_id": d, "voice_id": v},
            timings=timings,
        )

        # 1. Encode and normalize audio for ASR.
        audio_path, temp_paths, audio_metrics = _prepare_webrtc_asr_audio(audio)
        timings.update(audio_metrics)
        set_status(
            "audio_prepared",
            prompt="",
            response_text="",
            audio_bytes=0,
            audio_duration_s=audio_duration_s,
            training={"stage_id": s, "difficulty_id": d, "voice_id": v},
            timings=timings,
        )

        # 2. ASR
        asr_start = time.perf_counter()
        audio_format = str(timings.get("asr_format") or ASR_FORMAT)
        asr_sample_rate = int(timings.get("asr_sample_rate") or ASR_SAMPLE_RATE)
        asr_options = {
            "max_sentence_silence": ASR_MAX_SENTENCE_SILENCE_MS,
            "semantic_punctuation_enabled": ASR_SEMANTIC_PUNCTUATION_ENABLED,
            "multi_threshold_mode_enabled": ASR_MULTI_THRESHOLD_MODE_ENABLED,
        }
        timings.update(
            {
                "asr_max_sentence_silence_ms": ASR_MAX_SENTENCE_SILENCE_MS,
                "asr_semantic_punctuation_enabled": ASR_SEMANTIC_PUNCTUATION_ENABLED,
                "asr_multi_threshold_mode_enabled": ASR_MULTI_THRESHOLD_MODE_ENABLED,
                "asr_use_callback": ASR_USE_CALLBACK,
            }
        )
        try:
            prompt, asr_error = _recognize_audio_fast(audio_path, audio_format, asr_sample_rate, asr_options, timings)
        finally:
            _cleanup_temp_paths(temp_paths)
        timings["asr_s"] = _round_time(asr_start)

        if asr_error:
            set_status("asr_error", error=asr_error, timings=timings)
            return
        timings["prompt_chars"] = len(prompt)
        if not prompt:
            set_status("asr_empty", audio_duration_s=audio_duration_s, timings=timings)
            return
        set_status("asr_done", prompt=prompt, audio_duration_s=audio_duration_s, timings=timings)

        # 3. LLM (v2 + guardrails)
        prompt_start = time.perf_counter()
        training_prompt, training_summary = build_training_prompt_v2(s, DEFAULT_CUSTOMER_ID, d)
        voice_cfg = resolve_voice(v)
        st = {**training_summary, "voice_id": v, "voice": voice_cfg.get("label",v)}
        # 会话记忆（WebRTC 用内存 session，以 stage+diff 为 key）
        session_key = f"webrtc-{s}-{d}"
        raw_mem = get_recent_memory(session_key, limit=10)
        mem, _ = sanitize_recent_memory(raw_mem)
        mem_prompt = f"【会话记忆】\n{mem}" if mem else "【会话记忆】\n暂无历史对话。"
        guard_prompt = build_role_guard_prompt(training_summary.get("customer"))
        timings["prompt_build_s"] = _round_time(prompt_start)
        set_status("loaded", prompt=prompt, training=st, timings=timings)

        qwen_start = time.perf_counter()
        resp = Generation.call(model=os.getenv("DASHSCOPE_LLM_MODEL","qwen-turbo"),
            messages=[{"role":"system","content":f"{load_customer_profile()}\n\n{training_prompt}\n\n{mem_prompt}\n\n{guard_prompt}"},
                      {"role":"user","content":prompt}], result_format="message")
        timings["qwen_s"] = _round_time(qwen_start)

        if resp.status_code == 200:
            raw = resp.output.choices[0].message.content
            rt = clean_customer_reply(raw, training_summary.get("customer"))
            if not rt: rt = raw.strip() or "嗯，您先说说具体想怎么合作？"
            if is_role_reversed_sales_reply(rt): rt = build_customer_guardrail_reply(prompt, st)
        else: rt = "抱歉，系统开小差了。"
        timings["response_chars"] = len(rt)
        set_status("qwen_done", response_text=rt, timings=timings)

        # 4. Save turn (SQLite)
        save_start = time.perf_counter()
        try:
            save_turn(session_id=session_key, user_text=prompt, assistant_text=rt, training=st)
            timings["save_s"] = _round_time(save_start)
        except Exception as exc:
            timings["save_s"] = _round_time(save_start)
            timings["save_error"] = f"{type(exc).__name__}: {exc}"

        # 5. TTS with retry → yield
        tts_start = time.perf_counter()
        ab = synthesize_with_retry(rt, voice_cfg)
        timings["tts_s"] = _round_time(tts_start)
        timings["total_s"] = _round_time(overall_start)
        set_status("tts_done", audio_bytes=len(ab), timings=timings)
        yield (24000, np.frombuffer(ab, dtype=np.int16).reshape(1, -1))

    except Exception as exc:
        set_status("exception", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
#   FastRTC Stream（模块级，供 app_new_web mount）
# ═══════════════════════════════════════════════════════

stream = Stream(
    modality="audio", mode="send-receive",
    handler=ReplyOnPause(
        response,
        algo_options=AlgoOptions(
            audio_chunk_duration=VAD_AUDIO_CHUNK_DURATION,
            started_talking_threshold=VAD_STARTED_TALKING_THRESHOLD,
            speech_threshold=VAD_SPEECH_THRESHOLD,
            max_continuous_speech_s=VAD_MAX_CONTINUOUS_SPEECH_S,
        ),
        model_options=SileroVadOptions(
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        ),
        can_interrupt=False,
    ),
    concurrency_limit=5,
    additional_inputs=[
        gr.Dropdown(choices=stage_choices(), value=DEFAULT_STAGE_ID, label="训练阶段", interactive=True),
        gr.Dropdown(choices=difficulty_choices(), value=DEFAULT_DIFFICULTY_ID, label="难度等级", interactive=True),
        gr.Dropdown(choices=voice_choices(), value=DEFAULT_VOICE_ID, label="客户音色", interactive=True),
    ],
    ui_args={"title":"国际物流模拟客户陪练","subtitle":"WebRTC · VAD · Guardrails · P0 few-shot","full_screen":False},
)


# ═══════════════════════════════════════════════════════
#   辅助函数（供 app_new_web API）
# ═══════════════════════════════════════════════════════

def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf: wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcribe_uploaded_audio(audio_bytes: bytes, filename: str = "") -> str:
    suffix = Path(filename or "").suffix or ".webm"
    input_path = mp3_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f: f.write(audio_bytes); input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f: mp3_path = f.name
        subprocess.run(["ffmpeg","-y","-i",input_path,"-vn","-ac","1","-ar","16000","-codec:a","libmp3lame","-b:a","64k",mp3_path], check=True, capture_output=True, text=True)
        rec = Recognition(model=os.getenv("DASHSCOPE_ASR_MODEL","paraformer-realtime-v2"), format="mp3", sample_rate=16000)
        asr_resp = rec.call(mp3_path)
        if asr_resp.status_code != 200: return ""
        sents = asr_resp.get_sentence()
        return " ".join(x.get("text","") for x in sents).strip() if isinstance(sents, list) else (sents or {}).get("text","").strip()
    finally:
        for p in (input_path, mp3_path):
            if p:
                try: os.remove(p)
                except OSError: pass
