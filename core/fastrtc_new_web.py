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
import random
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


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_project_env(path: Path) -> None:
    """Load project .env even when python-dotenv is missing in the conda env."""
    if not path.exists():
        return
    override = _truthy_env(os.getenv("DOTENV_OVERRIDE"))
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "DOTENV_OVERRIDE" and _truthy_env(value):
                override = True
                break
    except OSError:
        pass
    if load_dotenv:
        load_dotenv(path, override=override)
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_project_env(PROJECT_DIR / ".env")

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
from fastrtc.reply_on_pause import AlgoOptions, create_message

# ── 训练模块 ────────────────────────────────────────────
try:
    from .conversation_store import (
        get_recent_memory,
        get_session_turns,
        save_session_evaluation,
        save_turn,
        update_turn_audio_bytes,
    )
    from .case_loader import get_case
    from .training_session import (
        advance_session_state,
        get_or_create_session_context,
        set_active_case_for_combo,
    )
    from .training_evaluator import evaluate_training_session
    from .training_config import (
        _label,
        build_training_prompt_from_case,
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
    from conversation_store import (
        get_recent_memory,
        get_session_turns,
        save_session_evaluation,
        save_turn,
        update_turn_audio_bytes,
    )
    from case_loader import get_case
    from training_session import (
        advance_session_state,
        get_or_create_session_context,
        set_active_case_for_combo,
    )
    from training_evaluator import evaluate_training_session
    from training_config import (
        _label,
        build_training_prompt_from_case,
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
_ACTIVE_STREAM_LOCK = threading.Lock()
_ACTIVE_STREAM_SELECTION = {
    "stage_id": DEFAULT_STAGE_ID,
    "difficulty_id": DEFAULT_DIFFICULTY_ID,
    "voice_id": DEFAULT_VOICE_ID,
    "case_id": "",
}


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
ASR_LEADING_SILENCE_MS = _env_int("WEBRTC_ASR_LEADING_SILENCE_MS", 350)
ASR_TRAILING_SILENCE_MS = _env_int("WEBRTC_ASR_TRAILING_SILENCE_MS", 4500)
ASR_MAX_SENTENCE_SILENCE_MS = _env_int("WEBRTC_ASR_MAX_SENTENCE_SILENCE_MS", 4500)
ASR_SEMANTIC_PUNCTUATION_ENABLED = _env_bool("WEBRTC_ASR_SEMANTIC_PUNCTUATION_ENABLED", False)
ASR_MULTI_THRESHOLD_MODE_ENABLED = _env_bool("WEBRTC_ASR_MULTI_THRESHOLD_MODE_ENABLED", True)
ASR_USE_CALLBACK = _env_bool("WEBRTC_ASR_USE_CALLBACK", False)
ASR_CALLBACK_FIRST_FINAL_TIMEOUT_S = _env_float("WEBRTC_ASR_CALLBACK_FIRST_FINAL_TIMEOUT_S", 12.0)
ASR_CALLBACK_GRACE_S = _env_float("WEBRTC_ASR_CALLBACK_GRACE_S", 3.0)
ASR_CALLBACK_FRAME_BYTES = _env_int("WEBRTC_ASR_CALLBACK_FRAME_BYTES", 6400)
ASR_SYNC_FALLBACK = _env_bool("WEBRTC_ASR_SYNC_FALLBACK", True)
ASR_CALLBACK_TIMEOUT_ERROR = "ASR callback timeout before first final sentence"
ASR_EMPTY_NOTICE = "\u6211\u8fd9\u8fb9\u6ca1\u542c\u6e05\uff0c\u8bf7\u518d\u8bf4\u4e00\u904d\u3002"
ASR_ERROR_NOTICE = "\u8bed\u97f3\u8bc6\u522b\u6682\u65f6\u5931\u8d25\uff0c\u8bf7\u518d\u8bf4\u4e00\u904d\u3002"
VAD_AUDIO_CHUNK_DURATION = _env_float("WEBRTC_AUDIO_CHUNK_DURATION", 0.45)
VAD_STARTED_TALKING_THRESHOLD = _env_float("WEBRTC_STARTED_TALKING_THRESHOLD", 0.18)
VAD_SPEECH_THRESHOLD = _env_float("WEBRTC_SPEECH_THRESHOLD", 0.12)
VAD_MAX_CONTINUOUS_SPEECH_S = _env_float("WEBRTC_MAX_CONTINUOUS_SPEECH_S", 30.0)
VAD_THRESHOLD = _env_float("WEBRTC_VAD_THRESHOLD", 0.40)
VAD_MIN_SPEECH_DURATION_MS = _env_int("WEBRTC_MIN_SPEECH_DURATION_MS", 300)
VAD_MIN_SILENCE_DURATION_MS = _env_int("WEBRTC_MIN_SILENCE_DURATION_MS", 4500)
VAD_REPLY_PAUSE_MS = _env_int("WEBRTC_REPLY_PAUSE_MS", VAD_MIN_SILENCE_DURATION_MS)
VAD_SPEECH_PAD_MS = _env_int("WEBRTC_SPEECH_PAD_MS", 600)
VAD_PREROLL_MS = _env_int("WEBRTC_PREROLL_MS", max(VAD_SPEECH_PAD_MS, 1500))

STAGE_IDS = set(load_stages())
DIFFICULTY_IDS = set(load_difficulties())
VOICE_IDS = set(load_voices())
AVATAR_IDS = set(load_avatars()) | {"auto"}

# ── 全局状态 ───────────────────────────────────────────
LAST_STATUS = {
    "time": None, "stage": "idle", "prompt": "", "response_text": "",
    "audio_bytes": 0, "error": "", "training": {}, "timings": {},
}
_EVAL_LOCK = threading.Lock()
_EVAL_JOBS: set[str] = set()


def set_active_stream_selection(
    stage_id: str | None = None,
    difficulty_id: str | None = None,
    voice_id: str | None = None,
    case_id: str | None = None,
) -> dict:
    """让外层 Dashboard 与 /stream 语音 handler 使用同一套训练选择。"""
    with _ACTIVE_STREAM_LOCK:
        if stage_id in STAGE_IDS:
            _ACTIVE_STREAM_SELECTION["stage_id"] = str(stage_id)
        if difficulty_id in DIFFICULTY_IDS:
            _ACTIVE_STREAM_SELECTION["difficulty_id"] = str(difficulty_id)
        if voice_id in VOICE_IDS:
            _ACTIVE_STREAM_SELECTION["voice_id"] = str(voice_id)
        _ACTIVE_STREAM_SELECTION["case_id"] = str(case_id or "")
        active = dict(_ACTIVE_STREAM_SELECTION)

    stage_label = _choice_label(stage_choices(), active["stage_id"])
    difficulty_label = _choice_label(difficulty_choices(), active["difficulty_id"])
    if active.get("case_id"):
        set_active_case_for_combo(stage_label, difficulty_label, active["case_id"])
    return active


def get_active_stream_selection() -> dict:
    with _ACTIVE_STREAM_LOCK:
        return dict(_ACTIVE_STREAM_SELECTION)


def set_status(stage: str, **kwargs):
    LAST_STATUS.update({"time": datetime.now().isoformat(timespec="seconds"), "stage": stage, "error": "", **kwargs})
    print(f"[{LAST_STATUS['time']}] {stage}: {kwargs}", flush=True)


def _merge_state_context(training: dict, context: dict | None) -> None:
    if not context:
        return
    for key in (
        "current_state",
        "previous_state",
        "turn_count",
        "last_triggered_events",
        "state_validation",
        "training_complete",
        "is_success",
        "is_failure",
        "final_state",
    ):
        if key in context:
            training[key] = context.get(key)


def _turn_metadata(
    training: dict,
    before_state: str | None,
    requested_next_state: str,
    triggered_events: list,
    score_notes: dict,
) -> dict:
    return {
        "case_id": training.get("case_id"),
        "before_state": before_state,
        "current_state": training.get("current_state"),
        "requested_next_state": requested_next_state,
        "previous_state": training.get("previous_state"),
        "state_validation": training.get("state_validation"),
        "triggered_events": triggered_events,
        "score_notes": score_notes,
        "training_complete": training.get("training_complete"),
        "final_state": training.get("final_state"),
        "is_success": training.get("is_success"),
        "is_failure": training.get("is_failure"),
    }


def _maybe_evaluate_completed_session(session_id: str, training: dict) -> None:
    if not training.get("training_complete") or not training.get("case_id"):
        return
    with _EVAL_LOCK:
        if session_id in _EVAL_JOBS:
            return
        _EVAL_JOBS.add(session_id)

    def worker() -> None:
        try:
            evaluation = evaluate_training_session(
                session_id=session_id,
                case_id=training.get("case_id"),
                source_call_id=training.get("source_call_id"),
                training_type=training.get("training_type") or training.get("stage"),
            )
            save_session_evaluation(session_id, evaluation)
            live_training = dict(LAST_STATUS.get("training") or {})
            live_training.update(training)
            live_training["session_id"] = session_id
            set_status("evaluation_done", evaluation=evaluation, training=live_training)
        except Exception as exc:
            set_status("evaluation_error", error=f"{type(exc).__name__}: {exc}")
        finally:
            with _EVAL_LOCK:
                _EVAL_JOBS.discard(session_id)

    threading.Thread(target=worker, daemon=True).start()


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


def parse_customer_payload(raw_text: str) -> dict:
    json_text = _extract_json_object_text(raw_text or "")
    if not json_text:
        return {}
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _history_for_prompt(session_id: str, limit: int = 8) -> list[dict]:
    try:
        turns = get_session_turns(session_id)[-limit:]
    except Exception:
        return []
    return [
        {"user_text": turn.get("user_text", ""), "assistant_text": turn.get("assistant_text", "")}
        for turn in turns
    ]


def _choice_label(choices: list[tuple[str, str]], value: str) -> str:
    for label, item_value in choices:
        if item_value == value:
            return label
    return value


def _stream_session_key(args: tuple, stage_id: str, difficulty_id: str) -> str:
    if len(args) >= 4 and isinstance(args[-4], str):
        candidate = args[-4].strip()
        if candidate and candidate not in STAGE_IDS and candidate not in DIFFICULTY_IDS and candidate not in VOICE_IDS:
            return f"webrtc-{candidate}"
    return f"webrtc-{stage_id}-{difficulty_id}"


def _build_session_training_prompt(
    session_id: str,
    stage_id: str,
    difficulty_id: str,
    preferred_case_id: str | None = None,
) -> tuple[str, dict, dict]:
    stage_label = _choice_label(stage_choices(), stage_id)
    difficulty_label = _choice_label(difficulty_choices(), difficulty_id)
    context = get_or_create_session_context(
        session_id,
        stage_label,
        difficulty_label,
        preferred_case_id=preferred_case_id,
        use_active_case=True,
    )
    case = get_case(context.get("case_id"))
    if not case:
        prompt, summary = build_training_prompt_v2(stage_id, DEFAULT_CUSTOMER_ID, difficulty_id)
        summary.update({"session_id": session_id, "case_id": None, "current_state": ""})
        return prompt, summary, context

    history = _history_for_prompt(session_id)
    prompt, summary = build_training_prompt_from_case(
        case,
        current_state=context.get("current_state"),
        history=history,
        stage_id=stage_id,
        difficulty_id=difficulty_id,
    )
    summary.update(
        {
            "session_id": session_id,
            "candidate_count": context.get("candidate_count", 0),
            "current_state": context.get("current_state"),
        }
    )
    return prompt, summary, context


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


def build_customer_quality_prompt(memory_text: str) -> str:
    recent = memory_text.strip() if memory_text else "暂无"
    return (
        "【客户智能与反复读要求】\n"
        "- 先判断销售本轮真实意图：寒暄、公司介绍、询价、问需求、问航线、问痛点、推进下一步。\n"
        "- 回复必须承接销售刚说的具体内容，不能只泛泛说“请问有什么事”“你们有什么优势”。\n"
        "- 禁止连续重复近几轮客户已经说过的句式或问题；如果销售重复寒暄，你要转向一个具体业务追问。\n"
        "- 每轮至少体现一个真实客户判断点：价格透明度、航线/目的港、货量频率、时效稳定性、异常处理、现有货代对比、付款/账期。\n"
        "- 客户可以防备、犹豫、追问，但要像有业务背景的人，而不是客服机器人。\n"
        "- 如果销售的话明显没说完，只回复“你先说完，我听着”，不要主动开始新话题。\n"
        f"【近几轮客户已说过的话，避免复读】\n{recent}"
    )


def _norm_reply(text: str) -> str:
    return re.sub(r"[\s，。！？!?、,.；;：:\"'“”‘’（）()\[\]{}]+", "", text or "")


def _recent_customer_replies(memory_text: str) -> list[str]:
    replies: list[str] = []
    for line in (memory_text or "").splitlines():
        if "客户：" in line:
            replies.append(line.split("客户：", 1)[1].strip())
    return replies[-4:]


def _looks_low_value_reply(reply: str) -> bool:
    compact = _norm_reply(reply)
    generic = (
        "请问有什么事",
        "有什么事",
        "你们有什么事",
        "你们主要做哪条线",
        "先说重点",
        "你们有什么优势",
        "具体有什么优势",
    )
    return any(item in compact for item in generic)


def _fallback_customer_reply(user_text: str, training: dict) -> str:
    text = user_text or ""
    if any(word in text for word in ("价格", "报价", "便宜", "费用", "成本")):
        return "价格我会看，但我更关心报价里哪些费用是固定的，哪些后面可能加收。"
    if any(word in text for word in ("时效", "多久", "延误", "快", "慢")):
        return "时效你别只说大概，我想知道这条线正常几天到，旺季延误你们怎么处理。"
    if any(word in text for word in ("欧美", "美国", "欧洲", "东南亚", "海运", "空运", "目的港", "航线")):
        return "这条线我们确实会关注，但我要先看你们目的港费用和异常处理是不是透明。"
    if any(word in text for word in ("需求", "发货", "出货", "货量", "合作")):
        return "我们现在有合作货代，你先说你们能在哪个环节比他们更稳。"
    state = training.get("current_state") or training.get("attitude") or ""
    if "warming" in str(state) or "open" in str(state):
        return "可以，你继续说具体一点。你们现在最有把握的是哪条航线？"
    return "你先把你们能解决的问题说具体点，别只介绍公司。"


def refine_customer_reply(reply: str, user_text: str, memory_text: str, training: dict) -> str:
    cleaned = (reply or "").strip()
    if not cleaned:
        return cleaned
    compact = _norm_reply(cleaned)
    recent = [_norm_reply(item) for item in _recent_customer_replies(memory_text)]
    repeated = bool(compact and any(compact == item or compact in item or item in compact for item in recent if item))
    if repeated or _looks_low_value_reply(cleaned):
        return _fallback_customer_reply(user_text, training)
    return cleaned


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


class ReplyOnStablePause(ReplyOnPause):
    """ReplyOnPause with pre-roll and continuous-silence pause detection."""

    def copy(self):
        return ReplyOnStablePause(
            self.fn,
            self.startup_fn,
            self.algo_options,
            self.model_options,
            self.can_interrupt,
            self.expected_layout,
            self.output_sample_rate,
            self.output_frame_size,
            self.input_sample_rate,
            self.model,
            self.needs_args,
        )

    def _append_preroll(self, state, audio: np.ndarray, sampling_rate: int) -> None:
        max_samples = max(0, int(sampling_rate * VAD_PREROLL_MS / 1000))
        if max_samples <= 0:
            return
        previous = getattr(state, "_preroll_audio", None)
        combined = audio if previous is None else np.concatenate((previous, audio))
        if combined.size > max_samples:
            combined = combined[-max_samples:]
        setattr(state, "_preroll_audio", combined)

    @staticmethod
    def _append_stream(state, audio: np.ndarray) -> None:
        if state.stream is None:
            state.stream = audio
        else:
            state.stream = np.concatenate((state.stream, audio))

    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state) -> bool:
        duration = len(audio) / sampling_rate
        if duration < self.algo_options.audio_chunk_duration:
            return False

        dur_vad, _ = self.model.vad((sampling_rate, audio), self.model_options)
        was_started = state.started_talking
        if dur_vad > self.algo_options.started_talking_threshold and not state.started_talking:
            state.started_talking = True
            setattr(state, "_stable_silence_s", 0.0)
            self.send_message_sync(create_message("log", "started_talking"))

        if state.started_talking:
            if not was_started:
                preroll = getattr(state, "_preroll_audio", None)
                if preroll is not None and preroll.size > 0:
                    self._append_stream(state, preroll)
                setattr(state, "_preroll_audio", None)
            self._append_stream(state, audio)
        else:
            self._append_preroll(state, audio, sampling_rate)
            state.buffer = None
            return False

        state.buffer = None
        current_duration = len(state.stream) / sampling_rate if state.stream is not None else 0.0
        if current_duration >= self.algo_options.max_continuous_speech_s:
            return True

        if dur_vad < self.algo_options.speech_threshold:
            stable_silence_s = getattr(state, "_stable_silence_s", 0.0) + duration
        else:
            stable_silence_s = 0.0
        setattr(state, "_stable_silence_s", stable_silence_s)
        return stable_silence_s >= max(0.0, VAD_REPLY_PAUSE_MS / 1000)


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
    leading_samples = max(0, int(int(audio[0]) * ASR_LEADING_SILENCE_MS / 1000))
    if leading_samples:
        source_pcm = np.concatenate((np.zeros(leading_samples, dtype=np.int16), source_pcm))
        metrics["audio_leading_silence_ms"] = ASR_LEADING_SILENCE_MS
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
    leading_silence_ms = max(0, ASR_LEADING_SILENCE_MS)
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
                f"adelay={leading_silence_ms}:all=1,apad=pad_dur={trailing_silence_s}",
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
        if leading_silence_ms:
            metrics["audio_leading_silence_ms"] = leading_silence_ms
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
        return prompt, "" if prompt else ASR_CALLBACK_TIMEOUT_ERROR

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
        prompt, error = _recognize_audio_callback(audio_path, audio_format, sample_rate, asr_options, timings)
        if error == ASR_CALLBACK_TIMEOUT_ERROR and ASR_SYNC_FALLBACK:
            timings["asr_callback_fallback_reason"] = error
            fallback_prompt, fallback_error = _recognize_audio_sync(audio_path, audio_format, sample_rate, asr_options, timings)
            timings["asr_callback_fallback_used"] = True
            if fallback_error:
                return "", f"{error}; sync fallback failed: {fallback_error}"
            return fallback_prompt, ""
        return prompt, error
    except Exception as exc:
        timings["asr_callback_error"] = f"{type(exc).__name__}: {exc}"
        if ASR_SYNC_FALLBACK:
            timings["asr_callback_fallback_reason"] = timings["asr_callback_error"]
            timings["asr_callback_fallback_used"] = True
            return _recognize_audio_sync(audio_path, audio_format, sample_rate, asr_options, timings)
        return "", f"{type(exc).__name__}: {exc}"


def _friendly_asr_error(message: str) -> str:
    raw = str(message or "").strip()
    compact = raw.lower()
    if "access denied" in compact or "account is in good standing" in compact:
        return (
            "DashScope ASR 鉴权失败：当前 DASHSCOPE_API_KEY 对实时语音识别无权限、账号欠费/未开通，"
            "或正在使用旧的系统环境变量 Key。请在项目 .env 中填入可用 Key，确认已开通 Paraformer 实时语音识别，"
            "然后重启服务。原始错误："
            f"{raw}"
        )
    return raw


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
    training_prompt, training_summary, session_context = _build_session_training_prompt(session_id, s, d)
    voice_cfg = resolve_voice(v)
    avatar_cfg = resolve_avatar_for_customer(training_summary.get("customer_id"), a)
    st = {**training_summary, "voice_id": v, "voice": voice_cfg.get("label",v),
          "avatar_id": avatar_cfg.get("id",a), "avatar": avatar_cfg.get("label",a),
          "session_id": session_id}
    raw_mem = get_recent_memory(session_id, limit=10)
    mem, removed = sanitize_recent_memory(raw_mem)
    mem_prompt = f"【当前会话记忆】\n{mem}" if mem else "【当前会话记忆】\n暂无历史对话。"
    avatar_prompt = f"【客户人物形象】\n形象：{avatar_cfg.get('label',a)}\n角色：{avatar_cfg.get('role','')}\n性格：{avatar_cfg.get('temperament','')}"
    guard_prompt = build_role_guard_prompt(training_summary.get("customer"))
    quality_prompt = build_customer_quality_prompt(mem)
    set_status("loaded", prompt=prompt, training={**st, "session_id": session_id})
    resp = Generation.call(model=os.getenv("DASHSCOPE_LLM_MODEL","qwen3.6-plus"),
        messages=[{"role":"system","content":f"{training_prompt}\n\n{avatar_prompt}\n\n{mem_prompt}\n\n{guard_prompt}\n\n{quality_prompt}"},
                  {"role":"user","content":prompt}],
        result_format="message",
        temperature=0.45,
        top_p=0.8,
        repetition_penalty=1.12,
        seed=random.randint(1, 2147483647),
    )
    payload = {}
    next_state = ""
    triggered_events = []
    score_notes = {}
    if resp.status_code == 200:
        raw = resp.output.choices[0].message.content
        payload = parse_customer_payload(raw)
        next_state = str(payload.get("next_state") or "")
        triggered_events = payload.get("triggered_events") if isinstance(payload.get("triggered_events"), list) else []
        score_notes = payload.get("score_notes") if isinstance(payload.get("score_notes"), dict) else {}
        rt = clean_customer_reply(raw, training_summary.get("customer"))
        if not rt: rt = raw.strip() or "嗯，您先说说具体想怎么合作？"
        rt = refine_customer_reply(rt, prompt, mem, st)
        gr = "role_reversed" if is_role_reversed_sales_reply(rt) else ("hostile" if is_hostile_or_confused_user_text(prompt) else "")
        if gr: rt = build_customer_guardrail_reply(prompt, st)
    else: raw = rt = "抱歉，系统开小差了。"; gr = ""
    before_state = training_summary.get("current_state")
    next_context = advance_session_state(session_id, next_state, triggered_events) if not gr else session_context
    _merge_state_context(st, next_context)
    set_status("qwen_done", response_text=rt, guardrail=gr or None, training=st)
    tid, tix = None, None
    try: tid, tix = save_turn(session_id=session_id, user_text=prompt, assistant_text=rt, training=st,
        metadata={"stage_id":s,"difficulty_id":d,"voice_id":v,"avatar_id":a,
                  **_turn_metadata(st, before_state, next_state, triggered_events, score_notes)})
    except Exception as e: set_status("save_error", error=str(e))
    else:
        _maybe_evaluate_completed_session(session_id, st)
    ab = synthesize_with_retry(rt, voice_cfg)
    if tid is not None: update_turn_audio_bytes(tid, len(ab))
    return {"prompt":prompt,"response_text":rt,"raw_response_text":raw,"guardrail":gr,"training":st,
            "turn_index":tix,"audio_bytes":ab,"sample_rate":24000,
            "next_state": next_state, "triggered_events": triggered_events, "score_notes": score_notes}


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
        active_selection = get_active_stream_selection()
        s = active_selection.get("stage_id") or s
        d = active_selection.get("difficulty_id") or d
        v = active_selection.get("voice_id") or v
        active_case_id = str(active_selection.get("case_id") or "")

        audio_duration_s = _audio_duration_s(audio)
        timings.update(_audio_level_metrics(audio))
        timings.update(
            {
                "vad_reply_pause_ms": VAD_REPLY_PAUSE_MS,
                "vad_preroll_ms": VAD_PREROLL_MS,
                "asr_leading_silence_ms": ASR_LEADING_SILENCE_MS,
            }
        )
        set_status(
            "received_audio",
            prompt="",
            response_text="",
            audio_bytes=0,
            audio_duration_s=audio_duration_s,
            audio_sample_rate=audio[0],
            training={"stage_id": s, "difficulty_id": d, "voice_id": v, "case_id": active_case_id},
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
            training={"stage_id": s, "difficulty_id": d, "voice_id": v, "case_id": active_case_id},
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
            notice = ASR_ERROR_NOTICE
            tts_start = time.perf_counter()
            voice_cfg = resolve_voice(v)
            ab = synthesize_with_retry(notice, voice_cfg)
            timings["tts_s"] = _round_time(tts_start)
            timings["total_s"] = _round_time(overall_start)
            set_status(
                "asr_error",
                error=_friendly_asr_error(asr_error),
                response_text=notice,
                audio_bytes=len(ab),
                timings=timings,
            )
            yield (24000, np.frombuffer(ab, dtype=np.int16).reshape(1, -1))
            return
        timings["prompt_chars"] = len(prompt)
        if not prompt:
            notice = ASR_EMPTY_NOTICE
            tts_start = time.perf_counter()
            voice_cfg = resolve_voice(v)
            ab = synthesize_with_retry(notice, voice_cfg)
            timings["tts_s"] = _round_time(tts_start)
            timings["total_s"] = _round_time(overall_start)
            set_status(
                "asr_empty",
                audio_duration_s=audio_duration_s,
                response_text=notice,
                audio_bytes=len(ab),
                timings=timings,
            )
            yield (24000, np.frombuffer(ab, dtype=np.int16).reshape(1, -1))
            return
        set_status("asr_done", prompt=prompt, audio_duration_s=audio_duration_s, timings=timings)

        # 3. LLM (JSONL case session + state machine + guardrails)
        prompt_start = time.perf_counter()
        session_key = _stream_session_key(args, s, d)
        if active_case_id:
            session_key = f"{session_key}-{active_case_id}"
        training_prompt, training_summary, session_context = _build_session_training_prompt(
            session_key,
            s,
            d,
            preferred_case_id=active_case_id or None,
        )
        voice_cfg = resolve_voice(v)
        st = {**training_summary, "voice_id": v, "voice": voice_cfg.get("label",v), "session_id": session_key}
        raw_mem = get_recent_memory(session_key, limit=10)
        mem, _ = sanitize_recent_memory(raw_mem)
        mem_prompt = f"【会话记忆】\n{mem}" if mem else "【会话记忆】\n暂无历史对话。"
        guard_prompt = build_role_guard_prompt(training_summary.get("customer"))
        quality_prompt = build_customer_quality_prompt(mem)
        timings["prompt_build_s"] = _round_time(prompt_start)
        set_status("loaded", prompt=prompt, training=st, timings=timings)

        qwen_start = time.perf_counter()
        resp = Generation.call(model=os.getenv("DASHSCOPE_LLM_MODEL","qwen3.6-plus"),
            messages=[{"role":"system","content":f"{training_prompt}\n\n{mem_prompt}\n\n{guard_prompt}\n\n{quality_prompt}"},
                      {"role":"user","content":prompt}],
            result_format="message",
            temperature=0.45,
            top_p=0.8,
            repetition_penalty=1.12,
            seed=random.randint(1, 2147483647),
        )
        timings["qwen_s"] = _round_time(qwen_start)

        if resp.status_code == 200:
            raw = resp.output.choices[0].message.content
            payload = parse_customer_payload(raw)
            next_state = str(payload.get("next_state") or "")
            triggered_events = payload.get("triggered_events") if isinstance(payload.get("triggered_events"), list) else []
            score_notes = payload.get("score_notes") if isinstance(payload.get("score_notes"), dict) else {}
            rt = clean_customer_reply(raw, training_summary.get("customer"))
            if not rt: rt = raw.strip() or "嗯，您先说说具体想怎么合作？"
            rt = refine_customer_reply(rt, prompt, mem, st)
            if is_role_reversed_sales_reply(rt):
                rt = build_customer_guardrail_reply(prompt, st)
                next_context = session_context
            else:
                next_context = advance_session_state(session_key, next_state, triggered_events)
        else:
            rt = "抱歉，系统开小差了。"
            raw = rt
            payload = {}
            next_state = ""
            triggered_events = []
            score_notes = {}
            next_context = session_context
        before_state = training_summary.get("current_state")
        _merge_state_context(st, next_context)
        timings["response_chars"] = len(rt)
        set_status(
            "qwen_done",
            response_text=rt,
            raw_response_text=raw,
            next_state=next_state,
            triggered_events=triggered_events,
            score_notes=score_notes,
            training=st,
            timings=timings,
        )

        # 4. Save turn (SQLite)
        save_start = time.perf_counter()
        try:
            save_turn(
                session_id=session_key,
                user_text=prompt,
                assistant_text=rt,
                training=st,
                metadata={
                    **_turn_metadata(st, before_state, next_state, triggered_events, score_notes),
                },
            )
            timings["save_s"] = _round_time(save_start)
            _maybe_evaluate_completed_session(session_key, st)
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
    handler=ReplyOnStablePause(
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
