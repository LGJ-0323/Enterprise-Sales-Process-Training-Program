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
import os
import re
import shutil
import subprocess
import tempfile
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
from dashscope.audio.asr import Recognition
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

STAGE_IDS = set(load_stages())
DIFFICULTY_IDS = set(load_difficulties())
VOICE_IDS = set(load_voices())
AVATAR_IDS = set(load_avatars()) | {"auto"}

# ── 全局状态 ───────────────────────────────────────────
LAST_STATUS = {
    "time": None, "stage": "idle", "prompt": "", "response_text": "",
    "audio_bytes": 0, "error": "", "training": {},
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
        raw = resp.output.choices[0].message.content; rt = strip_spoken_identity(raw, training_summary.get("customer"))
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
        if len(args) >= 3: stage_id, diff_id, voice_id = args[-3], args[-2], args[-1]
        elif len(args) == 2: stage_id, diff_id, voice_id = args[0], args[1], DEFAULT_VOICE_ID
        elif len(args) == 1: stage_id, diff_id, voice_id = args[0], DEFAULT_DIFFICULTY_ID, DEFAULT_VOICE_ID
        else: stage_id, diff_id, voice_id = DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID, DEFAULT_VOICE_ID
        s, d, v = stage_id or DEFAULT_STAGE_ID, diff_id or DEFAULT_DIFFICULTY_ID, voice_id or DEFAULT_VOICE_ID

        set_status("received_audio", prompt="", response_text="", audio_bytes=0)

        # 1. Save audio
        audio_data = audio_to_bytes(audio)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data); audio_path = f.name

        # 2. ASR
        rec = Recognition(model=os.getenv("DASHSCOPE_ASR_MODEL","paraformer-realtime-v2"), callback=None, format="mp3", sample_rate=audio[0])
        try: asr_resp = rec.call(audio_path)
        finally:
            try: os.remove(audio_path)
            except OSError: pass

        if asr_resp.status_code != 200: set_status("asr_error", error=asr_resp.message); return
        sentences = asr_resp.get_sentence()
        prompt = " ".join(x.get("text","") for x in sentences).strip() if isinstance(sentences,list) else (sentences or {}).get("text","").strip() if isinstance(sentences,dict) else ""
        if not prompt: set_status("asr_empty"); return
        set_status("asr_done", prompt=prompt)

        # 3. LLM (v2 + guardrails)
        training_prompt, training_summary = build_training_prompt_v2(s, DEFAULT_CUSTOMER_ID, d)
        voice_cfg = resolve_voice(v)
        st = {**training_summary, "voice_id": v, "voice": voice_cfg.get("label",v)}
        # 会话记忆（WebRTC 用内存 session，以 stage+diff 为 key）
        session_key = f"webrtc-{s}-{d}"
        raw_mem = get_recent_memory(session_key, limit=10)
        mem, _ = sanitize_recent_memory(raw_mem)
        mem_prompt = f"【会话记忆】\n{mem}" if mem else "【会话记忆】\n暂无历史对话。"
        guard_prompt = build_role_guard_prompt(training_summary.get("customer"))
        set_status("loaded", prompt=prompt, training=st)

        resp = Generation.call(model=os.getenv("DASHSCOPE_LLM_MODEL","qwen-turbo"),
            messages=[{"role":"system","content":f"{load_customer_profile()}\n\n{training_prompt}\n\n{mem_prompt}\n\n{guard_prompt}"},
                      {"role":"user","content":prompt}], result_format="message")

        if resp.status_code == 200:
            raw = resp.output.choices[0].message.content
            rt = strip_spoken_identity(raw, training_summary.get("customer"))
            if not rt: rt = raw.strip() or "嗯，您先说说具体想怎么合作？"
            if is_role_reversed_sales_reply(rt): rt = build_customer_guardrail_reply(prompt, st)
        else: rt = "抱歉，系统开小差了。"
        set_status("qwen_done", response_text=rt)

        # 4. Save turn (SQLite)
        try: save_turn(session_id=session_key, user_text=prompt, assistant_text=rt, training=st)
        except Exception: pass

        # 5. TTS with retry → yield
        ab = synthesize_with_retry(rt, voice_cfg)
        set_status("tts_done", audio_bytes=len(ab))
        yield (24000, np.frombuffer(ab, dtype=np.int16).reshape(1, -1))

    except Exception as exc:
        set_status("exception", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════
#   FastRTC Stream（模块级，供 app_new_web mount）
# ═══════════════════════════════════════════════════════

stream = Stream(
    modality="audio", mode="send-receive",
    handler=ReplyOnPause(response,
        algo_options=AlgoOptions(audio_chunk_duration=0.4, started_talking_threshold=0.05, speech_threshold=0.03, max_continuous_speech_s=6),
        model_options=SileroVadOptions(threshold=0.35, min_speech_duration_ms=120, min_silence_duration_ms=700, speech_pad_ms=250)),
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
