"""
fastrtc_test.py — ASR + LLM(P0 v2) + TTS 实时语音管道

职责：
  · response(): WebRTC 音频 → ASR → build_training_prompt_v2 → 千问 → TTS → 音频流
  · stream: FastRTC Stream 对象（供 app_new_web 挂载）

延迟优化：
  · build_training_prompt_v2 已通过 lru_cache 缓存，首次调用后不再重复加载 JSONL
  · system prompt 只在首次调用时组装，后续调用直接复用
"""

import os
import shutil
import tempfile
import time
import traceback
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

FFMPEG_BIN = os.getenv(
    "FFMPEG_BIN",
    r"D:\tools\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin",
)
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

# ── 训练配置 ──
try:
    from .training_config import (
        build_training_prompt_v2,
        difficulty_choices,
        resolve_voice,
        stage_choices,
        voice_choices,
    )
except ImportError:
    from training_config import (
        build_training_prompt_v2,
        difficulty_choices,
        resolve_voice,
        stage_choices,
        voice_choices,
    )


dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

DEFAULT_PROFILE_PATH = BASE_DIR / "prompts" / "customer_profile.md"
DEFAULT_STAGE_ID = os.getenv("TRAINING_STAGE_ID", "cold_call")
DEFAULT_CUSTOMER_ID = os.getenv("TRAINING_CUSTOMER_ID", "auto")
DEFAULT_DIFFICULTY_ID = os.getenv("TRAINING_DIFFICULTY_ID", "easy")
DEFAULT_VOICE_ID = os.getenv("TRAINING_VOICE_ID", "longsanshu_v3")

# ── 全局状态 ──
LAST_STATUS = {
    "time": None,
    "stage": "idle",
    "prompt": "",
    "response_text": "",
    "audio_bytes": 0,
    "error": "",
    "training": {},
}


def set_status(stage: str, **kwargs):
    LAST_STATUS.update(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "error": "",
            **kwargs,
        }
    )
    print(f"[{LAST_STATUS['time']}] {stage}: {kwargs}", flush=True)


def load_customer_profile() -> str:
    profile_path = Path(os.getenv("CUSTOMER_PROFILE_PATH", DEFAULT_PROFILE_PATH))
    try:
        return profile_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        set_status("profile_error", error=f"Cannot read profile: {exc}")
        return "你是一个中文语音模拟客户。请用自然、简洁的中文回答，每次回复 1 到 3 句话。"


def _normalize_response_args(audio_or_data, args):
    audio = getattr(audio_or_data, "audio", None) or audio_or_data
    stage_id = DEFAULT_STAGE_ID
    difficulty_id = DEFAULT_DIFFICULTY_ID
    voice_id = DEFAULT_VOICE_ID

    if len(args) >= 3:
        stage_id, difficulty_id, voice_id = args[-3:]
    elif len(args) == 2:
        stage_id, difficulty_id = args
    elif len(args) == 1:
        stage_id = args[0]

    return (
        audio,
        stage_id or DEFAULT_STAGE_ID,
        difficulty_id or DEFAULT_DIFFICULTY_ID,
        voice_id or DEFAULT_VOICE_ID,
    )


def response(
    audio: tuple[int, np.ndarray],
    *args,
):
    """FastRTC ReplyOnPause handler: ASR → LLM(v2) → TTS → yield audio

    FastRTC 可能传 4 或 5 个参数 (audio, [webrtc_id,] stage, difficulty, voice)。
    *args 吞掉多余的 webrtc_id。

    延迟: ASR(~500ms) + LLM(~1-3s) + TTS(~500ms)
    build_training_prompt_v2 已缓存，仅首次调用时加载 JSONL
    """
    try:
        # 从 args 中提取 stage/difficulty/voice（取最后 3 个）
        overall_start = time.perf_counter()
        audio, selected_stage_id, selected_difficulty_id, selected_voice_id = _normalize_response_args(audio, args)

        try:
            audio_duration_s = round(float(np.asarray(audio[1]).shape[-1]) / float(audio[0]), 2)
        except Exception:
            audio_duration_s = None
        set_status(
            "received_audio",
            prompt="",
            response_text="",
            audio_bytes=0,
            training={
                "stage_id": selected_stage_id,
                "difficulty_id": selected_difficulty_id,
                "voice_id": selected_voice_id,
            },
            audio_duration_s=audio_duration_s,
        )

        # 1. Audio → bytes
        audio_data = audio_to_bytes(audio)

        # 2. Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_file.write(audio_data)
            audio_path = audio_file.name

        # 3. ASR
        asr_start = time.perf_counter()
        recognition = Recognition(
            model=os.getenv("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v2"),
            callback=None,
            format="mp3",
            sample_rate=audio[0],
        )
        try:
            asr_response = recognition.call(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        if asr_response.status_code == 200:
            sentences = asr_response.get_sentence()
            if isinstance(sentences, list):
                prompt = " ".join(sentence.get("text", "") for sentence in sentences).strip()
            elif isinstance(sentences, dict):
                prompt = sentences.get("text", "").strip()
            else:
                prompt = ""
            set_status("asr_done", prompt=prompt)
            if not prompt:
                set_status("asr_empty", error="No text recognized from audio.")
                return
        else:
            set_status("asr_error", error=asr_response.message)
            return

        # 4. P0 v2: build_training_prompt_v2（lru_cache 已生效）
        training_prompt, training_summary = build_training_prompt_v2(
            selected_stage_id,
            DEFAULT_CUSTOMER_ID,
            selected_difficulty_id,
        )
        voice_config = resolve_voice(selected_voice_id)
        set_status(
            "training_loaded",
            prompt=prompt,
            training={
                **training_summary,
                "voice_id": selected_voice_id,
                "voice": voice_config.get("label", selected_voice_id),
            },
        )

        # 5. LLM
        qwen_response = Generation.call(
            model=os.getenv("DASHSCOPE_LLM_MODEL", "qwen-turbo"),
            messages=[
                {"role": "system", "content": f"{load_customer_profile()}\n\n{training_prompt}"},
                {"role": "user", "content": prompt},
            ],
            result_format="message",
        )

        if qwen_response.status_code == 200:
            response_text = qwen_response.output.choices[0].message.content
        else:
            response_text = "抱歉，系统开小差了。"
            set_status("qwen_error", error=qwen_response.message)

        set_status("qwen_done", response_text=response_text)

        # 6. TTS
        synthesizer = SpeechSynthesizer(
            model=voice_config.get("model") or os.getenv("DASHSCOPE_TTS_MODEL", "cosyvoice-v1"),
            voice=voice_config.get("voice") or os.getenv("DASHSCOPE_TTS_VOICE", "longxiaochun"),
            format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        )
        audio_bytes = synthesizer.call(response_text)
        if not audio_bytes:
            set_status("tts_error", error="DashScope TTS returned empty audio.")
            return

        set_status("tts_done", audio_bytes=len(audio_bytes))
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
        yield (24000, audio_array)

    except Exception as exc:
        set_status("exception", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return


# ── FastRTC Stream ────────────────────────────────────
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(
        response,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.4,
            started_talking_threshold=0.05,
            speech_threshold=0.03,
            max_continuous_speech_s=6,
        ),
        model_options=SileroVadOptions(
            threshold=0.35,
            min_speech_duration_ms=120,
            min_silence_duration_ms=700,
            speech_pad_ms=250,
        ),
    ),
    concurrency_limit=5,
    additional_inputs=[
        gr.Dropdown(
            choices=stage_choices(),
            value=DEFAULT_STAGE_ID,
            label="训练阶段",
            interactive=True,
        ),
        gr.Dropdown(
            choices=difficulty_choices(),
            value=DEFAULT_DIFFICULTY_ID,
            label="难度等级",
            interactive=True,
        ),
        gr.Dropdown(
            choices=voice_choices(),
            value=DEFAULT_VOICE_ID,
            label="客户音色",
            interactive=True,
        ),
    ],
    ui_args={
        "title": "国际物流模拟客户陪练",
        "subtitle": "阶段 / 难度 / 音色",
        "full_screen": False,
    },
)
