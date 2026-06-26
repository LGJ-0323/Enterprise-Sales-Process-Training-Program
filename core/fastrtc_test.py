import os
import shutil
import tempfile
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
import numpy as np
from dashscope import Generation
from dashscope.audio.asr import Recognition
from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
from fastrtc import ReplyOnPause, Stream, audio_to_bytes
from fastrtc.pause_detection.silero import SileroVadOptions
from fastrtc.reply_on_pause import AlgoOptions

try:
    from .training_config import build_training_prompt, resolve_voice
except ImportError:
    from training_config import build_training_prompt, resolve_voice


dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

DEFAULT_PROFILE_PATH = BASE_DIR / "prompts" / "customer_profile.md"
DEFAULT_STAGE_ID = os.getenv("TRAINING_STAGE_ID", "cold_call")
DEFAULT_CUSTOMER_ID = os.getenv("TRAINING_CUSTOMER_ID", "auto")
DEFAULT_DIFFICULTY_ID = os.getenv("TRAINING_DIFFICULTY_ID", "easy")
DEFAULT_VOICE_ID = os.getenv("TRAINING_VOICE_ID", "longsanshu_v3")

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


def response(audio: tuple[int, np.ndarray]):
    try:
        set_status("received_audio", prompt="", response_text="", audio_bytes=0)
        audio_data = audio_to_bytes(audio)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_file.write(audio_data)
            audio_path = audio_file.name

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

        training_prompt, training_summary = build_training_prompt(
            DEFAULT_STAGE_ID,
            DEFAULT_CUSTOMER_ID,
            DEFAULT_DIFFICULTY_ID,
        )
        set_status("training_loaded", prompt=prompt, training=training_summary)

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

        voice_config = resolve_voice(DEFAULT_VOICE_ID)
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
)
