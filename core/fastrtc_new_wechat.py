from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import time
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

FFMPEG_BIN = os.getenv(
    "FFMPEG_BIN",
    r"D:\tools\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin",
)
if not shutil.which("ffmpeg") and os.path.exists(os.path.join(FFMPEG_BIN, "ffmpeg.exe")):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition
from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

try:
    from .conversation_store import get_recent_memory, save_turn, update_turn_audio_bytes
    from .training_config import (
        build_training_prompt,
        build_training_prompt_v2,
        load_avatars,
        load_difficulties,
        load_stages,
        load_voices,
        resolve_avatar_for_customer,
        resolve_voice,
    )
except ImportError:
    from conversation_store import get_recent_memory, save_turn, update_turn_audio_bytes
    from training_config import (
        build_training_prompt,
        build_training_prompt_v2,
        load_avatars,
        load_difficulties,
        load_stages,
        load_voices,
        resolve_avatar_for_customer,
        resolve_voice,
    )


dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

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


def resolve_runtime_selection(*values):
    selected_stage_id = DEFAULT_STAGE_ID
    selected_difficulty_id = DEFAULT_DIFFICULTY_ID
    selected_voice_id = DEFAULT_VOICE_ID
    selected_avatar_id = DEFAULT_AVATAR_ID

    for value in values:
        if not isinstance(value, str):
            continue
        if value in STAGE_IDS:
            selected_stage_id = value
        elif value in DIFFICULTY_IDS:
            selected_difficulty_id = value
        elif value in VOICE_IDS:
            selected_voice_id = value
        elif value in AVATAR_IDS:
            selected_avatar_id = value

    return selected_stage_id, selected_difficulty_id, selected_voice_id, selected_avatar_id


def strip_spoken_identity(text: str, customer_name: str | None = None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    escaped_name = re.escape(customer_name.strip()) if customer_name else ""
    speaker_labels = [
        "客户",
        "模拟客户",
        "企业客户",
        "企业人员",
        "外贸主管",
        "物流经理",
        "供应链负责人",
        "采购负责人",
        "总经理",
        "AI",
        "助手",
    ]
    if escaped_name:
        speaker_labels.insert(0, escaped_name)

    label_pattern = "|".join(speaker_labels)
    suffix_name_pattern = r"[\u4e00-\u9fff]{1,4}(?:女士|先生|经理|主管|总监|主任|负责人|总)"
    prefix_pattern = rf"(?:{label_pattern}|{suffix_name_pattern})"
    speaker_prefix = re.compile(
        rf"^\s*[（(【\[]?\s*{prefix_pattern}\s*[）)】\]]?\s*[:：]\s*",
        re.IGNORECASE,
    )

    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = speaker_prefix.sub("", cleaned).strip()

    return cleaned


def is_role_reversed_sales_reply(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text or "")
    if not cleaned:
        return False

    identity_phrases = (
        "我们是雄达物流",
        "我是雄达物流",
        "我们这边是雄达物流",
        "我们这边就是雄达物流",
        "这边是雄达物流",
        "这边就是雄达物流",
        "我们雄达物流",
        "雄达物流这边",
        "雄达物流的销售",
    )
    if any(phrase in cleaned for phrase in identity_phrases):
        return True

    prospecting_patterns = (
        r"(想|想先|想了解|了解一下|确认一下).{0,12}(你们|贵司).{0,24}(需求|发货|出货)",
        r"(你们|贵司).{0,12}(有没有|是否有).{0,24}(发货|出货|物流需求)",
        r"(你们|贵司).{0,16}(最近|近期).{0,16}(有没有|是否有).{0,20}(海运|空运|发往|发到).{0,8}需求",
    )
    return any(re.search(pattern, cleaned) for pattern in prospecting_patterns)


def is_hostile_or_confused_user_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text or "")
    if not cleaned:
        return False
    triggers = (
        "他妈",
        "妈的",
        "在说啥",
        "说啥呢",
        "什么鬼",
        "搞什么",
        "搞啥",
        "搞错了吧",
        "你在说什么",
        "听不懂",
    )
    return any(trigger in cleaned for trigger in triggers)


def build_customer_guardrail_reply(user_text: str, training: dict[str, object]) -> str:
    text = user_text or ""
    if is_hostile_or_confused_user_text(text):
        return "我这边听着有点乱。你是雄达物流的销售吧？有事就直接说重点。"
    if "价格" in text or "低" in text or "便宜" in text:
        return "价格低我会关注，但我更关心费用是不是透明。你们旺季会不会临时加价？"
    if "雄达物流" in text:
        return "嗯，你好。你们主要做哪条线？我现在有合作货代，先说重点吧。"
    return "我这边时间不多。你先说说你们能解决什么具体问题？"


def sanitize_recent_memory(memory_text: str) -> tuple[str, int]:
    kept_lines: list[str] = []
    removed = 0
    for line in (memory_text or "").splitlines():
        if "客户：" in line:
            customer_text = line.split("客户：", 1)[1]
            if is_role_reversed_sales_reply(customer_text):
                removed += 1
                continue
        kept_lines.append(line)
    return "\n".join(kept_lines), removed


def build_role_guard_prompt(customer_name: str | None) -> str:
    display_name = customer_name or "当前客户"
    return (
        "【最高优先级角色校验】\n"
        f"- 你下一句必须是{display_name}作为企业客户的自然回应。\n"
        "- role=user 的消息全部来自雄达物流销售；role=assistant 的消息只能来自企业客户。\n"
        "- 如果历史记忆里出现客户自称雄达物流、询问对方发货需求等销售口吻，那是旧版本错误，必须忽略。\n"
        "- 如果销售说“我们是雄达物流”，你要把它理解为对方自我介绍，不能复述为自己的身份。\n"
        "- 禁止输出“我们是雄达物流”“想了解你们有没有发货需求”等销售话术。"
    )


def synthesize_with_retry(response_text: str, voice_config: dict, attempts: int = 3) -> bytes:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            synthesizer = SpeechSynthesizer(
                model=voice_config.get("model") or os.getenv("DASHSCOPE_TTS_MODEL", "cosyvoice-v1"),
                voice=voice_config.get("voice") or os.getenv("DASHSCOPE_TTS_VOICE", "longxiaochun"),
                format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            )
            audio_bytes = synthesizer.call(response_text)
            if audio_bytes:
                return audio_bytes
            last_error = RuntimeError("DashScope TTS returned empty audio.")
        except Exception as exc:
            last_error = exc
            set_status("tts_retry", error=f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}")
            time.sleep(0.6)

    raise RuntimeError(f"TTS failed after {attempts} attempts: {last_error}")


def _parse_asr_prompt(asr_response) -> str:
    sentences = asr_response.get_sentence()
    if isinstance(sentences, list):
        return " ".join(sentence.get("text", "") for sentence in sentences).strip()
    if isinstance(sentences, dict):
        return sentences.get("text", "").strip()
    return ""


def transcribe_audio_file(audio_path: str, audio_format: str = "mp3", sample_rate: int = 16000) -> str:
    recognition = Recognition(
        model=os.getenv("DASHSCOPE_ASR_MODEL", "paraformer-realtime-v2"),
        callback=None,
        format=audio_format,
        sample_rate=sample_rate,
    )
    asr_response = recognition.call(audio_path)
    if asr_response.status_code != 200:
        set_status("asr_error", error=asr_response.message)
        raise RuntimeError(asr_response.message)

    prompt = _parse_asr_prompt(asr_response)
    set_status("asr_done", prompt=prompt)
    if not prompt:
        set_status("asr_empty", error="No text recognized from audio.")
        raise ValueError("No text recognized from audio.")
    return prompt


def transcode_audio_to_mp3(input_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output_file:
        output_path = output_file.name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    return output_path


def transcribe_uploaded_audio(audio_bytes: bytes, filename: str = "") -> str:
    suffix = Path(filename or "").suffix or ".webm"
    input_path = ""
    mp3_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as input_file:
            input_file.write(audio_bytes)
            input_path = input_file.name

        mp3_path = transcode_audio_to_mp3(input_path)
        return transcribe_audio_file(mp3_path, audio_format="mp3", sample_rate=16000)
    finally:
        for path in (input_path, mp3_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def run_customer_turn(
    prompt: str,
    stage_id: str = DEFAULT_STAGE_ID,
    difficulty_id: str = DEFAULT_DIFFICULTY_ID,
    voice_id: str = DEFAULT_VOICE_ID,
    avatar_id: str = DEFAULT_AVATAR_ID,
    session_id: str = "h5-local",
) -> dict[str, object]:
    selected_stage_id, selected_difficulty_id, selected_voice_id, selected_avatar_id = resolve_runtime_selection(
        stage_id,
        difficulty_id,
        voice_id,
        avatar_id,
    )

    training_prompt, training_summary = build_training_prompt_v2(
        selected_stage_id,
        DEFAULT_CUSTOMER_ID,
        selected_difficulty_id,
    )
    voice_config = resolve_voice(selected_voice_id)
    avatar_config = resolve_avatar_for_customer(
        training_summary.get("customer_id"),
        selected_avatar_id,
    )
    training_state = {
        **training_summary,
        "voice_id": selected_voice_id,
        "voice": voice_config.get("label", selected_voice_id),
        "avatar_id": avatar_config.get("id", selected_avatar_id),
        "avatar": avatar_config.get("label", selected_avatar_id),
    }
    raw_recent_memory = get_recent_memory(session_id, limit=10)
    recent_memory, removed_memory_lines = sanitize_recent_memory(raw_recent_memory)
    if removed_memory_lines:
        set_status("memory_sanitized", removed_role_reversed_lines=removed_memory_lines)
    memory_prompt = (
        f"【当前会话记忆】\n{recent_memory}"
        if recent_memory
        else "【当前会话记忆】\n暂无历史对话。"
    )
    avatar_prompt = (
        "【客户人物形象】\n"
        f"形象：{avatar_config.get('label', selected_avatar_id)}\n"
        f"角色：{avatar_config.get('role', '')}\n"
        f"性格气质：{avatar_config.get('temperament', '')}\n"
        f"表现方式：{avatar_config.get('visual_style', '')}"
    )
    role_guard_prompt = build_role_guard_prompt(training_summary.get("customer"))
    set_status(
        "training_loaded",
        prompt=prompt,
        training={**training_state, "session_id": session_id},
    )

    qwen_response = Generation.call(
        model=os.getenv("DASHSCOPE_LLM_MODEL", "qwen-turbo"),
        messages=[
            {
                "role": "system",
                "content": (
                    f"{load_customer_profile()}\n\n"
                    f"{training_prompt}\n\n"
                    f"{avatar_prompt}\n\n"
                    f"{memory_prompt}\n\n"
                    f"{role_guard_prompt}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        result_format="message",
    )

    if qwen_response.status_code == 200:
        raw_response_text = qwen_response.output.choices[0].message.content
        response_text = strip_spoken_identity(raw_response_text, training_summary.get("customer"))
        if not response_text:
            response_text = raw_response_text.strip() or "嗯，您先说说具体想怎么合作？"
        if is_role_reversed_sales_reply(response_text):
            response_text = build_customer_guardrail_reply(prompt, training_state)
            guardrail_reason = "role_reversed_sales_reply"
        elif is_hostile_or_confused_user_text(prompt):
            response_text = build_customer_guardrail_reply(prompt, training_state)
            guardrail_reason = "hostile_or_confused_user_text"
        else:
            guardrail_reason = ""
    else:
        raw_response_text = "抱歉，系统开小差了。"
        response_text = raw_response_text
        guardrail_reason = ""
        set_status("qwen_error", error=qwen_response.message)

    qwen_status = {"response_text": response_text}
    if raw_response_text != response_text:
        qwen_status["raw_response_text"] = raw_response_text
    if guardrail_reason:
        qwen_status["guardrail"] = guardrail_reason
    set_status("qwen_done", **qwen_status)

    turn_id = None
    turn_index = None
    try:
        turn_id, turn_index = save_turn(
            session_id=session_id,
            user_text=prompt,
            assistant_text=response_text,
            training=training_state,
            metadata={
                "stage_id": selected_stage_id,
                "difficulty_id": selected_difficulty_id,
                "voice_id": selected_voice_id,
                "avatar_id": selected_avatar_id,
            },
        )
        set_status(
            "turn_saved",
            prompt=prompt,
            response_text=response_text,
            training={**training_state, "session_id": session_id, "turn_index": turn_index},
        )
    except Exception as exc:
        set_status("save_error", error=f"{type(exc).__name__}: {exc}")

    audio_bytes = synthesize_with_retry(response_text, voice_config)
    if turn_id is not None:
        update_turn_audio_bytes(turn_id, len(audio_bytes))
    set_status("tts_done", audio_bytes=len(audio_bytes))

    return {
        "prompt": prompt,
        "response_text": response_text,
        "raw_response_text": raw_response_text,
        "guardrail": guardrail_reason,
        "training": training_state,
        "turn_index": turn_index,
        "audio_bytes": audio_bytes,
        "sample_rate": 24000,
    }
