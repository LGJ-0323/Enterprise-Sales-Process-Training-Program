import os
import tempfile
import asyncio
import subprocess
import numpy as np
import soundfile as sf
import edge_tts
from dashscope.audio.asr import Recognition
from fastrtc import ReplyOnPause, Stream
from openai import OpenAI

# ========== 1. STT（不变） ==========
class DashScopeSTT:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    def stt(self, audio):
        sample_rate, audio_data = audio
        pcm_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
            f.write(pcm_bytes)
            temp_path = f.name
        try:
            recognition = Recognition(
                model='paraformer-realtime-v2',
                format='pcm',
                sample_rate=sample_rate,
                callback=None
            )
            result = recognition.call(temp_path)
            if result.status_code == 200:
                sentence_data = result.get_sentence()
                if sentence_data is None:
                    return ""
                if isinstance(sentence_data, list):
                    return " ".join(s.text for s in sentence_data if hasattr(s, 'text'))
                else:
                    return sentence_data.text
            else:
                raise Exception(f"语音识别失败: {result.message}")
        finally:
            os.unlink(temp_path)

# ========== 2. 改进版 TTS（支持动态采样率） ==========
class EdgeTTSModel:
    def __init__(self, voice="zh-CN-XiaoxiaoNeural"):
        self.voice = voice

    def stream_tts_sync(self, text, target_sample_rate=24000):
        """
        生成器，每次 yield (sample_rate, audio_array)
        target_sample_rate: 输出采样率，应与 fastrtc 期望的音频格式匹配
        """
        if not text.strip():
            return

        async def _generate_mp3(text, voice, out_path):
            communicate = edge_tts.Communicate(text, voice)
            with open(out_path, 'wb') as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            mp3_path = tmp_mp3.name

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_generate_mp3(text, self.voice, mp3_path))
            loop.close()

            # 转为 WAV：单声道，目标采样率
            wav_path = mp3_path + ".wav"
            subprocess.run([
                'ffmpeg', '-y', '-i', mp3_path,
                '-ac', '1',                     # 单声道
                '-ar', str(target_sample_rate), # 动态采样率
                '-f', 'wav', '-acodec', 'pcm_s16le',
                wav_path
            ], check=True, capture_output=True)

            audio_array, sr = sf.read(wav_path)
            if audio_array.size == 0 or np.max(np.abs(audio_array)) < 1e-6:
                print("[TTS 警告] 静音音频")
                return

            # fastrtc 需要 (sample_rate, audio_array)，audio_array 应为 float32
            yield (sr, audio_array.astype(np.float32))

            os.unlink(wav_path)
        finally:
            os.unlink(mp3_path)

# ========== 3. 主程序 ==========
def main():
    stt_model = DashScopeSTT()
    tts_model = EdgeTTSModel(voice="zh-CN-XiaoxiaoNeural")

    dashscope_client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    def echo(audio):
        # 获取输入音频的采样率，用于后续 TTS 输出
        input_sample_rate, audio_array = audio

        # 1. 语音识别
        prompt = stt_model.stt(audio)
        print(f"[识别] {prompt}")

        if not prompt.strip():
            return

        # 2. LLM 生成回复
        try:
            response = dashscope_client.chat.completions.create(
                model="qwen-turbo-latest",
                messages=[
                    {"role": "system", "content": "你是一个语音助手，请始终使用中文回复，语气自然、简洁。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
            )
            reply_text = response.choices[0].message.content
            print(f"[LLM] {reply_text}")
        except Exception as e:
            print(f"LLM 错误: {e}")
            return

        # 3. TTS 输出，采样率与输入相同
        for chunk in tts_model.stream_tts_sync(reply_text, target_sample_rate=input_sample_rate):
            yield chunk

    # 启动流（不再使用 output_sample_rate 参数）
    stream = Stream(
        ReplyOnPause(echo),
        modality="audio",
        mode="send-receive"
    )
    stream.ui.launch()

if __name__ == "__main__":
    main()