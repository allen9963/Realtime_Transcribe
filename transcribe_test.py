# import torch
# from qwen_asr import Qwen3ASRModel
# import numpy as np
# import soundfile as sf
#
# def load_audio(audio_path: str) -> np.ndarray:
#     """讀取音訊，確認為 16 kHz，並轉成單聲道 float32。"""
#     audio, sample_rate = sf.read(
#         audio_path,
#         dtype="float32",
#         always_2d=True,
#     )
#
#     if sample_rate != 16_000:
#         raise ValueError("音訊必須先重新取樣為 16 kHz。")
#
#     mono_audio = audio.mean(axis=1)
#     return np.ascontiguousarray(mono_audio, dtype=np.float32)
#
#
# # audio = load_audio("audio\\10s_test.wav")
# # audio = load_audio("audio\\20260408140049_20m_3.wav")
# audio = load_audio("audio\\Batch3266636984.wav")
# print(len(audio))
#
# model = Qwen3ASRModel.from_pretrained(
#     "Qwen/Qwen3-ASR-0.6B",
#     dtype=torch.bfloat16,
#     device_map="cpu",
#     # attn_implementation="flash_attention_2",
#     max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
#     max_new_tokens=4096, # Maximum number of tokens to generate. Set a larger value for long audio input.
# )
#
# results = model.transcribe(
#     audio="audio\\Batch3266636984.wav",
#     language=None, # set "English" to force the language
# )
#
# print(results[0].language)
# print(results[0].text)
# import llama_cpp
from crispasr import Session

import numpy as np
import soundfile as sf
import time
import os

# 限制 ggml Vulkan 只能看到 Intel GPU
os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"
os.environ["CRISPASR_QWEN3_ASR_BENCH"] = "1"

# 開啟詳細紀錄
os.environ["CRISPASR_VERBOSE"] = "1"

def load_audio(audio_path: str) -> np.ndarray:
    """讀取音訊，確認為 16 kHz，並轉成單聲道 float32。"""
    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    if sample_rate != 16_000:
        raise ValueError("音訊必須先重新取樣為 16 kHz。")

    mono_audio = audio.mean(axis=1)
    return np.ascontiguousarray(mono_audio, dtype=np.float32)


# audio = load_audio("audio\\10s_test.wav")
# audio = load_audio("audio\\20260408140049_20m_3.wav")
# audio = load_audio("audio\\Batch3266636984.wav")
audio = load_audio("audio\\Batch3266636984_clean.wav")

start = time.time()
# s = Session("./models/qwen3-asr-0.6b-q4_k.gguf")
s = Session(
    # "./models/rumble-2-flash-q4_k.gguf",
    "./models/qwen3-asr-0.6b-q4_k.gguf",
    backend="qwen3",
    n_threads=4,
    use_gpu=True,
    flash_attn=True,
    n_gpu_layers=-1,
)
s.set_beam_size(1)
print(f"backend: {s.backend}")
chunk_ms = 1
for i in range(0, len(audio), chunk_ms*16000):
    speech_pcm = audio[0:i+chunk_ms*16000]

    print("speech_pcm length: ", len(speech_pcm))
    text = ""
    start = time.time()
    for seg in s.transcribe(speech_pcm, sample_rate=16000):
        # print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
        text += seg.text
    print(text)
    print(time.time() - start)
    # segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
#
# text = transcribe_audio(session, audio, op)
# text = ""
# for seg in s.transcribe(audio, sample_rate=16000):
#     # print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
#     text += seg.text
# print(text)
# print(time.time() - start)





# import wave
# import numpy as np
# from pywebrtc_audio import AudioProcessor
# import time
#
#
# def denoise_wav(input_path: str, output_path: str, ns_level: int = 2):
#     # 讀取完整 WAV
#     with wave.open(input_path, "rb") as wav:
#         sample_rate = wav.getframerate()
#         num_channels = wav.getnchannels()
#         sample_width = wav.getsampwidth()
#         num_frames = wav.getnframes()
#
#         if sample_width != 2:
#             raise ValueError("音檔必須是 16-bit PCM WAV")
#
#         if sample_rate not in (16000, 32000, 48000):
#             raise ValueError("請先轉換為 16k、32k 或 48kHz")
#
#         if num_channels not in (1, 2):
#             raise ValueError("僅支援 Mono 或 Stereo")
#
#         audio = np.frombuffer(
#             wav.readframes(num_frames),
#             dtype="<i2",
#         ).copy()
#
#     # 建立 WebRTC 降噪器
#     processor = AudioProcessor(
#         sample_rate=sample_rate,
#         num_channels=num_channels,
#         noise_suppression=True,
#         ns_level=ns_level,
#         high_pass_filter=True,
#         echo_cancellation=False,
#         auto_gain_control=False,
#     )
#
#     # 一次處理整個音檔
#     cleaned_audio = processor.process(audio)
#
#     # 儲存降噪後的 WAV
#     with wave.open(output_path, "wb") as wav:
#         wav.setnchannels(num_channels)
#         wav.setsampwidth(2)
#         wav.setframerate(sample_rate)
#         wav.writeframes(cleaned_audio.tobytes())
#
#     return cleaned_audio
#
# start = time.time()
# path = "./audio/Batch3266636984.wav"
# denoise_wav(
#     input_path="./audio/Batch3266636984.wav",
#     output_path="./audio/Batch3266636984_clean.wav",
#     # input_path="./audio/20260408140049_20m_3.wav",
#     # output_path="./audio/20260408140049_20m_3_clean.wav",
#     ns_level=2,
# )
# print(time.time() - start)
