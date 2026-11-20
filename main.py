import numpy as np
import soundfile as sf

import time
import os
os.environ["TRANSCRIBE_NO_REPETITION_GUARD"] = "1"
import transcribe_cpp
from pywebrtc_audio import AudioProcessor
import wave

# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F32.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F16.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q8_0.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q4_K_M.gguf"
# MODEL_PATH = "models/whisper-small-F16.gguf"
# MODEL_PATH = "models/Qwen3-ASR-0.6B-Q8_0.gguf"
# MODEL_PATH = "models/whisper-small-Q4_K_M.gguf"
# MODEL_PATH = "models/Qwen3-ASR-0.6B-Q6_K.gguf"
# MODEL_PATH = "models/Qwen3-ASR-1.7B-Q4_K_M.gguf"
MODEL_PATH = "models/Qwen3-ASR-0.6B-Q5_K_M.gguf"
# MODEL_PATH = "models/Qwen3-ASR-0.6B-Q4_K_M.gguf"
print("CUDA 是否可用：", transcribe_cpp.backend_available("cuda"))
devices = transcribe_cpp.backends()
str_tmp = "Intel"
# str_tmp = "NVIDIA"
# 印出可用裝置
for device in devices:
    print(
        f"後端={device.kind}, "
        f"名稱={device.name}, "
        f"描述={device.description}"
    )
selected_device = next(
    (
        device
        for device in devices
        if device.kind == "vulkan"
        and str_tmp.casefold()
        in f"{device.name} {device.description}".casefold()
    ),
    None,
)
print(selected_device)

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


def transcribe_audio(session, audio: np.ndarray, o) -> str:
    """使用已建立的 session 轉錄音訊。"""
    try:
        result = session.run(audio)
        return result.text
    except transcribe_cpp.errors.OutputTruncated as e:
        print(dir(e))  # 看有哪些屬性
        print(getattr(e, "partial_result", None))  # 若有的話就是迴圈前的內容
        print(getattr(e, "status", None))
        print(getattr(e, "args", None))


# audio = load_audio("audio\\10s_test.wav")
# audio = load_audio("audio\\20260408140049_20m_3.wav")
# audio = load_audio("audio\\20260408140049_20m_3_clean.wav")
audio = load_audio("audio\\Batch3266636984.wav")
# audio = load_audio("audio\\Batch3266636984_clean.wav")
print(len(audio))

# Model 保持存活，就可以重複使用已載入的模型。
start = time.time()

with transcribe_cpp.Model(
    MODEL_PATH,
    backend="vulkan",
    device=selected_device,
) as model:
    op = None
    print(model.capabilities)
    chunk_ms = 1000 # ms
    chunk = max(1, int(16000 * chunk_ms / 1000))
    with model.session() as session:
        # text = transcribe_audio(session, audio, op)
        # print(text)
        # print(time.time() - start)

        for i in range(0, len(audio), chunk):
            start = time.time()
            text = transcribe_audio(session, audio[0 : i+chunk], op)

            print(f"[0~{len(audio[0 : i+chunk]) / 16000}]: {text}")
            print(time.time() - start)

# import transcribe_cpp as tc
#
# print("Python 套件版本：", tc.__version__)
# print("原生版本：", tc.native_version())
# print("原生提交：", tc.native_commit())
# print("函式庫路徑：", tc.library_path())
# print("原生套件來源：", tc.native_provider())

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
#     print(cleaned_audio.shape)
#     print(type(cleaned_audio))
#
#     # 儲存降噪後的 WAV
#     # with wave.open(output_path, "wb") as wav:
#     #     wav.setnchannels(num_channels)
#     #     wav.setsampwidth(2)
#     #     wav.setframerate(sample_rate)
#     #     wav.writeframes(cleaned_audio.tobytes())
#
#     return cleaned_audio
#
# audio = denoise_wav(input_path="audio\\Batch3266636984.wav", output_path="").astype(np.float32) / 32768.0
# with transcribe_cpp.Model(
#     MODEL_PATH,
#     backend="vulkan",
#     device=selected_device,
# ) as model:
#     op = None
#     print(model.capabilities)
#     chunk_ms = 1000 # ms
#     chunk = max(1, int(16000 * chunk_ms / 1000))
#     with model.session() as session:
#         with transcribe_cpp.Model(
#             "models/silero-vad-v6.2-F32.gguf", backend="cpu",
#         ) as vad:
#             # seg = audio[480800:1440000]
#             # # seg = audio[512800:1440000]
#             # print(seg.dtype, seg.shape, seg.flags["C_CONTIGUOUS"])
#             # print(seg.min(), seg.max(), np.abs(seg).mean(), np.isfinite(seg).all())
#             # text = transcribe_audio(session, seg, op)
#             # print(text)
#             # sf.write("seg2.wav", seg, 16000)  # 實際聽聽看，或丟給別的 ASR 跑
#             with vad.vad_session(n_threads=1) as v:
#                 result = v.run(
#                     audio,
#                     threshold=0.5,
#                     min_speech_ms=500,
#                     min_silence_ms=500,
#                     speech_pad_ms=30,
#                     max_speech_ms=0,
#                 )
#                 segments = []
#                 print(result)
#                 final_result = ""
#                 for segment in result.segments:
#                     start = segment.start_sample / 16000
#                     end = segment.end_sample / 16000
#                     speech_pcm = audio[segment.start_sample:segment.end_sample]
#                     print(f"語音 {start:.3f}～{end:.3f} 秒，{len(speech_pcm)} samples")
#                     text = transcribe_audio(session, speech_pcm, op)
#                     print(text)
#                     final_result += text
#
#                 print(final_result)
#             # chunk_ms = 30
#             # for i in range(0, len(audio), chunk_ms*16000):
#             #     speech_pcm = audio[i:i+chunk_ms*16000]
#             #
#             #     print("speech_pcm length: ", len(speech_pcm))
#             #     text = transcribe_audio(session, speech_pcm, op)
#             #     print(text)
#             #     segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
#             # start = time.time()
#             # text = transcribe_audio(session, audio, op)
#             #
#             # print(text)
#             # print(time.time() - start)
#
#         # for i in range(0, len(audio), chunk):
#         #     start = time.time()
#         #     text = transcribe_audio(session, audio[0 : i+chunk], op)
#         #
#         #     print(f"[0~{len(audio[0 : i+chunk]) / 16000}]: {text}")
#         #     print(time.time() - start)