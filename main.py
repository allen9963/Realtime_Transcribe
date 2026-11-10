import numpy as np
import soundfile as sf
import transcribe_cpp
import time


# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F32.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F16.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q8_0.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q4_K_M.gguf"
# MODEL_PATH = "models/whisper-small-F16.gguf"
MODEL_PATH = "models/Qwen3-ASR-0.6B-Q8_0.gguf"
print("CUDA 是否可用：", transcribe_cpp.backend_available("cuda"))
devices = transcribe_cpp.backends()
# str_tmp = "Intel"
str_tmp = "NVIDIA"
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
    result = session.run(audio)
    return result.text

# audio = load_audio("audio\\10s_test.wav")
audio = load_audio("audio\\20260408140049_20m_3.wav")
# audio = load_audio("audio\\recording-16k.wav")
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