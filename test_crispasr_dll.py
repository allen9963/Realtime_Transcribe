# # #
# # # import ctypes
# # # from pathlib import Path
# # # import transcribe_cpp
# # # from crispasr import _binding
# # #
# # # dll_path = Path(_binding._find_lib()).resolve()
# # #
# # # print(f"CrispASR DLL: {dll_path}")
# # # print(f"DLL exists: {dll_path.is_file()}")
# # #
# # # try:
# # #     library = ctypes.CDLL(str(dll_path))
# # #     print("CrispASR DLL loaded successfully")
# # # except OSError as error:
# # #     print(f"DLL loading failed: {error}")
# #
# #
# # import ctypes
# # from pathlib import Path
# #
# # # 先載入 CrispASR
# # from crispasr import _binding
# #
# # dll_path = Path(_binding._find_lib()).resolve()
# # crisp_library = ctypes.CDLL(str(dll_path))
# #
# # print("CrispASR DLL loaded successfully")
# #
# # # 再載入 transcribe.cpp
# # import transcribe_cpp
# #
# # print("transcribe.cpp imported successfully")
#
# import ctypes
# from ctypes import wintypes
#
# import transcribe_cpp
#
# kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
#
# kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
# kernel32.GetModuleHandleW.restype = wintypes.HMODULE
#
# kernel32.GetModuleFileNameW.argtypes = [
#     wintypes.HMODULE,
#     wintypes.LPWSTR,
#     wintypes.DWORD,
# ]
# kernel32.GetModuleFileNameW.restype = wintypes.DWORD
#
# dll_names = [
#     "ggml.dll",
#     "ggml-base.dll",
#     "ggml-cpu.dll",
#     "ggml-vulkan.dll",
# ]
#
# for dll_name in dll_names:
#     handle = kernel32.GetModuleHandleW(dll_name)
#
#     if not handle:
#         print(f"{dll_name}: Not loaded")
#         continue
#
#     path_buffer = ctypes.create_unicode_buffer(32768)
#
#     result = kernel32.GetModuleFileNameW(
#         handle,
#         path_buffer,
#         len(path_buffer),
#     )
#
#     if result:
#         print(f"{dll_name}: {path_buffer.value}")



import asyncio
from contextlib import ExitStack, asynccontextmanager, contextmanager
import json
import logging
import multiprocessing
from pathlib import Path
import signal
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
import time
import transcribe_cpp
import soundfile as sf
import numpy as np
app = FastAPI(title="SpeechTrack 即時轉錄 Demo")



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


def transcribe_audio(session, audio: np.ndarray) -> str:
    """使用已建立的 session 轉錄音訊。"""
    result = session.run(audio)
    return result.text

# audio = load_audio("audio\\10s_test.wav")
# audio = load_audio("audio\\20260408140049_20m_3.wav")
# audio = load_audio("audio\\recording-16k.wav")
# print(len(audio))

# Model 保持存活，就可以重複使用已載入的模型。
start = time.time()



class TranslationRequest(BaseModel):
    """翻譯端點接收轉錄原文和目標語言代碼。"""
    job_id: str
    file_path: str

@app.post("/test/transcribe")
def transcribe(payload: TranslationRequest, request: Request):
    path = payload.file_path.strip()
    print(path)
    audio = load_audio(path)
    with transcribe_cpp.Model(
            MODEL_PATH,
            backend="vulkan",
            device=selected_device,
    ) as model, transcribe_cpp.Model("models\\silero-vad-v6.2-F32.gguf", backend="cpu") as vad_model:
        if transcribe_cpp.Role.VAD not in vad_model.roles:
            raise RuntimeError("這份模型沒有 VAD 角色")
        op = None
        info = vad_model.vad_info
        start_time = time.time()
        with model.session() as session, vad_model.vad_session(n_threads=1) as vad:
            result = vad.run(
                audio,
                threshold=0.5,
                min_speech_ms=250,
                min_silence_ms=500,
                speech_pad_ms=30,
                max_speech_ms=0
            )
            result_json = []
            for segment in result.segments:
                start = segment.start_sample / info.sample_rate
                end = segment.end_sample / info.sample_rate
                speech_pcm = audio[segment.start_sample:segment.end_sample]
                print(f"語音 {start:.3f}～{end:.3f} 秒，{len(speech_pcm)} samples")
                text = transcribe_audio(session, speech_pcm)
                print(text)
                result_json.append({"start": round(start, 3), "end": round(end, 3), "text": text})
            print(time.time() - start_time)
            return {"type": "Result", "job_id": payload.job_id, "transcription": result_json}

            # for i in range(0, len(audio), chunk):
            #     start = time.time()
            #     text = transcribe_audio(session, audio[0: i + chunk], op)
            #
            #     print(f"[0~{len(audio[0: i + chunk]) / 16000}]: {text}")
            #     print(time.time() - start)


if __name__ == "__main__":
    # 預設只提供給本機瀏覽器。
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)