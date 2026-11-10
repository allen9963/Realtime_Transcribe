# from __future__ import annotations
#
# import argparse
# import array
# import sys
# import time
# import wave
#
# import transcribe_cpp
#
# # MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F32.gguf"
# # MODEL_PATH = "models/whisper-large-v3-turbo-Q8_0.gguf"
# MODEL_PATH = "models/Qwen3-ASR-0.6B-Q8_0.gguf"
#
# def calculate_device_select(device_name="Intel"):
#     devices = transcribe_cpp.backends()
#     str_tmp = device_name
#     # str_tmp = "NVIDIA"
#     # 印出可用裝置
#     for device in devices:
#         print(
#             f"後端={device.kind}, "
#             f"名稱={device.name}, "
#             f"描述={device.description}"
#         )
#     selected_device = next(
#         (
#             device
#             for device in devices
#             if device.kind == "vulkan"
#                and str_tmp.casefold() # 用來將字串轉換為適合「不區分大小寫的字串比較（case-insensitive comparison）」的格式
#                in f"{device.name} {device.description}".casefold()
#         ),
#         None,
#     )
#     print(selected_device)
#     return selected_device
#
# def load_wav_mono16k(path: str) -> array.array:
#     with wave.open(path, "rb") as w:
#         if w.getsampwidth() != 2 or w.getframerate() != 16000 or w.getnchannels() != 1:
#             raise SystemExit(
#                 f"{path}: expected 16 kHz 16-bit mono — resample first, e.g. "
#                 f"ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
#             )
#         pcm16 = array.array("h")
#         pcm16.frombytes(w.readframes(w.getnframes()))
#     if sys.byteorder == "big":
#         pcm16.byteswap()
#     return array.array("f", (s / 32768.0 for s in pcm16))
#
#
# def render(text: "transcribe_cpp.StreamText") -> None:
#     # One status line: committed text plain, tentative dimmed. \r-rewrite is
#     # enough for a demo; a real UI keys off StreamUpdate.committed_changed /
#     # tentative_changed to redraw minimally.
#     line = f"{text.committed}\x1b[2m{text.tentative}\x1b[0m"
#     print(f"\r\x1b[K{line}", end="", flush=True)
#
#
# def main() -> int:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", default=MODEL_PATH)
#     ap.add_argument("--audio", default="audio\\20260408140049_20m_3.wav")
#     ap.add_argument("--chunk-ms", type=int, default=1000,
#                     help="feed size in milliseconds (default: 250)")
#     ap.add_argument("--realtime", action="store_true",
#                     help="sleep each chunk's duration (simulate a live mic)")
#     ap.add_argument("--language", default=None)
#     args = ap.parse_args()
#
#     print(args)
#
#     pcm = load_wav_mono16k(args.audio)
#     chunk = max(1, int(16000 * args.chunk_ms / 1000))
#     print(chunk)
#
#     with transcribe_cpp.Model(args.model, backend="vulkan", device=calculate_device_select()) as model:
#         caps = model.capabilities
#         print(caps)
#         if not caps.supports_streaming:
#             raise SystemExit(
#                 f"{model.arch}/{model.variant} does not support streaming — "
#                 "use a streaming model (e.g. moonshine-streaming-tiny)"
#             )
#         print(f"model: {model.arch}/{model.variant} on {model.backend} | "
#               f"feeding {args.chunk_ms} ms chunks"
#               f"{' at realtime pace' if args.realtime else ''}")
#
#         with model.session() as session:
#             with session.stream(language=args.language, commit_policy="stable_prefix") as stream:
#                 for i in range(0, len(pcm), chunk):
#                     update = stream.feed(pcm[i:i + chunk])
#                     # print(update)
#                     if update.committed_changed or update.tentative_changed:
#                         print(update)
#                         print(stream.text())
#                         # render(stream.text())
#                     if args.realtime:
#                         time.sleep(args.chunk_ms / 1000)
#                 update = stream.finalize()
#                 final = stream.text()
#                 # print(final)
#                 render(final)
#                 print()  # land the cursor after the live line
#                 print(f"\nfinal ({update.audio_committed_ms / 1000:.1f}s audio, "
#                       f"revision {stream.revision}):\n{final.committed.strip()}")
#     return 0
#
#
# if __name__ == "__main__":
#     raise SystemExit(main())


import numpy as np
import soundfile as sf
import transcribe_cpp as tc
import time


# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F32.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F16.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q8_0.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q4_K_M.gguf"
# MODEL_PATH = "models/whisper-small-F16.gguf"
MODEL_PATH = "models/Qwen3-ASR-0.6B-Q8_0.gguf"
print("CUDA 是否可用：", tc.backend_available("cuda"))
devices = tc.backends()
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
# audio = load_audio("audio\\20260408140049_20m_3.wav")
# audio = load_audio("audio\\recording-16k.wav")

pcm = load_audio("audio\\recording-16k.wav")

# Model 保持存活，就可以重複使用已載入的模型。
start = time.time()
speech_region = []
with tc.Model("models\\silero-vad-v6.2-F32.gguf", backend="cpu") as model:
    if tc.Role.VAD not in model.roles:
        raise RuntimeError("這份模型沒有 VAD 角色")

    # info = model.vad_info
    # with model.vad_session(n_threads=1) as vad:
    #     result = vad.run(
    #         pcm,
    #         # threshold=0.5,
    #         # min_speech_ms=250,
    #         # min_silence_ms=500,
    #         # speech_pad_ms=30,
    #         # max_speech_ms=0,
    #         threshold=0.5,
    #         min_speech_ms=500,
    #         min_silence_ms=400,
    #         speech_pad_ms=30,
    #         max_speech_ms=8000,
    #     )
    #
    #     for segment in result.segments:
    #         start = segment.start_sample / info.sample_rate
    #         end = segment.end_sample / info.sample_rate
    #         speech_pcm = pcm[segment.start_sample:segment.end_sample]
    #         print(f"語音 {start:.3f}～{end:.3f} 秒，{len(speech_pcm)} samples")
    #
    #     print("處理的 frames：", len(result.probs))
    #     print("耗時：", vad.timings)
    info = model.vad_info
    chunk_samples = info.sample_rate * 32 // 1000
    print("chunk_samples: ", chunk_samples)
    print("info.frame_samples: ", info.frame_samples)
    # silence = 0
    silence_region = []
    def show_probabilities(result: tc.VadResult) -> bool:
        # global silence
        # for offset, probability in enumerate(result.probs):
        #     frame_index = result.first_frame + offset
        #     print("frame_index: ", frame_index)
        #     start_seconds = frame_index * info.frame_samples / info.sample_rate
        #     print(f"{start_seconds:.3f}s，語音機率={probability:.3f}")


        #     if probability < 0.5:
        #         silence += 1
        #     else:
        #         silence = 0
        # return silence >= 15
        if len(result.probs) > 0:
            probability = result.probs[0]
            if probability < 0.5:
                return True
            else:
                return False


    count = 0

    speech_start_time = -1
    speech_end_time = -1
    silence_count = 0
    with model.vad_session(n_threads=1) as vad:
        for start in range(0, len(pcm), chunk_samples):
            result = vad.stream_feed(pcm[start:start + chunk_samples])
            print("Result: ", result)
            print(f"[{start / info.sample_rate:.3f}~{(start + chunk_samples) / info.sample_rate:.3f}]")
            is_silence = show_probabilities(result)
            count+=1
            if is_silence:
                silence_count += 1
                # speech_region.append(f"[{start / info.sample_rate:.3f}~{(start + chunk_samples) / info.sample_rate:.3f}]")
            else:
                silence_count = 0
                if speech_start_time < 0:
                    print("123===============")
                    speech_start_time = start / info.sample_rate

            print("speech_start_time: ", speech_start_time)

            # if is_speech is False:
            #     # silence_region.append(f"[{start / info.sample_rate:.3f}~{(start + chunk_samples) / info.sample_rate:.3f}]")
            #     silence += 1
            # else:
            #     speech_start_time

            if silence_count > 15:
                speech_end_time = (start + chunk_samples) / info.sample_rate
                if speech_start_time >= 0:
                    speech_region.append(
                        [round(speech_start_time, 3), round(speech_end_time, 3)])
                speech_start_time = -1
                silence_count = 0
            print("silence_count: ", silence_count)
            print("==================")

        tail = vad.stream_flush()
        show_probabilities(tail)
        speech_end_time = (start + chunk_samples) / info.sample_rate
        if speech_start_time >= 0:
            speech_region.append(
                [round(speech_start_time, 3), round(speech_end_time, 3)])


        print(count)
        print(speech_region)

with tc.Model(
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

        # for i in range(0, len(audio), chunk):
        #     start = time.time()
        #     text = transcribe_audio(session, audio[0 : i+chunk], op)
        #
        #     print(f"[0~{len(audio[0 : i+chunk]) / 16000}]: {text}")
        #     print(time.time() - start)
        final_str = ""
        for i in speech_region:
            start = time.time()
            text = transcribe_audio(session, pcm[int(i[0] * 16000) : int(i[1] * 16000)], op)

            # print(f"[0~{len(pcm[0 : i+chunk]) / 16000}]: {text}")
            print(f"[{i[0]} ~ {i[1]}]: {text}")
            final_str += text
            print(time.time() - start)

        print("final_str: ", final_str)

# import transcribe_cpp as tc
#
# print("Python 套件版本：", tc.__version__)
# print("原生版本：", tc.native_version())
# print("原生提交：", tc.native_commit())
# print("函式庫路徑：", tc.library_path())
# print("原生套件來源：", tc.native_provider())