import numpy as np
import soundfile as sf
import transcribe_cpp
import time
import os


# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F32.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf"
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F16.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q8_0.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q4_K_M.gguf"
# MODEL_PATH = "models/whisper-small-F16.gguf"
MODEL_PATH = "models/Qwen3-ASR-0.6B-Q8_0.gguf"
# MODEL_PATH = "models/whisper-small-F16.gguf"
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

    print("Audio len: ",len(audio))
    result = session.run(audio)
    return result.text

# audio = load_audio("audio\\10s_test.wav")
# audio = load_audio("audio\\20260408140049_20m_3.wav")
# print(type(audio))
audio = load_audio("audio\\recording-16k.wav")

# Model 保持存活，就可以重複使用已載入的模型。
start = time.time()

with transcribe_cpp.Model(MODEL_PATH,backend="vulkan",device=selected_device) as transcribe_model, transcribe_cpp.Model("models\\silero-vad-v6.2-F32.gguf", backend="cpu") as vad_model:
    op = None
    print(transcribe_model.capabilities)
    chunk_ms = 1000 # ms
    chunk = max(1, int(16000 * chunk_ms / 1000))
    OVERLAP_MS = 20
    OVERLAP_SAMPLE = OVERLAP_MS * 16000
    with transcribe_model.session() as session, vad_model.vad_session(n_threads=1) as vad:
        # text = transcribe_audio(session, audio, op)
        # print(text)
        # print(time.time() - start)
        now_audio_segments_count = 0
        previous_audio_segments_count = 0
        final_str = []
        f_str = ""
        pre_partial = ""
        audio_data_start_sample = 0
        segment_start_sample = -1
        pre_segment_time = [0, 0]
        if transcribe_cpp.Role.VAD not in vad_model.roles:
            raise RuntimeError("這份模型沒有 VAD 角色")
        for i in range(0, len(audio), chunk):
            start = time.time()

            pcm = audio[audio_data_start_sample : i+chunk]
            # pcm = audio[0: i + chunk]
            print(f"pcm [0~{i+chunk}]")
            print(f"pcm[{audio_data_start_sample}~{i+chunk}]")
            print("Now_pcm_len: ", len(pcm))
            info = vad_model.vad_info
            result = vad.run(
                pcm,
                threshold=0.7,
                min_speech_ms=250,
                min_silence_ms=500,
                speech_pad_ms=30,
                max_speech_ms=0,
            )
            # if len(result.segments) > 0:
            #     print(result.segments[-1])
            #     now_audio_segments_count = len(result.segments)
            #
            #     for segment in result.segments:
            #         start = segment.start_sample / info.sample_rate
            #         end = segment.end_sample / info.sample_rate
            #         speech_pcm = pcm[segment.start_sample:segment.end_sample]
            #         print(f"語音 {start:.3f}～{end:.3f} 秒，{len(speech_pcm)} samples")
            #     start = time.time()
            #     text = transcribe_audio(session, speech_pcm, op)
            #
            #     if now_audio_segments_count > 1 or text == pre_partial:
            #         final_str.append(pre_partial)
            #         audio_data_start_sample = result.segments[0].end_sample
            #         f_str += pre_partial
            #
            #     print(f"[0~{len(audio[0 : i+chunk]) / 16000}]: {text}")
            #     print(time.time() - start)
            #
            #     previous_audio_segments_count = now_audio_segments_count
            #     pre_partial = text
            #
            #     print("======================================")
            segment_list = list(result.segments)
            if len(segment_list) > 0:
                now_audio_segments_count = len(result.segments)
                if now_audio_segments_count > 1:
                    if (pre_segment_time[0] == segment_list[0].start_sample) and (
                            pre_segment_time[1] == segment_list[0].end_sample):
                        audio_data_start_sample += segment_list[0].end_sample
                    else:
                        audio_data_start_sample += pre_segment_time[1]
                    f_str += text
                    text = ""
                    print("f_str Increase!!!!!!!!!!!!")
                for segment in result.segments:
                    start = segment.start_sample / info.sample_rate
                    end = segment.end_sample / info.sample_rate
                    speech_pcm = pcm[segment.start_sample:segment.end_sample]
                    print(f"語音 {start:.3f}～{end:.3f} 秒，{len(speech_pcm)} samples")

                start = time.time()
                text = transcribe_audio(session, speech_pcm, op)
                print("Partial str: ", text)
                print(time.time() - start)

                previous_audio_segments_count = now_audio_segments_count
                if previous_audio_segments_count > 0:
                    pre_segment_time[0] = segment_list[0].start_sample
                    pre_segment_time[1] = segment_list[0].end_sample
                print("====================")
        # print(final_str)
        f_str += text
        print("FINAL str: ", f_str)
# import transcribe_cpp as tc
#
# print("Python 套件版本：", tc.__version__)
# print("原生版本：", tc.native_version())
# print("原生提交：", tc.native_commit())
# print("函式庫路徑：", tc.library_path())
# print("原生套件來源：", tc.native_provider())