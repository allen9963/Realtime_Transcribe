import transcribe_cpp as tc
import argparse
import array
import sys
import time
import wave

def calculate_device_select(device_name="Intel"):
    devices = tc.backends()
    str_tmp = device_name
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
               and str_tmp.casefold() # 用來將字串轉換為適合「不區分大小寫的字串比較（case-insensitive comparison）」的格式
               in f"{device.name} {device.description}".casefold()
        ),
        None,
    )
    print(selected_device)
    return selected_device

def load_wav_mono16k(path: str) -> array.array:
    with wave.open(path, "rb") as w:
        params = w.getparams()
        print(params)
        if w.getsampwidth() != 2 or w.getframerate() != 16000 or w.getnchannels() != 1:
            raise SystemExit(
                f"{path}: expected 16 kHz 16-bit mono — resample first, e.g. "
                f"ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
            )
        pcm16 = array.array("h")
        pcm16.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        pcm16.byteswap()
    return array.array("f", (s / 32768.0 for s in pcm16))

pcm = load_wav_mono16k("audio\\recording-16k.wav")
print(len(pcm))

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
    speech_region = []
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
                        f"[{speech_start_time:.3f}~{speech_end_time:.3f}]")
                speech_start_time = -1
                silence_count = 0
            print("silence_count: ", silence_count)
            print("==================")

        tail = vad.stream_flush()
        show_probabilities(tail)


        print(count)
        print(speech_region)



        #[8.928~8.960]