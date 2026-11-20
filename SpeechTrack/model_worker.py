import signal
from contextlib import ExitStack, asynccontextmanager, contextmanager, suppress
from pathlib import Path

def run_model_process(connection, open_model, model_args):
    """在子程序中載入模型，並透過 Pipe 收送指令與結果。"""
    # Ctrl+C 由主程序處理，子程序在 Pipe 關閉時釋放模型。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with connection, ExitStack() as resources:
            try:
                commands = resources.enter_context(open_model(*model_args))
            except Exception as error:
                connection.send((False, f"{type(error).__name__}: {error}"))
                return

            # 模型載入完成後才通知主程序，原生例外只傳文字。
            connection.send((True, None))
            while True:
                command, args = connection.recv()
                # print("[DBF] command: ", command)
                try:
                    result = commands[command](*args)
                except Exception as error:
                    connection.send((False, f"{type(error).__name__}: {error}"))
                else:
                    connection.send((True, result))
    except (EOFError, OSError):
        # 主程序關閉 Pipe 時，由 context manager 釋放原生模型。
        pass


@contextmanager
def asr_commands(name: str, model_dir: str, sample_rate: int,):
    """啟動時載入 ASR，第一次辨識時才載入 Silero VAD。"""
    import numpy as np
    import soundfile as sf
    import transcribe_cpp

    MODEL_DIR = Path(model_dir)
    SAMPLE_RATE = sample_rate

    selected_device = next(
        (
            device
            for device in transcribe_cpp.backends()
            if device.kind == "vulkan"
            and "intel" in f"{device.name} {device.description}".casefold()
        ),
        None,
    )
    with ExitStack() as models, ExitStack() as recording:
        print("Model_name: ", name)
        asr_model = models.enter_context(transcribe_cpp.Model(
            MODEL_DIR / f"{name}.gguf", backend="vulkan", device=selected_device,
        ))
        vad_model = None
        session = vad = None

        def start():
            """為即時錄音或檔案轉錄建立 ASR 與 VAD session。"""
            nonlocal session, vad, vad_model
            recording.close()
            # 第一次需要分段時才載入 VAD，後續由同一個子程序重用。
            if vad_model is None:
                vad_model = models.enter_context(transcribe_cpp.Model(
                    MODEL_DIR / "silero-vad-v6.2-F32.gguf", backend="cpu",
                ))
            with ExitStack() as sessions:
                session = sessions.enter_context(asr_model.session())
                vad = sessions.enter_context(vad_model.vad_session(n_threads=1))
                recording.enter_context(sessions.pop_all())

        def score(frame: bytes) -> float | None:
            """將一包 PCM 送入 VAD 並回傳語音機率。"""
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
            probabilities = vad.stream_feed(samples).probs
            return probabilities[0] if probabilities else None

        def recognize(pcm: bytes) -> str:
            """將 PCM 轉成 float32，再交給 ASR session 辨識。"""
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            if not samples.any():
                return ""
            return session.run(samples).text.strip()

        def recognize_file(file_path: str) -> list[dict]:
            """依 VAD 的分段結果轉錄 16 kHz 音檔並回傳時間。"""
            audio, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
            if sample_rate != SAMPLE_RATE:
                raise ValueError("音訊必須先重新取樣為 16 kHz。")
            audio = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
            result = vad.run(
                audio,
                threshold=0.7,
                min_speech_ms=500,
                min_silence_ms=500,
                speech_pad_ms=30,
                max_speech_ms=0,
            )
            segments = []
            for segment in result.segments:
                start = segment.start_sample / SAMPLE_RATE
                end = segment.end_sample / SAMPLE_RATE
                speech_pcm = audio[segment.start_sample:segment.end_sample]
                print(f"語音 {start:.3f}～{end:.3f} 秒，{len(speech_pcm)} samples")
                text = session.run(speech_pcm).text.strip()
                print(text)
                segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
            return segments

        yield {
            "start": start,
            "stop": recording.close,
            "score": score,
            "recognize": recognize,
            "recognize_file": recognize_file,
        }

@contextmanager
def translation_commands(model_dir: str, language_relation: dict[str, str],):
    """在獨立的翻譯子程序中載入 Hy-MT2。"""
    from llama_cpp import Llama
    MODEL_DIR = Path(model_dir)
    LANGUAGE_RELATION = language_relation

    model = Llama(
        model_path=str(MODEL_DIR / "Hy-MT2-1.8B-Q4_K_M.gguf"),
        verbose=False,
        enable_thinking=False,
    )

    def translate(text: str, target: str) -> str:
        """用指定的語言翻譯原文，回傳模型產生的文字。"""
        language = LANGUAGE_RELATION[target]
        # prompt = f"Translate the following segment into {language}, without additional explanation：{text}"
        prompt = f"Translate the following text into {language}. Note that you should only output the translated result without any additional explanation:\n\n{text}"
        print(prompt)
        print("target: ", target)
        print("LANGUAGE_RELATION: ", language)
        result = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        translation = result["choices"][0]["message"]["content"]
        print(result)
        print(translation)
        return translation.strip()

    try:
        yield {"translate": translate}
    finally:
        model.close()