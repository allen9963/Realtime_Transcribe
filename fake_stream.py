"""執行 python fake_stream.py，然後開啟 http://127.0.0.1:8000。"""

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
import soundfile as sf
import numpy as np


# WebSocket 每則二進位訊息固定為 32 ms 的 16 kHz 單聲道 PCM16LE。
SAMPLE_RATE = 16_000
FRAME_SAMPLES = SAMPLE_RATE * 32 // 1000                                 # 512
FRAME_BYTES = FRAME_SAMPLES * 2                                          # 1024
UPDATE_BYTES = FRAME_BYTES * 32  # 累積約 1 秒音訊後更新暫定文字。            # 32768
SEGMENT_BYTES = FRAME_BYTES * 320  # 每約 10 秒定稿，限制重複辨識的音訊長度。   # 327680

SILENCE_CHUNK = 16 # 32ms * 14
SILENCE_BYTES = SILENCE_CHUNK * FRAME_BYTES

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_NAMES = ("whisper-small-F16", "Qwen3-ASR-0.6B-Q8_0", "whisper-large-v3-turbo-Q4_K_M", "whisper-large-v3-turbo-Q8_0", "whisper-small-Q4_K_M", "Qwen3-ASR-0.6B-Q4_K_M")
# 所有轉錄模型共用 MADLAD 的目標語言清單。
TRANSLATION_TARGETS = ("en", "ja", "ko", "zh", "de", "fr", "es")
LANGUAGE_RELATION = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}
logger = logging.getLogger("uvicorn.error")


class ModelWorkerError(RuntimeError):
    """模型子程序載入、推論或通訊失敗。"""


def run_model_process(connection, open_model, model_args):
    """子程序內載入模型，透過 Pipe 收送指令和可序列化的結果。"""
    # Ctrl+C 由主程序處理，子程序等待 Pipe 關閉再釋放模型。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with connection, ExitStack() as resources: # 建立一個 ExitStack
            try:
                commands = resources.enter_context(open_model(*model_args)) # 動態地將所有檔案加入 stack 管理，因為ASR需要load兩個model
            except Exception as error:
                connection.send((False, f"{type(error).__name__}: {error}"))
                return

            # 模型載入完成後才通知主程序。原生例外只傳文字，避免主程序反序列化時載入 DLL。
            connection.send((True, None))
            while True:
                command, args = connection.recv() # 接收ASR或是Translate動作以及需要的參數
                print("[DBF] command: ", command)
                try:
                    result = commands[command](*args)
                except Exception as error:
                    connection.send((False, f"{type(error).__name__}: {error}"))
                else:
                    connection.send((True, result))
    except (EOFError, OSError):
        # 主程序關閉 Pipe 時，離開 context manager 並釋放原生模型。
        pass


class ModelProcess:
    """使用 spawn 啟動單一模型子程序，並串行處理該程序的指令。"""

    def __init__(self, name, open_model, *model_args):
        context = multiprocessing.get_context("spawn")
        self._lock = Lock()
        self._connection, child = context.Pipe()  # context.Pipe()是建立父程序和子程序的的溝通管道，self._connection為父程序從送資料的變數，child是子程序從送資料的變數
        self._process = context.Process(
            target=run_model_process, args=(child, open_model, model_args), name=name
        )
        try:
            self._process.start()
        except Exception as error:
            self._connection.close()
            self._process.close()
            raise ModelWorkerError(f"{name} 無法啟動：{error}") from error
        finally:
            child.close()

        # 等待就緒，避免切換至一個載入失敗的模型。
        try:
            self._receive()
        except (EOFError, OSError, ModelWorkerError) as error:
            self.close()
            raise ModelWorkerError(f"{name} 模型載入失敗：{error}") from error

    def is_alive(self) -> bool:
        """判斷是否還能使用目前的子程序。"""
        process = self._process
        return process is not None and process.is_alive()

    def _receive(self):
        """等待單一回覆，模型卡住時讓呼叫端能結束子程序。"""
        if not self._connection.poll(120):
            raise TimeoutError("等待模型子程序回覆逾時")
        success, result = self._connection.recv()
        if not success:
            raise ModelWorkerError(result)
        return result

    def call(self, command: str, *args):
        """每個模型一次處理一個請求，保持指令與回覆成對。"""
        with self._lock:
            if not self.is_alive():
                raise ModelWorkerError("模型子程序已停止")
            try:
                self._connection.send((command, args))
                return self._receive()
            except (EOFError, OSError) as error:
                self._close()
                raise ModelWorkerError(f"模型子程序通訊失敗：{error}") from error

    def close(self) -> None:
        """關閉通道，等待子程序釋放模型後退出。"""
        with self._lock:
            self._close()

    def _close(self) -> None:
        """持有鎖時執行關閉；逾時才終止子程序。"""
        process = self._process
        if process is None:
            return
        self._connection.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()
        self._process = None




@contextmanager
def asr_commands(name: str):
    """僅在 ASR 子程序載入 transcribe.cpp、選定的 ASR 模型及 Silero VAD。"""
    import numpy as np
    import transcribe_cpp

    devices = transcribe_cpp.backends()
    selected_device = next(
        (
            device
            for device in devices
            if device.kind == "vulkan"
            and "intel" in f"{device.name} {device.description}".casefold()
        ),
        None,
    )
    with ExitStack() as models, ExitStack() as recording:
        asr_model = models.enter_context(transcribe_cpp.Model(
            MODEL_DIR / f"{name}.gguf", backend="vulkan", device=selected_device,
        ))
        vad_model = models.enter_context(transcribe_cpp.Model(
            MODEL_DIR / "silero-vad-v6.2-F32.gguf", backend="cpu",
        ))
        session = vad = None

        def start():
            """每次錄音建立新的 ASR 與 VAD session，避免帶入上一場錄音的狀態。"""
            nonlocal session, vad
            recording.close()
            with ExitStack() as sessions:
                session = sessions.enter_context(asr_model.session())
                vad = sessions.enter_context(vad_model.vad_session(n_threads=1))
                recording.enter_context(sessions.pop_all())

        def score(frame: bytes) -> float | None:
            """將一包 PCM 交給子程序中的 VAD session，回傳語音機率。"""
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
            probabilities = vad.stream_feed(samples).probs
            return probabilities[0] if probabilities else None

        def recognize(pcm: bytes) -> str:
            """將 PCM 轉成 float32，再透過子程序中的 ASR session 取得文字。"""
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            if not samples.any():
                return ""
            return session.run(samples).text.strip()

        def recognize_file(file_path: str) -> list:

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

            audio = load_audio(file_path)
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
                start = segment.start_sample / SAMPLE_RATE
                end = segment.end_sample / SAMPLE_RATE
                speech_pcm = audio[segment.start_sample:segment.end_sample]
                print(f"語音 {start:.3f}～{end:.3f} 秒，{len(speech_pcm)} samples")
                text = session.run(speech_pcm).text.strip()
                print(text)
                result_json.append({"start": round(start, 3), "end": round(end, 3), "text": text})
            return result_json




        yield {"start": start, "stop": recording.close, "score": score, "recognize": recognize, "recognize_file": recognize_file}


@contextmanager
def translation_commands():
    """僅在翻譯子程序載入 CrispASR 和 MADLAD 模型。"""
    # from crispasr import Session
    # from langdetect import DetectorFactory, LangDetectException, detect
    #
    # DetectorFactory.seed = 0
    # with Session(
    #     model_path=str(MODEL_DIR / "madlad400-3b-mt-q4_k.gguf"),
    #     backend="madlad",
    #     n_threads=4,
    # ) as model:
    #     def translate(text: str, target: str) -> str:
    #         """根據實際轉錄文字判斷來源語言，並在同一個 session 翻譯。"""
    #         try:
    #             source = detect(text)
    #         except LangDetectException:
    #             source = "en"  # 純數字等無法判定語言的文字沿用原本的英文預設。
    #         return model.translate_text(text=text, src_lang=source, tgt_lang=target).strip()
    #
    #     yield {"translate": translate}
    from llama_cpp import Llama

    llm = Llama(
        # model_path="models/Qwen3.5-2B-Q4_K_M.gguf",
        # model_path="models/Qwen3-0.6B-Q8_0.gguf",
        # model_path="models/Index-Translate-2B.Q8_0.gguf",
        model_path="models/Hy-MT2-1.8B-Q4_K_M.gguf",

        verbose=False,
        enable_thinking=False
    )

    def translate(text: str, target: str) -> str:
        # user_input = f"Please translate the following text into {LANGUAGE_RELATION[target]} and output the translation directly without any explanation. \n\n{text}"
        # user_input = f"Please translate the following text into {LANGUAGE_RELATION[target]} and output the translation directly without any explanation. \n\n{text} /no_think"
        user_input = f"Translate the following segment into {LANGUAGE_RELATION[target]}, without additional explanation：{text}"
        print(user_input)
        # print(instruction)
        print("target: ", target)
        print("LANGUAGE_RELATION: ", LANGUAGE_RELATION[target])
        result = llm.create_chat_completion(
            messages=[
                # {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0.0,
        )
        print(result)
        print(result['choices'][0]['message']['content'])
        return result['choices'][0]['message']['content'].strip()

    # from pathlib import Path
    #
    # import ctranslate2
    # import sentencepiece as spm
    # import time
    #
    # model_dir = Path("models/small100-ctranslate2-int8")
    #
    # translator = ctranslate2.Translator(
    #     str(model_dir),
    #     device="cpu",
    #     compute_type="int8_float32",
    #     inter_threads=1,
    #     intra_threads=4,
    # )
    # tokenizer = spm.SentencePieceProcessor(
    #     model_file=str(model_dir / "sentencepiece.bpe.model")
    # )
    #
    # def translate(text: str, target: str) -> str:
    #     source_tokens = (
    #             [f"__{target}__"]
    #             + tokenizer.encode(text, out_type=str)
    #             + ["</s>"]
    #     )
    #     print(source_tokens)
    #     result = translator.translate_batch(
    #         [source_tokens],
    #         beam_size=5,
    #         max_decoding_length=512,
    #         length_penalty=1.2,
    #         coverage_penalty=0.1,
    #         return_end_token=True,
    #     )
    #     translation = tokenizer.decode(result[0].hypotheses[0])
    #     print(translation)
    #     # print(time.time() - start)
    #     return translation

    yield {"translate": translate}


class ModelState:
    """持有目前的模型，並在切換模型與開始錄音之間互斥。"""

    def __init__(self, model: ModelProcess):
        self.name = MODEL_NAMES[0]
        self.model = model
        self.lock = asyncio.Lock()
        self.recording = False

    async def select(self, name: str) -> None:
        """先成功載入新模型，才更新狀態並釋放舊模型。"""
        if name == self.name and self.model.is_alive():
            return
        replacement = await run_in_threadpool(ModelProcess, "SpeechTrack-ASR", asr_commands, name)
        previous = self.model
        self.model, self.name = replacement, name
        await run_in_threadpool(previous.close)

    def settings(self) -> dict:
        """提供目前的轉錄模型名稱和共用翻譯模型的目標語言。"""
        return {
            "model": self.name,
            "targets": TRANSLATION_TARGETS,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """分別啟動轉錄和翻譯程序，服務關閉時等兩者釋放模型。"""
    asr = await run_in_threadpool(ModelProcess, "SpeechTrack-ASR", asr_commands, MODEL_NAMES[0])
    state = ModelState(asr)
    app.state.asr = state
    try:
        translation_model = await run_in_threadpool(ModelProcess, "SpeechTrack-MT", translation_commands)
        app.state.translation_model = translation_model
        try:
            # 轉錄後可用 HTTP 請求呼叫同一個 FastAPI 的翻譯路由。
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://speechtrack") as client:
                app.state.translation_client = client
                yield
        finally:
            await run_in_threadpool(translation_model.close)
    finally:
        # 即使翻譯模型載入失敗，也要關閉已啟動的 ASR 子程序。
        await run_in_threadpool(state.model.close)


app = FastAPI(title="SpeechTrack 即時轉錄 Demo", lifespan=lifespan)

class Health(BaseModel):
    server_status: str

@app.get("/v1/server_ready", response_model=Health)
def health(request: Request):
    if request.app.state.translation_model.is_alive() and request.app.state.asr.model is not None:
        return {"server_status": "ready"}


class TranslationRequest(BaseModel):
    """翻譯端點接收轉錄原文和目標語言代碼。"""

    text: str
    target_lang: str


@app.post("/v1/translate")
async def translate_text(payload: TranslationRequest, request: Request) -> dict[str, str]:
    """將文字送給載入 MADLAD 的翻譯子程序。"""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="請提供要翻譯的文字。")
    if payload.target_lang not in TRANSLATION_TARGETS:
        raise HTTPException(status_code=422, detail="不支援這個翻譯語言。")
    try:
        translation = await run_in_threadpool(
            request.app.state.translation_model.call, "translate", text, payload.target_lang
        )
    except ModelWorkerError as error:
        logger.exception("翻譯子程序執行失敗")
        raise HTTPException(status_code=503, detail="翻譯失敗。") from error
    return {"translation": translation}


@asynccontextmanager
async def asr_recording(model: ModelProcess):
    """錄音開始和結束時，由 ASR 子程序管理原生 session。"""
    await run_in_threadpool(model.call, "start")
    try:
        yield
    finally:
        if model.is_alive():
            await run_in_threadpool(model.call, "stop")


async def send_translation(
    websocket: WebSocket, final: str, segment_id: int, revision: int, text: str, target: str
) -> None:
    """呼叫翻譯 API，並把結果送回原文對應的清單列。"""
    client: httpx.AsyncClient = websocket.app.state.translation_client
    message = {
        "type": "translation", "final": final, "segment_id": segment_id, "revision": revision, "target": target,
    }
    try:
        response = await client.post("/v1/translate", json={"text": text, "target_lang": target})
        response.raise_for_status()
        message["translation"] = response.json()["translation"]
    except (httpx.HTTPError, ValueError, KeyError):
        # 翻譯失敗時，讓前端保留原文並顯示這一列的錯誤。
        logger.exception("翻譯 API 呼叫失敗")
        message["error"] = "翻譯失敗。"
    await websocket.send_json(message)


async def send_transcripts(
    websocket: WebSocket,
    model: ModelProcess,
    frames: asyncio.Queue[bytes | str | None],
    target: str,
) -> None:
    """收集目前的音訊段落，反覆辨識以更新暫定文字。"""
    pcm = bytearray()
    last_size = 0
    silence_chunk = 0
    # 同一段原文更新或切換語言時提升版本，讓譯文能對應到正確的清單列。
    last_text = ""
    segment_id = 1
    revision = 0
    async with asr_recording(model):
        while True:
            frame = await frames.get()

            if isinstance(frame, bytes):
                print("PER FRAME SIZE: ", len(frame))
            # 推論期間仍能接收音訊。一次取出已排隊的封包，略過過時的暫定結果。

            while isinstance(frame, bytes) and frame:
                # 每包 32 ms 音訊由 ASR 子程序內的 VAD 判定語音機率。
                probability = await run_in_threadpool(model.call, "score", frame)
                if probability is not None:
                    print(f"語音機率 {probability:.2f}")
                    silence_chunk = silence_chunk + 1 if probability < 0.9 else 0

                pcm.extend(frame)
                print("[DBG] PCM length: ", len(pcm))
                print("[DBG] silence_chunk: ", silence_chunk)
                # if len(pcm) == SEGMENT_BYTES or frames.empty() or silence_chunk >= 16:
                if frames.empty() or silence_chunk >= SILENCE_CHUNK:
                    break
                frame = frames.get_nowait()

            if frame == b"":
                return  # 用戶端已斷線，捨棄剩餘音訊。
            # 語言指令和音訊共用佇列，依 WebSocket 的接收順序處理。
            target_changed = isinstance(frame, str) and frame != target
            if isinstance(frame, str):
                target = frame
            stopping = frame is None
            # final = stopping or len(pcm) == SEGMENT_BYTES
            final = stopping or silence_chunk >= SILENCE_CHUNK or len(pcm) >= SEGMENT_BYTES
            new_audio_bytes = len(pcm) - last_size
            refresh_translation = target_changed and bool(last_text)
            if pcm and (final or new_audio_bytes >= UPDATE_BYTES or refresh_translation):
                print("[DBG] last_size: ", last_size)
                if new_audio_bytes > 0 and (
                    new_audio_bytes >= UPDATE_BYTES or (final and len(pcm) > SILENCE_BYTES)
                ):
                    last_text = await run_in_threadpool(model.call, "recognize", bytes(pcm))
                    last_size = len(pcm)

                # 先讓前端顯示原文，翻譯 API 回應後再送出譯文。
                revision += 1
                print("[DBG] result[text]: ", last_text)
                await websocket.send_json({
                    "type": "transcript", "final": final, "text": last_text, "target": target,
                    "segment_id": segment_id, "revision": revision,
                })
                if last_text and target:
                    await send_translation(websocket, final, segment_id, revision, last_text, target)

                if final:
                    pcm.clear()
                    last_size = 0
                    silence_chunk = 0
                    last_text = ""
                    segment_id += 1
                    revision = 0
                print("=============")
            if stopping:
                return


@app.websocket("/v1/audio_transcribe")
async def audio_transcribe(websocket: WebSocket):
    """同一條連線接收 JSON 操作指令與原始 PCM 封包。"""
    await websocket.accept()
    state: ModelState = websocket.app.state.asr
    # 佇列可放 PCM、語言代碼或停止訊號，保留一個位置給停止訊號。
    frames: asyncio.Queue[bytes | str | None] = asyncio.Queue(maxsize=2049)
    worker: asyncio.Task | None = None
    recording_target = ""
    try:
        # 讓獨立的前端檔案直接取得後端使用的 PCM 規格與模型清單。
        await websocket.send_json({
            "type": "ready",
            **state.settings(),
            "config": {
                "sampleRate": SAMPLE_RATE,
                "frameSamples": FRAME_SAMPLES,
                "models": MODEL_NAMES,
            },
        })
        while True:
            message = await websocket.receive()
            # print(message)
            if message["type"] == "websocket.disconnect":
                break
            if worker is not None and worker.done():
                await worker  # 若辨識工作失敗，交由外層回報並釋放錄音。

            frame = message.get("bytes")
            if frame is not None:
                if worker is None or len(frame) != FRAME_BYTES:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"請先開始錄音，並傳送每包 {FRAME_BYTES} bytes 的 PCM。",
                    })
                    continue
                if frames.qsize() == frames.maxsize - 1:
                    await websocket.send_json({"type": "error", "message": "待辨識音訊過多。"})
                    await websocket.close(code=1013)
                    break
                frames.put_nowait(frame)
                continue

            try:
                control = json.loads(message["text"])
                if not isinstance(control, dict):
                    raise ValueError("控制訊息必須是 JSON 物件。")
                action = control.get("type")
                # 錄音開始和錄音中的切換使用同一份語言清單。
                if action in ("start", "set_target"):
                    target = control.get("target", "")
                    if target != "" and target not in TRANSLATION_TARGETS:
                        raise ValueError("不支援這個翻譯語言。")

                if action == "set_target":
                    if worker is None:
                        raise ValueError("請先開始錄音。")
                    if frames.qsize() == frames.maxsize - 1:
                        raise ValueError("待處理音訊過多，請稍後再切換語言。")
                    frames.put_nowait(target)
                    recording_target = target
                    continue

                if action == "stop" and worker is not None:
                    frames.put_nowait(None)
                    await worker
                    worker = None
                    state.recording = False
                    await websocket.send_json({"type": "stopped"})
                    continue
                async with state.lock:
                    if action == "select_model":
                        name = control.get("model")
                        if name not in MODEL_NAMES:
                            raise ValueError("未知的轉錄模型。")
                        if state.recording:
                            raise ValueError("請先停止錄音再切換模型。")
                        try:
                            await state.select(name)
                        except ModelWorkerError:
                            # 載入失敗時保留原模型，讓使用者可以繼續操作。
                            logger.exception("無法載入選擇的模型")
                            raise ValueError("模型載入失敗。") from None
                        reply = {"type": "model", **state.settings()}
                    elif action == "start":
                        if state.recording:
                            raise ValueError("目前已有錄音正在進行。")
                        if control.get("model") != state.name:
                            raise ValueError("模型已切換，請確認選單後再開始。")
                        # 上一場若因子程序意外退出而中止，開始新錄音時重新載入。
                        await state.select(state.name)
                        recording_target = target
                        state.recording = True
                        worker = asyncio.create_task(
                            send_transcripts(websocket, state.model, frames, target)
                        )
                        reply = {"type": "started"}
                    else:
                        raise ValueError("目前無法執行這個指令。")
                await websocket.send_json(reply)
            except ValueError as error:
                reply = {"type": "error", "message": str(error), **state.settings()}
                if worker is not None:
                    reply["target"] = recording_target
                await websocket.send_json(reply)
    except WebSocketDisconnect:
        pass
    except ModelWorkerError:
        logger.exception("音訊辨識失敗")
        await websocket.send_json({"type": "error", "message": "音訊辨識失敗。"})
        await websocket.close(code=1011)
    finally:
        if worker is not None:
            # 等正在執行的推論結束，再允許其他連線切換模型。
            while not frames.empty():
                frames.get_nowait()
            frames.put_nowait(b"")
            await asyncio.gather(worker, return_exceptions=True)
            state.recording = False



class FileTranscribeRequest(BaseModel):
    """翻譯端點接收轉錄原文和目標語言代碼。"""
    job_id: str
    file_path: str


@app.post("/v1/file_transcribe")
async def file_transcribe(payload: FileTranscribeRequest, request: Request):
    """將文字送給載入 MADLAD 的翻譯子程序。"""
    file_path = payload.file_path.strip()
    if not file_path:
        raise HTTPException(status_code=422, detail="請提供要翻譯的文字。")
    try:
        print("request.app.state.asr.model: ", request.app.state.asr.model)
        async with asr_recording(request.app.state.asr.model):
            result_json = await run_in_threadpool(
                request.app.state.asr.model.call, "recognize_file", file_path
            )
        return {"type": "Result", "job_id": payload.job_id, "transcription": result_json}
    except ModelWorkerError as error:
        logger.exception("翻譯子程序執行失敗")
        raise HTTPException(status_code=503, detail="翻譯失敗。") from error



# 放在 API 路由之後，讓既有的 WebSocket 路徑優先處理。
app.mount("/", StaticFiles(directory=PROJECT_DIR / "frontend", html=True), name="frontend")


if __name__ == "__main__":
    # 預設只提供給本機瀏覽器。
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
