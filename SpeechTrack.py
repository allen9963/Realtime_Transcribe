"""執行 python SpeechTrack.py，然後開啟 http://127.0.0.1:8000。"""

import asyncio
from contextlib import ExitStack, asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
import opencc


converter = opencc.OpenCC('s2t.json')


# 每個音訊封包是 32 ms 的 16 kHz 單聲道 PCM16LE。
SAMPLE_RATE = 16_000
FRAME_SAMPLES = SAMPLE_RATE * 32 // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
UPDATE_BYTES = FRAME_BYTES * 32
SEGMENT_BYTES = FRAME_BYTES * 320
SILENCE_CHUNK = 16
SILENCE_BYTES = SILENCE_CHUNK * FRAME_BYTES
MAX_AUDIO_FRAMES = 2048

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
FRONTEND_DIR = PROJECT_DIR / "frontend"
# 以同一個名稱載入和回報預設轉錄模型。
DEFAULT_ASR_MODEL = "Qwen3-ASR-0.6B-Q6_K"
MODEL_NAMES = (
    "whisper-small-F16",
    "Qwen3-ASR-0.6B-Q8_0",
    "whisper-large-v3-turbo-Q4_K_M",
    "whisper-large-v3-turbo-Q8_0",
    "whisper-small-Q4_K_M",
    "Qwen3-ASR-0.6B-Q4_K_M",
    DEFAULT_ASR_MODEL,
)
# 翻譯選單的語言代碼對應到 Hy-MT2 提示詞中的語言名稱。
TRANSLATION_TARGETS = ("en", "ja", "ko", "zh-Hant", "zh-Hans", "de", "fr", "es")
LANGUAGE_RELATION = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-Hant" : "Chinese",
    "zh-Hans" : "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}
logger = logging.getLogger("uvicorn.error")


class ModelWorkerError(RuntimeError):
    """模型子程序載入、推論或通訊失敗。"""


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


class ModelProcess:
    """使用 spawn 啟動模型子程序，依序處理該程序的指令。"""

    def __init__(self, name, open_model, *model_args):
        context = multiprocessing.get_context("spawn")
        self._lock = Lock()
        self._connection, child = context.Pipe()
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

        # 等待模型載入完成再供服務使用。
        try:
            self._receive()
        except (EOFError, OSError, ModelWorkerError) as error:
            self.close()
            raise ModelWorkerError(f"{name} 模型載入失敗：{error}") from error

    def is_alive(self) -> bool:
        """檢查模型子程序是否仍在執行。"""
        process = self._process
        return process is not None and process.is_alive()

    def _receive(self):
        """等待單一回覆，讓呼叫端能處理模型推論逾時。"""
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
        """關閉通道並等子程序釋放模型。"""
        with self._lock:
            self._close()

    def _close(self) -> None:
        """持有鎖時關閉程序，超時才強制停止。"""
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
    """啟動時載入 ASR，第一次辨識時才載入 Silero VAD。"""
    import numpy as np
    import soundfile as sf
    import transcribe_cpp

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
def translation_commands():
    """在獨立的翻譯子程序中載入 Hy-MT2。"""
    from llama_cpp import Llama

    model = Llama(
        model_path=str(MODEL_DIR / "Hy-MT2-1.8B-Q4_K_M.gguf"),
        verbose=False,
        enable_thinking=False,
    )

    def translate(text: str, target: str) -> str:
        """用指定的語言翻譯原文，回傳模型產生的文字。"""
        language = LANGUAGE_RELATION[target]
        prompt = f"Translate the following segment into {language}, without additional explanation：{text}"
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


class ModelState:
    """保存已載入的 ASR 模型，協調模型切換與辨識。"""

    def __init__(self, model: ModelProcess, name: str):
        self.name = name
        self.model = model
        self.lock = asyncio.Lock()
        self.recording = False

    async def select(self, name: str) -> None:
        """完成新模型載入後才更新設定並釋放舊模型。"""
        if name == self.name and self.model.is_alive():
            return
        replacement = await run_in_threadpool(ModelProcess, "SpeechTrack-ASR", asr_commands, name)
        previous = self.model
        self.model, self.name = replacement, name
        await run_in_threadpool(previous.close)

    def settings(self) -> dict:
        """回傳目前的轉錄模型與支援的翻譯語言。"""
        return {"model": self.name, "targets": TRANSLATION_TARGETS}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服務啟動時只預載預設 ASR 模型。"""
    asr = await run_in_threadpool(ModelProcess, "SpeechTrack-ASR", asr_commands, DEFAULT_ASR_MODEL)
    state = ModelState(asr, DEFAULT_ASR_MODEL)
    app.state.asr = state
    # 空字串代表不翻譯，翻譯模型在第一筆翻譯時才載入。
    app.state.translate_target_language = ""
    app.state.translation_model = None
    app.state.translation_lock = asyncio.Lock()
    try:
        # 即時轉錄的譯文走同一個 FastAPI 服務的翻譯端點。
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://speechtrack") as client:
            app.state.translation_client = client
            yield
    finally:
        try:
            translation_model = app.state.translation_model
            if translation_model is not None:
                await run_in_threadpool(translation_model.close)
        finally:
            await run_in_threadpool(state.model.close)


@asynccontextmanager
async def asr_recording(model: ModelProcess):
    """用錄音或檔案工作所需的 ASR session 執行辨識。"""
    await run_in_threadpool(model.call, "start")
    try:
        yield
    finally:
        if model.is_alive():
            await run_in_threadpool(model.call, "stop")


app = FastAPI(title="SpeechTrack 即時轉錄", lifespan=lifespan)


class Health(BaseModel):
    """表示服務的就緒狀態。"""

    server_status: str


@app.get("/v1/server_ready", response_model=Health)
def health(request: Request):
    """以 ASR 子程序判斷服務能否開始轉錄。"""
    if not request.app.state.asr.model.is_alive():
        raise HTTPException(status_code=503, detail="轉錄模型尚未就緒。")
    return {"server_status": "ready"}


class TranslationTargetRequest(BaseModel):
    """接收自動翻譯使用的目標語言。"""

    target_lang: str


@app.post("/v1/set_translate_target_language")
async def set_translate_target_language(
    payload: TranslationTargetRequest, request: Request
) -> dict[str, str]:
    """更新服務內的自動翻譯語言。"""
    target = payload.target_lang
    if target and target not in TRANSLATION_TARGETS:
        raise HTTPException(status_code=422, detail="不支援這個翻譯語言。")

    request.app.state.translate_target_language = target
    return {"target_lang": target}


class TranslationRequest(BaseModel):
    """接收要翻譯的原文和目標語言代碼。"""

    text: str
    target_lang: str


@app.post("/v1/translate")
async def translate_text(payload: TranslationRequest, request: Request) -> dict[str, str]:
    """第一次翻譯時載入 Hy-MT2，後續請求重用同一個模型。"""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="請提供要翻譯的文字。")
    if payload.target_lang not in TRANSLATION_TARGETS:
        raise HTTPException(status_code=422, detail="不支援這個翻譯語言。")
    state = request.app.state
    try:
        # 多筆翻譯同時到達時，只啟動一個翻譯子程序。
        async with state.translation_lock:
            model = state.translation_model
            if model is None:
                model = await run_in_threadpool(
                    ModelProcess, "SpeechTrack-MT", translation_commands
                )
                state.translation_model = model
        translation = await run_in_threadpool(model.call, "translate", text, payload.target_lang)
    except ModelWorkerError as error:
        logger.exception("翻譯子程序執行失敗")
        raise HTTPException(status_code=503, detail="翻譯失敗。") from error
    return {"translation": translation}


def recording_settings(app: FastAPI) -> dict:
    """提供模型清單、目前的翻譯語言和 PCM 封包規格。"""
    return {
        **app.state.asr.settings(),
        "target_lang": app.state.translate_target_language,
        "config": {
            "sampleRate": SAMPLE_RATE,
            "frameSamples": FRAME_SAMPLES,
            "models": MODEL_NAMES,
        },
    }


@app.get("/v1/audio_transcribe/config")
async def get_recording_settings(request: Request):
    """頁面載入時透過 HTTP 讀取錄音設定。"""
    return recording_settings(request.app)


class ModelSelection(BaseModel):
    """指定待機時要載入的轉錄模型。"""

    model: str


@app.post("/v1/audio_transcribe/model")
async def select_model(payload: ModelSelection, request: Request):
    """先切換模型，再回覆目前可用的錄音設定。"""
    state: ModelState = request.app.state.asr
    if payload.model not in MODEL_NAMES:
        raise HTTPException(status_code=422, detail="未知的轉錄模型。")

    async with state.lock:
        if state.recording:
            raise HTTPException(status_code=409, detail="請先停止錄音再切換模型。")
        try:
            await state.select(payload.model)
        except ModelWorkerError as error:
            logger.exception("無法載入選擇的模型")
            raise HTTPException(status_code=503, detail="模型載入失敗。") from error
        return recording_settings(request.app)


class FileTranscribeRequest(BaseModel):
    """接收檔案轉錄的工作編號與音檔路徑。"""

    job_id: str
    file_path: str


@app.post("/v1/file_transcribe")
async def file_transcribe(payload: FileTranscribeRequest, request: Request):
    """檔案轉錄和即時錄音輪流使用 ASR 子程序中的辨識 session。"""
    state: ModelState = request.app.state.asr
    async with state.lock:
        if state.recording:
            raise HTTPException(status_code=409, detail="請先停止錄音再轉錄檔案。")
        file_path = payload.file_path.strip()
        if not file_path:
            raise HTTPException(status_code=422, detail="請提供要翻譯的文字。")
        try:
            async with asr_recording(state.model):
                transcription = await run_in_threadpool(state.model.call, "recognize_file", file_path)
        except ModelWorkerError as error:
            logger.exception("檔案轉錄子程序執行失敗")
            raise HTTPException(status_code=503, detail="翻譯失敗。") from error
        return {"type": "Result", "job_id": payload.job_id, "transcription": transcription}


@dataclass
class Recording:
    """每條錄音連線各自保存待處理音訊與停止狀態。"""

    frames: asyncio.Queue[bytes | None] = field(
        default_factory=lambda: asyncio.Queue(MAX_AUDIO_FRAMES + 1)
    )
    stopping: bool = False
    disconnected: bool = False

    def discard_audio(self) -> None:
        """斷線時丟棄待處理音訊，喚醒等候封包的辨識工作。"""
        self.disconnected = True
        while not self.frames.empty():
            self.frames.get_nowait()
        self.frames.put_nowait(None)


def read_control(message: dict) -> dict:
    """從 WebSocket 訊息讀取 JSON 指令。"""
    text = message.get("text")
    if text is None:
        raise ValueError("請傳送 JSON 控制指令。")
    try:
        control = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("無法讀取 JSON 控制指令。") from None
    if not isinstance(control, dict):
        raise ValueError("控制訊息必須是 JSON 物件。")
    return control


async def receive_message(websocket: WebSocket) -> dict:
    """將斷線交給錄音流程統一清理。"""
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    return message


async def receive_audio(websocket: WebSocket, recording: Recording) -> None:
    """收取 PCM 和停止指令，直到最後一筆轉錄與翻譯完成。"""
    while True:
        message = await receive_message(websocket)
        frame = message.get("bytes")
        try:
            if frame is not None:
                if recording.stopping:
                    raise ValueError("錄音已停止接收音訊。")
                if len(frame) != FRAME_BYTES:
                    raise ValueError(f"每包 PCM 必須是 {FRAME_BYTES} bytes。")
                if recording.frames.qsize() >= MAX_AUDIO_FRAMES:
                    raise asyncio.QueueFull
                recording.frames.put_nowait(frame)
                continue

            control = read_control(message)
            if control.get("type") == "stop":
                if not recording.stopping:
                    recording.stopping = True
                    recording.frames.put_nowait(None)
            else:
                raise ValueError("目前無法執行這個指令。")
        except ValueError as error:
            await websocket.send_json({"type": "error", "message": str(error)})


async def send_translation(
    websocket: WebSocket, final: bool, segment_id: int, revision: int, text: str, target: str
) -> None:
    """呼叫翻譯 API，將譯文送回同一個轉錄段落與版本。"""
    client: httpx.AsyncClient = websocket.app.state.translation_client
    message = {
        "type": "translation",
        "final": final,
        "segment_id": segment_id,
        "revision": revision,
        "target": target,
    }
    try:
        response = await client.post("/v1/translate", json={"text": text, "target_lang": target})
        response.raise_for_status()
        message["translation"] = response.json()["translation"] if target != "zh-Hant" else converter.convert(response.json()["translation"])
    except (httpx.HTTPError, ValueError, KeyError):
        # 回傳對應段落的翻譯錯誤，讓前端保留原文。
        logger.exception("翻譯 API 呼叫失敗")
        message["error"] = "翻譯失敗。"
    await websocket.send_json(message)


async def send_transcripts(websocket: WebSocket, model: ModelProcess, recording: Recording) -> None:
    """以既有的 VAD 和音訊長度規則更新暫定與定稿結果。"""
    state = websocket.app.state
    pcm = bytearray()
    last_size = 0
    silence_chunk = 0
    last_text = ""
    segment_id = 1
    revision = 0

    while True:
        frame = await recording.frames.get()
        while frame is not None:
            probability = await run_in_threadpool(model.call, "score", frame)
            if probability is not None:
                silence_chunk = silence_chunk + 1 if probability < 0.9 else 0
            pcm.extend(frame)
            if recording.frames.empty() or silence_chunk >= SILENCE_CHUNK:
                break
            frame = recording.frames.get_nowait()

        if recording.disconnected:
            return
        stopping = frame is None
        final = stopping or silence_chunk >= SILENCE_CHUNK or len(pcm) >= SEGMENT_BYTES
        new_audio_bytes = len(pcm) - last_size
        if pcm and (final or new_audio_bytes >= UPDATE_BYTES):
            if new_audio_bytes > 0 and (
                new_audio_bytes >= UPDATE_BYTES or (final and len(pcm) > SILENCE_BYTES)
            ):
                last_text = await run_in_threadpool(model.call, "recognize", bytes(pcm))
                last_size = len(pcm)
            if recording.disconnected:
                return

            revision += 1
            await websocket.send_json({
                "type": "transcript",
                "text": converter.convert(last_text),
                "final": final,
                "segment_id": segment_id,
                "revision": revision,
                "target": state.translate_target_language,
            })
            # 呼叫翻譯 API 前讀取服務目前的語言設定。
            target = state.translate_target_language
            if last_text and target and not recording.disconnected:
                await send_translation(websocket, final, segment_id, revision, last_text, target)

            if final:
                pcm.clear()
                last_size = silence_chunk = 0
                last_text = ""
                segment_id += 1
                revision = 0
        if stopping:
            return


async def stream_recording(websocket: WebSocket, model: ModelProcess, recording: Recording) -> None:
    """收音與辨識並行，等最後一筆結果送出後清理錄音。"""
    async with asr_recording(model):
        await websocket.send_json({"type": "started"})
        receiver = asyncio.create_task(receive_audio(websocket, recording))
        transcriber = asyncio.create_task(send_transcripts(websocket, model, recording))
        try:
            finished, _ = await asyncio.wait(
                (receiver, transcriber), return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                task.result()
        finally:
            receiver.cancel()
            if not transcriber.done():
                # 等已開始的推論結束，再讓共用子程序釋放這場錄音。
                recording.discard_audio()
            await asyncio.gather(receiver, transcriber, return_exceptions=True)


async def close_with_error(websocket: WebSocket, message: str, code: int, **details) -> None:
    """在連線仍可用時回覆錯誤後關閉。"""
    with suppress(WebSocketDisconnect):
        await websocket.send_json({"type": "error", "message": message, **details})
        await websocket.close(code=code)


@app.websocket("/v1/audio_transcribe")
async def audio_transcribe(websocket: WebSocket):
    """每條 WebSocket 處理一次錄音並使用已載入的模型。"""
    await websocket.accept()
    state: ModelState = websocket.app.state.asr
    try:
        await websocket.send_json({"type": "ready", **recording_settings(websocket.app)})
        control = read_control(await receive_message(websocket))
        if control.get("type") != "start":
            raise ValueError("請先開始錄音。")
        recording = Recording()

        # 使用同一把鎖協調 HTTP 模型切換與錄音開始。
        async with state.lock:
            if state.recording:
                raise ValueError("目前已有錄音正在進行。")
            if control.get("model") != state.name:
                raise ValueError("模型已切換，請確認選單後再開始。")
            model = state.model
            if not model.is_alive():
                raise ModelWorkerError("轉錄模型程序已停止。")
            state.recording = True
        try:
            await stream_recording(websocket, model, recording)
        finally:
            state.recording = False

        # 最後一段的原文與譯文已送出，再通知前端並關閉連線。
        await websocket.send_json({"type": "stopped"})
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except ValueError as error:
        await close_with_error(websocket, str(error), 1008, model=state.name)
    except asyncio.QueueFull:
        await close_with_error(websocket, "待辨識音訊過多。", 1013)
    except (ModelWorkerError, TimeoutError):
        logger.exception("音訊辨識失敗")
        await close_with_error(websocket, "音訊辨識失敗。請確認模型後再錄音。", 1011)


@app.get("/app.js", include_in_schema=False)
def frontend_script():
    """共用既有網頁，並提供 SpeechTrack 的錄音與連線邏輯。"""
    return FileResponse(
        FRONTEND_DIR / "speechtrack.js",
        media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


# API 與新版腳本先配對，其他前端資源由原有目錄提供。
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # 直接執行時，檔案轉錄的相對音檔路徑以專案目錄為起點。
    os.chdir(PROJECT_DIR)
    uvicorn.run(app, host="127.0.0.1", port=8000)
