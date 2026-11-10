"""執行 python fake_stream.py，然後開啟 http://127.0.0.1:8000。"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
import numpy as np
import transcribe_cpp


# WebSocket 每則二進位訊息固定為 32 ms 的 16 kHz 單聲道 PCM16LE。
SAMPLE_RATE = 16_000
FRAME_SAMPLES = SAMPLE_RATE * 32 // 1000                                 # 512
FRAME_BYTES = FRAME_SAMPLES * 2                                          # 1024
UPDATE_BYTES = FRAME_BYTES * 32  # 累積約 1 秒音訊後更新暫定文字。            # 32768
SEGMENT_BYTES = FRAME_BYTES * 320  # 每約 8 秒定稿，限制重複辨識的音訊長度。   # 327680

SILENCE_CHUNK = 14 # 32ms * 14
SILENCE_BYTES = SILENCE_CHUNK * FRAME_BYTES

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_NAMES = ("whisper-small-F16", "Qwen3-ASR-0.6B-Q8_0")
logger = logging.getLogger("uvicorn.error")


# =============== Translate ==================
from pathlib import Path

import ctranslate2
import sentencepiece as spm
import time


model_dir = Path("models/small100-ctranslate2-int8")

translator = ctranslate2.Translator(
    str(model_dir),
    device="cpu",
    compute_type="int8_float32",
    inter_threads=1,
    intra_threads=4,
)
tokenizer = spm.SentencePieceProcessor(
    model_file=str(model_dir / "sentencepiece.bpe.model")
)


def translate(text:str, target_language:str):
    # text = "Could you tell me where the station is?"
    # text = "阿里巴巴"
    # target_language = "en"

    start = time.time()
    # SMaLL-100 places the target-language token at the beginning of the source.
    source_tokens = (
            [f"__{target_language}__"]
            + tokenizer.encode(text, out_type=str)
            + ["</s>"]
    )

    result = translator.translate_batch(
        [source_tokens],
        beam_size=5,
        max_decoding_length=256,
    )
    translation = tokenizer.decode(result[0].hypotheses[0])
    print(translation)
    print(time.time() - start)
    return translation

# =============== Translate ==================


def load_model(name: str) -> transcribe_cpp.Model:
    """從專案現有的 GGUF 檔案載入模型，由套件選擇運算裝置。"""
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
    return transcribe_cpp.Model(MODEL_DIR / f"{name}.gguf", backend="vulkan", device=selected_device)


class ModelState:
    """持有目前的模型，並在切換模型與開始錄音之間互斥。"""

    def __init__(self, model: transcribe_cpp.Model):
        self.name = MODEL_NAMES[0]
        self.model = model
        self.lock = asyncio.Lock()
        self.recording = False

    async def select(self, name: str) -> None:
        """先成功載入新模型，才更新狀態並釋放舊模型。"""
        if name == self.name:
            return
        replacement = await run_in_threadpool(load_model, name)
        previous = self.model
        self.model, self.name = replacement, name
        await run_in_threadpool(previous.close)

    def settings(self) -> dict:
        """讓前端依目前模型的能力顯示可用的翻譯語言。"""
        capabilities = self.model.capabilities
        return {
            "model": self.name,
            "targets": capabilities.translate_target_languages
            if capabilities.supports_translate else (),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入預設模型，關閉服務時釋放模型。"""
    state = ModelState(await run_in_threadpool(load_model, MODEL_NAMES[0]))
    app.state.asr = state
    try:
        # 沿用專案範例，以 CPU 載入共用的 Silero VAD 模型。
        vad_model = await run_in_threadpool(
            transcribe_cpp.Model, MODEL_DIR / "silero-vad-v6.2-F32.gguf", backend="cpu"
        )
        app.state.vad_model = vad_model
        try:
            yield
        finally:
            await run_in_threadpool(vad_model.close)
    finally:
        # 即使 VAD 載入失敗，也要釋放已載入的 ASR 模型。
        await run_in_threadpool(state.model.close)


app = FastAPI(title="SpeechTrack 即時轉錄 Demo", lifespan=lifespan)


def recognize(session: transcribe_cpp.Session, pcm: bytes, target: str) -> dict:
    """將 PCM16LE 轉為模型需要的 float32，保留原文與選用的譯文。"""
    print("PCM DATA length: ", len(pcm))
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    result = {"text": "", "translation": ""}
    if not samples.any():
        return result  # 略過全零音訊，避免對靜音產生文字。
    result["text"] = session.run(samples).text.strip()
    if target:
        result["translation"] = translate(
            text=result["text"], target_language=target
        ).text.strip()
    return result


async def send_transcripts(
    websocket: WebSocket,
    model: transcribe_cpp.Model,
    frames: asyncio.Queue[bytes | None],
    target: str,
) -> None:
    """收集目前的音訊段落，反覆辨識以更新暫定文字。"""
    pcm = bytearray()
    last_size = 0
    vad_model = websocket.app.state.vad_model
    silence_chunk = 0
    last_transcription_str = ""
    with model.session() as session, vad_model.vad_session(n_threads=1) as vad:
        while True:
            frame = await frames.get()

            if frame is not None:
                print("PER FRAME SIZE: ", len(frame))
            # 推論期間仍能接收音訊。一次取出已排隊的封包，略過過時的暫定結果。

            while frame:
                # ================ VAD ==========================
                chunk = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                result = await run_in_threadpool(vad.stream_feed, chunk)
                # 每包 32 ms 音訊對應一個語音機率，先印出來觀察。
                if result.probs:
                    print(f"語音機率 {result.probs[0]:.2f}")
                if result.probs[0] < 0.9:
                    silence_chunk += 1
                else:
                    silence_chunk = 0
                # ================ VAD ==========================

                pcm.extend(frame)
                print("[DBG] PCM length: ", len(pcm))
                print("[DBG] silence_chunk: ", silence_chunk)
                # if len(pcm) == SEGMENT_BYTES or frames.empty() or silence_chunk >= 16:
                if frames.empty() or silence_chunk >= SILENCE_CHUNK:
                    break
                frame = frames.get_nowait()

            if frame == b"":
                return  # 用戶端已斷線，捨棄剩餘音訊。
            stopping = frame is None
            # final = stopping or len(pcm) == SEGMENT_BYTES
            final = stopping or silence_chunk >= SILENCE_CHUNK #or len(pcm) >= SEGMENT_BYTES
            if pcm and (final or len(pcm) - last_size >= UPDATE_BYTES):
                # if len(pcm) != last_size and len(pcm) - last_size >= UPDATE_BYTES:
                print("[DBG] last_size: ", last_size)


                if (len(pcm) != last_size and len(pcm) - last_size >= UPDATE_BYTES) or (final and len(pcm) > SILENCE_BYTES):
                    result = await run_in_threadpool(recognize, session, bytes(pcm), target)
                else:
                    result = {"text": last_transcription_str, "translation": ""}

                print("[DBG] result[text]: ", result["text"])
                await websocket.send_json({"type": "transcript", "final": final, **result})

                last_transcription_str = result['text']
                last_size = len(pcm)
                if final:
                    pcm.clear()
                    last_size = 0
                    silence_chunk = 0
                    last_transcription_str = ""
                print("=============")
            if stopping:
                return


@app.websocket("/test/audio_transcribe")
async def audio_transcribe(websocket: WebSocket):
    """同一條連線接收 JSON 操作指令與原始 PCM 封包。"""
    await websocket.accept()
    state: ModelState = websocket.app.state.asr
    state_vad: ModelState = websocket.app.state.vad_model
    # 保留一個佇列位置供停止訊號使用。其餘位置最多存約 65 秒的待辨識音訊。
    frames: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2049)
    worker: asyncio.Task | None = None
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
                        except transcribe_cpp.TranscribeError:
                            # 載入失敗時保留原模型，讓使用者可以繼續操作。
                            logger.exception("無法載入選擇的模型")
                            raise ValueError("模型載入失敗。") from None
                        reply = {"type": "model", **state.settings()}
                    elif action == "start":
                        target = control.get("target", "")
                        if state.recording:
                            raise ValueError("目前已有錄音正在進行。")
                        if control.get("model") != state.name:
                            raise ValueError("模型已切換，請確認選單後再開始。")
                        if target != "" and target not in state.settings()["targets"]:
                            raise ValueError("目前模型不支援這個翻譯語言。")
                        state.recording = True
                        worker = asyncio.create_task(
                            send_transcripts(websocket, state.model, frames, target)
                        )
                        reply = {"type": "started"}
                    else:
                        raise ValueError("目前無法執行這個指令。")
                await websocket.send_json(reply)
            except ValueError as error:
                await websocket.send_json({
                    "type": "error", "message": str(error), **state.settings(),
                })
    except WebSocketDisconnect:
        pass
    except transcribe_cpp.TranscribeError:
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


# 放在 API 路由之後，讓既有的 WebSocket 路徑優先處理。
app.mount("/", StaticFiles(directory=PROJECT_DIR / "frontend", html=True), name="frontend")


if __name__ == "__main__":
    # 預設只提供給本機瀏覽器。
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
