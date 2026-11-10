# import os, uuid, tempfile, shutil, subprocess
# import logging
# from typing import Optional
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
#
#
# # add by allen start
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import asyncio
# import time
# from starlette.websockets import WebSocketDisconnect
# # from llama_cpp import Llama
# from contextlib import asynccontextmanager
# import uvicorn
# # add by allen end
#
# # transcribe.cpp lib start
# import argparse
# import array
# import sys
# import time
# import wave
#
# import transcribe_cpp
#
# # transcribe.cpp lib end
#
# ######### Debug Log section #########
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=os.getenv("LOG_LEVEL", "warning").upper(),
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
# )
#
#
# ######### Fastapi setting #########
# app = FastAPI(title="Whisper Server", version="1.0")
#
# # add by allen start
# app.router.redirect_slashes = True
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or ["http://localhost:5173"] for security
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# class streaming_transcribe_model_param(BaseModel):
#     model_path: str
#     chunk_ms: int
#     realtime: bool = True
#     language: str = "auto"  # 預設值
#
# MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F32.gguf"
# # MODEL_PATH = "models/whisper-large-v3-turbo-Q8_0.gguf"
#
# param_setting = {
#     "model_path": MODEL_PATH,
#     "chunk_ms": 20,
#     "realtime": True,
#     "language": "auto"
# }
#
# param = streaming_transcribe_model_param(**param_setting)
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # 【Startup 階段】：這段程式碼會在應用程式「開始接收請求前」執行
#     print("應用程式啟動中：正在載入 ML 模型...")
#     with transcribe_cpp.Model(param.model, backend="vulkan", device=calculate_device_select()) as model:
#         caps = model.capabilities
#         print(caps)
#         if not caps.supports_streaming:
#             raise SystemExit(
#                 f"{model.arch}/{model.variant} does not support streaming — "
#                 "use a streaming model (e.g. moonshine-streaming-tiny)"
#             )
#         print(f"model: {model.arch}/{model.variant} on {model.backend} | "
#               f"feeding {param.chunk_ms} ms chunks"
#               f"{' at realtime pace' if param.realtime else ''}")
#     # ml_models["sentiment_analysis"] = lambda text: f"模型分析結果：{text}"
#
#     yield  # 👈 程式會在這裡暫停，開始接收客戶端請求
#
#     # 【Shutdown 階段】：這段程式碼會在應用程式「停止運作時」執行
#     print("應用程式關閉中：正在釋放資源...")
#     # ml_models.clear()
#
#
# # 3. 將 lifespan 函式傳入 FastAPI 實例中
# app = FastAPI(lifespan=lifespan)
#
#
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
#                and str_tmp.casefold()
#                in f"{device.name} {device.description}".casefold()
#         ),
#         None,
#     )
#     print(selected_device)
#     return selected_device
#
# def render(text: "transcribe_cpp.StreamText") -> None:
#     # One status line: committed text plain, tentative dimmed. \r-rewrite is
#     # enough for a demo; a real UI keys off StreamUpdate.committed_changed /
#     # tentative_changed to redraw minimally.
#     line = f"{text.committed}\x1b[2m{text.tentative}\x1b[0m"
#     print(f"\r\x1b[K{line}", end="", flush=True)
#
#
# def main() -> None:
#     uvicorn.run(
#         app,
#         host="127.0.0.1",  # 只接受本機連線
#         port=8000,
#         reload=False,
#         workers=1,
#     )
#
#
# if __name__ == "__main__":
#     main()

"""錄音網頁與 FastAPI 雛型；安裝 fastapi、uvicorn 後執行 python demo.py。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import transcribe_cpp
from pydantic import BaseModel
import time
import numpy as np
import array

# 雛型只接收錄音並回報大小；尚未解碼音訊、執行 VAD 或語音辨識。
logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 10 * 1024 * 1024

# ASR_model = None
# VAD_model = None

class streaming_transcribe_model_param(BaseModel):
    model_path: str
    chunk_ms: int
    realtime: bool = True
    language: None # 預設值

MODEL_PATH = "models/nemotron-3.5-asr-streaming-0.6b-F32.gguf"
# MODEL_PATH = "models/whisper-large-v3-turbo-Q8_0.gguf"

param_setting = {
    "model_path": MODEL_PATH,
    "chunk_ms": 320,
    "realtime": True,
    "language": None
}

param = streaming_transcribe_model_param(**param_setting)

def calculate_device_select(device_name="Intel"):
    devices = transcribe_cpp.backends()
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
               and str_tmp.casefold()
               in f"{device.name} {device.description}".casefold()
        ),
        None,
    )
    print(selected_device)
    return selected_device


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """管理應用程式啟動與關閉，供後續放入共用模型資源。"""
    # Startup：之後可在這裡載入 ASR / VAD 模型，供各個請求使用。
    app.state.asr_model = None
    app.state.vad_model = None
    app.state.ready = True
    logger.info("錄音服務已啟動；ASR / VAD 尚未串接。")
    # 【Startup 階段】：這段程式碼會在應用程式「開始接收請求前」執行
    print("應用程式啟動中：正在載入 ML 模型...")
    with transcribe_cpp.Model(param.model_path, backend="vulkan", device=calculate_device_select()) as model:
        caps = model.capabilities
        print(caps)
        if not caps.supports_streaming:
            raise SystemExit(
                f"{model.arch}/{model.variant} does not support streaming — "
                "use a streaming model (e.g. moonshine-streaming-tiny)"
            )
        print(f"model: {model.arch}/{model.variant} on {model.backend} | "
              f"feeding {param.chunk_ms} ms chunks"
              f"{' at realtime pace' if param.realtime else ''}")
        app.state.asr_model = model
        # ml_models["sentiment_analysis"] = lambda text: f"模型分析結果：{text}"

        yield  # 👈 程式會在這裡暫停，開始接收客戶端請求

        # 【Shutdown 階段】：這段程式碼會在應用程式「停止運作時」執行
        app.state.asr_model = None
        app.state.ready = False
        logger.info("錄音服務已關閉。")
        # ml_models.clear()


# 只建立一個 FastAPI 實例，確保 lifespan 與路由都掛在同一個 app。
app = FastAPI(title="SpeechTrack 錄音 Demo", version="0.1.0", lifespan=lifespan)

# 將 HTML / CSS / JavaScript 放在同一檔案，方便先閱讀完整的前後端流程。
PAGE_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SpeechTrack · 錄音測試台</title>
  <style>
    /* 基本配色與排版：使用系統字型，不依賴外部套件。 */
    :root { color-scheme: light; font-family: system-ui, "Microsoft JhengHei", sans-serif; color: #172b43; background: #f3f6fa; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 48px 24px; }
    main { max-width: 1040px; margin: auto; }
    .brand { color: #2463d4; font-size: 14px; font-weight: 750; letter-spacing: 2px; }
    h1 { margin: 12px 0; font-size: clamp(28px, 4vw, 38px); }
    .subtitle, .hint { color: #62738a; line-height: 1.7; }
    .subtitle { margin: 0 0 28px; }
    .layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(240px, 1fr); gap: 22px; align-items: start; }
    .card { padding: 26px; border: 1px solid #e0e7f0; border-radius: 18px; background: white; box-shadow: 0 8px 28px #203c6210; }
    h2 { font-size: 18px; margin: 0 0 18px; }
    .recording-top { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    #status { padding: 7px 12px; border-radius: 999px; font-size: 13px; background: #eef2f7; color: #52627b; }
    #status[data-state="recording"], #status[data-state="error"] { color: #b42335; background: #fff0f1; }
    #status[data-state="success"] { color: #15704d; background: #eaf8f1; }
    #timer { display: block; margin: 28px 0 22px; font-size: 46px; font-weight: 650; font-variant-numeric: tabular-nums; letter-spacing: 2px; }
    .actions { display: flex; flex-wrap: wrap; gap: 12px; }
    button { border: 0; border-radius: 10px; padding: 13px 22px; font: inherit; font-weight: 650; cursor: pointer; }
    #start-button { color: white; background: #2463d4; }
    #stop-button { color: #b42335; background: #fff0f1; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    button:focus-visible, a:focus-visible { outline: 3px solid #94b9ff; outline-offset: 3px; }
    .hint { margin: 16px 0 0; font-size: 13px; }
    .result { border-top: 1px solid #e7edf5; padding-top: 24px; margin-top: 24px; }
    label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 650; }
    textarea { display: block; width: 100%; min-height: 175px; padding: 16px; resize: vertical; border: 1px solid #dce5f0; border-radius: 10px; background: #f9fbfe; color: #263b57; font: inherit; line-height: 1.7; }
    #playback { margin-top: 22px; }
    audio { display: block; width: 100%; margin: 10px 0; }
    a { color: #2463d4; font-size: 14px; }
    .badge { display: inline-block; padding: 4px 9px; border-radius: 6px; color: #78869a; background: #f1f4f8; font-size: 12px; font-weight: 500; }
    fieldset { margin: 22px 0 0; padding: 0; border: 0; }
    .field { margin-top: 20px; }
    .field:first-child { margin-top: 0; }
    input { width: 100%; padding: 10px 12px; border: 1px solid #e0e7f0; border-radius: 8px; font: inherit; }
    input:disabled { color: #7f8ca0; background: #f5f7fa; }
    /* 小螢幕改成單欄，讓錄音按鈕與文字區保持可讀。 */
    @media (max-width: 720px) { body { padding: 28px 16px; } .layout { grid-template-columns: 1fr; } .card { padding: 22px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="brand">SPEECHTRACK / DEMO</div>
      <h1>錄音測試台</h1>
      <p class="subtitle">從麥克風錄音，測試前端與後端的音訊傳送流程。</p>
    </header>

    <div class="layout">
      <section class="card" aria-labelledby="recorder-heading">
        <div class="recording-top">
          <h2 id="recorder-heading">麥克風錄音</h2>
          <span id="status" role="status" aria-live="polite">準備就緒</span>
        </div>
        <output id="timer" aria-label="錄音時間">00:00</output>
        <div class="actions">
          <button id="start-button" type="button">開始錄音</button>
          <button id="stop-button" type="button" disabled>停止錄音</button>
        </div>
        <p class="hint">錄音時持續傳送：每包 20 ms、16 kHz、單聲道 PCM16。首次使用請允許麥克風權限。</p>

        <!-- 回放只使用瀏覽器中的錄音資料，不另存到伺服器。 -->
        <div id="playback" hidden>
          <label for="audio-preview">錄音回放</label>
          <audio id="audio-preview" controls></audio>
          <a id="download-link">下載這段錄音</a>
        </div>

        <div class="result">
          <label for="text-output">文字顯示區</label>
          <textarea id="text-output" readonly placeholder="錄音期間會在這裡顯示後端回覆。"></textarea>
          <p class="hint">停止後可回放或下載；超過 10 MB 的錄音仍持續傳送，但不保留本機回放。</p>
        </div>
      </section>

      <aside class="card" aria-labelledby="vad-heading">
        <h2 id="vad-heading">VAD 參數設定 <span class="badge">預留</span></h2>
        <p class="hint">語音活動偵測功能尚未啟用，下方數值為預留範例。</p>
        <!-- 預留控制項先停用，不會送到後端或影響錄音。 -->
        <fieldset disabled aria-label="尚未啟用的 VAD 設定">
          <div class="field">
            <label for="vad-threshold">語音判定門檻</label>
            <input id="vad-threshold" type="number" min="0" max="1" step="0.05" value="0.5">
          </div>
          <div class="field">
            <label for="vad-silence">靜音持續時間（ms）</label>
            <input id="vad-silence" type="number" min="0" step="100" value="500">
          </div>
          <div class="field">
            <label for="vad-speech">最短語音長度（ms）</label>
            <input id="vad-speech" type="number" min="0" step="50" value="250">
          </div>
        </fieldset>
      </aside>
    </div>
  </main>

  <script>
    // 傳輸規格：16 kHz、單聲道、little-endian PCM16；20 ms = 320 samples = 640 bytes。
    const SAMPLE_RATE = 16000;
    const FRAME_SAMPLES = 320;
    const FRAME_BYTES = FRAME_SAMPLES * 2;
    const MAX_PENDING_BYTES = SAMPLE_RATE * 2 * 2;
    const MAX_PREVIEW_BYTES = 10 * 1024 * 1024;
    const startButton = document.getElementById("start-button");
    const stopButton = document.getElementById("stop-button");
    const statusLabel = document.getElementById("status");
    const timerLabel = document.getElementById("timer");
    const textOutput = document.getElementById("text-output");
    const audioPreview = document.getElementById("audio-preview");
    const downloadLink = document.getElementById("download-link");
    const playback = document.getElementById("playback");
    let recording = null;
    let previewUrl = null;

    // Worklet 在音訊執行緒累積取樣，不使用 setInterval(20) 或壓縮的 MediaRecorder。
    const WORKLET_SOURCE = `
      class PcmRecorder extends AudioWorkletProcessor {
        constructor() {
          super();
          this.recording = true;
          this.offset = 0;
          this.buffer = new ArrayBuffer(640);
          this.view = new DataView(this.buffer);
          this.port.onmessage = (event) => {
            if (event.data.type !== "stop" || !this.recording) return;
            this.recording = false;
            // 尾端不足 320 samples 時補零，最後一包仍固定為 20 ms。
            if (this.offset > 0) this.sendFrame();
            this.port.postMessage({ type: "stopped" });
          };
        }

        // 轉移完整封包的所有權，下一包改用新的緩衝區。
        sendFrame() {
          this.port.postMessage({ type: "audio", buffer: this.buffer }, [this.buffer]);
          this.buffer = new ArrayBuffer(640);
          this.view = new DataView(this.buffer);
          this.offset = 0;
        }

        process(inputs) {
          if (!this.recording) return false;
          const channels = inputs[0];
          if (!channels || channels.length === 0) return true;
          // 處理實際區塊長度；即使來源有多聲道，也先平均成單聲道。
          for (let index = 0; index < channels[0].length; index++) {
            let sample = 0;
            for (const channel of channels) sample += channel[index];
            sample = Math.max(-1, Math.min(1, sample / channels.length));
            const pcm = Math.round(sample < 0 ? sample * 32768 : sample * 32767);
            this.view.setInt16(this.offset * 2, pcm, true);
            this.offset++;
            if (this.offset === 320) this.sendFrame();
          }
          // 不寫入 outputs，保持靜音，避免麥克風聲音從喇叭播放。
          return true;
        }
      }
      registerProcessor("pcm-recorder", PcmRecorder);
    `;

    // 後端文字使用 value 顯示，避免當作 HTML 執行。
    function setStatus(message, state = "idle") {
      statusLabel.textContent = message;
      statusLabel.dataset.state = state;
    }

    // 所有停止及錯誤路徑都釋放麥克風、音訊節點與 WebSocket。
    function releaseRecording(session) {
      session.closed = true;
      clearInterval(session.timerId);
      clearTimeout(session.connectTimeout);
      clearTimeout(session.stopTimeout);
      session.rejectConnection?.(new Error("錄音已取消。"));
      session.resolveStop?.();
      session.source?.disconnect();
      session.source = null;
      if (session.worklet) {
        session.worklet.port.onmessage = null;
        session.worklet.port.close();
        session.worklet.disconnect();
        session.worklet = null;
      }
      session.microphone?.getTracks().forEach((track) => track.stop());
      session.microphone = null;
      if (session.context && session.context.state !== "closed") {
        session.context.close().catch(() => {});
      }
      session.context = null;
      session.socket?.close();
      session.socket = null;
    }

    // PCM 沒有檔案標頭；只在本機回放／下載時加上 WAV 標頭。
    function showPreview(session) {
      if (!session.previewAvailable || session.previewChunks.length === 0) return;
      const header = new ArrayBuffer(44);
      const view = new DataView(header);
      const writeLabel = (offset, label) => {
        for (let index = 0; index < label.length; index++) view.setUint8(offset + index, label.charCodeAt(index));
      };
      writeLabel(0, "RIFF");
      view.setUint32(4, 36 + session.totalBytes, true);
      writeLabel(8, "WAVE");
      writeLabel(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, SAMPLE_RATE, true);
      view.setUint32(28, SAMPLE_RATE * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeLabel(36, "data");
      view.setUint32(40, session.totalBytes, true);
      const wav = new Blob([header, ...session.previewChunks], { type: "audio/wav" });
      session.previewChunks = [];
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(wav);
      audioPreview.src = previewUrl;
      downloadLink.href = previewUrl;
      downloadLink.download = "recording-16k.wav";
      playback.hidden = false;
    }

    // 發生問題時停止整段錄音，不悄悄丟棄封包後繼續辨識。
    function failRecording(session, error) {
      if (recording !== session) return;
      recording = null;
      releaseRecording(session);
      showPreview(session);
      startButton.disabled = false;
      stopButton.disabled = true;
      textOutput.value = `錄音處理失敗：${error.message}`;
      setStatus("錄音已中止", "error");
    }

    // 一次錄音共用一條連線；收到中間辨識結果時只更新文字，不關閉連線。
    function connectSocket(session) {
      return new Promise((resolve, reject) => {
        const url = new URL("/api/audio", window.location.href);
        url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        url.searchParams.set("mime_type", "audio/pcm");
        url.searchParams.set("sample_rate", String(SAMPLE_RATE));
        url.searchParams.set("channels", "1");
        url.searchParams.set("format", "s16le");
        url.searchParams.set("chunk_ms", "20");
        const socket = new WebSocket(url);
        session.socket = socket;
        session.rejectConnection = reject;
        const onFailure = (message) => {
          if (session.closed) return;
          reject(new Error(message));
          failRecording(session, new Error(message));
        };
        session.connectTimeout = setTimeout(() => onFailure("WebSocket 連線逾時。"), 10000);
        socket.addEventListener("open", () => {
          if (session.closed) return;
          clearTimeout(session.connectTimeout);
          session.rejectConnection = null;
          resolve();
        });
        socket.addEventListener("message", (event) => {
          if (session.closed) return;
          try {
            const result = JSON.parse(event.data);
            if (result.type === "error") throw new Error(result.detail || "後端處理失敗。");
            if (result.type !== "result" || typeof result.text !== "string") {
              throw new Error("後端必須回傳 type=result，且 text 必須是字串。");
            }
            textOutput.value = result.text;
          } catch (error) {
            failRecording(session, error);
          }
        });
        socket.addEventListener("error", () => onFailure("WebSocket 連線失敗，請確認後端服務。"));
        socket.addEventListener("close", () => onFailure("WebSocket 在錄音期間中斷。"));
      });
    }

    // 每個 Worklet 封包直接送出，不等待後端回覆才傳下一包。
    function sendFrame(session, buffer) {
      if (recording !== session || session.closed) return;
      try {
        if (buffer.byteLength !== FRAME_BYTES) throw new Error("錄音封包必須是 320 samples／640 bytes。");
        if (session.socket.readyState !== WebSocket.OPEN) throw new Error("WebSocket 尚未連線或已中斷。");
        if (session.socket.bufferedAmount + buffer.byteLength > MAX_PENDING_BYTES) {
          throw new Error("音訊傳送積壓超過兩秒，請檢查網路或後端處理速度。");
        }
        session.socket.send(buffer);
        session.totalBytes += buffer.byteLength;
        session.framesSent++;
        // 長錄音仍持續傳送；僅限制本機回放的記憶體用量。
        if (session.previewAvailable && session.totalBytes <= MAX_PREVIEW_BYTES) {
          session.previewChunks.push(buffer);
        } else {
          session.previewAvailable = false;
          session.previewChunks = [];
        }
      } catch (error) {
        failRecording(session, error);
      }
    }

    // 先等最後一包音訊送出，再關閉連線與裝置。
    async function stopRecording() {
      const session = recording;
      if (!session || session.state !== "recording") return;
      session.state = "stopping";
      stopButton.disabled = true;
      clearInterval(session.timerId);
      setStatus("正在停止錄音…");
      try {
        const stopped = new Promise((resolve, reject) => {
          session.resolveStop = resolve;
          session.stopTimeout = setTimeout(() => reject(new Error("音訊處理器停止逾時。")), 2000);
        });
        session.worklet.port.postMessage({ type: "stop" });
        await stopped;
        if (recording !== session) return;
        recording = null;
        releaseRecording(session);
        showPreview(session);
        startButton.disabled = false;
        setStatus(`錄音已停止，送出 ${session.framesSent} 包`, "success");
      } catch (error) {
        failRecording(session, error);
      }
    }

    // 每次按下開始才取得麥克風；AudioContext 負責將裝置音訊重採樣為 16 kHz。
    startButton.addEventListener("click", async () => {
      if (recording) return;
      const session = {
        state: "starting", closed: false, previewAvailable: true,
        previewChunks: [], totalBytes: 0, framesSent: 0,
      };
      recording = session;
      startButton.disabled = true;
      textOutput.value = "";
      timerLabel.textContent = "00:00";
      audioPreview.pause();
      playback.hidden = true;
      setStatus("正在準備錄音…");
      try {
        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
          throw new Error("請透過 localhost 或 HTTPS 開啟網頁，才能使用麥克風。");
        }
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass || !window.AudioWorkletNode) throw new Error("此瀏覽器不支援 AudioWorklet 錄音。");
        session.context = new AudioContextClass({ sampleRate: SAMPLE_RATE, latencyHint: "interactive" });
        if (session.context.sampleRate !== SAMPLE_RATE || !session.context.audioWorklet) {
          throw new Error("此瀏覽器無法提供 16 kHz 的 AudioWorklet 錄音。");
        }
        // 在使用者點擊的啟動流程中啟用音訊，避免自動播放限制。
        await session.context.resume();
        if (session.closed) return;
        session.microphone = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: { ideal: 1 }, sampleRate: { ideal: SAMPLE_RATE } },
        });
        if (session.closed) { releaseRecording(session); return; }
        await connectSocket(session);
        if (session.closed) return;
        const workletUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "application/javascript" }));
        try {
          await session.context.audioWorklet.addModule(workletUrl);
        } finally {
          URL.revokeObjectURL(workletUrl);
        }
        if (session.closed) return;
        session.worklet = new AudioWorkletNode(session.context, "pcm-recorder", {
          numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
          channelCount: 1, channelCountMode: "explicit",
        });
        session.worklet.port.onmessage = (event) => {
          if (session.closed) return;
          if (event.data.type === "audio") sendFrame(session, event.data.buffer);
          if (event.data.type === "stopped") session.resolveStop?.();
        };
        session.worklet.addEventListener("processorerror", () => failRecording(session, new Error("音訊處理器發生錯誤。")));
        session.source = session.context.createMediaStreamSource(session.microphone);
        session.state = "recording";
        session.source.connect(session.worklet);
        session.worklet.connect(session.context.destination);
        // 過期裝置事件不得影響下一次錄音。
        session.microphone.getAudioTracks()[0].addEventListener("ended", () => {
          if (recording === session) stopRecording();
        });
        const startedAt = Date.now();
        session.timerId = setInterval(() => {
          const seconds = Math.floor((Date.now() - startedAt) / 1000);
          timerLabel.textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
        }, 250);
        stopButton.disabled = false;
        setStatus("錄音中，持續傳送音訊", "recording");
      } catch (error) {
        if (error.name === "NotAllowedError") error = new Error("無法使用麥克風，請確認瀏覽器權限後再試。");
        failRecording(session, error);
      }
    });

    // 停止按鈕及離開頁面都會釋放錄音資源。
    stopButton.addEventListener("click", stopRecording);
    window.addEventListener("pagehide", () => {
      if (recording) {
        const session = recording;
        recording = null;
        releaseRecording(session);
      }
      // 頁面由瀏覽器快取恢復時，按鈕仍可開始下一次錄音。
      startButton.disabled = false;
      stopButton.disabled = true;
      playback.hidden = true;
      setStatus("錄音已停止");
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """由同一個服務提供網頁，前端可直接使用相對路徑呼叫 API。"""
    return HTMLResponse(PAGE_HTML)


@app.get("/api/health")
async def health(request: Request) -> dict:
    """確認服務已經完成 lifespan 啟動階段，並呈現辨識串接狀態。"""
    return {"ready": request.app.state.ready, "asr_ready": False, "vad_enabled": False}


@app.websocket("/api/audio")
async def receive_audio(websocket: WebSocket) -> None:
    """每個二進位訊息是一段完整錄音；回傳 JSON，支援同一連線重複傳送。"""
    await websocket.accept()
    # 瀏覽器的 WebSocket 無法自訂 Content-Type，改從查詢參數取得音訊格式。
    mime_type = websocket.query_params.get("mime_type", "audio/wav").split(";", 1)[0].strip().lower()
    if not mime_type.startswith("audio/"):
        await websocket.send_json({"type": "error", "detail": "請指定 audio/* 的音訊格式。"})
        await websocket.close(code=1008)
        return

    def render(text: "transcribe_cpp.StreamText") -> None:
        # One status line: committed text plain, tentative dimmed. \r-rewrite is
        # enough for a demo; a real UI keys off StreamUpdate.committed_changed /
        # tentative_changed to redraw minimally.
        line = f"{text.committed}\x1b[2m{text.tentative}\x1b[0m"
        print(f"\r\x1b[K{line}", end="", flush=True)
    audio_buffer = np.zeros(0, dtype=np.int16)
    try:

        options = transcribe_cpp.ParakeetStreamOptions(
            att_context_right=0
        )

        if not app.state.asr_model.accepts(options):
            raise RuntimeError("Model does not support these options")
        with app.state.asr_model.session() as session:
            with session.stream(language=param.language, commit_policy="stable_prefix", family=options) as stream:
                while True:
                    # 使用 receive 區分二進位資料、誤傳的文字訊息及正常斷線。
                    # message = await websocket.receive()
                    # if message["type"] == "websocket.disconnect":
                    #     return
                    # audio_data = message.get("bytes")
                    # if audio_data is None:
                    #     await websocket.send_json({"type": "error", "detail": "請傳送二進位音訊資料。"})
                    #     await websocket.close(code=1003)
                    #     return
                    # if not audio_data:
                    #     await websocket.send_json({"type": "error", "detail": "錄音資料不可為空。"})
                    #     await websocket.close(code=1008)
                    #     return
                    # if len(audio_data) > MAX_AUDIO_BYTES:
                    #     await websocket.send_json({"type": "error", "detail": "錄音大小不可超過 10 MB。"})
                    #     await websocket.close(code=1009)
                    #     return

                    pcm_bytes = await websocket.receive_bytes()
                    chunk = np.frombuffer(pcm_bytes, dtype=np.int16)
                    audio_buffer = np.concatenate((audio_buffer, chunk))
                    print(len(chunk))


                    update = stream.feed(array.array("f", (s / 32768.0 for s in chunk)))
                    print(update)
                    if update.committed_changed or update.tentative_changed:
                        print(update)
                        print(stream.text())
                        # render(stream.text())
                    # if param.realtime:
                    #     time.sleep(param.chunk_ms / 1000)
                    # update = stream.finalize()
                    final = stream.text()
                    # print(final)
                    render(final)
                    print()  # land the cursor after the live line
                    print(f"\nfinal ({update.audio_committed_ms / 1000:.1f}s audio, "
                          f"revision {stream.revision}):\n{final.committed.strip()}")

                    # 串接位置：依 mime_type 解碼、轉成模型需要的取樣率，再呼叫 ASR / VAD。
                    # 目前只回報接收結果，不驗證編碼內容、不存檔，也不產生假辨識文字。
                    await websocket.send_json({
                        "type": "result",
                        "text": final.full,
                        "bytes_received": len(audio_buffer),
                        "mime_type": mime_type,
                        "asr_ready": True,
                    })
    except WebSocketDisconnect:
        # 客戶端在後端回覆前離線時，直接結束本次連線處理。
        logger.info("錄音 WebSocket 已中斷。")
    except Exception as e:
        print("STT Realtime WS Error:", e)
    finally:
        running = False
        try:
            await websocket.close()
        except:
            pass
        print("STT WS closed")


if __name__ == "__main__":
    # 本機啟動入口；瀏覽器開啟 http://localhost:8000 即可使用。
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
