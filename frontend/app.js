const model = document.querySelector("#model");
const target = document.querySelector("#target");
const record = document.querySelector("#record");
const statusLabel = document.querySelector("#status");
// 定稿與暫定各自顯示原文和選用的譯文。
const finalTextOutput = document.querySelector("#final-text");
const partialTextOutput = document.querySelector("#partial-text");
const finalTranslationOutput = document.querySelector("#final-translation");
const partialTranslationOutput = document.querySelector("#partial-translation");
const translationOutputs = document.querySelectorAll(".translation-output");
const translationHint = document.querySelector("#translation-hint");

const socketUrl = new URL("/test/audio_transcribe", location.href);
socketUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
const socket = new WebSocket(socketUrl);
// WebSocket 連線就緒時取得後端的模型名稱與 PCM 規格。
let config;
let phase = "connecting";
let capture = null;
let finalText = [];
let finalTranslation = [];

// 模型載入和錄音期間鎖定設定，等伺服器確認後再開放操作。
function setPhase(next, message) {
  phase = next;
  statusLabel.textContent = message;
  record.disabled = next !== "ready" && next !== "recording";
  record.textContent = next === "recording" ? "停止錄音" : "開始錄音";
  model.disabled = next !== "ready";
  target.disabled = next !== "ready" || target.options.length === 1;
}

function applySettings(message) {
  const previousTarget = target.value;
  model.value = message.model;
  target.replaceChildren(new Option("不翻譯", ""));
  for (const language of message.targets) {
    target.add(new Option(language === "en" ? "英文" : language, language));
  }
  target.value = message.targets.includes(previousTarget) ? previousTarget : "";
  translationHint.textContent = message.targets.length ? "" : "目前模型僅支援轉錄。";
}

function send(type, settings = {}) {
  socket.send(JSON.stringify({type, ...settings}));
}

// 結束錄音、連線中斷或錄音失敗時，都釋放麥克風與音訊節點。
function releaseCapture() {
  const session = capture;
  capture = null;
  if (!session) return;
  clearTimeout(session.stopTimer);
  session.stream?.getTracks().forEach(track => track.stop());
  session.source?.disconnect();
  session.node?.port.close();
  session.node?.disconnect();
  session.context?.close().catch(() => {});
}

function finishCapture() {
  if (!capture) return;
  releaseCapture();
  if (socket.readyState === WebSocket.OPEN) send("stop");
}

function stopRecording() {
  setPhase("stopping", "正在完成最後一段轉錄…");
  capture.node.port.postMessage("stop");
  // 如果音訊執行緒已停止，逾時後仍送出停止指令。
  capture.stopTimer = setTimeout(finishCapture, 1500);
}

async function startRecording() {
  setPhase("preparing", "正在準備麥克風…");
  const session = {};
  capture = session;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error("請用支援 AudioWorklet 的瀏覽器，從 localhost 或 HTTPS 開啟。");
    }
    // 由 Web Audio 將裝置取樣率轉為 16 kHz，再由 Worklet 組成 PCM 封包。
    session.context = new AudioContext({sampleRate: config.sampleRate});
    if (session.context.sampleRate !== config.sampleRate) {
      throw new Error("瀏覽器無法提供 16 kHz 音訊。");
    }
    await session.context.resume();
    await session.context.audioWorklet.addModule("/pcm-worklet.js");
    if (capture !== session) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {channelCount: {ideal: 1}, sampleRate: {ideal: config.sampleRate}},
    });
    if (capture !== session) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    session.stream = stream;
    session.source = session.context.createMediaStreamSource(stream);
    session.node = new AudioWorkletNode(session.context, "pcm-recorder", {
      channelCount: 1,
      channelCountMode: "explicit",
      processorOptions: {frameSamples: config.frameSamples},
    });
    session.node.port.onmessage = ({data}) => {
      if (capture !== session || socket.readyState !== WebSocket.OPEN) return;
      if (data === "stopped") finishCapture();
      else socket.send(data);
    };
    session.node.onprocessorerror = () => {
      if (capture !== session) return;
      setPhase("stopping", "音訊處理器發生錯誤，正在停止錄音…");
      finishCapture();
    };
    send("start", {model: model.value, target: target.value});
  } catch (error) {
    if (capture !== session) return;
    releaseCapture();
    setPhase("ready", error.message);
  }
}

// 暫定框只保留最新結果。收到定稿時累加到定稿框並清空暫定框。
function showTranscript(message) {
  console.log("transcript message: ", message)
  if (message.final) {
    if (message.text) finalText.push(message.text);
    if (message.translation) finalTranslation.push(message.translation);
    finalTextOutput.textContent = finalText.join("\n");
    finalTranslationOutput.textContent = finalTranslation.join("\n");
  }
  partialTextOutput.textContent = message.final ? "" : message.text;
  partialTranslationOutput.textContent = message.final ? "" : message.translation;
}

socket.addEventListener("message", ({data}) => {
  const message = JSON.parse(data);
  switch (message.type) {
    case "ready":
    case "model":
      if (message.type === "ready") {
        config = message.config;
        model.replaceChildren(...config.models.map(name => new Option(name, name)));
      }
      applySettings(message);
      setPhase("ready", "可以開始錄音");
      break;
    case "started":
      // 每次開始錄音時，同時清除兩個顯示框的舊結果。
      finalText = [];
      finalTranslation = [];
      finalTextOutput.textContent = "";
      partialTextOutput.textContent = "";
      finalTranslationOutput.textContent = "";
      partialTranslationOutput.textContent = "";
      for (const output of translationOutputs) {
        output.hidden = !target.value;
      }
      capture.source.connect(capture.node);
      capture.node.connect(capture.context.destination);
      setPhase("recording", "錄音中");
      break;
    case "transcript":
      showTranscript(message);
      break;
    case "stopped":
      setPhase("ready", "錄音已停止");
      break;
    case "error":
      if (message.model) applySettings(message);
      if (phase === "preparing") releaseCapture();
      if (phase === "recording" || phase === "stopping") statusLabel.textContent = message.message;
      else setPhase("ready", message.message);
      break;
  }
});
socket.addEventListener("close", () => {
  releaseCapture();
  setPhase("offline", "已與服務中斷連線，請重新整理網頁。");
});
model.addEventListener("change", () => {
  setPhase("selecting", "正在載入模型…");
  send("select_model", {model: model.value});
});
record.addEventListener("click", () => {
  if (phase === "recording") stopRecording();
  else if (phase === "ready") startRecording();
});
window.addEventListener("pagehide", () => {
  releaseCapture();
  socket.close();
});
