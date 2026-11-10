const model = document.querySelector("#model");
const target = document.querySelector("#target");
const record = document.querySelector("#record");
const statusLabel = document.querySelector("#status");
// 清單每列包含原文和譯文，暫定列可以就地更新。
const transcriptFeed = document.querySelector("#transcript-feed");
const transcriptList = document.querySelector("#transcript-list");
const transcriptEmpty = document.querySelector("#transcript-empty");
const transcriptTemplate = document.querySelector("#transcript-row");
const translationHint = document.querySelector("#translation-hint");
// 以中文顯示後端回傳的語言，選單的值仍使用語言代碼。
const translationLabels = {
  en: "英文",
  ja: "日文",
  ko: "韓文",
  zh: "中文",
  de: "德文",
  fr: "法文",
  es: "西班牙文",
};

const socketUrl = new URL("/test/audio_transcribe", location.href);
socketUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
const socket = new WebSocket(socketUrl);
// WebSocket 連線就緒時取得後端的模型名稱與 PCM 規格。
let config;
let phase = "connecting";
let capture = null;
// 依段落編號保存清單列及其最新版本，讓後收到的譯文找到原文。
const transcriptRows = new Map();

// 模型僅能在待機時切換，翻譯語言在錄音時也可調整。
function setPhase(next, message) {
  phase = next;
  statusLabel.textContent = message;
  record.disabled = next !== "ready" && next !== "recording";
  record.textContent = next === "recording" ? "停止錄音" : "開始錄音";
  model.disabled = next !== "ready";
  target.disabled = (next !== "ready" && next !== "recording") || target.options.length === 1;
}

function applySettings(message) {
  const previousTarget = target.value;
  model.value = message.model;
  target.replaceChildren(new Option("不翻譯", ""));
  for (const language of message.targets) {
    target.add(new Option(translationLabels[language] ?? language, language));
  }
  target.value = message.targets.includes(previousTarget) ? previousTarget : "";
  translationHint.textContent = message.targets.length
    ? "錄音中也能切換語言，已定稿的譯文會保留。"
    : "目前沒有可用的翻譯語言。";
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

// 暫定與定稿共用同一列。原文更新時保留同語言的上一版譯文。
function showTranscript(message) {
  const entry = transcriptRows.get(message.segment_id);
  if (entry && message.revision <= entry.revision) return;
  if (!message.text) {
    entry?.row.remove();
    transcriptRows.delete(message.segment_id);
    transcriptEmpty.hidden = transcriptList.childElementCount > 0;
    return;
  }

  const row = entry?.row ?? transcriptTemplate.content.firstElementChild.cloneNode(true);
  row.classList.toggle("is-partial", !message.final);
  row.querySelector(".transcript-draft").hidden = message.final;
  row.querySelector(".transcript-text").textContent = message.text;

  // 切換語言或關閉翻譯時，才清除這一列原有的譯文。
  if (entry && entry.target !== message.target) {
    const translation = row.querySelector(".transcript-translation");
    translation.hidden = true;
    translation.textContent = "";
  }
  if (!entry) transcriptList.append(row);
  transcriptRows.set(message.segment_id, {
    row, revision: message.revision, final: message.final, target: message.target,
  });
  transcriptEmpty.hidden = true;
}

// 忽略舊版本的譯文，以及切換語言後才收到的舊語言暫定譯文。
function showTranslation(message) {
  const entry = transcriptRows.get(message.segment_id);
  if (!entry || entry.revision !== message.revision || entry.target !== message.target) return;
  if (!entry.final && message.target !== target.value) return;

  const output = entry.row.querySelector(".transcript-translation");
  // 空結果或翻譯失敗時，優先保留已顯示的譯文。
  const text = message.translation || (output.hidden ? message.error : "");
  if (!text) return;
  output.textContent = text;
  output.hidden = false;
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
      // 開始新錄音時清空上一場的清單。
      transcriptList.replaceChildren();
      transcriptEmpty.hidden = false;
      transcriptRows.clear();
      capture.source.connect(capture.node);
      capture.node.connect(capture.context.destination);
      setPhase("recording", "錄音中");
      break;
    case "transcript":
    case "translation": {
      // 原文或譯文更新後，只有原本停在底部時才跟隨最新結果。
      const follow = transcriptFeed.scrollHeight - transcriptFeed.scrollTop
        - transcriptFeed.clientHeight < 32;
      if (message.type === "transcript") showTranscript(message);
      else showTranslation(message);
      if (follow) transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
      break;
    }
    case "stopped":
      setPhase("ready", "錄音已停止");
      break;
    case "error":
      if (message.model) applySettings(message);
      // 後端拒絕切換時，顯示仍在使用的語言。
      if (message.target !== undefined) target.value = message.target;
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
// 待機時的選擇會隨 start 傳送，錄音中的變更透過同一條 WebSocket 即時送出。
target.addEventListener("change", () => {
  if (phase === "recording" && socket.readyState === WebSocket.OPEN) {
    send("set_target", {target: target.value});
  }
});
record.addEventListener("click", () => {
  if (phase === "recording") stopRecording();
  else if (phase === "ready") startRecording();
});
window.addEventListener("pagehide", () => {
  releaseCapture();
  socket.close();
});
