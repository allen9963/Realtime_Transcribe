const model = document.querySelector("#model");
// 只在這支前端腳本隱藏模型區塊並套用兩欄版面。
const modelSection = model.closest("section");
modelSection.hidden = true;
modelSection.closest("main").classList.add("speechtrack-layout");
const target = document.querySelector("#target");
const record = document.querySelector("#record");
const statusLabel = document.querySelector("#status");
const transcriptFeed = document.querySelector("#transcript-feed");
const transcriptList = document.querySelector("#transcript-list");
const transcriptEmpty = document.querySelector("#transcript-empty");
const transcriptTemplate = document.querySelector("#transcript-row");
const translationHint = document.querySelector("#translation-hint");
const responseLog = document.querySelector("#response-log");
const responseEmpty = document.querySelector("#response-empty");

// 語言選單在頁面初始化時即可使用，選擇會留到下次錄音。
const translationLabels = {
  en: "英文",
  ja: "日文",
  ko: "韓文",
  "zh-Hant": "繁體中文",
  "zh-Hans": "簡體中文",
  de: "德文",
  fr: "法文",
  es: "西班牙文",
};
for (const [language, label] of Object.entries(translationLabels)) {
  target.add(new Option(label, language));
}
target.disabled = false;
const translationInstructions = "隨時可切換語言，下一筆翻譯會使用新的選擇。";
translationHint.textContent = translationInstructions;

const socketUrl = new URL("/v1/audio_transcribe", location.href);
socketUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
let config;
let activeModel;
let activeTarget = "";
let targetChanged = false;
let targetUpdate;
let phase = "loading";
let session = null;
const transcriptRows = new Map();

// 錄音準備、模型切換及收尾期間，翻譯語言選單仍可操作。
function setPhase(next, message) {
  phase = next;
  statusLabel.textContent = message;
  record.disabled = next !== "ready" && next !== "recording";
  record.textContent = next === "recording" ? "停止錄音" : "開始錄音";
  model.disabled = next !== "ready";
}

// 依序顯示 HTTP 與 WebSocket 的原始回應。
function logResponse(data) {
  const follow = responseLog.scrollHeight - responseLog.scrollTop - responseLog.clientHeight < 32;
  const response = document.createElement("pre");
  response.textContent = data;
  responseEmpty.hidden = true;
  responseLog.append(response);
  if (follow) responseLog.scrollTop = responseLog.scrollHeight;
}

function parseResponse(data) {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

async function requestJson(path, options) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const raw = await response.text();
  logResponse(raw);
  const data = parseResponse(raw);
  if (!response.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  if (!data || typeof data !== "object") throw new Error("後端未提供 JSON 回應。");
  return data;
}

async function requestSettings(path, options) {
  // 錄音設定須包含取樣率與封包大小。
  const settings = await requestJson(path, options);
  if (!settings.config) throw new Error("後端未提供錄音設定。");
  return settings;
}

function showActiveModel(name) {
  activeModel = name;
  model.value = name;
}

async function loadSettings() {
  setPhase("loading", "正在取得錄音設定…");
  try {
    const settings = await requestSettings("/v1/audio_transcribe/config");
    config = settings.config;
    model.replaceChildren(...config.models.map(name => new Option(name, name)));
    showActiveModel(settings.model);
    // 開頁回應抵達前若已操作選單，保留使用者正在提交的選擇。
    activeTarget = settings.target_lang;
    if (!targetChanged) target.value = activeTarget;
    setPhase("ready", "可以開始錄音");
  } catch (error) {
    setPhase("offline", `無法取得錄音設定。${error.message}`);
  }
}

async function selectModel() {
  setPhase("selecting", "正在載入模型…");
  try {
    const settings = await requestSettings("/v1/audio_transcribe/model", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model: model.value}),
    });
    showActiveModel(settings.model);
    setPhase("ready", "可以開始錄音");
  } catch (error) {
    model.value = activeModel;
    setPhase("ready", error.message);
  }
}

function send(current, type, settings = {}) {
  current.socket.send(JSON.stringify({type, ...settings}));
}

// 停止收音後釋放麥克風，連線繼續接收最後一段結果。
function releaseAudio(current) {
  clearTimeout(current.stopTimer);
  current.stream?.getTracks().forEach(track => track.stop());
  current.source?.disconnect();
  current.node?.port.close();
  current.node?.disconnect();
  current.context?.close().catch(() => {});
  current.stream = current.source = current.node = current.context = null;
}

function closeSession(current, message) {
  if (session !== current) return;
  session = null;
  releaseAudio(current);
  current.socket?.close();
  setPhase("ready", message);
}

function finishAudio(current) {
  if (session !== current || !current.node) return;
  releaseAudio(current);
  if (current.socket.readyState === WebSocket.OPEN) send(current, "stop");
}

function stopRecording(current) {
  setPhase("stopping", "正在完成最後一段轉錄…");
  // 等 Worklet 送完尾包。若音訊處理已中止，仍要通知後端收尾。
  current.stopTimer = setTimeout(() => finishAudio(current), 1500);
  current.node.port.postMessage("stop");
}

async function startRecording() {
  setPhase("preparing", "正在準備麥克風…");
  const current = {model: model.value};
  session = current;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error("請用支援 AudioWorklet 的瀏覽器，從 localhost 或 HTTPS 開啟。");
    }
    current.context = new AudioContext({sampleRate: config.sampleRate});
    if (current.context.sampleRate !== config.sampleRate) {
      throw new Error("瀏覽器無法提供 16 kHz 音訊。");
    }
    await current.context.resume();
    if (session !== current) return;
    await current.context.audioWorklet.addModule("/pcm-worklet.js");
    if (session !== current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {channelCount: {ideal: 1}, sampleRate: {ideal: config.sampleRate}},
    });
    if (session !== current) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    current.stream = stream;
    current.source = current.context.createMediaStreamSource(stream);
    current.node = new AudioWorkletNode(current.context, "pcm-recorder", {
      channelCount: 1,
      channelCountMode: "explicit",
      processorOptions: {frameSamples: config.frameSamples},
    });
    current.node.port.onmessage = ({data}) => {
      if (session !== current || !current.node || current.socket?.readyState !== WebSocket.OPEN) return;
      if (data === "stopped") finishAudio(current);
      else current.socket.send(data);
    };
    current.node.onprocessorerror = () => {
      if (session !== current) return;
      if (phase === "recording" || phase === "stopping") {
        setPhase("stopping", "音訊處理器發生錯誤，正在停止錄音…");
        finishAudio(current);
      } else {
        closeSession(current, "音訊處理器發生錯誤。");
      }
    };

    // 先完成待處理的語言設定，再為這場錄音建立 WebSocket。
    await targetUpdate;
    if (session !== current) return;
    statusLabel.textContent = "正在建立錄音連線…";
    current.socket = new WebSocket(socketUrl);
    current.socket.addEventListener("message", ({data}) => receiveResponse(current, data));
    current.socket.addEventListener("close", () => {
      closeSession(current, current.error ?? "錄音連線已中斷，可以重新開始錄音。");
    });
  } catch (error) {
    closeSession(current, error.message);
  }
}

// 原文和譯文使用同一列，等待新的譯文期間保留已有的文字。
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
  if (entry && !message.target) {
    const translation = row.querySelector(".transcript-translation");
    translation.hidden = true;
    translation.textContent = "";
  }
  if (!entry) transcriptList.append(row);
  transcriptRows.set(message.segment_id, {row, revision: message.revision, final: message.final});
  transcriptEmpty.hidden = true;
}

function showTranslation(message) {
  const entry = transcriptRows.get(message.segment_id);
  // 語言可能在原文送出後才變更，仍用段落編號和版本找到同一列。
  if (!entry || entry.revision !== message.revision) return;
  if (!entry.final && message.target !== target.value) return;

  const output = entry.row.querySelector(".transcript-translation");
  const text = message.translation || (output.hidden ? message.error : "");
  if (!text) return;
  output.textContent = text;
  output.hidden = false;
}

function receiveResponse(current, data) {
  if (session !== current) return;
  logResponse(data);
  const message = parseResponse(data);
  switch (message?.type) {
    case "ready":
      send(current, "start", {model: current.model});
      break;
    case "started":
      transcriptList.replaceChildren();
      transcriptEmpty.hidden = false;
      transcriptRows.clear();
      current.source.connect(current.node);
      current.node.connect(current.context.destination);
      setPhase("recording", "錄音中");
      break;
    case "transcript":
    case "translation": {
      const follow = transcriptFeed.scrollHeight - transcriptFeed.scrollTop
        - transcriptFeed.clientHeight < 32;
      if (message.type === "transcript") showTranscript(message);
      else showTranslation(message);
      if (follow) transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
      break;
    }
    case "stopped":
      closeSession(current, "錄音已停止");
      break;
    case "error":
      if (message.model) showActiveModel(message.model);
      if (phase === "preparing") closeSession(current, message.message);
      else {
        current.error = message.message;
        statusLabel.textContent = message.message;
      }
      break;
  }
}

model.addEventListener("change", selectModel);
target.addEventListener("change", () => {
  targetChanged = true;
  const selectedTarget = target.value;
  // 依序送出每次選擇，避免較早的請求覆蓋較新的語言。
  const update = targetUpdate.then(async () => {
    let hint = translationInstructions;
    try {
      const settings = await requestJson("/v1/set_translate_target_language", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({target_lang: selectedTarget}),
      });
      activeTarget = settings.target_lang;
    } catch (error) {
      hint = `無法設定翻譯語言。${error.message}`;
    }
    if (targetUpdate === update) {
      target.value = activeTarget;
      translationHint.textContent = hint;
    }
  });
  targetUpdate = update;
});
record.addEventListener("click", () => {
  if (phase === "recording") stopRecording(session);
  else if (phase === "ready") startRecording();
});
window.addEventListener("pagehide", () => {
  if (session) closeSession(session, "可以開始錄音");
});

// 先讀取目前的語言設定，接著處理選單變更。
targetUpdate = loadSettings();
