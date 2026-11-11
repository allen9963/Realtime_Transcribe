# SpeechTrack 即時轉錄 Demo 修改紀錄

這份紀錄依聊天中的實作順序整理。前幾項從專案根目錄的 Demo 開始，後續翻譯、清單與子程序調整都以本目錄的版本為準。 [目前的後端](fake_stream.py) [目前的前端](frontend/index.html)

## 修改順序

### 1. 建立 FastAPI 錄音與轉錄 Demo

- 建立 `fake_stream.py`，提供錄音按鈕、模型選單、轉錄文字與翻譯語言設定。最初支援 `whisper-small-F16` 和 `Qwen3-ASR-0.6B-Q8_0`，啟動時載入 Whisper，停止錄音後可切換模型。 [後端](fake_stream.py)
- 瀏覽器透過 `/test/audio_transcribe` 傳送 16 kHz 單聲道 PCM16LE。每包固定 **32 ms、512 個取樣、1,024 bytes**。停止錄音時，前端將不足一包的尾端補零。 [音訊組包](frontend/pcm-worklet.js)
- 兩個初始模型以累積音訊重跑辨識的方式更新暫定文字。初版翻譯使用 Whisper 的英文翻譯能力。 [後端](fake_stream.py)

### 2. 拆分前端與後端

- 將原本內嵌在 Python 的 HTML、CSS、操作程式與 AudioWorklet 拆成獨立檔案，並修正內嵌 JavaScript 的換行轉義問題。FastAPI 透過靜態檔案提供頁面，前端從 WebSocket 的 `ready` 訊息取得模型清單與音訊規格。 [網頁](frontend/index.html) [樣式](frontend/styles.css) [前端程式](frontend/app.js) [音訊處理](frontend/pcm-worklet.js)

### 3. 分開顯示 Partial 與 Final

- 中央區域一度分成兩個顯示框。Partial 只顯示最新的暫定文字，Final 累積定稿段落。定稿時清空 Partial，啟用翻譯時在對應框內顯示譯文。後續在第 6 項改成逐列呈現。 [現行顯示程式](frontend/app.js)

### 4. 在 lifespan 載入 Silero VAD

- 加入 `silero-vad-v6.2-F32.gguf` 的 CPU 載入與關閉清理。這一階段將模型放在 `app.state.vad_model`，並加入模型資料夾的備援路徑。 [VAD 測試範例](vad_test.py)
- 目前版本在 ASR 子程序內為每次錄音建立 VAD session，逐包取得語音機率，並以連續靜音或停止錄音作為定稿條件。 [現行 VAD 與定稿流程](fake_stream.py)

### 5. 擴充到七種翻譯語言

- 在本目錄的 Demo 加入英文 `en`、日文 `ja`、韓文 `ko`、中文 `zh`、德文 `de`、法文 `fr`、西班牙文 `es`。當時由 SMaLL-100 翻譯 ASR 輸出的文字，讓兩個轉錄模型共用語言清單。 [前端語言選單](frontend/app.js) [後端語言設定](fake_stream.py)
- 修正翻譯函式已回傳字串卻再次取用 `.text` 的錯誤。當時的 Final 若沿用同段 Partial，也會一併沿用譯文。 [現行翻譯流程](fake_stream.py)

### 6. 改成 Android ListView 風格並支援錄音中切換語言

- 中央區域改成可捲動清單，每列上方放原文、下方放譯文。最新 Partial 在同一列更新，定稿時保留該列，下一段再新增一列。 [清單結構](frontend/index.html) [清單樣式](frontend/styles.css)
- 加入 WebSocket 的 `set_target` 指令，讓錄音中可切換翻譯語言或選擇「不翻譯」。新語言會更新目前的 Partial，已定稿項目保留原有譯文。 [前端切換邏輯](frontend/app.js) [後端指令處理](fake_stream.py)

### 7. 新增翻譯 API，原文與譯文分開回傳

- 新增 `POST /test/translate`，接收 `text` 與 `target`，回傳 `translation`。後端先送出 WebSocket 的 `transcript` 訊息，再呼叫同一個 FastAPI 服務的翻譯 API，完成後另外送出 `translation` 訊息。 [翻譯 API 與 WebSocket](fake_stream.py)
- 原文與譯文使用 `segment_id` 和 `revision` 配對。前端把譯文補在對應列，並跳過過期的回覆。翻譯失敗時保留原文。 [前端配對邏輯](frontend/app.js)

### 8. 修正譯文更新時的閃爍

- 同一段、同一語言的原文更新或定稿時，保留上一版譯文，等新版譯文到達後直接替換。空結果或翻譯失敗時保留已顯示的譯文。切換語言或關閉翻譯時，清除該列舊譯文。 [譯文顯示邏輯](frontend/app.js)

### 9. 用 spawn 隔離 ggml DLL

- 用 Python `multiprocessing` 的 `spawn` 啟動兩個獨立子程序。轉錄程序用 `transcribe_cpp` 載入選定的 ASR 模型和 Silero VAD，翻譯程序用 `crispasr` 載入 MADLAD。FastAPI 主程序透過 Pipe 傳遞音訊、文字和指令，兩套原生函式庫各自在自己的程序內載入。 [子程序實作](fake_stream.py) [Python spawn 文件](https://docs.python.org/3.11/library/multiprocessing.html#contexts-and-start-methods)
- 切換 ASR 模型時，先等新程序載入成功，再關閉舊程序。載入失敗時保留原模型。服務關閉時會等待子程序釋放模型。MADLAD 依實際輸入文字偵測來源語言。 [模型切換與資源管理](fake_stream.py)

## 驗證紀錄

- 用本機音檔測試 PCM 封包、Whisper 與 Qwen 的 WebSocket 轉錄，以及 Partial 和 Final 的譯文。用模擬 WebSocket 訊息驗證舊譯文保留與過期回覆處理。 [前端顯示程式](frontend/app.js) [後端](fake_stream.py)
- 在 Windows 檢查主程序與兩個子程序載入的 DLL，確認兩套 ggml 位於不同程序。實測模型切換成功、切換失敗時保留原模型、ASR 子程序退出後可重新載入，以及服務關閉後兩個子程序均已退出。 [子程序管理](fake_stream.py) [DLL 檢查範例](test_crispasr_dll.py)
