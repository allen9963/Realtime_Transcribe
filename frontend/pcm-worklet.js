// 根據取樣數組成固定長度的 PCM 封包，停止時把未滿的尾包補零。
class PcmRecorder extends AudioWorkletProcessor {
  constructor({processorOptions}) {
    super();
    this.frameSamples = processorOptions.frameSamples;
    this.active = true;
    this.resetFrame();
    this.port.onmessage = () => {
      this.active = false;
      if (this.offset > 0) this.sendFrame();
      this.port.postMessage("stopped");
    };
  }

  resetFrame() {
    this.frame = new DataView(new ArrayBuffer(this.frameSamples * 2));
    this.offset = 0;
  }

  sendFrame() {
    const buffer = this.frame.buffer;
    this.port.postMessage(buffer, [buffer]);
    this.resetFrame();
  }

  process(inputs) {
    if (!this.active) return false;
    // AudioWorkletNode 已把輸入混成單聲道。輸出維持靜音，避免麥克風回授。
    for (const sample of inputs[0][0] ?? []) {
      const clipped = Math.max(-1, Math.min(1, sample));
      const pcm = Math.round(clipped * (clipped < 0 ? 32768 : 32767));
      this.frame.setInt16(this.offset * 2, pcm, true);
      this.offset++;
      if (this.offset === this.frameSamples) this.sendFrame();
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PcmRecorder);
