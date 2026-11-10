# import numpy as np
# import soundfile as sf
# from crispasr import Session
# import time
#
# MODEL_PATH = "models/omniasr-ctc-1b-v2-q8_0.gguf"
# AUDIO_PATH = "audio\\recording-16k.wav"
#
# # 讀取音訊
# audio, sample_rate = sf.read(
#     AUDIO_PATH,
#     dtype="float32"
# )
#
# # 建立 OmniASR Session
# with Session(
#     model_path=MODEL_PATH,
#     backend="omniasr",
#     n_threads=4,
# ) as model:
#
#     print(f"Backend: {model.backend}")
#     start = time.time()
#     segments = model.transcribe(
#         np.asarray(audio, dtype=np.float32),
#         sample_rate=sample_rate,
#     )
#
#     for segment in segments:
#         print(
#             f"[{segment.start:.2f} - {segment.end:.2f}] "
#             f"{segment.text}"
#         )
#     print(time.time() - start)
#
# # import numpy as np
# # import soundfile as sf
# # from crispasr import Session
# # import time
# #
# # MODEL_PATH = "models/omniasr-ctc-300m-v2.gguf"
# # AUDIO_PATH = "audio\\recording-16k.wav"
# #
# # # 讀取音訊
# # audio, sample_rate = sf.read(
# #     AUDIO_PATH,
# #     dtype="float32"
# # )
# #
# # # 建立 OmniASR Session
# # with Session(
# #         model_path="models\\m2m100-418m-q8_0.gguf",
# #         backend="m2m100",
# #         n_threads=4,
# # ) as model:
# #
# #     print(f"Backend: {model.backend}")
# #     start = time.time()
# #     segments = model.transcribe(
# #         np.asarray(audio, dtype=np.float32),
# #         sample_rate=sample_rate,
# #     )
# #
# #     for segment in segments:
# #         print(
# #             f"[{segment.start:.2f} - {segment.end:.2f}] "
# #             f"{segment.text}"
# #         )
# #     print(time.time() - start)



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

# text = "Could you tell me where the station is?"
text = "其中，我最推薦 SMaLL-100 與 PLaMo 3 Translate 0.8B，因為兩者不需要額外建立語言偵測模型，也能支援多國語言互譯。"
target_language = "en"

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