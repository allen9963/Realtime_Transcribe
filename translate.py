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


# import numpy as np
# # import soundfile as sf
# from crispasr import Session
# import time
#
# MODEL_PATH = "models/omniasr-ctc-300m-v2.gguf"
# AUDIO_PATH = "audio\\recording-16k.wav"
#
# # # 讀取音訊
# # # audio, sample_rate = sf.read(
# # #     AUDIO_PATH,
# # #     dtype="float32"
# # # )
# #
# # 建立 OmniASR Session
# texts = "And um, and NVIDIA's market cap was eight billion dollars when Stripe launched in twenty eleven."
# # texts = "This year is twenty eleven."
# # texts = "Today is March twenty six, twenty twenty one."
# with Session(
#         model_path="models\\m2m100-418m-q4_k.gguf",
#         backend="m2m100",
#         n_threads=4,
# ) as model:
#
#     print(f"Backend: {model.backend}")
#     start = time.time()
#     segments = model.translate_text(
#         # text="学英语，很多人叫我做翻译，我没时间。我可以把它介绍给我的老师和同事去做。而当时的想法，我觉得应该会挣钱。",
#         # src_lang="zh",
#         # tgt_lang="en",
#         text=texts,
#         src_lang="en",
#         tgt_lang="zh",
#     )
#     print(segments)
#     print(time.time() - start)



# from pathlib import Path
#
# import ctranslate2
# import sentencepiece as spm
# import time
#
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
# # text = "Could you tell me where the station is?"
# # text = "学英语，很多人叫我做翻译，我没时间。我可以把它介绍给我的老师和同事去做。而当时的想法，我觉得应该会挣钱。"
# # target_language = "en"
#
# text = "And um, and NVIDIA's market cap was eight billion dollars when Stripe launched in twenty eleven."
# # target_language = "zh"
#
# # text = "in twenty eleven."
# # text = "That was a very big move for them, and when we went to the US as a family, I think the the key part that my parents always instilled upon me is."
# # text = "Nine billion."
# target_language = "zh"
#
# start = time.time()
# # SMaLL-100 places the target-language token at the beginning of the source.
# source_tokens = (
#     [f"__{target_language}__"]
#     + tokenizer.encode(text, out_type=str)
#     + ["</s>"]
# )
# print(source_tokens)
# result = translator.translate_batch(
#     [source_tokens],
#     beam_size=5,
#     max_decoding_length=256,
#     length_penalty=1.2,
#     coverage_penalty=0.1,
#     return_end_token=True,
# )
# translation = tokenizer.decode(result[0].hypotheses[0])
# print(translation)
# print(time.time() - start)

# from transformers import M2M100ForConditionalGeneration
# from tokenization_small100 import SMALL100Tokenizer
#
# hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
# chinese_text = "生活就像一盒巧克力。"
#
# model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
# tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
#
# # translate Hindi to French
# tokenizer.tgt_lang = "fr"
# encoded_hi = tokenizer(hi_text, return_tensors="pt")
# generated_tokens = model.generate(**encoded_hi)
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# # => "La vie est comme une boîte de chocolat."
#
# # translate Chinese to English
# tokenizer.tgt_lang = "en"
# encoded_zh = tokenizer(chinese_text, return_tensors="pt")
# generated_tokens = model.generate(**encoded_zh)
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# # => "Life is like a box of chocolate."


import numpy as np
# import soundfile as sf
from langdetect import detect, detect_langs, DetectorFactory

DetectorFactory.seed = 0
from crispasr import Session
import time
# from words2num2 import words2num, alpha2digits

# MODEL_PATH = "models/omniasr-ctc-300m-v2.gguf"
# AUDIO_PATH = "audio\\recording-16k.wav"

# # 讀取音訊
# # audio, sample_rate = sf.read(
# #     AUDIO_PATH,
# #     dtype="float32"
# # )
#
# 建立 OmniASR Session
with Session(
    model_path="models\\madlad400-3b-mt-q4_k.gguf",
    backend="madlad",
    n_threads=4,
) as model:
    start = time.time()
    # texts = "And um, and NVIDIA's market cap was nine billion dollars when Stripe launched in twenty eleven."
    # texts = "This year is twenty eleven."
    texts = "That was a very big move for them, and when we went to the US as a family, I think the the key part that my parents always instilled upon me is."
    # texts = "one billion"
    language = detect(texts)
    confidence = detect_langs(texts)
    # print(language)
    # print(alpha2digits(texts, lang=language))
    # texts = alpha2digits(texts, lang=language)

    print(texts)
    print(f"Language: {language}")
    print(f"Scores: {confidence}")
    print(f"Backend: {model.backend}")

    segments = model.translate_text(
        # text="学英语，很多人叫我做翻译，我没时间。我可以把它介绍给我的老师和同事去做。而当时的想法，我觉得应该会挣钱。",
        text=texts,
        src_lang=language,
        tgt_lang="zh",
    )
    print(segments)
    print(time.time() - start)