
import time
# import transcribe_cpp
from llama_cpp import Llama
llm = Llama(
    # model_path="models/Qwen3.5-2B-Q4_K_M.gguf",
    # model_path="models/translategemma-4b-it-Q4_K_M.gguf",
    # model_path="models/gemma-4-E2B-it-assistant.F16.gguf",
    # model_path="models/gemma-3-1b-it-Q4_K_M.gguf",
    # model_path="models/gemma-3-1b-it-Q8_0.gguf",
    # model_path="models/Qwen3-0.6B-Q8_0.gguf",
    # model_path="models/Index-Translate-2B.Q8_0.gguf",
    # model_path="models/madlad400-3b-mt-q4_k_m.gguf",
    # chat_format="llama-2"
    # chat_format="gemma",
    # model_path="models/gemma-4-E2B-it.Q8_0-pleQ4_K.gguf",
    # model_path="models/plamo-3-translate-0.8b-Q4_K_M.gguf",
    model_path="models/Hy-MT2-1.8B-Q4_K_M.gguf",
    # model_path="models/Hy-MT2-1.8B-1.25Bit.gguf",
    verbose=True,
    enable_thinking=False
)
start = time.time()
# text = "学英语，很多人叫我做翻译，我没时间。我可以把它介绍给我的老师和同事去做。而当时的想法，我觉得应该会挣钱。"
# trg_lang = "en"
# src_lang = "zh"
# # src_lang = "Chinese"
# # trg_lang = "English"
# # text = "Today is March twenty six, twenty twenty one."
text = "And um, and NVIDIA's market cap was eight billion dollars when Stripe launched in twenty eleven."
# text = "And um, and NVIDIA's market cap was nine billion dollars when Stripe launched in twenty eleven."
# text = "That was a very big move for them, and when we went to the US as a family, I think the the key part that my parents always instilled upon me is."
# text = "Nine billion."

# trg_lang = "zh"
# src_lang = "en"
src_lang = "English"
trg_lang = "Chinese"
# trg_lang = "Chinese, Mandarin(Traditional, Taiwan)"

# instruction = f"""You are a professional translator. Only translate the following text into {trg_lang} /no_think.
# """
# instruction = f"""Please translate the following text into {trg_lang} and output the translation directly without any explanation.
# # """
#
# user_input = f"Please translate the following text into {trg_lang} and output the translation directly without any explanation. \n\n{text}  \n/no_think"

# user_input = f"Please translate the following text into {trg_lang} and output the translation directly without any explanation. \n\n{text}"
user_input = f"Translate the following segment into {trg_lang}, without additional explanation：{text}"
# user_input = f"Translate the following text into {trg_lang}. Note that you should only output the translated result without any additional explanation:{text}"
# user_input = f"<2{trg_lang}> {text}"
# print(instruction)
result = llm.create_chat_completion(
    messages = [
      # {"role": "system", "content": instruction},
      {
          "role": "user",
          "content": user_input
      }
    ],
    temperature=0.0,
    # messages=[
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "source_lang_code": "en",
    #                 "target_lang_code": "zh",
    #                 "text": text
    #             }
    #         ],
    #     }
    # ]
)


# result = llm.create_chat_completion(
#     messages = [
#       {"role": "system", "content": "[plamo-translate]\ntarget_lang=zh"},
#       {
#             "role": "user",
#             "content": text,
#       }
#     ],
#     temperature=0.0,
#     # chat_template_kwargs={
#     #
#     # },
#     max_tokens=512,
#     stop=["<|plamo:tag|>", "<|plamo:bos|>"]
#     # messages=[
#     #     {
#     #         "role": "user",
#     #         "content": [
#     #             {
#     #                 "type": "text",
#     #                 # "source_lang_code": src_lang,
#     #                 # "target_lang_code": trg_lang,
#     #                 "src_lang": src_lang,
#     #                 "tgt_lang": trg_lang,
#     #                 "text": text
#     #             }
#     #         ],
#     #     }
#     # ]
# )
#
print(result)
print(result['choices'][0]['message']['content'])
print(time.time() - start)