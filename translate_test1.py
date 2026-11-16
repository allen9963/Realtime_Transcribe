import ctypes
import os

import llama_cpp as lc
import time

MODEL_PATH = "models/madlad400-3b-mt-q4_k_m.gguf"


def tokenize(vocab, text: str) -> list[int]:
    """將翻譯輸入轉換為 Token ID。"""
    raw_text = text.encode("utf-8")
    capacity = 512
    tokens = (lc.llama_token * capacity)()

    count = lc.llama_tokenize(
        vocab,
        raw_text,
        len(raw_text),
        tokens,
        capacity,
        True,   # 使用模型設定的特殊 Token
        True,   # 解析 <2zh> 等特殊 Token
    )

    if count <= 0:
        raise ValueError(f"Tokenization failed: {count}")

    return list(tokens[:count])


def process_batch(ctx, tokens, is_encoder, start_pos=0):
    """建立 llama_batch，依模型階段執行 Encoder 或 Decoder。"""
    batch = lc.llama_batch_init(len(tokens), 0, 1)

    try:
        batch.n_tokens = len(tokens)

        for i, token_id in enumerate(tokens):
            batch.token[i] = token_id
            batch.pos[i] = start_pos + i

            # Encoder 和 Decoder 必須使用相同 Sequence ID
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0

            batch.logits[i] = int(
                not is_encoder and i == len(tokens) - 1
            )

        if is_encoder:
            result = lc.llama_encode(ctx, batch)
        else:
            result = lc.llama_decode(ctx, batch)

        if result != 0:
            raise RuntimeError(f"Model processing failed: {result}")

    finally:
        lc.llama_batch_free(batch)


def detokenize(vocab, token_ids):
    """將生成的 Token ID 還原成 UTF-8 文字。"""
    if not token_ids:
        return ""

    tokens = (lc.llama_token * len(token_ids))(*token_ids)
    buffer = ctypes.create_string_buffer(65536)

    size = lc.llama_detokenize(
        vocab,
        tokens,
        len(token_ids),
        buffer,
        len(buffer),
        True,
        False,
    )

    if size < 0:
        raise RuntimeError(f"Detokenization failed: {size}")

    return buffer.raw[:size].decode("utf-8", errors="replace")


def translate(text: str, target_lang="zh", max_tokens=128):
    lc.llama_backend_init()

    model = None
    ctx = None
    sampler = None

    try:
        model_params = lc.llama_model_default_params()
        model_params.n_gpu_layers = 0

        model = lc.llama_model_load_from_file(
            os.fsencode(MODEL_PATH),
            model_params,
        )

        if not model:
            raise RuntimeError("Failed to load model")

        if not lc.llama_model_has_encoder(model):
            raise RuntimeError("Model does not have an encoder")

        ctx_params = lc.llama_context_default_params()
        ctx_params.n_ctx = 1024
        ctx_params.n_batch = 512

        ctx = lc.llama_init_from_model(model, ctx_params)

        if not ctx:
            raise RuntimeError("Failed to create context")

        vocab = lc.llama_model_get_vocab(model)

        # MADLAD 使用目標語言標記
        prompt = f"<2{target_lang}> {text}"
        input_tokens = tokenize(vocab, prompt)

        # Step 1: Encoder
        process_batch(
            ctx,
            input_tokens,
            is_encoder=True,
        )

        # Step 2: Decoder 起始 Token
        start_token = lc.llama_model_decoder_start_token(model)

        if start_token < 0:
            start_token = lc.llama_vocab_bos(vocab)

        # Greedy decoding
        sampler = lc.llama_sampler_init_greedy()

        output_tokens = []
        current_token = start_token

        # Step 3: Decoder 逐 Token 生成
        for position in range(max_tokens):
            process_batch(
                ctx,
                [current_token],
                is_encoder=False,
                start_pos=position,
            )

            next_token = lc.llama_sampler_sample(
                sampler,
                ctx,
                -1,
            )

            if lc.llama_vocab_is_eog(vocab, next_token):
                break

            output_tokens.append(next_token)
            current_token = next_token

        return detokenize(vocab, output_tokens)

    finally:
        if sampler:
            lc.llama_sampler_free(sampler)
        if ctx:
            lc.llama_free(ctx)
        if model:
            lc.llama_model_free(model)

        lc.llama_backend_free()


if __name__ == "__main__":
    start = time.time()
    result = translate(
        text="And um, and NVIDIA's market cap was eight billion dollars when Stripe launched in twenty eleven.",
        target_lang="zh",
    )
    print("time:", time.time() - start)
    print("翻譯結果:", result)