#!/usr/bin/env python3
"""Patch pwilkin/openmoss for MOSS-TTS-v1.5 prompt + audio_start priming."""
from pathlib import Path
import sys

PIPELINE = Path("openmoss/src/pipeline.cpp")


def patch_delay_h(text: str) -> str:
    old = """    std::vector<int32_t> extract_audio_codes(int32_t & n_vq_out, int32_t & t_audio) const;

private:"""
    new = """    std::vector<int32_t> extract_audio_codes(int32_t & n_vq_out, int32_t & t_audio) const;

    // MOSS-TTS v1.5: <|audio_start|> is the first generated text token (not prefilled).
    void prime_audio_start();

private:"""
    if old not in text:
        raise SystemExit("delay.h: extract_audio_codes anchor not found")
    return text.replace(old, new, 1)


def patch_delay_cpp(text: str) -> str:
    insert_before = "DelayStep DelayState::step(const float * text_logits,"
    method = """
void DelayState::prime_audio_start() {
    if (!m_history.empty() &&
        m_history.back().front() == m_dims.audio_start_token_id) {
        return;
    }
    std::vector<int32_t> row(1 + m_dims.n_vq, m_dims.audio_pad_code);
    row[0] = m_dims.audio_start_token_id;
    m_history.push_back(std::move(row));
    m_is_audio       = true;
    m_audio_length   = 1;
    m_delayed_length = -1;
    m_is_stopping    = false;
}

"""
    if insert_before not in text:
        raise SystemExit("delay.cpp: DelayState::step anchor not found")
    return text.replace(insert_before, method + insert_before, 1)


def patch_pipeline(text: str) -> str:
    old_prompt = """    const std::string im_start    = id_to_token(tok, d.im_start_token_id);
    const std::string im_end      = id_to_token(tok, d.im_end_token_id);
    const std::string audio_start = id_to_token(tok, d.audio_start_token_id);

    std::string body = build_user_inst(req, reference_block);
    std::string out;
    out += im_start + "user\\n" + body + im_end + "\\n"
         + im_start + "assistant\\n" + audio_start;
    return out;"""

    new_prompt = """    const std::string im_start = id_to_token(tok, d.im_start_token_id);
    const std::string im_end   = id_to_token(tok, d.im_end_token_id);

    std::string body = build_user_inst(req, reference_block);
    std::string out;
    // MOSS-TTS v1.5: HF apply_chat_template(..., add_generation_prompt=True)
    // ends at "<|im_start|>assistant\\n" — <|audio_start|> is generated next.
    out += im_start + "user\\n" + body + im_end + "\\n"
         + im_start + "assistant\\n";
    return out;"""

    if old_prompt not in text:
        raise SystemExit("pipeline.cpp: build_prompt_text block not found")
    text = text.replace(old_prompt, new_prompt, 1)

    old_loop = """    // ── 3. Autoregressive loop ─────────────────────────────────────────────
    auto t_gen = clock_t_::now();
    int32_t pos = prompt_len;
    int32_t step = 0;
    DelayStep last_step;
    for (; step < req.max_new_tokens; ++step) {"""

    new_loop = """    // ── 3. Prime <|audio_start|> (v1.5 first generated text token) ────────
    int32_t pos = prompt_len;
    if (prompt_len > 0 &&
        grid[size_t(prompt_len - 1) * size_t(1 + n_vq)] != d.audio_start_token_id) {
        state.prime_audio_start();
        std::vector<int32_t> row(1 + n_vq, d.audio_pad_code);
        row[0] = d.audio_start_token_id;

        auto prime_emb = model.compute_input_embeddings(row.data(), 1);
        llama_decode_embeddings(model.backbone_ctx(),
                                prime_emb.data(),
                                /*n_tokens=*/1, hidden,
                                /*pos_start=*/prompt_len,
                                /*output_last=*/true);
        pos = prompt_len + 1;
        std::fprintf(stderr, "[generate] primed audio_start at pos %d\\n", prompt_len);
    }

    // ── 4. Autoregressive loop ─────────────────────────────────────────────
    auto t_gen = clock_t_::now();
    int32_t step = 0;
    DelayStep last_step;
    for (; step < req.max_new_tokens; ++step) {"""

    if old_loop not in text:
        raise SystemExit("pipeline.cpp: autoregressive loop anchor not found")
    text = text.replace(old_loop, new_loop, 1)

    # Renumber comment sections after insert (optional cosmetic)
    text = text.replace(
        "    // ── 4. Extract audio codes",
        "    // ── 5. Extract audio codes",
        1,
    )
    text = text.replace(
        "    // ── 5. Codec decode",
        "    // ── 6. Codec decode",
        1,
    )
    return text


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("openmoss")
    pipeline = root / "src/pipeline.cpp"
    delay_h = root / "include/openmoss/delay.h"
    delay_cpp = root / "src/delay.cpp"

    pipeline.write_text(patch_pipeline(pipeline.read_text()))
    delay_h.write_text(patch_delay_h(delay_h.read_text()))
    delay_cpp.write_text(patch_delay_cpp(delay_cpp.read_text()))
    print(f"Patched {root} for MOSS-TTS v1.5")


if __name__ == "__main__":
    main()
