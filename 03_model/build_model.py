"""
build_model.py

Builds a from-scratch MarianMT model matching the exact architecture of
the paper's original checkpoint-524316 run (confirmed via its saved
config.json at
../../MPETokenization/Paralleldata/results/checkpoint-524316/config.json):
6 encoder + 6 decoder layers, 8 attention heads, d_model=512,
ffn_dim=2048, Swish activation, shared encoder/decoder embeddings, static
(sinusoidal) position embeddings -- but wired to MoVoC_Tok's 120,000-token
shared Ge'ez-script+English tokenizer instead of that run's bespoke
63,050-token vocabulary (120,000 still satisfies the >=63,050 requirement).

This is a genuinely new model (random init), not a fine-tune of any
pretrained checkpoint -- there is no pretrained MarianMT checkpoint
anywhere with a 120k MoVoC_Tok-compatible vocabulary, and the whole point
of this run is to reproduce the paper's own from-scratch training setup,
not a transfer-learning experiment.

USAGE
    python build_model.py --outdir ./init_model
"""

import argparse

from tokenizers import Tokenizer
from transformers import MarianConfig, MarianMTModel, PreTrainedTokenizerFast

MOVOC_TOK_JSON = "/homes/neumann/teklehaymanot/MoVoC_Tok/04_validation/tokenizer.json"
DIRECTION_TOKENS = [">>eng<<", ">>amh<<", ">>tir<<"]


def build_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer.from_file(MOVOC_TOK_JSON)
    base_vocab_size = tok.get_vocab_size()

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=DIRECTION_TOKENS,
        model_input_names=["input_ids", "attention_mask"],
    )
    # MoVoC_Tok has no dedicated <pad> token. The paper's own real run used
    # pad_token_id=63049 -- the LAST index of its 63,050-token vocab, not a
    # small/shared id -- and trained successfully, which confirms Marian's
    # static (sinusoidal) position embedding does not have the same
    # padding_idx-indexes-a-small-table bug that forced EnTiMT to reuse
    # <unk> as pad for its NLLB/M2M100 transplant. So we replicate the real,
    # proven-working pattern here: add one new dedicated <pad> token at the
    # end of the vocabulary, rather than overload an existing token.
    fast_tok.add_special_tokens({"pad_token": "<pad>"})
    assert fast_tok.pad_token_id == base_vocab_size + len(DIRECTION_TOKENS), (
        f"expected pad token appended after direction tokens, got id "
        f"{fast_tok.pad_token_id} vs expected {base_vocab_size + len(DIRECTION_TOKENS)}"
    )
    return fast_tok


def build_model(vocab_size: int, pad_token_id: int, eos_token_id: int, decoder_start_token_id: int) -> MarianMTModel:
    config = MarianConfig(
        vocab_size=vocab_size,
        decoder_vocab_size=vocab_size,
        d_model=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        activation_function="swish",
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_position_embeddings=512,
        share_encoder_decoder_embeddings=True,
        static_position_embeddings=True,
        normalize_embedding=False,
        add_final_layer_norm=False,
        scale_embedding=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=0,
        decoder_start_token_id=decoder_start_token_id,
        forced_eos_token_id=eos_token_id,
    )
    return MarianMTModel(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./init_model")
    args = parser.parse_args()

    print("--- building MoVoC_Tok tokenizer wrapper (+ direction tags + <pad>) ---")
    tokenizer = build_tokenizer()
    vocab_size = len(tokenizer)
    print(f"  final vocab size: {vocab_size} (MoVoC_Tok base 120,000 + "
          f"{len(DIRECTION_TOKENS)} direction tags + 1 pad token)")
    print(f"  pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}, "
          f"bos_token_id={tokenizer.bos_token_id}, unk_token_id={tokenizer.unk_token_id}")

    print("--- building from-scratch MarianMT model (random init) ---")
    model = build_model(
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model built, {n_params:,} parameters")

    print(f"--- saving to {args.outdir} ---")
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print("done.")


if __name__ == "__main__":
    main()
