"""
evaluate_tigre_zeroshot.py

Evaluates the trained en<->am/en<->ti model on English<->Tigre translation
WITHOUT any Tigre fine-tuning -- Tigre never appears in training data
anywhere in this project. This tests whether the model's shared MoVoC_Tok
vocabulary (which does cover Tigre) and its exposure to two *other*
Ge'ez-script languages (Amharic, Tigrinya) gives it any zero-shot transfer
to a third, unseen one.

Only 45 real English-Tigre sentence pairs exist anywhere in this project
(Tatoeba-sourced, via HornMT) -- this is reported explicitly as a small
qualitative check, not a statistically robust benchmark. No Tigre data of
any larger scale exists to evaluate against; this project does not
fabricate a larger evaluation set to make the result look more robust than
it is.

Since MoVoC_Tok has no dedicated Tigre direction tag baked into training
(only >>amh<</>>tir<</>>eng<< were used), Tigre is evaluated in the same
direction-tag scheme by reusing ">>tir<<" (the closest linguistic relative
among trained directions) to signal "produce Ge'ez-script output" -- this
choice is reported explicitly, not hidden, since it materially affects the
zero-shot setup.

USAGE
    python evaluate_tigre_zeroshot.py --model_dir ../04_training/mt_output/final
"""

import argparse
import json

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TATOEBA_EN_TIG_EN = "/homes/neumann/teklehaymanot/TigrinyaTokenizer/MPETokenization/Paralleldata/HornMT/data/Tatoeba.en-tig.en"
TATOEBA_EN_TIG_TIG = "/homes/neumann/teklehaymanot/TigrinyaTokenizer/MPETokenization/Paralleldata/HornMT/data/Tatoeba.en-tig.tig"
MAX_LENGTH = 128


def read_parallel(en_path, tig_path):
    en_lines = [l.strip() for l in open(en_path, encoding="utf-8") if l.strip()]
    tig_lines = [l.strip() for l in open(tig_path, encoding="utf-8") if l.strip()]
    n = min(len(en_lines), len(tig_lines))
    return en_lines[:n], tig_lines[:n]


def translate(model, tokenizer, texts, direction_tag, device):
    outputs = []
    for text in texts:
        inputs = tokenizer(f"{direction_tag} {text}", return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH, num_beams=4)
        outputs.append(tokenizer.decode(out_ids[0], skip_special_tokens=True))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../04_training/mt_output/final")
    parser.add_argument("--report_out", default="./tigre_zeroshot_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- loading model from {args.model_dir} (device={device}) ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    print(f"--- loading real 45-pair English-Tigre Tatoeba set ---")
    en_lines, tig_lines = read_parallel(TATOEBA_EN_TIG_EN, TATOEBA_EN_TIG_TIG)
    print(f"  {len(en_lines)} pairs (zero-shot -- Tigre never appeared in training)")

    print("--- en -> tig (zero-shot, using >>tir<< direction tag) ---")
    en_to_tig_hyp = translate(model, tokenizer, en_lines, ">>tir<<", device)
    bleu_en_tig = sacrebleu.corpus_bleu(en_to_tig_hyp, [tig_lines])
    chrf_en_tig = sacrebleu.corpus_chrf(en_to_tig_hyp, [tig_lines])

    print("--- tig -> en (zero-shot) ---")
    tig_to_en_hyp = translate(model, tokenizer, tig_lines, ">>eng<<", device)
    bleu_tig_en = sacrebleu.corpus_bleu(tig_to_en_hyp, [en_lines])
    chrf_tig_en = sacrebleu.corpus_chrf(tig_to_en_hyp, [en_lines])

    report = {
        "eval_set": "Tatoeba en-tig (via HornMT), 45 pairs -- zero-shot, Tigre absent from training",
        "n_pairs": len(en_lines),
        "en_to_tig": {
            "bleu": bleu_en_tig.score,
            "chrf": chrf_en_tig.score,
            "examples": [
                {"source": s, "hypothesis": h, "reference": r}
                for s, h, r in list(zip(en_lines, en_to_tig_hyp, tig_lines))[:10]
            ],
        },
        "tig_to_en": {
            "bleu": bleu_tig_en.score,
            "chrf": chrf_tig_en.score,
            "examples": [
                {"source": s, "hypothesis": h, "reference": r}
                for s, h, r in list(zip(tig_lines, tig_to_en_hyp, en_lines))[:10]
            ],
        },
    }

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nen->tig: BLEU={bleu_en_tig.score:.3f} chrF={chrf_en_tig.score:.3f}")
    print(f"tig->en: BLEU={bleu_tig_en.score:.3f} chrF={chrf_tig_en.score:.3f}")
    print(f"\nReport written to {args.report_out}")
    print("\nNOTE: 45 pairs is a small qualitative check, not a robust benchmark.")


if __name__ == "__main__":
    main()
