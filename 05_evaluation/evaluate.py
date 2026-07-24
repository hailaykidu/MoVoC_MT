"""
evaluate.py

Evaluates the trained en<->am/en<->ti model on its held-out dev sets
(2,000 pairs each, never seen during training) in both directions per
language, using BLEU and chrF (sacrebleu).

USAGE
    python evaluate.py --model_dir ../04_training/mt_output/final
"""

import argparse
import json

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_LENGTH = 128


def read_parallel(en_path, other_path):
    en_lines = [l.strip() for l in open(en_path, encoding="utf-8") if l.strip()]
    other_lines = [l.strip() for l in open(other_path, encoding="utf-8") if l.strip()]
    n = min(len(en_lines), len(other_lines))
    return en_lines[:n], other_lines[:n]


def translate(model, tokenizer, texts, direction_tag, device, batch_size=16):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{direction_tag} {t}" for t in texts[i:i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True).to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH, num_beams=4)
        outputs.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
    return outputs


def eval_direction(model, tokenizer, src_lines, tgt_lines, direction_tag, device):
    hyps = translate(model, tokenizer, src_lines, direction_tag, device)
    bleu = sacrebleu.corpus_bleu(hyps, [tgt_lines])
    chrf = sacrebleu.corpus_chrf(hyps, [tgt_lines])
    return bleu.score, chrf.score, hyps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../04_training/mt_output/final")
    parser.add_argument("--dev_en_am", default="../02_cleaning/clean_am/dev.en")
    parser.add_argument("--dev_am", default="../02_cleaning/clean_am/dev.am")
    parser.add_argument("--dev_en_ti", default="../02_cleaning/clean_ti/dev.en")
    parser.add_argument("--dev_ti", default="../02_cleaning/clean_ti/dev.ti")
    parser.add_argument("--report_out", default="./eval_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- loading model from {args.model_dir} (device={device}) ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    report = {}

    print("--- Amharic dev set (2,000 held-out pairs) ---")
    dev_en_am, dev_am = read_parallel(args.dev_en_am, args.dev_am)
    bleu, chrf, hyps = eval_direction(model, tokenizer, dev_en_am, dev_am, ">>amh<<", device)
    print(f"  en->am: BLEU={bleu:.3f} chrF={chrf:.3f}")
    report["en_to_am"] = {"bleu": bleu, "chrf": chrf, "n": len(dev_am)}

    bleu, chrf, hyps = eval_direction(model, tokenizer, dev_am, dev_en_am, ">>eng<<", device)
    print(f"  am->en: BLEU={bleu:.3f} chrF={chrf:.3f}")
    report["am_to_en"] = {"bleu": bleu, "chrf": chrf, "n": len(dev_en_am)}

    print("--- Tigrinya dev set (2,000 held-out pairs) ---")
    dev_en_ti, dev_ti = read_parallel(args.dev_en_ti, args.dev_ti)
    bleu, chrf, hyps = eval_direction(model, tokenizer, dev_en_ti, dev_ti, ">>tir<<", device)
    print(f"  en->ti: BLEU={bleu:.3f} chrF={chrf:.3f}")
    report["en_to_ti"] = {"bleu": bleu, "chrf": chrf, "n": len(dev_ti)}

    bleu, chrf, hyps = eval_direction(model, tokenizer, dev_ti, dev_en_ti, ">>eng<<", device)
    print(f"  ti->en: BLEU={bleu:.3f} chrF={chrf:.3f}")
    report["ti_to_en"] = {"bleu": bleu, "chrf": chrf, "n": len(dev_en_ti)}

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to {args.report_out}")


if __name__ == "__main__":
    main()
