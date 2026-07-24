"""
train_mt.py

Trains the from-scratch MarianMT model (../03_model/init_model, MoVoC_Tok
tokenizer) on English<->Amharic and English<->Tigrinya, bidirectionally,
mixing both language pairs into one multilingual model distinguished by
direction tags (">>amh<<" / ">>tir<<" / ">>eng<<") prepended to the
encoder input -- the same pattern EnTiMT already validated, extended from
one language pair to two.

Tigre is deliberately NOT included anywhere in this training data -- it is
evaluated separately, zero-shot, in ../05_evaluation/evaluate_tigre_zeroshot.py.

Hyperparameters otherwise match the paper's original checkpoint-524316 run
(lr=5e-5, linear schedule, 500 warmup steps, 3 epochs) except batch size:
this run's combined bidirectional en-am + en-ti data is ~3x the original
single-direction corpus, and MoVoC_Tok's vocab is ~1.9x larger (120k vs
63k, which slows the output projection layer specifically), so
per_device_train_batch_size is raised from the paper's 8 to 32 to keep
wall-clock time within the SLURM job's time limit -- a deliberate,
disclosed deviation, not an oversight.

USAGE
    python train_mt.py --output_dir ./mt_output
"""

import argparse

import numpy as np
import sacrebleu
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_DIR = "../03_model/init_model"
MAX_LENGTH = 128


def read_parallel(en_path, other_path):
    en_lines = open(en_path, encoding="utf-8").read().split("\n")
    other_lines = open(other_path, encoding="utf-8").read().split("\n")
    n = min(len(en_lines), len(other_lines))
    return en_lines[:n], other_lines[:n]


def build_bidirectional_dataset(en_lines, other_lines, other_tag):
    sources, targets = [], []
    for en, other in zip(en_lines, other_lines):
        if not en or not other:
            continue
        sources.append(f"{other_tag} {en}")
        targets.append(other)
        sources.append(">>eng<< " + other)
        targets.append(en)
    return Dataset.from_dict({"source": sources, "target": targets})


def make_preprocess_fn(tokenizer):
    eos_id = tokenizer.eos_token_id

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["source"], max_length=MAX_LENGTH, truncation=True
        )
        labels = tokenizer(
            examples["target"], max_length=MAX_LENGTH, truncation=True
        )
        model_inputs["input_ids"] = [
            ids[: MAX_LENGTH - 1] + [eos_id] for ids in model_inputs["input_ids"]
        ]
        model_inputs["attention_mask"] = [
            mask[: MAX_LENGTH - 1] + [1] for mask in model_inputs["attention_mask"]
        ]
        model_inputs["labels"] = [
            ids[: MAX_LENGTH - 1] + [eos_id] for ids in labels["input_ids"]
        ]
        return model_inputs

    return preprocess


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
        return {"bleu": bleu.score, "chrf": chrf.score}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_en_am", default="../02_cleaning/clean_am/train.en")
    parser.add_argument("--train_am", default="../02_cleaning/clean_am/train.am")
    parser.add_argument("--dev_en_am", default="../02_cleaning/clean_am/dev.en")
    parser.add_argument("--dev_am", default="../02_cleaning/clean_am/dev.am")
    parser.add_argument("--train_en_ti", default="../02_cleaning/clean_ti/train.en")
    parser.add_argument("--train_ti", default="../02_cleaning/clean_ti/train.ti")
    parser.add_argument("--dev_en_ti", default="../02_cleaning/clean_ti/dev.en")
    parser.add_argument("--dev_ti", default="../02_cleaning/clean_ti/dev.ti")
    parser.add_argument("--output_dir", default="./mt_output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=lambda x: x.lower() != "false", default=True)
    args = parser.parse_args()

    print("--- loading from-scratch model + MoVoC_Tok tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    print("--- building bidirectional en<->am and en<->ti datasets ---")
    train_en_am, train_am = read_parallel(args.train_en_am, args.train_am)
    dev_en_am, dev_am = read_parallel(args.dev_en_am, args.dev_am)
    train_en_ti, train_ti = read_parallel(args.train_en_ti, args.train_ti)
    dev_en_ti, dev_ti = read_parallel(args.dev_en_ti, args.dev_ti)

    train_am_ds = build_bidirectional_dataset(train_en_am, train_am, ">>amh<<")
    train_ti_ds = build_bidirectional_dataset(train_en_ti, train_ti, ">>tir<<")
    dev_am_ds = build_bidirectional_dataset(dev_en_am, dev_am, ">>amh<<")
    dev_ti_ds = build_bidirectional_dataset(dev_en_ti, dev_ti, ">>tir<<")

    train_ds = concatenate_datasets([train_am_ds, train_ti_ds]).shuffle(seed=args.seed)
    dev_ds = concatenate_datasets([dev_am_ds, dev_ti_ds])
    print(f"  train: {len(train_ds)} examples ({len(train_am_ds)} am-derived + {len(train_ti_ds)} ti-derived)")
    print(f"  dev: {len(dev_ds)} examples ({len(dev_am_ds)} am-derived + {len(dev_ti_ds)} ti-derived)")

    preprocess = make_preprocess_fn(tokenizer)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["source", "target"])
    dev_ds = dev_ds.map(preprocess, batched=True, remove_columns=["source", "target"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        fp16=args.fp16,
        logging_steps=100,
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print("--- training ---")
    trainer.train()

    print("--- saving final model ---")
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")


if __name__ == "__main__":
    main()
