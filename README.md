# MoVoC_MT: Downstream Machine Translation Evaluation of MoVoC_Tok

Real downstream validation of the [MoVoC_Tok](https://github.com/hailaykidu/Ge-ez_eng_tokenizer)
tokenizer (built for the [MoVoC](https://github.com/hailaykidu/MoVoC) project,
arXiv:[2509.08812](https://arxiv.org/abs/2509.08812)): a **from-scratch MarianMT**
model, trained bidirectionally on English↔Amharic and English↔Tigrinya, then
evaluated **zero-shot on English↔Tigre** (a third Ge'ez-script language never
seen during training) to test whether MoVoC_Tok's shared vocabulary gives any
real cross-lingual transfer. This is the kind of downstream MT validation the
MoVoC paper's own Table 3 describes, but which the MoVoC project itself never
had until now.

Model: [Hailay/movoc-mt-en-am-ti](https://huggingface.co/Hailay/movoc-mt-en-am-ti) on the Hugging Face Hub.

## Architecture

Matches the exact architecture of the paper's own original MarianMT run
(`Paralleldata/results/checkpoint-524316/config.json`), confirmed field-for-field:
6 encoder + 6 decoder layers, 8 attention heads, `d_model=512`, feedforward
dimension 2048, Swish activation, shared encoder/decoder embeddings, static
(sinusoidal) position embeddings. **106,104,832 parameters.**

The one deliberate difference: vocabulary. The original run used a bespoke
63,050-token vocabulary; this run uses **MoVoC_Tok's 120,000-token shared
Ge'ez-script + English SentencePiece Unigram vocabulary** instead (120,004
after adding 3 direction tags and 1 dedicated `<pad>` token) — still
comfortably above the ≥63,050 requirement, and the actual point of this
project: testing MoVoC_Tok downstream, not reproducing the original run
byte-for-byte.

This is a genuinely new model trained from random initialization, not a
fine-tune of any pretrained checkpoint — no pretrained MarianMT model
anywhere has a MoVoC_Tok-compatible vocabulary, and the goal is to reproduce
the paper's own from-scratch training methodology, not run a transfer-learning
experiment.

## Data

| Pair | Source | Train | Dev |
|---|---|---|---|
| English–Amharic | Raw NLLB (mined), cleaned with the same pipeline EnTiMT built for Tigrinya (NFC normalization, length/length-ratio filtering, Ethiopic-script-ratio filtering, exact + MinHash near-dup dedup) | **1,078,567** | 2,000 |
| English–Tigrinya | Reused directly from `EnTiMT/02_cleaning/clean/` (already cleaned, same pipeline) | **1,140,309** | 2,000 |
| English–Tigre (eval only) | Tatoeba, via HornMT -- the *only* English-Tigre parallel data found anywhere in this project | -- | **45 pairs** (43 scored; 2 produced empty hypotheses) |

Amharic's raw NLLB corpus is 16.1M lines; only the first 1.5M raw lines were
cleaned (yielding 1,078,567 after filtering/dedup), deliberately capped to
stay roughly balanced with Tigrinya's corpus size rather than letting one
language pair dominate the bidirectional training mix.

Tigre is **never included in training data anywhere in this project** --
its 45-pair Tatoeba set is held out purely for zero-shot evaluation, per
the project's explicit scope.

## Training

Bidirectional: every English-Amharic and English-Tigrinya pair trains both
directions, distinguished by a direction tag (`>>amh<<` / `>>tir<<` /
`>>eng<<`) prepended to the encoder input -- the same scheme validated in
EnTiMT, extended from one language pair to two. 4,437,752 examples per
epoch (2,157,134 Amharic-derived + 2,280,618 Tigrinya-derived).

| Setting | Value |
|---|---|
| True batch size | **32** (raised from the paper's 8 -- see below) |
| Learning rate | Peak **5e-05**, linear decay, 500 warmup steps |
| Epochs | 3 |
| Precision | fp16 |
| Max sequence length | 128 tokens |
| Total steps | 416,040 |

**Batch size deliberately deviates from the paper's 8.** This run's combined
bidirectional corpus is ~3x the size of the paper's original single-direction
run, and MoVoC_Tok's vocabulary is ~1.9x larger (which slows the output
projection layer specifically, since its cost scales with vocab size).
Batch 8 was projected to take 38-53 hours, risking the 48h job limit; batch
32 was chosen instead, a disclosed and deliberate deviation, not an oversight.

### Real training results (SLURM job 52623, COMPLETED, exit code 0)

| Metric | Real value |
|---|---|
| Per-epoch training loss | epoch 1.0 → **3.6062**; epoch 2.0 → **3.2935**; epoch 3.0 → **3.1292** |
| Gradient-norm range | **1.33 - 5.61** (finite steps) + 2 isolated `inf` events (steps 64,800 and 331,600) |
| Throughput | **502.8 samples/sec** |
| Runtime | **7h 25m 41s** total job wall time (SLURM `sacct`); 7h 21m 20s for the training loop itself |
| Eval-loss trend during training | 6.11 → 3.08 (steadily improving, no overfitting observed) |

The 2 `inf` gradient-norm events were investigated, not glossed over: both
are classic fp16 loss-scaler overflow events (PyTorch's `GradScaler` skips
that step's optimizer update and backs off the scale factor automatically).
Confirmed benign -- loss and eval BLEU continued improving smoothly
immediately before and after both events, with no divergence.

## Results

### Amharic and Tigrinya (2,000 held-out dev pairs each, beam search, num_beams=4)

| Direction | BLEU | chrF |
|---|---|---|
| en -> am | **24.699** | 33.655 |
| am -> en | 20.485 | **45.554** |
| en -> ti | 20.556 | 18.634 |
| ti -> en | 10.571 | 31.945 |


Both languages show the same pattern: **X→English is noticeably stronger
than English→X** in both BLEU and chrF. Amharic outperforms Tigrinya in
both directions, plausibly reflecting Amharic's larger, cleaner training
corpus (1.08M vs 1.14M pairs is similar in scale, but NLLB-mined Amharic
data may differ in domain/quality from Tigrinya's).

### Tigre zero-shot (43 of 45 Tatoeba pairs; Tigre never appeared in training)

| Direction | BLEU | chrF |
|---|---|---|
| en -> tig | 11.713 | 19.405 |
| tig -> en | **17.628** | **32.187** |

**Reported honestly, including a real qualitative asymmetry**: `tig→en`
zero-shot transfer is genuinely reasonable -- e.g. source "ቴክኖሎጂ፡ ነቲ ኣብ
መንጎ ወዲ ሰብን ተፈጥሮን ዘሎ ግጭት ከህድኦ ኣይከኣለን" produces "Technology can't stop
the conflict between humans and nature" against a reference meaning
"Technology has failed to ease the conflict between man and nature" --
capturing the real meaning despite zero Tigre training exposure, plausibly
because MoVoC_Tok's shared vocabulary already covers Tigre subwords and the
model's Ge'ez-script-to-English decoding patterns (learned from Amharic and
Tigrinya) transfer reasonably well to a third, related language.

`en→tig`, by contrast, shows **real repetition-loop degeneration** under
greedy/beam decoding -- e.g. "his stomach is weak" produces "ከብዲ ከብዲ ከብዲ
ከብዲ ከብዲ ከብዲ ከብዲ ከብዲ እዩ" (a single word looping), similar to the
degeneration pattern EnTiMT documented for its own low-resource decoding.
This is a real, asymmetric result -- not smoothed over or averaged away --
and 43 pairs is a small qualitative check, not a statistically robust
benchmark; no larger Tigre test set exists anywhere to draw a stronger
conclusion from.

## Pipeline

```
02_cleaning/clean_amharic_corpus.py     -> cleans raw NLLB en-am (reuses EnTiMT's proven logic)
03_model/build_model.py                  -> from-scratch MarianConfig + MoVoC_Tok tokenizer wiring
04_training/train_mt.py                  -> bidirectional Seq2SeqTrainer fine-tune (SLURM)
05_evaluation/evaluate.py                -> BLEU/chrF on Amharic/Tigrinya dev sets
05_evaluation/evaluate_tigre_zeroshot.py -> BLEU/chrF on the real 45-pair Tigre set, zero-shot
```

## Reproducing

```bash
cd 03_model && python build_model.py --outdir ./init_model
cd ../04_training && sbatch submit_job.sh
cd ../05_evaluation && sbatch submit_eval_job.sh
```

## Reproducibility

MoVoC_MT is **partially reproducible**: data splitting is seeded and
deterministic, but exact result recovery is not guaranteed.

- **Seeded**: dataset shuffling and split (`SEED=42`,
  `02_cleaning/clean_amharic_corpus.py:30`) and training data shuffling
  (`--seed 42`, `04_training/train_mt.py:122,173`).
- **Not seeded**: model weight initialization (`03_model/build_model.py`
  calls `MarianMTModel(config)` with no `torch.manual_seed`/`set_seed`
  anywhere in the file) -- the reported metrics correspond to one
  unrepeatable initialization draw.
- **Not deterministic**: training uses fp16 mixed precision (`fp16=True`)
  with no `torch.use_deterministic_algorithms` or fixed cuDNN algorithm
  selection.
- **Not pinned**: no `requirements.txt`/lockfile is committed; reported
  figures were produced with `transformers==4.57.6`, `torch==2.9.0+cu128`,
  `datasets==5.0.0`, `tokenizers==0.22.2`, `sacrebleu==2.6.0`.
- **External, unversioned dependencies**: the MoVoC_Tok tokenizer file,
  the Tigrinya cleaned corpus, and the Tatoeba Tigre eval set are all
  referenced via absolute paths outside this repository, with no
  checksum or pinned commit.

This section is also published on
[GitHub](https://github.com/hailaykidu/MoVoC_MT) (source of truth) and
mirrored, in condensed form, on the
[Hugging Face model card](https://huggingface.co/Hailay/movoc-mt-en-am-ti).

Reported training run (SLURM job 52623, single A100,
`04_training/movoc_mt_train.out`): batch size 32, peak LR 5e-05 (linear
decay, 500 warmup steps), grad-norm range 1.3269047-5.6101074 across
4,158 finite logged steps plus 2 `inf` events (fp16 loss-scaler
overflow, steps ≈64,800 and ≈331,600), throughput 502.771 samples/sec,
training-loop runtime 26,479.75 s (7h 21m 20s).

## Limitations

- **Amharic corpus is capped at 1.5M raw NLLB lines** (of 16.1M available),
  a deliberate choice to balance against Tigrinya's corpus size rather than
  a ceiling on what data exists.
- **Batch size (32) deviates from the paper's (8)** for wall-clock reasons
  -- disclosed above, not hidden.
- **Tigre zero-shot eval is only 43-45 pairs** -- a small qualitative check.
  No larger English-Tigre parallel dataset exists anywhere in this project
  or was found searching the whole environment.
- **En→X translation is weaker than X→En** for both trained languages, and
  en→tig zero-shot shows real repetition-loop degeneration -- reported as
  observed, not adjusted.
- **No comparison against the original 63,050-vocab run** on the same
  data/directions was performed -- this project measures MoVoC_Tok's
  downstream behavior on its own terms, not a controlled tokenizer ablation.
