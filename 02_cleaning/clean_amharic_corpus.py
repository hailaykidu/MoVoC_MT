"""
clean_amharic_corpus.py

Cleans the raw NLLB English-Amharic parallel corpus using the same
normalization/filtering/dedup logic already proven in
../../EnTiMT/02_cleaning/clean_corpus.py (reused verbatim, not
reimplemented, since it already fixed a real str.splitlines() alignment
bug and correctly handles Ethiopic-script consistency filtering).

USAGE
    python clean_amharic_corpus.py \
        --en-file /path/to/NLLB.am-en.en \
        --am-file /path/to/NLLB.am-en.am \
        --outdir ./clean
"""

import argparse
import json
import random
import re
import unicodedata
import zlib
from pathlib import Path

MIN_LINE_CHARS = 2
MAX_LINE_CHARS = 2000
MIN_LEN_RATIO = 0.25
MAX_LEN_RATIO = 4.0
DEV_HOLDOUT = 2000
SEED = 42

ETHIOPIC_RANGES = [
    (0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F),
]
SCRIPT_CONSISTENCY_MIN_RATIO = 0.3

CONTROL_CHAR_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A)) + "]"
)
WHITESPACE_RE = re.compile(r"\s+")
VERSE_NUMBER_RE = re.compile(r"^\d+(:\d+)?\s+")
ALT_PHRASING_RE = re.compile(r"\[([^\[\]/]*)//([^\[\]/]*)\]")


def normalize_line(line: str, strip_mined_artifacts: bool = True) -> str:
    line = unicodedata.normalize("NFC", line)
    line = CONTROL_CHAR_RE.sub("", line)
    if strip_mined_artifacts:
        line = ALT_PHRASING_RE.sub(lambda m: m.group(1).strip(), line)
        line = VERSE_NUMBER_RE.sub("", line)
    line = WHITESPACE_RE.sub(" ", line).strip()
    return line


def ethiopic_ratio(line: str) -> float:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return 1.0
    n_ethiopic = sum(
        1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ETHIOPIC_RANGES)
    )
    return n_ethiopic / len(letters)


_MERSENNE_PRIME = (1 << 61) - 1
_PERM_COEFFS = [(1103515245, 12345), (214013, 2531011), (69069, 1234567),
                (1664525, 1013904223), (22695477, 1), (1103527590, 24691)]


def word_shingles(text: str, k: int = 3):
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash_signature(shingle_set):
    if not shingle_set:
        return None
    base_hashes = [zlib.crc32(s.encode("utf-8")) for s in shingle_set]
    return tuple(
        min((a * h + b) % _MERSENNE_PRIME for h in base_hashes)
        for a, b in _PERM_COEFFS
    )


def load_pairs(en_path: Path, am_path: Path):
    # plain "\n"-splitting, not str.splitlines() -- see EnTiMT/02_cleaning
    # clean_corpus.py for why splitlines() silently desyncs en/am line counts
    en_lines = en_path.read_text(encoding="utf-8").split("\n")
    am_lines = am_path.read_text(encoding="utf-8").split("\n")
    n = min(len(en_lines), len(am_lines))
    pairs = []
    for en, am in zip(en_lines[:n], am_lines[:n]):
        en = normalize_line(en)
        am = normalize_line(am)
        if not (MIN_LINE_CHARS <= len(en) <= MAX_LINE_CHARS):
            continue
        if not (MIN_LINE_CHARS <= len(am) <= MAX_LINE_CHARS):
            continue
        ratio = len(en) / len(am)
        if not (MIN_LEN_RATIO <= ratio <= MAX_LEN_RATIO):
            continue
        if ethiopic_ratio(am) < SCRIPT_CONSISTENCY_MIN_RATIO:
            continue
        pairs.append((en, am))
    return pairs, n


def dedup(pairs):
    seen_exact = set()
    exact_deduped = []
    for en, am in pairs:
        key = (en, am)
        if key in seen_exact:
            continue
        seen_exact.add(key)
        exact_deduped.append((en, am))

    seen_sigs = set()
    near_deduped = []
    for en, am in exact_deduped:
        sig = minhash_signature(word_shingles(en + " ||| " + am))
        if sig is None or sig not in seen_sigs:
            if sig is not None:
                seen_sigs.add(sig)
            near_deduped.append((en, am))
    return exact_deduped, near_deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en-file", required=True)
    parser.add_argument("--am-file", required=True)
    parser.add_argument("--outdir", default="./clean")
    parser.add_argument("--max-raw-lines", type=int, default=None,
                         help="optional cap on raw lines read, for a faster dev run")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("--- loading + line-level filtering ---")
    en_path, am_path = Path(args.en_file), Path(args.am_file)
    if args.max_raw_lines:
        # read-and-cap without loading the full 16M-line file twice
        en_lines = en_path.read_text(encoding="utf-8").split("\n")[:args.max_raw_lines]
        am_lines = am_path.read_text(encoding="utf-8").split("\n")[:args.max_raw_lines]
        tmp_en, tmp_am = outdir / "_tmp.en", outdir / "_tmp.am"
        tmp_en.write_text("\n".join(en_lines), encoding="utf-8")
        tmp_am.write_text("\n".join(am_lines), encoding="utf-8")
        pairs, raw_n = load_pairs(tmp_en, tmp_am)
        tmp_en.unlink()
        tmp_am.unlink()
    else:
        pairs, raw_n = load_pairs(en_path, am_path)
    print(f"  {raw_n} raw lines read, {len(pairs)} pairs survived line-level filters")

    print("--- dedup ---")
    exact_deduped, near_deduped = dedup(pairs)
    print(f"  after exact dedup: {len(exact_deduped)}")
    print(f"  after near-dup (MinHash) dedup: {len(near_deduped)}")

    random.seed(SEED)
    random.shuffle(near_deduped)
    dev = near_deduped[:DEV_HOLDOUT]
    train = near_deduped[DEV_HOLDOUT:]

    for split_name, split_data in [("train", train), ("dev", dev)]:
        with open(outdir / f"{split_name}.en", "w", encoding="utf-8") as fen, \
             open(outdir / f"{split_name}.am", "w", encoding="utf-8") as fam:
            for en, am in split_data:
                fen.write(en + "\n")
                fam.write(am + "\n")

    report = {
        "raw_lines": raw_n,
        "after_line_filters": len(pairs),
        "after_exact_dedup": len(exact_deduped),
        "after_near_dup_dedup": len(near_deduped),
        "train": len(train),
        "dev": len(dev),
    }
    with open(outdir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\ntrain: {len(train)} pairs, dev: {len(dev)} pairs")
    print(f"Cleaning report written to {outdir / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
