"""
IndicTrans2 FLORES-200 Translation Testing
Hindi <-> Malayalam using ai4bharat/indictrans2-indic-indic-dist-320M model

Requirements:
    pip install torch transformers==4.46.0 sentencepiece sacremoses IndicTransToolkit datasets

FLORES dataset files needed (place in same directory or update FLORES_DIR):
    - hin_Deva.devtest
    - mal_Mlym.devtest

Download FLORES-200 devtest from:
    https://github.com/facebookresearch/flores/tree/main/flores200
    or via HuggingFace: load_dataset("facebook/flores", "mal_Mlym")
"""

import os
import csv
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# ─────────────────────────── Configuration ──────────────────────────────────

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"   # ← changed

# Language tags used by IndicTrans2
HIN_TAG = "hin_Deva"
MAL_TAG = "mal_Mlym"

# Path to the directory containing the .devtest files
FLORES_DIR = Path(__file__).parent  # default: same folder as this script

# Output directory for CSV files
OUTPUT_DIR = Path(__file__).parent / "output"

# Batch size for inference (reduce if OOM)
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────── Helpers ────────────────────────────────────────

def load_flores_file(path: Path) -> list[str]:
    """Load a FLORES devtest file (one sentence per line, UTF-8)."""
    if not path.exists():
        raise FileNotFoundError(
            f"FLORES file not found: {path}\n"
            "Download from https://github.com/facebookresearch/flores/tree/main/flores200 "
            "or via `load_dataset('facebook/flores', 'mal_Mlym')`"
        )
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    print(f"  Loaded {len(lines)} sentences from {path.name}")
    return lines


def translate(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model,
    tokenizer,
    ip: IndicProcessor,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    """Translate a list of sentences using IndicTrans2."""
    translations = []
    total = len(sentences)

    for start in range(0, total, batch_size):
        batch = sentences[start : start + batch_size]
        end = min(start + batch_size, total)
        print(f"    Translating sentences {start + 1}–{end} / {total} ...", end="\r")

        # Pre-process
        batch_inputs = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize
        inputs = tokenizer(
            batch_inputs,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode
        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Post-process
        processed = ip.postprocess_batch(decoded, lang=tgt_lang)
        translations.extend(processed)

    print()  # newline after progress
    return translations


def save_csv(
    output_path: Path,
    input_texts: list[str],
    translations: list[str],
    references: list[str],
):
    """Save results to a CSV with columns: input, translation, reference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["input", "translation", "reference"])
        for inp, trans, ref in zip(input_texts, translations, references):
            writer.writerow([inp, trans, ref])
    print(f"  Saved → {output_path}  ({len(input_texts)} rows)")


# ─────────────────────────── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IndicTrans2 FLORES-200 evaluation (Hindi ↔ Malayalam)")
    parser.add_argument("--flores-dir", type=Path, default=FLORES_DIR,
                        help="Directory containing hin_Deva.devtest and mal_Mlym.devtest")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory to write output CSV files")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Limit sentences for quick testing (default: all 1012)")
    args = parser.parse_args()

    print("=" * 65)
    print(" IndicTrans2 Distilled 320M  |  FLORES-200  |  Hindi ↔ Malayalam")
    print("=" * 65)
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Device : {DEVICE}")
    print(f"  Batch  : {args.batch_size}")
    print()

    # ── Load FLORES files ──────────────────────────────────────────────────
    print("▶ Loading FLORES devtest files …")
    hin_file = args.flores_dir / "hin_Deva.devtest"
    mal_file = args.flores_dir / "mal_Mlym.devtest"

    hin_sentences = load_flores_file(hin_file)
    mal_sentences = load_flores_file(mal_file)

    if len(hin_sentences) != len(mal_sentences):
        raise ValueError(
            f"Sentence count mismatch: Hindi={len(hin_sentences)}, Malayalam={len(mal_sentences)}"
        )

    if args.max_sentences:
        hin_sentences = hin_sentences[: args.max_sentences]
        mal_sentences = mal_sentences[: args.max_sentences]
        print(f"  (Limited to first {args.max_sentences} sentences)")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n▶ Loading model: {MODEL_NAME} …")
    print("  (This will download ~700 MB on first run)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    print("  Model loaded successfully.")

    ip = IndicProcessor(inference=True)

    # ── Direction 1: Hindi → Malayalam ────────────────────────────────────
    print(f"\n▶ Direction 1: Hindi ({HIN_TAG}) → Malayalam ({MAL_TAG})")
    hin_to_mal = translate(
        hin_sentences, HIN_TAG, MAL_TAG,
        model, tokenizer, ip, args.batch_size
    )
    save_csv(
        args.output_dir / "hin_to_mal.csv",
        input_texts=hin_sentences,
        translations=hin_to_mal,
        references=mal_sentences,
    )

    # ── Direction 2: Malayalam → Hindi ────────────────────────────────────
    print(f"\n▶ Direction 2: Malayalam ({MAL_TAG}) → Hindi ({HIN_TAG})")
    mal_to_hin = translate(
        mal_sentences, MAL_TAG, HIN_TAG,
        model, tokenizer, ip, args.batch_size
    )
    save_csv(
        args.output_dir / "mal_to_hin.csv",
        input_texts=mal_sentences,
        translations=mal_to_hin,
        references=hin_sentences,
    )

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  All done! Output files:")
    for name in ["hin_to_mal.csv", "mal_to_hin.csv"]:
        p = args.output_dir / name
        print(f"    {p}")
    print("=" * 65)


if __name__ == "__main__":
    main()
