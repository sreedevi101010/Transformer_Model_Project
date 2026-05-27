import os

# ---------------- DISABLE GPU ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch
import numpy as np
import evaluate
import gc

# ---------------- DEVICE ----------------
BERT_DEVICE = "cpu"
print(f"Using device: {BERT_DEVICE}")

# ---------------- LOAD METRICS ----------------
print("\nLoading evaluation metrics...")

sacrebleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
bertscore_metric = evaluate.load("bertscore")
comet_metric = evaluate.load("comet")
meteor_metric = evaluate.load("meteor")

# ---------------- BLEURT ----------------
bleurt_metric = evaluate.load(
    "bleurt",
    module_type="metric",
    checkpoint="BLEURT-20-D12"
)

print("All metrics loaded successfully!")

# ---------------- FUNCTION ----------------
def evaluate_translation_from_single_file(
        input_csv_path,
        output_csv_path,
        predictions_column="translated_text",
        source_column="text",
        references_column="Reference_text",
        src_lang_for_bert="hi",
        tgt_lang_for_bert="ml"
):

    print(f"\nStarting Evaluation: {input_csv_path}")

    # ---------------- CHECK FILE ----------------

    if not os.path.exists(input_csv_path):
        print(f"File not found: {input_csv_path}")
        return

    # ---------------- READ CSV ----------------

    try:

        df = pd.read_csv(input_csv_path, encoding='utf-8')

        # FIX BOM ISSUE
        df.columns = df.columns.str.replace('\ufeff', '')

    except Exception as e:

        print(f"CSV Read Error: {e}")
        return

    # ---------------- CHECK COLUMNS ----------------

    required_cols = [
        predictions_column,
        source_column,
        references_column
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:

        print(f"Missing columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return

    # ---------------- DATA ----------------

    predictions = df[predictions_column].astype(str).tolist()
    sources = df[source_column].astype(str).tolist()
    references_raw = df[references_column].astype(str).tolist()

    total = len(predictions)

    print(f"\nCalculating metrics for {total} samples...")

    # ---------------- SCORE LISTS ----------------

    bleu_scores = []
    chrf_scores = []
    bert_f1_scores = []
    comet_scores = []
    meteor_scores = []
    bleurt_scores = []

    # ---------------- LOOP ----------------

    for i in range(total):

        prediction = predictions[i]
        source = sources[i]
        reference = references_raw[i]

        # ---------------- EMPTY REFERENCE ----------------

        if not reference.strip():

            print(f"Skipping row {i} (empty reference)")

            bleu_scores.append(np.nan)
            chrf_scores.append(np.nan)
            bert_f1_scores.append(np.nan)
            comet_scores.append(np.nan)
            meteor_scores.append(np.nan)
            bleurt_scores.append(np.nan)

            continue

        # ---------------- BLEU ----------------

        try:

            bleu_result = sacrebleu_metric.compute(
                predictions=[prediction],
                references=[[reference]]
            )

            bleu_scores.append(bleu_result["score"])

        except Exception as e:

            print(f"BLEU Error at row {i}: {e}")
            bleu_scores.append(np.nan)

        # ---------------- chrF ----------------

        try:

            chrf_result = chrf_metric.compute(
                predictions=[prediction],
                references=[[reference]]
            )

            chrf_scores.append(chrf_result["score"])

        except Exception as e:

            print(f"chrF Error at row {i}: {e}")
            chrf_scores.append(np.nan)

        # ---------------- BERTScore ----------------

        try:

            bert_result = bertscore_metric.compute(
                predictions=[prediction],
                references=[reference],
                lang=tgt_lang_for_bert,
                device=BERT_DEVICE
            )

            bert_f1_scores.append(bert_result["f1"][0])

        except Exception as e:

            print(f"BERTScore Error at row {i}: {e}")
            bert_f1_scores.append(np.nan)

        # ---------------- COMET ----------------

        try:

            comet_result = comet_metric.compute(
                predictions=[prediction],
                references=[reference],
                sources=[source],
                gpus=0
            )

            comet_scores.append(comet_result["scores"][0])

        except Exception as e:

            print(f"COMET Error at row {i}: {e}")
            comet_scores.append(np.nan)

        # ---------------- METEOR ----------------

        try:

            meteor_result = meteor_metric.compute(
                predictions=[prediction],
                references=[reference]
            )

            meteor_scores.append(meteor_result["meteor"])

        except Exception as e:

            print(f"METEOR Error at row {i}: {e}")
            meteor_scores.append(np.nan)

        # ---------------- BLEURT ----------------

        try:

            bleurt_result = bleurt_metric.compute(
                predictions=[prediction],
                references=[reference]
            )

            bleurt_scores.append(bleurt_result["scores"][0])

        except Exception as e:

            print(f"BLEURT Error at row {i}: {e}")
            bleurt_scores.append(np.nan)

        # ---------------- PROGRESS ----------------

        if i % 10 == 0:

            print(f"Processed {i}/{total}")

    # ---------------- SAVE RESULTS ----------------

    df["BLEU"] = bleu_scores
    df["chrF"] = chrf_scores
    df["BERTScore_F1"] = bert_f1_scores
    df["COMET"] = comet_scores
    df["METEOR"] = meteor_scores
    df["BLEURT"] = bleurt_scores

    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"\nSaved results to: {output_csv_path}")

    # ---------------- FINAL SCORES ----------------

    print("\nFinal Aggregate Scores")
    print("----------------------")

    print(f"BLEU:      {np.nanmean(bleu_scores):.4f}")
    print(f"chrF:      {np.nanmean(chrf_scores):.4f}")
    print(f"BERTScore: {np.nanmean(bert_f1_scores):.4f}")
    print(f"COMET:     {np.nanmean(comet_scores):.4f}")
    print(f"METEOR:    {np.nanmean(meteor_scores):.4f}")
    print(f"BLEURT:    {np.nanmean(bleurt_scores):.4f}")

    # ---------------- CLEANUP ----------------

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# ---------------- MAIN ----------------

if __name__ == "__main__":

    input_path = "hin_to_mal.csv"
    output_path = "hin_to_mal_eval_results.csv"

    evaluate_translation_from_single_file(
        input_csv_path=input_path,
        output_csv_path=output_path,
        predictions_column="translated_text",
        source_column="text",
        references_column="Reference_text",
        src_lang_for_bert="hi",
        tgt_lang_for_bert="ml"
    )
