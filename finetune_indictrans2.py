import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from IndicTransToolkit.processor import IndicProcessor


# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME       = "ai4bharat/indictrans2-indic-indic-dist-320M"
DATA_PATH        = "dataset_train.csv"
OUTPUT_DIR_ML2HI = "./indictrans2-mal-hindi"
OUTPUT_DIR_HI2ML = "./indictrans2-hindi-mal"

SRC_LANG_ML = "mal_Mlym"
TGT_LANG_HI = "hin_Deva"
SRC_LANG_HI = "hin_Deva"
TGT_LANG_ML = "mal_Mlym"

MAX_INPUT_LEN  = 256
MAX_TARGET_LEN = 256
BATCH_SIZE     = 8
GRAD_ACCUM     = 4
EPOCHS         = 3
LR             = 5e-5
WARMUP_STEPS   = 200
LOGGING_STEPS  = 50
SAVE_STEPS     = 500
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16           = torch.cuda.is_available()


# ── Dataset ───────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, ip, src_lang, tgt_lang,
                 src_col, tgt_col):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.ip        = ip
        self.src_lang  = src_lang
        self.tgt_lang  = tgt_lang
        self.src_col   = src_col
        self.tgt_col   = tgt_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = str(self.data.loc[idx, self.src_col])
        tgt_text = str(self.data.loc[idx, self.tgt_col])

        src_processed = self.ip.preprocess_batch(
            [src_text], src_lang=self.src_lang, tgt_lang=self.tgt_lang
        )[0]

        tgt_processed = self.ip.preprocess_batch(
            [tgt_text], src_lang=self.tgt_lang, tgt_lang=self.src_lang,
            is_target=True
        )[0]

        return {"src": src_processed, "tgt": tgt_processed}


def collate_fn(batch, tokenizer):
    src_texts = [b["src"] for b in batch]
    tgt_texts = [b["tgt"] for b in batch]

    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    with tokenizer.as_target_tokenizer() if hasattr(tokenizer, "as_target_tokenizer") else _noop():
        labels = tokenizer(
            text_target=tgt_texts,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    label_ids = labels["input_ids"]
    # Replace padding token id with -100 so loss ignores them
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = label_ids
    return model_inputs


class _noop:
    """No-op context manager for tokenizers that don't have as_target_tokenizer."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ── Training loop ─────────────────────────────────────────────────────────────

def finetune(src_lang, tgt_lang, src_col, tgt_col, output_dir, df):
    print(f"\n{'='*60}")
    print(f"  Finetuning: {src_lang}  →  {tgt_lang}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(DEVICE)

    ip = IndicProcessor(inference=False)

    dataset = TranslationDataset(
        dataframe=df,
        tokenizer=tokenizer,
        ip=ip,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_col=src_col,
        tgt_col=tgt_col,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=FP16)

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=FP16):
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if step % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOGGING_STEPS == 0:
                    avg = epoch_loss / step
                    print(f"  Epoch {epoch} | Step {global_step} | Loss {avg:.4f}")

                if global_step % SAVE_STEPS == 0:
                    ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"  Saved checkpoint → {ckpt}")

        print(f"Epoch {epoch} done | Avg loss: {epoch_loss/len(loader):.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nFinal model saved to: {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    assert "malayalam" in df.columns and "hindi" in df.columns, \
        "CSV must contain 'malayalam' and 'hindi' columns."
    df = df.dropna(subset=["malayalam", "hindi"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    finetune(
        src_lang=SRC_LANG_ML, tgt_lang=TGT_LANG_HI,
        src_col="malayalam",  tgt_col="hindi",
        output_dir=OUTPUT_DIR_ML2HI, df=df,
    )

    finetune(
        src_lang=SRC_LANG_HI, tgt_lang=TGT_LANG_ML,
        src_col="hindi",      tgt_col="malayalam",
        output_dir=OUTPUT_DIR_HI2ML, df=df,
    )

    print("\nAll done!")
