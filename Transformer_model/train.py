import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import sentencepiece as spm

from config import get_config
from dataset import TranslationDataset
from model import build_transformer


def ensure_artifact_dirs(config):
    base_dir = Path(config.get("artifact_dir", "artifacts"))
    tokenizer_dir = base_dir / "tokenizers"
    corpus_dir = base_dir / "corpus"
    weights_dir = base_dir / "weights"
    final_dir = base_dir / "final"

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "tokenizer_dir": tokenizer_dir,
        "corpus_dir": corpus_dir,
        "weights_dir": weights_dir,
        "final_dir": final_dir,
    }


def get_weights_file_path(config, epoch: str):
    dirs = ensure_artifact_dirs(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(dirs["weights_dir"] / model_filename)


def latest_weights_file_path(config):
    dirs = ensure_artifact_dirs(config)
    weights_files = list(dirs["weights_dir"].glob(f"{config['model_basename']}*.pt"))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def get_final_model_path(config):
    dirs = ensure_artifact_dirs(config)
    return str(dirs["final_dir"] / "final_model.pt")


def get_all_sentences(ds, lang):
    for item in ds:
        text = item[lang]
        if isinstance(text, str) and text.strip():
            yield text.strip()


def get_or_build_tokenizer(config, ds, lang):
    dirs = ensure_artifact_dirs(config)
    model_path = dirs["tokenizer_dir"] / f"tokenizer_{lang}.model"
    vocab_path = dirs["tokenizer_dir"] / f"tokenizer_{lang}.vocab"
    corpus_file = dirs["corpus_dir"] / f"{lang}_corpus.txt"

    if not model_path.exists():
        print(f"Training tokenizer for {lang}...")

        with open(corpus_file, "w", encoding="utf-8") as f:
            for sentence in get_all_sentences(ds, lang):
                f.write(sentence + "\n")

        spm.SentencePieceTrainer.train(
            input=str(corpus_file),
            model_prefix=str((dirs["tokenizer_dir"] / f"tokenizer_{lang}")),
            vocab_size=config.get("vocab_size", 8000),
            model_type="unigram",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            bos_piece="[SOS]",
            eos_piece="[EOS]",
            character_coverage=1.0,
        )

        print(f"Saved tokenizer model: {model_path}")
        print(f"Saved tokenizer vocab : {vocab_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def get_ds(config):
    df = pd.read_csv(config["train_csv"])
    df = df.dropna(subset=[config["lang_src"], config["lang_tgt"]])

    df[config["lang_src"]] = df[config["lang_src"]].astype(str)
    df[config["lang_tgt"]] = df[config["lang_tgt"]].astype(str)

    ds_raw = Dataset.from_pandas(df)

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw = ds_raw.select(range(train_ds_size))
    val_ds_raw = ds_raw.select(range(train_ds_size, train_ds_size + val_ds_size))

    train_ds = TranslationDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq"],
    )

    val_ds = TranslationDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq"],
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq"],
        config["seq"],
        d_model=config["d_model"],
    )
    return model


def run_validation(model, val_dataloader, device, max_print=5):
    model.eval()
    print("\nValidation samples:")

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_print:
                break

            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]

            print(f"SRC: {src_text}")
            print(f"TGT: {tgt_text}")
            print("-" * 80)


def save_checkpoint(model, optimizer, epoch, global_step, config, epoch_tag):
    model_filename = get_weights_file_path(config, epoch_tag)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }

    try:
        torch.save(checkpoint, model_filename)
        print(f"Saved checkpoint: {model_filename}")
    except Exception as e:
        print(f"Warning: failed to save checkpoint at {model_filename}")
        print(f"Reason: {e}")


def save_final_model(model, tokenizer_src, tokenizer_tgt, config):
    final_model_path = get_final_model_path(config)
    final_payload = {
        "model_state_dict": model.state_dict(),
        "src_lang": config["lang_src"],
        "tgt_lang": config["lang_tgt"],
        "seq": config["seq"],
        "d_model": config["d_model"],
        "src_vocab_size": tokenizer_src.get_piece_size(),
        "tgt_vocab_size": tokenizer_tgt.get_piece_size(),
    }

    try:
        torch.save(final_payload, final_model_path)
        print(f"Final model saved at: {final_model_path}")
    except Exception as e:
        print(f"Warning: failed to save final model")
        print(f"Reason: {e}")


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dirs = ensure_artifact_dirs(config)
    print(f"Artifacts directory: {dirs['base_dir'].resolve()}")

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(
        config,
        tokenizer_src.get_piece_size(),
        tokenizer_tgt.get_piece_size(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    pad_idx = tokenizer_tgt.piece_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

    initial_epoch = 0
    global_step = 0

    if config.get("preload") is not None:
        model_filename = (
            latest_weights_file_path(config)
            if config["preload"] == "latest"
            else get_weights_file_path(config, config["preload"])
        )

        if model_filename is not None and Path(model_filename).exists():
            print(f"Loading model: {model_filename}")
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            initial_epoch = state["epoch"] + 1
            global_step = state["global_step"]

    for epoch in range(initial_epoch, config["num_epochs"]):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output,
                encoder_mask,
                decoder_input,
                decoder_mask,
            )
            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, proj_output.size(-1)),
                label.view(-1),
            )

            batch_iterator.set_postfix({"loss": f"{loss.item():6.4f}"})

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1

        save_checkpoint(model, optimizer, epoch, global_step, config, f"{epoch:02d}")
        save_checkpoint(model, optimizer, epoch, global_step, config, "latest")

        run_validation(model, val_dataloader, device)

    save_final_model(model, tokenizer_src, tokenizer_tgt, config)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()

    # sensible defaults if missing in config.py
    config.setdefault("artifact_dir", "artifacts")
    config.setdefault("model_basename", "tmodel_")
    config.setdefault("vocab_size", 8000)

    train_model(config)
