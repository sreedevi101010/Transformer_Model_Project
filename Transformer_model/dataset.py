import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq):
        super().__init__()
        self.seq = seq
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.piece_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.piece_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.piece_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair = self.ds[idx]

        src_text = pair[self.src_lang]
        tgt_text = pair[self.tgt_lang]

        # SentencePiece token ids
        enc_input_tokens = self.tokenizer_src.encode(src_text, out_type=int)
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text, out_type=int)

        # Truncate to fit sequence length
        # encoder_input = [SOS] + src + [EOS]
        enc_input_tokens = enc_input_tokens[: self.seq - 2]

        # decoder_input = [SOS] + tgt
        # label         = tgt + [EOS]
        dec_input_tokens = dec_input_tokens[: self.seq - 1]

        enc_num_padding_tokens = self.seq - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq, f"Encoder size mismatch: {encoder_input.size(0)} vs {self.seq}"
        assert decoder_input.size(0) == self.seq, f"Decoder size mismatch: {decoder_input.size(0)} vs {self.seq}"
        assert label.size(0) == self.seq, f"Label size mismatch: {label.size(0)} vs {self.seq}"

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": ((decoder_input != self.pad_token.item()).unsqueeze(0).int() & causal_mask(decoder_input.size(0))),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
# ###
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# class TranslationDataset(Dataset):
#     def __init__(self, ds, ds1, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq):
#         super().__init__()
#         self.seq = seq
#         self.ds = ds
#         self.ds1 = ds1
#         self.tokenizer_src = tokenizer_src
#         self.tokenizer_tgt = tokenizer_tgt
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang

#         self.sos_token = torch.tensor([tokenizer_tgt.piece_to_id("[SOS]")], dtype=torch.int64)
#         self.eos_token = torch.tensor([tokenizer_tgt.piece_to_id("[EOS]")], dtype=torch.int64)
#         self.pad_token = torch.tensor([tokenizer_tgt.piece_to_id("[PAD]")], dtype=torch.int64)

#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, idx):
#         src_target_pair = self.ds[idx]
#         src_target_pair1 = self.ds1[idx]
#         src_text = src_target_pair[self.src_lang]
#         tgt_text = src_target_pair1[self.tgt_lang]

#         # Tokenize input and target texts
#         enc_input_tokens = self.tokenizer_src.encode(src_text)
#         dec_input_tokens = self.tokenizer_tgt.encode(tgt_text)

#         # Ensure EOS is always present
#         if not dec_input_tokens or dec_input_tokens[-1] != self.eos_token.item():
#             dec_input_tokens.append(self.eos_token.item())

#         # Ensure encoder input has EOS too
#         if not enc_input_tokens or enc_input_tokens[-1] != self.eos_token.item():
#             enc_input_tokens.append(self.eos_token.item())

#         # Truncate to max sequence length (keep space for <s> and </s>)
#         enc_input_tokens = enc_input_tokens[:self.seq - 2]
#         dec_input_tokens = dec_input_tokens[:self.seq - 2]  # Space for <s> and EOS

#         # Compute padding lengths (Ensure non-negative)
#         enc_num_padding_tokens = max(0, self.seq - len(enc_input_tokens) - 2)
#         dec_num_padding_tokens = max(0, self.seq - len(dec_input_tokens) - 2)

#         # Construct encoder input with <s> and </s>
#         encoder_input = torch.cat([
#             self.sos_token,
#             torch.tensor(enc_input_tokens, dtype=torch.int64),
#             self.eos_token,
#             torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64),
#         ], dim=0)

#         # Construct decoder input with <s> and <EOS>
#         decoder_input = torch.cat([
#             self.sos_token,
#             torch.tensor(dec_input_tokens, dtype=torch.int64),
#             self.eos_token,
#             torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64),
#         ], dim=0)

#         # Construct labels (shifted decoder output)
#         label = torch.cat([
#             torch.tensor(dec_input_tokens, dtype=torch.int64),
#             self.eos_token,
#             torch.tensor([self.pad_token.item()] * max(0, self.seq - len(dec_input_tokens) - 1), dtype=torch.int64),
#         ], dim=0)

#         # Ensure tensors are of correct length
#         assert encoder_input.size(0) == self.seq, f"Encoder size mismatch: {encoder_input.size(0)} vs {self.seq}"
#         assert decoder_input.size(0) == self.seq, f"Decoder size mismatch: {decoder_input.size(0)} vs {self.seq}"
#         assert label.size(0) == self.seq, f"Label size mismatch: {label.size(0)} vs {self.seq}"

#         return {
#             "encoder_input": encoder_input,  # (seq)
#             "decoder_input": decoder_input,  # (seq)
#             "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq)
#             "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq) & (1, seq, seq)
#             "label": label,  # (seq)
#             "src_text": src_text,
#             "tgt_text": tgt_text,
#         }

# def causal_mask(size):
#     mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
#     return mask == 0
# ence too long")
