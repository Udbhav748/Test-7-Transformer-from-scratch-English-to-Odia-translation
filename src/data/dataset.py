from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from configs.base import PAD_ID
from src.tokenization.train_tokenizer import EN_TOKENIZER_PATH, OR_TOKENIZER_PATH
from src.tokenization.tokenizer_utils import encode, load_tokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        split_path: Path,
        en_tokenizer_path: Path = EN_TOKENIZER_PATH,
        or_tokenizer_path: Path = OR_TOKENIZER_PATH,
    ):
        df = pd.read_parquet(split_path)
        en_tok = load_tokenizer(en_tokenizer_path)
        or_tok = load_tokenizer(or_tokenizer_path)

        self.src_ids = [torch.tensor(encode(en_tok, s), dtype=torch.long) for s in df["src"]]
        self.tgt_ids = [torch.tensor(encode(or_tok, t), dtype=torch.long) for t in df["tgt"]]

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int):
        return self.src_ids[idx], self.tgt_ids[idx]


def _pad_batch(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(seq.size(0) for seq in sequences)
    padded = torch.full((len(sequences), max_len), PAD_ID, dtype=torch.long)
    # pad_mask: True at PAD positions, matching nn.MultiheadAttention's
    # key_padding_mask convention where True positions are ignored
    pad_mask = torch.ones((len(sequences), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        padded[i, :seq.size(0)] = seq
        pad_mask[i, :seq.size(0)] = False
    return padded, pad_mask


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    src_seqs, tgt_seqs = zip(*batch)
    src_ids, src_pad_mask = _pad_batch(list(src_seqs))
    tgt_ids, tgt_pad_mask = _pad_batch(list(tgt_seqs))
    return src_ids, src_pad_mask, tgt_ids, tgt_pad_mask
