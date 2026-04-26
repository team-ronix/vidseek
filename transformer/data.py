import re
import os
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    @staticmethod
    def tokenize(text):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return []

        text = str(text).lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def build(self, sentences):
        counter = Counter()

        for sent in sentences:
            tokens = self.tokenize(sent)
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence: str):
        tokens = self.tokenize(sentence)
        if not tokens:
            return [self.word2idx[self.UNK_TOKEN]]
        return [
            self.word2idx.get(tok, self.word2idx[self.UNK_TOKEN])
            for tok in tokens
        ]

    def __len__(self):
        return len(self.word2idx)
class STSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 128):
        self.vocab = vocab
        self.max_len = max_len
        self.pairs = []

        for _, row in df.iterrows():
            s1 = str(row["sentence1"]) if pd.notna(row["sentence1"]) else ""
            s2 = str(row["sentence2"]) if pd.notna(row["sentence2"]) else ""

            ids_a = vocab.encode(s1)[:max_len]
            ids_b = vocab.encode(s2)[:max_len]

            score = float(row["score"]) if pd.notna(row["score"]) else 0.0
            if score > 1.0:
                score = score / 5.0

            self.pairs.append((ids_a, ids_b, score))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ids_a, ids_b, score = self.pairs[idx]
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(ids_b, dtype=torch.long),
            torch.tensor(score, dtype=torch.float),
        )
def collate_fn(batch):
    seqs_a, seqs_b, scores = zip(*batch)

    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_b = pad_sequence(list(seqs_b), batch_first=True, padding_value=0)

    mask_a = (padded_a != 0).long()
    mask_b = (padded_b != 0).long()

    scores = torch.stack(scores)

    return padded_a, padded_b, mask_a, mask_b, scores
def get_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str | None = None,
    batch_size: int = 32,
    max_len: int = 128,
    min_freq: int = 1,
    num_workers: int = 0,
):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    vocab = Vocabulary(min_freq=min_freq)

    all_sents = (
        train_df["sentence1"].fillna("").tolist()
        + train_df["sentence2"].fillna("").tolist()
    )
    vocab.build(all_sents)

    train_ds = STSDataset(train_df, vocab, max_len)
    val_ds = STSDataset(val_df, vocab, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    loaders = {"train": train_loader, "val": val_loader}

    if test_path:
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            test_ds = STSDataset(test_df, vocab, max_len)

            loaders["test"] = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
        else:
            print(f"[WARNING] Test file not found: {test_path}")

    return loaders, vocab