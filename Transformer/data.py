import re
import os
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3,
        }
        self.idx2word = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
            2: self.CLS_TOKEN,
            3: self.SEP_TOKEN,
        }

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
            counter.update(self.tokenize(sent))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence: str):
        tokens = self.tokenize(sentence)
        if not tokens:
            return [self.word2idx[self.UNK_TOKEN]]
        return [self.word2idx.get(tok, self.word2idx[self.UNK_TOKEN]) for tok in tokens]

    def __len__(self):
        return len(self.word2idx)


# ── helper: tokenize a sentence into [CLS] body... [SEP] ids ─────────────────

def _make_ids(sentence: str, vocab: Vocabulary, max_len: int):
    cls_id = vocab.word2idx[vocab.CLS_TOKEN]
    sep_id = vocab.word2idx[vocab.SEP_TOKEN]
    body   = vocab.encode(sentence)[: max_len - 2]
    return [cls_id] + body + [sep_id]


# ── STS regression dataset (original, kept for Spearman evaluation) ───────────

class STSDataset(Dataset):
    """Sentence pairs with normalized similarity scores for STS evaluation."""

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 128):
        self.pairs = []
        for _, row in df.iterrows():
            s1 = str(row["sentence1"]) if pd.notna(row["sentence1"]) else ""
            s2 = str(row["sentence2"]) if pd.notna(row["sentence2"]) else ""

            ids_a = _make_ids(s1, vocab, max_len)
            ids_b = _make_ids(s2, vocab, max_len)

            score = float(row["score"]) if pd.notna(row["score"]) else 0.0
            if score > 1.0:
                score = score / 5.0       # [0,5] → [0,1]
            score = score * 2.0 - 1.0    # [0,1] → [-1,1]  (matches cosine range)

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


# ── Sentence pair dataset — for Multiple Negatives Ranking Loss ───────────────

class SentencePairDataset(Dataset):
    """Positive sentence pairs for MNR Loss training.

    Dataset format conversion
    ─────────────────────────
    Input:  sentence1, sentence2, score   (score in [0,1] or [0,5])
    Output: (ids_a, ids_b) pairs where score >= pos_threshold

    In-batch negatives: during a forward pass, each (a_i, p_i) pair treats
    every other p_j in the same batch as a negative for a_i.  No explicit
    negative labels or triplet construction needed.

    Larger batches → more negatives → harder task → better embeddings.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_len: int   = 128,
        pos_threshold: float = 0.65,   # score in [0,1]; pairs below this are dropped
    ):
        self.pairs = []
        for _, row in df.iterrows():
            score = float(row["score"]) if pd.notna(row["score"]) else 0.0
            if score > 1.0:
                score = score / 5.0   # normalize [0,5] → [0,1]
            if score < pos_threshold:
                continue              # skip non-positive pairs

            s1 = str(row["sentence1"]) if pd.notna(row["sentence1"]) else ""
            s2 = str(row["sentence2"]) if pd.notna(row["sentence2"]) else ""
            self.pairs.append((_make_ids(s1, vocab, max_len),
                               _make_ids(s2, vocab, max_len)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ids_a, ids_b = self.pairs[idx]
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(ids_b, dtype=torch.long),
        )


# ── Triplet dataset — for TripletLoss (alternative to MNR) ───────────────────

class TripletDataset(Dataset):
    """(anchor, positive, negative) triplets built from STS pair data.

    Dataset format conversion
    ─────────────────────────
    Input:  sentence1, sentence2, score
    Output: (anchor, positive, negative) triplets where
            · positive = sentence paired with score >= pos_threshold
            · negative = sentence paired with score <= neg_threshold
              (falls back to random sentence if no explicit negatives exist)

    Hard negative mining
    ────────────────────
    After initial training, replace random negatives with hard negatives by
    calling mine_hard_negatives() from evaluate_search.py.  Hard negatives
    are sentences the model incorrectly scores as similar — they provide the
    strongest gradient signal.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_len: int         = 128,
        pos_threshold: float = 0.0,
        neg_threshold: float = 0.3,
    ):
        all_sentences = list({
            s for col in ("sentence1", "sentence2")
            for s in df[col].fillna("").tolist()
        })

        # Build anchor → positives and anchor → negatives maps
        positives: dict[str, list[str]] = {}
        negatives: dict[str, list[str]] = {}

        for _, row in df.iterrows():
            score = float(row["score"]) if pd.notna(row["score"]) else 0.0
            if score > 1.0:
                score = score / 5.0
            s1 = str(row["sentence1"]) if pd.notna(row["sentence1"]) else ""
            s2 = str(row["sentence2"]) if pd.notna(row["sentence2"]) else ""

            if score >= pos_threshold:
                positives.setdefault(s1, []).append(s2)
                positives.setdefault(s2, []).append(s1)
            if score <= neg_threshold:
                negatives.setdefault(s1, []).append(s2)
                negatives.setdefault(s2, []).append(s1)

        self.triplets = []
        for anchor, pos_list in positives.items():
            neg_pool = negatives.get(
                anchor,
                [s for s in all_sentences if s not in pos_list and s != anchor],
            )
            if not neg_pool:
                continue
            for pos in pos_list:
                neg = random.choice(neg_pool)
                self.triplets.append((
                    _make_ids(anchor, vocab, max_len),
                    _make_ids(pos,    vocab, max_len),
                    _make_ids(neg,    vocab, max_len),
                ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        ids_a, ids_p, ids_n = self.triplets[idx]
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(ids_p, dtype=torch.long),
            torch.tensor(ids_n, dtype=torch.long),
        )


# ── collate functions ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate for STSDataset — returns (ids_a, ids_b, mask_a, mask_b, scores)."""
    seqs_a, seqs_b, scores = zip(*batch)
    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_b = pad_sequence(list(seqs_b), batch_first=True, padding_value=0)
    return (
        padded_a, padded_b,
        (padded_a != 0).long(), (padded_b != 0).long(),
        torch.stack(scores),
    )


def collate_pair_fn(batch):
    """Collate for SentencePairDataset — returns (ids_a, ids_b, mask_a, mask_b)."""
    seqs_a, seqs_b = zip(*batch)
    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_b = pad_sequence(list(seqs_b), batch_first=True, padding_value=0)
    return (
        padded_a, padded_b,
        (padded_a != 0).long(), (padded_b != 0).long(),
    )


def collate_triplet_fn(batch):
    """Collate for TripletDataset — returns (a, p, n, mask_a, mask_p, mask_n)."""
    seqs_a, seqs_p, seqs_n = zip(*batch)
    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_p = pad_sequence(list(seqs_p), batch_first=True, padding_value=0)
    padded_n = pad_sequence(list(seqs_n), batch_first=True, padding_value=0)
    return (
        padded_a, padded_p, padded_n,
        (padded_a != 0).long(), (padded_p != 0).long(), (padded_n != 0).long(),
    )


# ── DataLoader factories ──────────────────────────────────────────────────────

def _build_vocab(train_df: pd.DataFrame, min_freq: int) -> Vocabulary:
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(
        train_df["sentence1"].fillna("").tolist()
        + train_df["sentence2"].fillna("").tolist()
    )
    return vocab


def get_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str | None = None,
    batch_size: int = 32,
    max_len: int = 128,
    min_freq: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    vocab: Vocabulary | None = None,   # pass a pre-built vocab to reuse it
):
    """DataLoaders for STS regression / Spearman evaluation."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df   = pd.read_csv(val_path,   low_memory=False)

    if vocab is None:
        vocab = _build_vocab(train_df, min_freq)

    loaders = {
        "train": DataLoader(STSDataset(train_df, vocab, max_len),
                            batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=pin_memory),
        "val":   DataLoader(STSDataset(val_df, vocab, max_len),
                            batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=pin_memory),
    }

    if test_path:
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path, low_memory=False)
            loaders["test"] = DataLoader(
                STSDataset(test_df, vocab, max_len),
                batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            print(f"[WARNING] Test file not found: {test_path}")

    return loaders, vocab


def get_pair_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str | None = None,
    batch_size: int = 64,
    max_len: int = 128,
    min_freq: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    pos_threshold: float = 0.65,
    sample_size: int | None = None,   # cap training set; None = use all
    vocab: Vocabulary | None = None,
):
    """DataLoaders for MNR Loss embedding training (positive pairs only)."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df   = pd.read_csv(val_path,   low_memory=False)

    if sample_size is not None and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    if vocab is None:
        vocab = _build_vocab(train_df, min_freq)

    loaders = {
        "train": DataLoader(
            SentencePairDataset(train_df, vocab, max_len, pos_threshold),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate_pair_fn, num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            SentencePairDataset(val_df, vocab, max_len, pos_threshold),
            batch_size=batch_size, shuffle=False,
            collate_fn=collate_pair_fn, num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path, low_memory=False)
        loaders["test"] = DataLoader(
            SentencePairDataset(test_df, vocab, max_len, pos_threshold),
            batch_size=batch_size, shuffle=False,
            collate_fn=collate_pair_fn, num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders, vocab


def get_triplet_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str | None = None,
    batch_size: int = 32,
    max_len: int = 128,
    min_freq: int = 1,
    num_workers: int = 0,
    pos_threshold: float = 0.65,
    neg_threshold: float = 0.3,
    vocab: Vocabulary | None = None,
):
    """DataLoaders for TripletLoss training."""
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    if vocab is None:
        vocab = _build_vocab(train_df, min_freq)

    loaders = {
        "train": DataLoader(
            TripletDataset(train_df, vocab, max_len, pos_threshold, neg_threshold),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate_triplet_fn, num_workers=num_workers,
        ),
        "val": DataLoader(
            TripletDataset(val_df, vocab, max_len, pos_threshold, neg_threshold),
            batch_size=batch_size, shuffle=False,
            collate_fn=collate_triplet_fn, num_workers=num_workers,
        ),
    }

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        loaders["test"] = DataLoader(
            TripletDataset(test_df, vocab, max_len, pos_threshold, neg_threshold),
            batch_size=batch_size, shuffle=False,
            collate_fn=collate_triplet_fn, num_workers=num_workers,
        )

    return loaders, vocab
