import re
import os
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


class Vocabulary:
    # maps words <-> integer ids
    # special tokens: PAD=0, UNK=1, CLS=2, SEP=3
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
        # lowercase, strip punctuation, split on whitespace
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return []
        text = str(text).lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def build(self, sentences):
        # count all words, then add those that appear at least min_freq times
        counter = Counter()
        for sent in sentences:
            counter.update(self.tokenize(sent))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence: str):
        # convert sentence -> list of token ids, unknown words -> UNK id
        tokens = self.tokenize(sentence)
        if not tokens:
            return [self.word2idx[self.UNK_TOKEN]]
        return [self.word2idx.get(tok, self.word2idx[self.UNK_TOKEN]) for tok in tokens]

    def __len__(self):
        return len(self.word2idx)


#  helpers 

def _make_ids(sentence: str, vocab: Vocabulary, max_len: int) -> list:
    # encode a sentence and wrap with [CLS] . .  [SEP], truncated to max_len
    cls_id = vocab.word2idx[vocab.CLS_TOKEN]
    sep_id = vocab.word2idx[vocab.SEP_TOKEN]
    body   = vocab.encode(sentence)[: max_len - 2]
    return [cls_id] + body + [sep_id]


def _pad_and_mask(*seq_lists):
    # pad each group of sequences and compute their attention masks
    padded = [pad_sequence(list(seqs), batch_first=True, padding_value=0) for seqs in seq_lists]
    masks  = [(p != 0).long() for p in padded]
    return padded, masks


#  datasets

class PairDataset(Dataset):
    # Loads (anchor, positive) pairs from any CSV with "anchor" and "positive" columns.

    def __init__(self, path: str, vocab: Vocabulary, max_len: int = 128):
        encode = lambda s: _make_ids(str(s), vocab, max_len)
        df = pd.read_csv(path)
        self.samples = [
            (encode(a), encode(p))
            for a, p in zip(df["anchor"], df["positive"])
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return tuple(torch.tensor(ids, dtype=torch.long) for ids in self.samples[idx])


class STSDataset(Dataset):
    # Sentence pairs with similarity scores for evaluation.
    # Normalizes scores from [0,5] or [0,1] -> [-1,1] to match cosine similarity range.

    def __init__(self, path: str, vocab: Vocabulary, max_len: int = 128):
        encode = lambda s: _make_ids(str(s) if pd.notna(s) else "", vocab, max_len)
        df = pd.read_csv(path)
        self.pairs = []
        for _, row in df.iterrows():
            ids_a = encode(row["sentence1"])
            ids_b = encode(row["sentence2"])
            score = float(row["score"]) if pd.notna(row["score"]) else 0.0
            if score > 1.0:
                score = score / 5.0       # [0, 5] -> [0, 1]
            score = score * 2.0 - 1.0    # [0, 1] -> [-1, 1]
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


#  collate functions 

def collate_pair(batch):
    # AllNLIDataset (pair) -> (ids_a, ids_b, mask_a, mask_b)
    seqs_a, seqs_b = zip(*batch)
    padded, masks = _pad_and_mask(seqs_a, seqs_b)
    return (*padded, *masks)


def collate_sts(batch):
    # STSDataset -> (ids_a, ids_b, mask_a, mask_b, scores)
    seqs_a, seqs_b, scores = zip(*batch)
    padded, masks = _pad_and_mask(seqs_a, seqs_b)
    return (*padded, *masks, torch.stack(scores))


#  DataLoader factories 

def build_vocab(path: str, min_freq: int = 2) -> Vocabulary:
    # build vocabulary from all sentences (anchor + positive + negative) in the CSV
    vocab = Vocabulary(min_freq=min_freq)
    df = pd.read_csv(path)
    # include negatives so their tokens stay in vocab even though we don't train on them
    sentences = df["anchor"].tolist() + df["positive"].tolist() + df["negative"].tolist()
    vocab.build(sentences)
    return vocab


def get_pair_loader(
    path: str,
    vocab: Vocabulary,
    batch_size: int = 64,
    max_len: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    # DataLoader for MNR training - (anchor, positive) pairs
    return DataLoader(
        PairDataset(path, vocab, max_len),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_pair,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_sts_loaders(
    paths: dict,
    vocab: Vocabulary,
    batch_size: int = 64,
    max_len: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict:
    # build one DataLoader per split (val, test) from STS benchmark CSV files
    loaders = {}
    for split, path in paths.items():
        if os.path.exists(path):
            loaders[split] = DataLoader(
                STSDataset(path, vocab, max_len),
                batch_size=batch_size, shuffle=False,
                collate_fn=collate_sts,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        else:
            print(f"[WARNING] STS file not found: {path}")
    return loaders
