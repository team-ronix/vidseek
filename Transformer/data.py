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


# ── helper ────────────────────────────────────────────────────────────────────

def _make_ids(sentence: str, vocab: Vocabulary, max_len: int):
    cls_id = vocab.word2idx[vocab.CLS_TOKEN]
    sep_id = vocab.word2idx[vocab.SEP_TOKEN]
    body   = vocab.encode(sentence)[: max_len - 2]
    return [cls_id] + body + [sep_id]


#  AllNLI datasets 

class AllNLIDataset(Dataset):
    """(anchor, positive) pairs from AllNLI.jsonl for MNR training.

    Each line: [anchor, positive, negative] — the explicit negative is
    discarded here because MNR uses every other positive in the batch as
    a negative for free, giving B-1 negatives per anchor per step.
    """

    def __init__(self, path: str, vocab: Vocabulary, max_len: int = 128):
        self.pairs = []
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            self.pairs.append((
                _make_ids(str(row["anchor"]),   vocab, max_len),
                _make_ids(str(row["positive"]), vocab, max_len),
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ids_a, ids_p = self.pairs[idx]
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(ids_p, dtype=torch.long),
        )


class AllNLITripletDataset(Dataset):
    """(anchor, positive, negative) triplets from AllNLI.jsonl for Triplet training.

    AllNLI provides explicit NLI-derived negatives (contradictions), which are
    semantically hard by construction — far stronger than randomly sampled ones.
    """

    def __init__(self, path: str, vocab: Vocabulary, max_len: int = 128):
        self.triplets = []
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            self.triplets.append((
                _make_ids(str(row["anchor"]),   vocab, max_len),
                _make_ids(str(row["positive"]), vocab, max_len),
                _make_ids(str(row["negative"]), vocab, max_len),
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


#  AllNLI labeled-pair dataset (for Contrastive / CoSENT / CosineMSE)

class AllNLILabeledPairDataset(Dataset):
    """Creates (sent_a, sent_b, label) pairs from AllNLI.

    Each row contributes two samples:
        (anchor, positive) → label 1.0
        (anchor, negative) → label 0.0

    Used by ContrastiveLoss, CoSENTLoss, and CosineMSELoss which all expect
    a scalar label / score alongside the two sentence embeddings.
    """

    def __init__(self, path: str, vocab: Vocabulary, max_len: int = 128):
        self.pairs = []
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            anchor   = _make_ids(str(row["anchor"]),   vocab, max_len)
            positive = _make_ids(str(row["positive"]), vocab, max_len)
            negative = _make_ids(str(row["negative"]), vocab, max_len)
            self.pairs.append((anchor, positive, 1.0))
            self.pairs.append((anchor, negative, 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ids_a, ids_b, label = self.pairs[idx]
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(ids_b, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


#  STS evaluation dataset

class STSDataset(Dataset):
    """Sentence pairs with similarity scores for Spearman evaluation.

    Handles both [0, 5] and [0, 1] score scales; normalises to [-1, 1]
    to match the cosine similarity range.
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 128):
        self.pairs = []
        for _, row in df.iterrows():
            s1 = str(row["sentence1"]) if pd.notna(row["sentence1"]) else ""
            s2 = str(row["sentence2"]) if pd.notna(row["sentence2"]) else ""

            ids_a = _make_ids(s1, vocab, max_len)
            ids_b = _make_ids(s2, vocab, max_len)

            score = float(row["score"]) if pd.notna(row["score"]) else 0.0
            if score > 1.0:
                score = score / 5.0       # [0, 5] → [0, 1]
            score = score * 2.0 - 1.0    # [0, 1] → [-1, 1]

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

def collate_fn(batch):
    """STSDataset → (ids_a, ids_b, mask_a, mask_b, scores)."""
    seqs_a, seqs_b, scores = zip(*batch)
    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_b = pad_sequence(list(seqs_b), batch_first=True, padding_value=0)
    return (
        padded_a, padded_b,
        (padded_a != 0).long(), (padded_b != 0).long(),
        torch.stack(scores),
    )


def collate_pair_fn(batch):
    """AllNLIDataset → (ids_a, ids_b, mask_a, mask_b)."""
    seqs_a, seqs_b = zip(*batch)
    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_b = pad_sequence(list(seqs_b), batch_first=True, padding_value=0)
    return (
        padded_a, padded_b,
        (padded_a != 0).long(), (padded_b != 0).long(),
    )


def collate_labeled_fn(batch):
    """AllNLILabeledPairDataset → (ids_a, ids_b, mask_a, mask_b, labels)."""
    seqs_a, seqs_b, labels = zip(*batch)
    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_b = pad_sequence(list(seqs_b), batch_first=True, padding_value=0)
    return (
        padded_a, padded_b,
        (padded_a != 0).long(), (padded_b != 0).long(),
        torch.stack(labels),
    )


def collate_triplet_fn(batch):
    """AllNLITripletDataset → (ids_a, ids_p, ids_n, mask_a, mask_p, mask_n)."""
    seqs_a, seqs_p, seqs_n = zip(*batch)
    padded_a = pad_sequence(list(seqs_a), batch_first=True, padding_value=0)
    padded_p = pad_sequence(list(seqs_p), batch_first=True, padding_value=0)
    padded_n = pad_sequence(list(seqs_n), batch_first=True, padding_value=0)
    return (
        padded_a, padded_p, padded_n,
        (padded_a != 0).long(), (padded_p != 0).long(), (padded_n != 0).long(),
    )


#  DataLoader factories 

def build_vocab_from_allnli(path: str, min_freq: int = 2) -> Vocabulary:
    """Build vocabulary from all sentences (anchor + positive + negative) in AllNLI.csv."""
    vocab = Vocabulary(min_freq=min_freq)
    df = pd.read_csv(path)
    sentences = (
        df["anchor"].tolist() + df["positive"].tolist() + df["negative"].tolist()
    )
    vocab.build(sentences)
    return vocab


def get_allnli_pair_loader(
    path: str,
    vocab: Vocabulary,
    batch_size: int = 64,
    max_len: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """DataLoader for MNR training — (anchor, positive) pairs from AllNLI."""
    return DataLoader(
        AllNLIDataset(path, vocab, max_len),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_pair_fn,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_allnli_labeled_pair_loader(
    path: str,
    vocab: Vocabulary,
    batch_size: int = 64,
    max_len: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """DataLoader for Contrastive/CoSENT/CosineMSE — labeled (a, b, label) pairs."""
    return DataLoader(
        AllNLILabeledPairDataset(path, vocab, max_len),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_labeled_fn,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_allnli_triplet_loader(
    path: str,
    vocab: Vocabulary,
    batch_size: int = 64,
    max_len: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """DataLoader for Triplet fine-tuning — (anchor, positive, negative) from AllNLI."""
    return DataLoader(
        AllNLITripletDataset(path, vocab, max_len),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_triplet_fn,
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
    """DataLoaders for STS evaluation.

    paths example:
        {'val':  'data/sts-222/stsb_validation.csv',
         'test': 'data/sts-222/stsb_test.csv'}
    """
    loaders = {}
    for split, path in paths.items():
        if os.path.exists(path):
            loaders[split] = DataLoader(
                STSDataset(pd.read_csv(path), vocab, max_len),
                batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        else:
            print(f"[WARNING] STS file not found: {path}")
    return loaders
