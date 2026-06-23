"""Semantic search evaluation and hard negative mining.

Metrics
───────
Recall@K      - fraction of queries where the true positive is in the top-K results
MRR@K         - Mean Reciprocal Rank: mean(1 / rank_of_first_hit)
Spearman ρ    - rank correlation between cosine similarity and gold STS scores

Hard negative mining
────────────────────
After initial training with random negatives, call mine_hard_negatives() to
find sentences that the *current* model incorrectly ranks as similar.
Re-train with these harder negatives for significantly better embeddings.

Usage
─────
python evaluate_search.py
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pad_sequence

from data import Vocabulary
from models.model.transformer import Transformer

# ── encoding helper ───────────────────────────────────────────────────────────

def encode_sentences(
    model: Transformer,
    sentences: list[str],
    vocab: Vocabulary,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a list of raw sentences into L2-normalized embeddings.

    Returns a (N, d_model) tensor of unit vectors.
    """
    model.eval()
    cls_id = vocab.word2idx[vocab.CLS_TOKEN]
    sep_id = vocab.word2idx[vocab.SEP_TOKEN]
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            seqs  = []
            for s in chunk:
                body = vocab.encode(s)[: max_len - 2]
                seqs.append(torch.tensor([cls_id] + body + [sep_id], dtype=torch.long))

            padded = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)
            mask   = (padded != 0).long()
            emb    = model.encode(padded, mask, normalize=True)   # (chunk, d)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)   # (N, d)


# ── semantic search metrics ───────────────────────────────────────────────────

def recall_at_k(
    query_embs:   torch.Tensor,   # (Q, d)
    corpus_embs:  torch.Tensor,   # (C, d)
    query_ids:    list[int],      # index of each query sentence in the corpus
    relevant_ids: list[int],      # index of the true positive for each query
    k: int = 10,
) -> float:
    """Recall@K: fraction of queries whose true positive appears in top-K results.

    The query sentence itself is excluded from its own result set so a trivial
    self-match never inflates the score.
    """
    sim = torch.matmul(query_embs, corpus_embs.T)   # (Q, C)

    # Mask self-similarity so the query can't retrieve itself
    for i, qid in enumerate(query_ids):
        sim[i, qid] = -1e9

    topk = sim.topk(k, dim=-1).indices   # (Q, K)

    hits = sum(
        1 for i, rel in enumerate(relevant_ids)
        if rel in topk[i].tolist()
    )
    return hits / len(query_ids)


def mean_reciprocal_rank(
    query_embs:   torch.Tensor,
    corpus_embs:  torch.Tensor,
    query_ids:    list[int],
    relevant_ids: list[int],
) -> float:
    """MRR: mean(1 / rank_of_first_relevant_document).

    Rank 1 means the first result is the true positive (perfect retrieval).
    """
    sim = torch.matmul(query_embs, corpus_embs.T)

    for i, qid in enumerate(query_ids):
        sim[i, qid] = -1e9

    sorted_idx = sim.argsort(dim=-1, descending=True)   # (Q, C)

    mrr = 0.0
    for i, rel in enumerate(relevant_ids):
        rank = (sorted_idx[i] == rel).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1.0 / rank

    return mrr / len(query_ids)


# ── hard negative mining ──────────────────────────────────────────────────────

def mine_hard_negatives(
    model: Transformer,
    df: pd.DataFrame,
    vocab: Vocabulary,
    max_len: int,
    device: torch.device,
    pos_threshold: float = 0.65,
    top_k: int = 5,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Find hard negatives using the current model's embedding space.

    Algorithm
    ─────────
    1. Encode all unique sentences in df.
    2. For each positive pair (anchor a, positive p):
       a. Compute cosine similarity between a and every corpus sentence.
       b. Exclude a itself and all known positives of a.
       c. Take the top_k most similar remaining sentences → hard negatives.
    3. Return a DataFrame with columns [anchor, positive, hard_negative].

    Why hard negatives improve training
    ────────────────────────────────────
    Random negatives are easy - the model quickly separates them and gets no
    gradient. Hard negatives are sentences the model *currently* thinks are
    similar to the anchor but shouldn't be. Training on them forces the model
    to make finer-grained distinctions, which directly improves retrieval
    precision at the top of the ranked list (Recall@1, MRR).

    Usage with TripletLoss
    ───────────────────────
    hard_df = mine_hard_negatives(model, train_df, vocab, ...)
    # hard_df has columns: anchor, positive, hard_negative
    # Feed to TripletDataset or add to your SentencePairDataset as extra negatives.
    """
    # Collect all unique sentences
    all_sentences = list({
        s for col in ("sentence1", "sentence2")
        for s in df[col].fillna("").tolist()
    })
    sent_to_idx = {s: i for i, s in enumerate(all_sentences)}

    print(f"Mining hard negatives - encoding {len(all_sentences)} sentences ...")
    corpus_embs = encode_sentences(model, all_sentences, vocab, max_len, batch_size, device)

    # Build anchor → known positives map
    anchor_to_pos: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        score = float(row["score"]) if pd.notna(row["score"]) else 0.0
        if score > 1.0:
            score /= 5.0
        s1 = str(row["sentence1"]) if pd.notna(row["sentence1"]) else ""
        s2 = str(row["sentence2"]) if pd.notna(row["sentence2"]) else ""
        if score >= pos_threshold:
            anchor_to_pos.setdefault(s1, set()).add(s2)
            anchor_to_pos.setdefault(s2, set()).add(s1)

    triplets = []
    for anchor, pos_set in tqdm(anchor_to_pos.items(), desc="Mining"):
        if anchor not in sent_to_idx:
            continue

        a_idx  = sent_to_idx[anchor]
        a_emb  = corpus_embs[a_idx].unsqueeze(0)             # (1, d)
        sims   = torch.matmul(a_emb, corpus_embs.T).squeeze(0)  # (C,)

        # Mask anchor and all known positives
        sims[a_idx] = -1e9
        for pos in pos_set:
            if pos in sent_to_idx:
                sims[sent_to_idx[pos]] = -1e9

        hard_neg_idxs = sims.topk(top_k).indices.tolist()
        hard_negs     = [all_sentences[i] for i in hard_neg_idxs]

        for pos in pos_set:
            for hn in hard_negs:
                triplets.append({"anchor": anchor, "positive": pos, "hard_negative": hn})

    result = pd.DataFrame(triplets)
    print(f"Mined {len(result):,} hard-negative triplets")
    return result


# ── full evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    val_path: str,
    vocab: Vocabulary,
    config: dict,
    device: torch.device,
):
    """Run full semantic search evaluation on the validation set."""
    df = pd.read_csv(val_path)

    # Build corpus from all unique sentences in the file
    all_sentences = list({
        s for col in ("sentence1", "sentence2")
        for s in df[col].fillna("").tolist()
    })
    sent_to_idx = {s: i for i, s in enumerate(all_sentences)}

    # Load model
    model = Transformer(
        vocab_size=len(vocab),
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
        dropout=0.0,
        pooling=config["pooling"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"Encoding {len(all_sentences)} corpus sentences ...")
    corpus_embs = encode_sentences(
        model, all_sentences, vocab, config["max_len"], batch_size=64, device=device
    )

    # Positive pairs only as ground truth for retrieval
    pos_df = df.copy()
    pos_df["score_01"] = pos_df["score"].apply(
        lambda s: float(s) / 5.0 if float(s) > 1.0 else float(s)
    )
    pos_df = pos_df[pos_df["score_01"] >= config["pos_threshold"]].reset_index(drop=True)

    query_ids    = [sent_to_idx[s] for s in pos_df["sentence1"].tolist()]
    relevant_ids = [sent_to_idx[s] for s in pos_df["sentence2"].tolist()]
    query_embs   = corpus_embs[query_ids]

    r1  = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=1)
    r5  = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=5)
    r10 = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=10)
    mrr = mean_reciprocal_rank(query_embs, corpus_embs, query_ids, relevant_ids)

    # Spearman on all pairs (not just positives)
    s1_list  = df["sentence1"].fillna("").tolist()
    s2_list  = df["sentence2"].fillna("").tolist()
    scores   = df["score"].apply(lambda s: float(s) / 5.0 if float(s) > 1.0 else float(s)).tolist()

    embs_1 = encode_sentences(model, s1_list, vocab, config["max_len"], 64, device)
    embs_2 = encode_sentences(model, s2_list, vocab, config["max_len"], 64, device)
    cos_sims = (embs_1 * embs_2).sum(-1).numpy()

    rho, _ = spearmanr(cos_sims, scores)

    print(f"\nSemantic Search Evaluation on {val_path}")
    print(f"  Recall@1   : {r1:.4f}   ({r1*100:.1f}%)")
    print(f"  Recall@5   : {r5:.4f}   ({r5*100:.1f}%)")
    print(f"  Recall@10  : {r10:.4f}   ({r10*100:.1f}%)")
    print(f"  MRR        : {mrr:.4f}")
    print(f"  Spearman ρ : {rho:.4f}")

    return {"recall@1": r1, "recall@5": r5, "recall@10": r10, "mrr": mrr, "spearman": rho}


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import pickle
    import pandas as pd

    CONFIG = {
        "train_path":    "data/stsb_train.csv",
        "val_path":      "data/stsb_validation.csv",
        "pos_threshold": 0.8,
        "max_len":       128,
        "d_model":       256,
        "n_layers":      4,
        "n_heads":       4,
        "d_ff":          512,
        "pooling":       "mean",
        "save_path":     "best_model.pt",
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load vocabulary saved by train.py
    with open("vocab.pkl", "rb") as _vf:
        vocab = pickle.load(_vf)

    # 2. Load the trained model
    from models.model.transformer import Transformer
    model = Transformer(
        vocab_size=len(vocab),
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        d_ff=CONFIG["d_ff"],
        max_len=CONFIG["max_len"],
        pooling=CONFIG["pooling"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(CONFIG["save_path"], map_location=DEVICE))

    # 3. Load training data to mine hard negatives from
    train_df = pd.read_csv(CONFIG["train_path"])

    # 4. Run hard negative mining - finds triplets the model currently gets wrong
    hard_triplets_df = mine_hard_negatives(
        model=model,
        df=train_df,
        vocab=vocab,
        max_len=CONFIG["max_len"],
        device=DEVICE,
        pos_threshold=CONFIG["pos_threshold"],
        top_k=5,
    )

    # 5. Save the hard negatives for train_triplet.py
    os.makedirs("data", exist_ok=True)
    hard_triplets_df.to_csv("data/train_hard_negatives.csv", index=False)
    print(f"Saved {len(hard_triplets_df):,} hard-negative triplets to data/train_hard_negatives.csv")
