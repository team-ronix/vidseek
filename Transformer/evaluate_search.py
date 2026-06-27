# evaluation helpers: Recall@K, MRR, and Spearman correlation

import torch
import pandas as pd
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pad_sequence

from data import Vocabulary


def _normalize_scores(scores: pd.Series) -> pd.Series:
    # STS scores can be [0,5] or [0,1] — normalize to [-1,1] to match cosine similarity
    scores_01 = scores.apply(lambda s: float(s) / 5.0 if float(s) > 1.0 else float(s))
    return scores_01 * 2.0 - 1.0


def encode_sentences(model, sentences, vocab, max_len, chunk_size, device):
    # encode sentences in chunks to avoid running out of memory on large datasets
    model.eval()
    cls_id = vocab.word2idx[vocab.CLS_TOKEN]
    sep_id = vocab.word2idx[vocab.SEP_TOKEN]
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(sentences), chunk_size):
            chunk = sentences[start : start + chunk_size]
            seqs = []
            for s in chunk:
                body = vocab.encode(s)[: max_len - 2]
                seqs.append(torch.tensor([cls_id] + body + [sep_id], dtype=torch.long))
            padded = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)
            mask = (padded != 0).long()
            embeddings.append(model.encode(padded, mask, normalize=True).cpu())

    return torch.cat(embeddings, dim=0)


def _sim_chunks(query_embs, corpus_embs, query_ids, relevant_ids, chunk):
    # yield (sim_matrix, relevant_chunk) per chunk of queries
    # mask out self-similarity so a sentence can't retrieve itself
    for start in range(0, len(query_ids), chunk):
        end = min(start + chunk, len(query_ids))
        sim = torch.matmul(query_embs[start:end], corpus_embs.T)
        for j, qid in enumerate(query_ids[start:end]):
            sim[j, qid] = -1e9
        yield sim, relevant_ids[start:end]


def recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=10, chunk=64):
    # check if the relevant sentence appears in the top-k results for each query
    hits = 0
    for sim, rel_chunk in _sim_chunks(query_embs, corpus_embs, query_ids, relevant_ids, chunk):
        topk = sim.topk(k, dim=-1).indices
        hits += sum(1 for j, rel in enumerate(rel_chunk) if rel in topk[j].tolist())
    return hits / len(query_ids)


def mean_reciprocal_rank(query_embs, corpus_embs, query_ids, relevant_ids, chunk=64):
    # average 1/rank across all queries — MRR=1.0 means the answer is always ranked first
    mrr = 0.0
    for sim, rel_chunk in _sim_chunks(query_embs, corpus_embs, query_ids, relevant_ids, chunk):
        sorted_idx = sim.argsort(dim=-1, descending=True)
        for j, rel in enumerate(rel_chunk):
            rank = (sorted_idx[j] == rel).nonzero(as_tuple=True)[0].item() + 1
            mrr += 1.0 / rank
    return mrr / len(query_ids)


def evaluate(model, csv_path, vocab, max_len, device, pos_threshold, batch_size=64):
    df = pd.read_csv(csv_path)

    # sort for deterministic index assignment across runs
    all_sentences = sorted({
        s for col in ("sentence1", "sentence2")
        for s in df[col].fillna("").tolist()
    })
    sent_to_idx = {s: i for i, s in enumerate(all_sentences)}

    print(f"    Encoding {len(all_sentences)} sentences ...")
    corpus_embs = encode_sentences(model, all_sentences, vocab, max_len, batch_size, device)

    # normalize scores to [-1, 1] so they match the cosine similarity range
    scores_11 = _normalize_scores(df["score"])

    # only keep pairs similar enough to count as "positive" for retrieval evaluation
    pos_mask = scores_11 >= pos_threshold
    pos_df   = df[pos_mask & df["sentence1"].isin(sent_to_idx) & df["sentence2"].isin(sent_to_idx)]

    query_ids    = [sent_to_idx[s] for s in pos_df["sentence1"].tolist()]
    relevant_ids = [sent_to_idx[s] for s in pos_df["sentence2"].tolist()]
    query_embs   = corpus_embs[query_ids]

    r1  = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=1)
    r5  = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=5)
    r10 = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=10)
    mrr = mean_reciprocal_rank(query_embs, corpus_embs, query_ids, relevant_ids)

    # reuse corpus_embs to get Spearman — no need to re-encode all sentences
    idx1     = [sent_to_idx[s] for s in df["sentence1"].fillna("").tolist()]
    idx2     = [sent_to_idx[s] for s in df["sentence2"].fillna("").tolist()]
    embs_1   = corpus_embs[idx1]
    embs_2   = corpus_embs[idx2]
    cos_sims = (embs_1 * embs_2).sum(-1).numpy()
    del embs_1, embs_2
    rho, _ = spearmanr(cos_sims, scores_11.tolist())
    rho = float(rho) if rho == rho else 0.0  # rho == rho is False for NaN

    return {
        "recall@1":  round(r1, 4),
        "recall@5":  round(r5, 4),
        "recall@10": round(r10, 4),
        "mrr":       round(mrr, 4),
        "spearman":  round(rho, 4),
    }
