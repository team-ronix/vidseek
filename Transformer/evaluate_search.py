# evaluation helpers: Recall@K, MRR, and Spearman correlation

import torch
import pandas as pd
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pad_sequence

from data import Vocabulary
from utils import normalize_scores


def encode_sentences(model, sentences, vocab, max_len, batch_size, device):
    model.eval()
    cls_id = vocab.word2idx[vocab.CLS_TOKEN]
    sep_id = vocab.word2idx[vocab.SEP_TOKEN]
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            seqs = []
            for s in chunk:
                body = vocab.encode(s)[: max_len - 2]
                seqs.append(torch.tensor([cls_id] + body + [sep_id], dtype=torch.long))
            padded = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)
            mask = (padded != 0).long()
            embeddings.append(model.encode(padded, mask, normalize=True).cpu())

    return torch.cat(embeddings, dim=0)


def recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=10, chunk=64):
    # process in chunks to avoid OOM on large corpora
    hits = 0
    for start in range(0, len(query_ids), chunk):
        end = min(start + chunk, len(query_ids))
        sim = torch.matmul(query_embs[start:end], corpus_embs.T)
        for j, qid in enumerate(query_ids[start:end]):
            sim[j, qid] = -1e9  # mask out the query itself
        topk = sim.topk(k, dim=-1).indices
        hits += sum(
            1 for j, rel in enumerate(relevant_ids[start:end])
            if rel in topk[j].tolist()
        )
    return hits / len(query_ids)


def mean_reciprocal_rank(query_embs, corpus_embs, query_ids, relevant_ids, chunk=64):
    mrr = 0.0
    for start in range(0, len(query_ids), chunk):
        end = min(start + chunk, len(query_ids))
        sim = torch.matmul(query_embs[start:end], corpus_embs.T)
        for j, qid in enumerate(query_ids[start:end]):
            sim[j, qid] = -1e9
        sorted_idx = sim.argsort(dim=-1, descending=True)
        for j, rel in enumerate(relevant_ids[start:end]):
            rank = (sorted_idx[j] == rel).nonzero(as_tuple=True)[0].item() + 1
            mrr += 1.0 / rank
    return mrr / len(query_ids)


def evaluate(model, csv_path, vocab, max_len, device, pos_threshold=0.3, batch_size=64):
    df = pd.read_csv(csv_path)

    # collect all unique sentences from both columns
    all_sentences = list({
        s for col in ("sentence1", "sentence2")
        for s in df[col].fillna("").tolist()
    })
    sent_to_idx = {s: i for i, s in enumerate(all_sentences)}

    print(f"    Encoding {len(all_sentences)} sentences ...")
    corpus_embs = encode_sentences(model, all_sentences, vocab, max_len, batch_size, device)

    scores_11 = normalize_scores(df["score"])

    # only keep pairs that are "positive" (similar enough) for retrieval metrics
    pos_mask = scores_11 >= pos_threshold
    pos_df = df[pos_mask & df["sentence1"].isin(sent_to_idx) & df["sentence2"].isin(sent_to_idx)]

    query_ids = [sent_to_idx[s] for s in pos_df["sentence1"].tolist()]
    relevant_ids = [sent_to_idx[s] for s in pos_df["sentence2"].tolist()]
    query_embs = corpus_embs[query_ids]

    r1  = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=1)
    r5  = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=5)
    r10 = recall_at_k(query_embs, corpus_embs, query_ids, relevant_ids, k=10)
    mrr = mean_reciprocal_rank(query_embs, corpus_embs, query_ids, relevant_ids)

    # reuse corpus_embs instead of re-encoding to save memory
    idx1 = [sent_to_idx[s] for s in df["sentence1"].fillna("").tolist()]
    idx2 = [sent_to_idx[s] for s in df["sentence2"].fillna("").tolist()]
    embs_1 = corpus_embs[idx1]
    embs_2 = corpus_embs[idx2]
    cos_sims = (embs_1 * embs_2).sum(-1).numpy()
    del embs_1, embs_2
    rho, _ = spearmanr(cos_sims, scores_11.tolist())
    rho = float(rho) if rho == rho else 0.0

    return {
        "recall@1":  round(r1, 4),
        "recall@5":  round(r5, 4),
        "recall@10": round(r10, 4),
        "mrr":       round(mrr, 4),
        "spearman":  round(rho, 4),
    }
