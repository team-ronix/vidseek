import numpy as np
from collections import Counter
try:
    from .tokenizer import tokenize
except ImportError:
    from tokenizer import tokenize


class BM25Encoder:
    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 min_df: int = 1, max_df_ratio: float = 0.95):
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocabulary: dict = {}
        self._idf: np.ndarray = np.array([])
        self._avg_dl: float = 0.0

    def fit(self, corpus: list) -> "BM25Encoder":
        tokenized = [tokenize(doc) for doc in corpus]
        N = len(tokenized)

        self._avg_dl = sum(len(t) for t in tokenized) / N if N > 0 else 1.0

        df: Counter = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1

        self.vocabulary = {
            term: idx
            for idx, term in enumerate(sorted(
                t for t, f in df.items()
                if f >= self.min_df and f / N <= self.max_df_ratio
            ))
        }
        V = len(self.vocabulary)

        self._idf = np.zeros(V)
        for term, idx in self.vocabulary.items():
            d = df[term]
            self._idf[idx] = np.log((N - d + 0.5) / (d + 0.5) + 1.0)

        print(f"[BM25Encoder] fit complete | vocab={V} | avgdl={self._avg_dl:.1f} tokens")
        return self

    def encode(self, sentence: str) -> np.ndarray:
        tokens = tokenize(sentence)
        if not tokens:
            return np.zeros(len(self.vocabulary))

        tf_counts = Counter(tokens)
        sent_len = len(tokens)
        vec = np.zeros(len(self.vocabulary))

        for term, tf in tf_counts.items():
            if term not in self.vocabulary:
                continue
            idx = self.vocabulary[term]
            denom = tf + self.k1 * (1.0 - self.b + self.b * sent_len / self._avg_dl)
            vec[idx] = self._idf[idx] * tf * (self.k1 + 1.0) / denom

        return vec

    def encode_as_dict(self, sentence: str) -> dict:
        vec = self.encode(sentence)
        return {int(i): float(v) for i, v in enumerate(vec) if v > 0.0}
    def save(self, path: str) -> None:
        np.savez(
            path,
            vocab_terms=np.array(list(self.vocabulary.keys())),
            vocab_indices=np.array(list(self.vocabulary.values())),
            idf=self._idf,
            avg_dl=np.array([self._avg_dl]),
            k1=np.array([self.k1]),
            b=np.array([self.b]),
        )

    @classmethod
    def load(cls, path: str) -> "BM25Encoder":
        data = np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=True)
        enc = cls(
            k1=float(data["k1"][0]),
            b=float(data["b"][0]),
        )
        terms = [k.decode() if isinstance(k, bytes) else str(k) for k in data["vocab_terms"].tolist()]
        enc.vocabulary = dict(zip(terms, [int(v) for v in data["vocab_indices"].tolist()]))
        enc._idf = data["idf"]
        enc._avg_dl = float(data["avg_dl"][0])
        return enc
