import sys
import time
import numpy as np
from collections import Counter
from .tokenizer import tokenize


class LSAEncoder:
    def __init__(self, n_components: int = 128,
                 min_df: int = 1, max_df_ratio: float = 0.95,
                 n_iter: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.n_iter = n_iter
        self.random_state = random_state
        self.vocabulary: dict = {}
        self._idf: np.ndarray = np.array([])
        self._U_k: np.ndarray = np.array([])
        self._S_k: np.ndarray = np.array([])
        self._k: int = 0

    def fit_transform(self, corpus: list) -> np.ndarray:
        t0 = time.time()
        tokenized = [tokenize(doc) for doc in corpus]
        N = len(tokenized)

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

        self._idf = np.array([
            np.log((1 + N) / (1 + df[term])) + 1.0
            for term in sorted(self.vocabulary, key=self.vocabulary.get)
        ])

        # Build sparse COO TF-IDF — stores only non-zero entries to avoid OOM
        row_list, col_list, val_list = [], [], []
        for j, tokens in enumerate(tokenized):
            if not tokens:
                continue
            tf_counts = Counter(tokens)
            total = len(tokens)
            for term, count in tf_counts.items():
                if term in self.vocabulary:
                    i = self.vocabulary[term]
                    row_list.append(i)
                    col_list.append(j)
                    val_list.append((count / total) * self._idf[i])

        rows = np.array(row_list, dtype=np.int32)
        cols = np.array(col_list, dtype=np.int32)
        vals = np.array(val_list, dtype=np.float64)
        nnz = len(vals)

        print(f"  Sparse TF-IDF: shape=({V}, {N})  nnz={nnz:,}  density={nnz/(V*N)*100:.3f}%")

        self._k = min(self.n_components, V - 1, N - 1)
        print(f"  Running randomized SVD  k={self._k}  n_iter={self.n_iter} ...")
        self._U_k, self._S_k, Vt_k = self._randomized_svd(rows, cols, vals, V, N, self._k)

        print(f"[LSAEncoder] fit done in {time.time()-t0:.1f}s | vocab={V} | k={self._k}")
        return Vt_k.T  # (N, k)

    @staticmethod
    def _spmm(rows, cols, vals, X, out_rows):
        p = X.shape[1]
        result = np.zeros((out_rows, p))
        for l in range(p):
            result[:, l] = np.bincount(rows, weights=vals * X[cols, l], minlength=out_rows)
        return result

    @staticmethod
    def _spmm_t(rows, cols, vals, X, out_rows):
        p = X.shape[1]
        result = np.zeros((out_rows, p))
        for l in range(p):
            result[:, l] = np.bincount(cols, weights=vals * X[rows, l], minlength=out_rows)
        return result

    def _randomized_svd(self, rows, cols, vals, V, N, k):
        p = 10
        kp = k + p
        rng = np.random.default_rng(self.random_state)

        Omega = rng.standard_normal((N, kp))
        Y = self._spmm(rows, cols, vals, Omega, V)

        for it in range(self.n_iter):
            Q, _ = np.linalg.qr(Y)
            Z = self._spmm_t(rows, cols, vals, Q, N)
            Q2, _ = np.linalg.qr(Z)
            Y = self._spmm(rows, cols, vals, Q2, V)
            sys.stdout.write(f"\r    power iter {it+1}/{self.n_iter} ...")
            sys.stdout.flush()
        print()

        Q, _ = np.linalg.qr(Y)
        B = self._spmm_t(rows, cols, vals, Q, N).T
        U_B, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_B

        return U[:, :k], S[:k], Vt[:k, :]

    def fit(self, corpus: list) -> "LSAEncoder":
        self.fit_transform(corpus)
        return self

    def encode(self, sentence: str) -> np.ndarray:
        q_tfidf = self._tfidf_vector(sentence)
        safe_S = np.where(self._S_k > 1e-12, self._S_k, 1e-12)
        dense = (self._U_k.T @ q_tfidf) / safe_S
        norm = np.linalg.norm(dense)
        return dense / norm if norm > 1e-12 else dense

    def _tfidf_vector(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(len(self.vocabulary))
        tf_counts = Counter(tokens)
        total = len(tokens)
        vec = np.zeros(len(self.vocabulary))
        for term, count in tf_counts.items():
            if term in self.vocabulary:
                i = self.vocabulary[term]
                vec[i] = (count / total) * self._idf[i]
        return vec

    def save(self, path: str) -> None:
        np.savez(
            path,
            vocab_terms=np.array(list(self.vocabulary.keys())),
            vocab_indices=np.array(list(self.vocabulary.values())),
            idf=self._idf,
            U_k=self._U_k,
            S_k=self._S_k,
            n_components=np.array([self.n_components]),
        )
        print(f"[LSAEncoder] saved to {path}.npz")

    @classmethod
    def load(cls, path: str) -> "LSAEncoder":
        data = np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=True)
        enc = cls(n_components=int(data["n_components"][0]))
        terms = [k.decode() if isinstance(k, bytes) else str(k) for k in data["vocab_terms"].tolist()]
        enc.vocabulary = dict(zip(terms, [int(v) for v in data["vocab_indices"].tolist()]))
        enc._idf = data["idf"]
        enc._U_k = data["U_k"]
        enc._S_k = data["S_k"]
        enc._k = len(data["S_k"])
        print(f"[LSAEncoder] loaded from {path}")
        return enc

    def __repr__(self) -> str:
        return f"LSAEncoder(n_components={self.n_components}, vocab_size={len(self.vocabulary)}, k={self._k})"
