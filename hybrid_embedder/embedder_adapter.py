import numpy as np
from pathlib import Path

from .main import HybridEmbedder
from Storage.VectorStore import VectorStore, VideoVector

# Pre-trained models live at Hybrid_embedder/models/model_{lsa,bm25}.npz
# Path is resolved relative to this file so it works from any working dir.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PREFIX = str(_PROJECT_ROOT / "Hybrid_embedder" / "models" / "model")


class HybridEmbedderAdapter:
    """
    Thin wrapper around the pre-trained HybridEmbedder.

    The models are fixed (pre-trained vocabulary, fixed LSA rank).
    No corpus fitting needed — call load() then use.

    encode() returns (dense_vector, sparse_vector) matching the
    evaluation-notebook interface:

        dense, sparse = adapter.encode(sentence)
        # dense : (k,)  float32  L2-normalised LSA projection
        # sparse: (V,)  float64  BM25 weighted TF vector
    """

    def __init__(self, transcripts=None):
        self.transcripts: list[dict] = transcripts or []
        self.embeddings:  list[np.ndarray] = []
        self.metadata:    list[dict] = []
        self._embedder:   HybridEmbedder | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        """Load pre-trained LSA + BM25 models. Returns True on success."""
        lsa_path = Path(f"{_MODEL_PREFIX}_lsa.npz")
        if not lsa_path.exists():
            print(f"[HybridEmbedderAdapter] model not found at {lsa_path}")
            return False
        try:
            self._embedder = HybridEmbedder.load(_MODEL_PREFIX)
            return True
        except Exception as exc:
            print(f"[HybridEmbedderAdapter] load failed: {exc}")
            return False

    @property
    def dimension(self) -> int:
        """Actual LSA rank of the loaded model (e.g. 128)."""
        if self._embedder is None:
            raise RuntimeError("Call load() first.")
        return int(self._embedder.lsa._k)

    # ------------------------------------------------------------------ encode
    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (dense_vector, sparse_vector) for one sentence.

        dense  : (k,)  — L2-normalised LSA projection
        sparse : (V,)  — BM25 weighted TF vector (full vocab, float64)
        """
        if self._embedder is None:
            raise RuntimeError("Call load() first.")
        return self._embedder.encode(text)

    def transform_single_text(self, text: str) -> np.ndarray:
        """Dense-only encoding used for HNSW querying."""
        dense, _ = self.encode(text)
        return dense

    # ------------------------------------------------------------------ hybrid score
    def hybrid_score(
        self,
        dense_sim: float,
        q_sparse: np.ndarray,
        doc_text: str,
        alpha: float = 0.9,
    ) -> float:
        """
        Combine HNSW dense cosine with BM25 sparse cosine.

            score = (1 - alpha) * dense_cosine + alpha * sparse_cosine
        """
        _, doc_sparse = self.encode(doc_text)
        norm_q = np.linalg.norm(q_sparse)
        norm_d = np.linalg.norm(doc_sparse)
        if norm_q > 1e-12 and norm_d > 1e-12:
            sparse_sim = float(np.dot(q_sparse, doc_sparse) / (norm_q * norm_d))
        else:
            sparse_sim = 0.0
        return (1.0 - alpha) * dense_sim + alpha * sparse_sim

    # ------------------------------------------------------------------ batch indexing
    def transform(self) -> None:
        """Encode self.transcripts; results stored in self.embeddings / self.metadata."""
        if self._embedder is None:
            raise RuntimeError("Call load() before transform().")
        self.embeddings.clear()
        self.metadata.clear()
        for item in self.transcripts:
            dense, _ = self.encode(item["text"])
            self.embeddings.append(dense)
            self.metadata.append({
                "type":       "transcript",
                "text":       item["text"],
                "video_path": item["video_path"],
                "start_time": item["start"],
                "end_time":   item["end"],
            })

    def save_embeddings(self, vector_store: VectorStore) -> None:
        for i, embedding in enumerate(self.embeddings):
            vector_store.storeVector(VideoVector(
                id=f"hybrid_embedding_{i}",
                embedding=embedding.tolist(),
                metadata=self.metadata[i],
            ))
