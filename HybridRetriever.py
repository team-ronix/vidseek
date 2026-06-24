import sys
from pathlib import Path

import numpy as np

from Storage.VectorStore import VideoVector
from Storage.HNSW import HNSWVectorStore

_HYBRID_DIR = Path(__file__).resolve().parent / "hybrid_embedder"
if str(_HYBRID_DIR) not in sys.path:
    sys.path.insert(0, str(_HYBRID_DIR))

from main import HybridEmbedder  # type: ignore

_MODEL_PREFIX = str(Path(__file__).resolve().parent / "hybrid_embedder" / "models" / "model")
_HNSW_ROOT    = "./data/hybrid_hnsw_index"
_LSA_DIM      = 128   # matches hybrid_embedder/models/model_lsa.npz n_components
_ALPHA        = 0.9   # hybrid = (1 - alpha)*dense_cos + alpha*sparse_cos


def _bm25_cosine(q_vec: np.ndarray, doc_sparse: dict) -> float:
    """Cosine similarity between a dense BM25 query vector and a stored sparse dict."""
    dot = sum(float(q_vec[int(k)]) * float(v)
              for k, v in doc_sparse.items()
              if 0 <= int(k) < len(q_vec))
    q_norm   = float(np.linalg.norm(q_vec))
    doc_norm = float(np.sqrt(sum(float(v) ** 2 for v in doc_sparse.values())))
    denom = q_norm * doc_norm
    return dot / denom if denom > 1e-8 else 0.0


class HybridRetriever:
    """
    Encodes documents with HybridEmbedder (LSA dense + BM25 sparse),
    stores both vectors, and retrieves with the hybrid scoring formula:

        score = (1 - alpha) * dense_cosine + alpha * sparse_cosine

    Relies on pre-trained models in hybrid_embedder/models/.
    """

    def __init__(self, ocr_results: dict, transcripts: list):
        self.ocr_results = ocr_results
        self.transcripts = transcripts
        self._embeddings: list = []
        self._metadata: list  = []
        self._embedder: HybridEmbedder | None = None

    # -- Model loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._embedder is not None:
            return
        lsa_path  = Path(f"{_MODEL_PREFIX}_lsa.npz")
        bm25_path = Path(f"{_MODEL_PREFIX}_bm25.npz")
        if not lsa_path.exists() or not bm25_path.exists():
            raise RuntimeError(
                f"Pre-trained HybridEmbedder models not found at {_MODEL_PREFIX}_lsa.npz / _bm25.npz"
            )
        self._embedder = HybridEmbedder.load(_MODEL_PREFIX)

    # -- Indexing ------------------------------------------------------------

    def transform(self) -> None:
        """Encode all OCR words and transcript segments; store dense+sparse."""
        self._ensure_loaded()

        for key, occurrences in self.ocr_results.items():
            dense, bm25_vec = self._embedder.encode(key)
            sparse = self._embedder.bm25.encode_as_dict(key)
            for occ in occurrences:
                self._embeddings.append(dense)
                self._metadata.append({
                    "type":          "ocr",
                    "text":          key,
                    "video_path":    occ["video_path"],
                    "start_time":    occ["start_time"],
                    "end_time":      occ["end_time"],
                    "sparse_vector": sparse,
                })

        for item in self.transcripts:
            dense, _ = self._embedder.encode(item["text"])
            sparse   = self._embedder.bm25.encode_as_dict(item["text"])
            self._embeddings.append(dense)
            self._metadata.append({
                "type":          "transcript",
                "text":          item["text"],
                "video_path":    item["video_path"],
                "start_time":    item["start"],
                "end_time":      item["end"],
                "sparse_vector": sparse,
            })

    def save_embeddings(self, store) -> None:
        for i, emb in enumerate(self._embeddings):
            store.storeVector(VideoVector(
                id        = f"hybrid_{i}",
                embedding = emb.tolist(),
                metadata  = self._metadata[i],
            ))

    # -- Retrieval -----------------------------------------------------------

    def query_hybrid(
        self,
        text:    str,
        top_k:   int   = 10,
        alpha:   float = _ALPHA,
    ) -> list[tuple[float, dict]]:
        """
        Hybrid retrieval:
          1. HNSW dense search for candidates
          2. Re-rank with: (1-alpha)*dense_cos + alpha*bm25_cos
        Returns list of (hybrid_score, metadata) sorted best-first.
        """
        self._ensure_loaded()
        q_dense, q_bm25 = self._embedder.encode(text)

        n_candidates = max(top_k * 4, 40)
        store = HNSWVectorStore(index_root=_HNSW_ROOT, dimension=_LSA_DIM, source="hybrid")
        _, metas, sims = store.query(q_dense.tolist(), top_k=n_candidates)

        scored: list[tuple[float, dict]] = []
        for meta, dense_sim in zip(metas[0], sims[0]):
            sparse_doc   = meta.get("sparse_vector", {})
            sparse_cos   = _bm25_cosine(q_bm25, sparse_doc) if sparse_doc else 0.0
            hybrid_score = (1.0 - alpha) * float(dense_sim) + alpha * sparse_cos
            scored.append((hybrid_score, meta))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def encode_query_dense(self, text: str) -> np.ndarray:
        """Dense-only query vector (for raw HNSW search without re-ranking)."""
        self._ensure_loaded()
        dense, _ = self._embedder.encode(text)
        return dense
