import numpy as np
from .lsa_encoder import LSAEncoder
from .bm25_encoder import BM25Encoder


class HybridEmbedder:
    def __init__(self, n_components: int = 64,
                 k1: float = 1.5, b: float = 0.75,
                 min_df: int = 1, max_df_ratio: float = 0.95):
        self.lsa = LSAEncoder(n_components=n_components, min_df=min_df, max_df_ratio=max_df_ratio)
        self.bm25 = BM25Encoder(k1=k1, b=b, min_df=min_df, max_df_ratio=max_df_ratio)

    def fit(self, corpus: list) -> "HybridEmbedder":
        self.lsa.fit(corpus)
        self.bm25.fit(corpus)
        return self

    def encode(self, sentence: str) -> tuple:
        return self.lsa.encode(sentence), self.bm25.encode(sentence)

    def encode_for_db(self, sentence: str) -> dict:
        dense, _ = self.encode(sentence)
        return {
            "dense_vector": dense.tolist(),
            "sparse_vector": self.bm25.encode_as_dict(sentence),
        }

    def save(self, prefix: str = "hybrid_model") -> None:
        self.lsa.save(f"{prefix}_lsa")
        self.bm25.save(f"{prefix}_bm25")

    @classmethod
    def load(cls, prefix: str = "hybrid_model") -> "HybridEmbedder":
        obj = cls.__new__(cls)
        obj.lsa = LSAEncoder.load(f"{prefix}_lsa")
        obj.bm25 = BM25Encoder.load(f"{prefix}_bm25")
        return obj
