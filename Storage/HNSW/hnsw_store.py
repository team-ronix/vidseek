import numpy as np

from Storage.VectorStore import VectorStore, VideoVector
from Storage.SQL.Repositories.VectorMetadataRepository import VectorMetadataRepository
from .hnsw_index import HNSWIndex


class HNSWVectorStore(VectorStore):
    """Vector store backed by the custom HNSW index + SQL metadata.

    Call ``commit()`` after finishing a batch of ``storeVector`` calls so
    that the in-RAM vectors and graph are flushed to disk atomically.
    If the process crashes before ``commit()``, the index reverts cleanly
    to the previous checkpoint — no partial or half-inserted state.
    """

    def __init__(
        self,
        index_root: str = "./data/hnsw_index",
        dimension: int = 384,
        M: int = 16,
        ef_construction: int = 200,
    ) -> None:
        self.index = HNSWIndex(
            root_dir=index_root,
            dimension=dimension,
            M=M,
            ef_construction=ef_construction,
        )

    def storeVector(self, vector: VideoVector) -> None:
        node_id = self.index.insert(np.array(vector.embedding, dtype=np.float32))

        repo = VectorMetadataRepository()
        try:
            repo.upsert(
                vector_id=node_id,
                external_id=str(vector.id) if vector.id is not None else None,
                metadata=vector.metadata or {},
            )
        finally:
            repo.close()

    def commit(self) -> None:
        """Flush all in-RAM vectors and graph to disk atomically."""
        self.index.save()

    def query(self, query_embedding: list[float], top_k: int = 5):
        hits = self.index.query(np.array(query_embedding, dtype=np.float32), top_k=top_k)

        ids_flat          = [str(vec_id) for vec_id, _ in hits]
        vector_ids        = [vec_id for vec_id, _ in hits]
        flat_similarities = [float(sim) for _, sim in hits]

        repo = VectorMetadataRepository()
        try:
            id_to_meta = repo.get_by_vector_ids(vector_ids)
        finally:
            repo.close()

        metas_flat = [id_to_meta.get(int(vec_id), {}) for vec_id in vector_ids]
        return [ids_flat], [metas_flat], [flat_similarities]
