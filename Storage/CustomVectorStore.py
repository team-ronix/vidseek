from pathlib import Path

from .VectorStore import VectorStore, VideoVector
from .IVF.two_level_index import TwoLevelIVFIndex
from Storage.SQL.Repositories.VectorMetadataRepository import VectorMetadataRepository


class CustomVectorStore(VectorStore):
    """Drop-in replacement for ChromaDB using custom 2-level IVF + SQL metadata."""

    def __init__(self, index_root: str = "./data/custom_ivf"):
        self.index = TwoLevelIVFIndex(root_dir=index_root)

    def storeVector(self, vector: VideoVector):
        insert_res = self.index.insert(vector.embedding)

        repo = VectorMetadataRepository()
        try:
            repo.upsert(
                vector_id=insert_res.vector_id,
                external_id=str(vector.id) if vector.id is not None else None,
                metadata=vector.metadata or {},
            )
        finally:
            repo.close()

    def query(self, query_embedding: list[float], top_k: int = 5):
        hits = self.index.query(query_embedding, top_k=top_k)

        ids_flat = [str(vec_id) for vec_id, _ in hits]
        vector_ids = [vec_id for vec_id, _ in hits]
        flat_similarities = [float(sim) for _, sim in hits]

        repo = VectorMetadataRepository()
        try:
            id_to_meta = repo.get_by_vector_ids(vector_ids)
        finally:
            repo.close()

        metas_flat = [id_to_meta.get(int(vec_id), {}) for vec_id in vector_ids]

        return [ids_flat], [metas_flat], [flat_similarities]
