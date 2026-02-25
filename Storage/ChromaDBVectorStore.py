import chromadb
from .VectorStore import VectorStore, VideoVector

class ChromaDBVectorStore(VectorStore):
    def __init__(self):
        self._client = chromadb.PersistentClient(path="./data")

    def storeVector(self, vector: VideoVector):
        collection = self._client.get_or_create_collection(name="video_segments")
        collection.add(
            ids=[vector.id],
            embeddings=[vector.embedding],
            metadatas=[vector.metadata]
        )

    def query(self, query_embedding: list[float], top_k: int = 5):
        collection = self._client.get_collection(name="video_segments")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['ids'], results['metadatas'], results['distances']