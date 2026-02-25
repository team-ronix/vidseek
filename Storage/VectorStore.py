
@dataclass
class VideoVector:
    id: str
    embedding: list[float]
    metadata: dict

class VectorStore:
    def __init__(self):
        pass

    def storeVector(self, vector: VideoVector):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def query(self, query_embedding: list[float], top_k: int = 5):
        raise NotImplementedError("This method should be implemented by subclasses.")