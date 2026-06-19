from Transformer import Transformer
from Storage.CustomVectorStore import CustomVectorStore

query = "March"

transformer = Transformer({}, [])
embedding = transformer.transform_single_text(query)
print(f"searching query '{query}'")

vector_store = CustomVectorStore()
ids, metadatas, distances = vector_store.query(embedding, top_k=5)
print("Search results:")
for id, metadata, distance in zip(ids[0], metadatas[0], distances[0]):
    print(f"ID: {id}, Metadata: {metadata}, Distance: {distance}")