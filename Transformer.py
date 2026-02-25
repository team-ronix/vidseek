from sentence_transformers import SentenceTransformer
import json
class Transformer:
    def __init__(self, ocr_results, transcripts, model_id = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_id)
        self.ocr_results = ocr_results
        self.transcripts = transcripts
        self.embeddings = []
        self.metadata = []

    def transform(self):
        # Process OCR results
        for key, occurrences in self.ocr_results.items():
            embedding = self.model.encode(key)
            for occ in occurrences:
                self.embeddings.append(embedding)
                self.metadata.append({
                    'type': 'ocr', 
                    'text': key, 
                    'video_path': occ['video_path'], 
                    'start_time': occ['start_time'],
                    'end_time': occ['end_time']
                })
        for item in self.transcripts:
            self.embeddings.append(self.model.encode(item['text']))
            self.metadata.append({
                'type': 'transcript',
                'text': item['text'],
                'video_path': item['video_path'], 
                'start_time': item['start'],
                'end_time': item['end']
            })
            
    def get_embeddings(self):
        return self.embeddings
    
    def get_metadata(self):
        return self.metadata
    
    def save_metadata(self, output_path='metadata.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
    def save_embeddings(self, VectorStore: VectorStore):
        for i, embedding in enumerate(self.embeddings):
            vector = VideoVector(
                id=f"embedding_{i}",
                embedding=embedding.tolist(),
                metadata=self.metadata[i]
            )
            VectorStore.storeVector(vector)

    def transform_single_text(self, text):
        return self.model.encode(text)