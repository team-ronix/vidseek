from sentence_transformers import SentenceTransformer

class Transformer:
    def __init__(self, ocr_results, transcripts, model_id = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_id)
        self.ocr_results = ocr_results
        self.transcripts = transcripts
        self.embeddings = []
        self.metadata = []

    def transform(self):
        for key, value in zip(self.ocr_results.keys(), self.ocr_results.values()):
            self.embeddings.append(self.model.encode(key))
            self.metadata.append({
                'type': 'ocr', 
                'video_path': value['video_path'], 
                'start_time': value['start_time'],
                'end_time': value['end_time']
            })
        for item in self.transcripts:
            self.embeddings.append(self.model.encode(item['text']))
            self.metadata.append({
                'type': 'transcript', 
                'video_path': item['video_path'], 
                'start_time': item['start'],
                'end_time': item['end']
            })
            
    def get_embeddings(self):
        return self.embeddings
    
    def get_metadata(self):
        return self.metadata