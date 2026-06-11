import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

class SentenceSegmentation:
    def __init__(self, video_path, transcript_json, similarity_threshold=0.75):
        self.video_path = video_path
        with open(transcript_json, 'r') as f:
            self.data = json.load(f)
        self.chunks = self.data.get('chunks', [])
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.segments = []
        self.groups = []

    def group_chunks_by_topic(self):
        if not self.chunks:
            return
        
        current_group = [self.chunks[0]]
        for i in range(1, len(self.chunks)):
            prev_text = self.chunks[i-1]['text']
            curr_text = self.chunks[i]['text']
            
            emb1 = self.model.encode(prev_text, convert_to_tensor=True)
            emb2 = self.model.encode(curr_text, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            
            if similarity >= self.similarity_threshold:
                current_group.append(self.chunks[i])
            else:
                self.groups.append(current_group)
                current_group = [self.chunks[i]]
        self.groups.append(current_group)

    def build_segments(self):
        segments = []
        for group in self.groups:
            start_time = group[0].get('timestamp', [0, 0])[0]
            end_time = group[-1].get('timestamp', [0, 0])[1]
            
            # Handle NoneType timestamps safely
            safe_start = round(start_time, 2) if start_time is not None else 0.0
            safe_end = round(end_time, 2) if end_time is not None else safe_start
            
            segments.append({
                "text": " ".join(c["text"] for c in group).strip(),
                "start": safe_start,
                "end": safe_end,
                "video_path": self.video_path
            })
        self.segments = segments

    def segment(self):
        self.group_chunks_by_topic()
        self.build_segments()
        return self.segments

    def save_segments(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.segments, f, indent=4)
            
    def load_segments(self, input_path):
        with open(input_path, 'r') as f:
            self.segments = json.load(f)