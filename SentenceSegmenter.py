import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SentenceSegmentation:
    def __init__(self, transcript_json, similarity_threshold=0.75):
        with open(transcript_json, "r", encoding="utf-8") as f:
                self.transcript = json.load(f)
        self.similarity_threshold = similarity_threshold
        self.groups = None
        self.segments = None

    def group_chunks_by_topic(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = self.transcript["chunks"]
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts)

        groups = []
        current_group = [chunks[0]]

        for i in range(1, len(chunks)):
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            if sim >= self.similarity_threshold:
                current_group.append(chunks[i])
            else:
                groups.append(current_group)
                current_group = [chunks[i]]

        groups.append(current_group)
        self.groups = groups

    def build_segments(self):
        segments = []
        for group in self.groups:
            segments.append({
                "text": " ".join(c["text"] for c in group).strip(),
                "start": round(group[0]["timestamp"][0], 2),
                "end": round(group[-1]["timestamp"][1], 2)
            })
        self.segments = segments

    def segment(self):
        self.group_chunks_by_topic()
        self.build_segments()
        return self.segments

    # -------------------------------
    # Save the segmented file into JSON
    # -------------------------------
    def save_segments(self, output_path="segmented_transcript.json"):
        if self.segments is None:
            raise ValueError("Run segment() first before saving.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.segments, f, indent=2, ensure_ascii=False)
        print(f"Segmented transcript saved to {output_path}")