import json
import sys
import pickle
import torch
from pathlib import Path

_TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "Transformer"
if str(_TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_DIR))

from models.model.transformer import Transformer
from data import _make_ids

_MODEL_DIR = _TRANSFORMER_DIR / "reuslt_AllNLI+specter"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MAX_LEN = 128


class SentenceSegmentation:
    def __init__(self, video_path, transcript_json, similarity_threshold=0.75):
        self.video_path = video_path
        with open(transcript_json, 'r') as f:
            self.data = json.load(f)
        self.chunks = self.data.get('chunks', [])
        self.similarity_threshold = similarity_threshold

        with open(_MODEL_DIR / "vocab.pkl", "rb") as vf:
            self._vocab = pickle.load(vf)

        self._model = Transformer(
            vocab_size=len(self._vocab),
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_ff=1536,
            max_len=_MAX_LEN,
            pooling="mean",
        ).to(_DEVICE)
        self._model.load_state_dict(
            torch.load(_MODEL_DIR / "best_model.pt", map_location=_DEVICE)
        )
        self._model.eval()

        self.segments = []
        self.groups = []

    def _encode(self, text: str) -> torch.Tensor:
        ids = _make_ids(text, self._vocab, _MAX_LEN)
        input_ids = torch.tensor([ids], dtype=torch.long, device=_DEVICE)
        with torch.no_grad():
            emb = self._model.encode(input_ids)  # (1, d_model), L2-normalized
        return emb[0]  # (d_model,)

    def group_chunks_by_topic(self):
        if not self.chunks:
            return

        current_group = [self.chunks[0]]
        for i in range(1, len(self.chunks)):
            prev_text = self.chunks[i-1]['text']
            curr_text = self.chunks[i]['text']

            emb1 = self._encode(prev_text)
            emb2 = self._encode(curr_text)
            # dot product == cosine similarity for L2-normalized vectors
            similarity = (emb1 * emb2).sum().item()

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
