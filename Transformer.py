import json
import sys
import pickle
import torch
from pathlib import Path

from Storage.VectorStore import VectorStore, VideoVector

_TRANSFORMER_DIR = Path(__file__).resolve().parent / "Transformer"
if str(_TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_DIR))

from models import Transformer as TransformerModel  # type: ignore
from data import _make_ids  # type: ignore

_VOCAB_PATH  = _TRANSFORMER_DIR / "results" / "allnli_specter" / "vocab.pkl"
_MODEL_PATH  = _TRANSFORMER_DIR / "results" / "allnli_specter" / "best_model.pt"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MAX_LEN = 128


class Transformer:
    def __init__(self, transcripts):
        self.transcripts = transcripts
        self.embeddings = []
        self.metadata = []

        with open(_VOCAB_PATH, "rb") as vf:
            self._vocab = pickle.load(vf)

        self._model = TransformerModel(
            vocab_size=len(self._vocab),
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_ff=1536,
            max_len=_MAX_LEN,
            pooling="mean",
        ).to(_DEVICE)
        self._model.load_state_dict(
            torch.load(_MODEL_PATH, map_location=_DEVICE, weights_only=True)
        )
        self._model.eval()

    def _encode(self, text: str) -> torch.Tensor:
        ids = _make_ids(text, self._vocab, _MAX_LEN)
        input_ids = torch.tensor([ids], dtype=torch.long, device=_DEVICE)
        with torch.no_grad():
            emb = self._model.encode(input_ids)  # (1, d_model), L2-normalized
        return emb[0]  # (d_model,)

    def transform(self):
        for item in self.transcripts:
            self.embeddings.append(self._encode(item['text']))
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

    def transform_single_text(self, text: str) -> torch.Tensor:
        return self._encode(text)
