import json
import re
import sys
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

_TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "Transformer"
if str(_TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMER_DIR))

from models.model.transformer import Transformer
from data import _make_ids

_MODEL_DIR = _TRANSFORMER_DIR / "results" / "allnli_specter"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MAX_LEN = 128


#  helpers 

def _smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    return np.convolve(np.pad(signal, radius, mode="reflect"), k, mode="valid").astype(np.float32)


def _find_valleys(signal: np.ndarray, min_prominence: float, alpha: float) -> List[int]:
    threshold = float(signal.mean()) - alpha * (float(signal.std()) + 1e-8)
    valleys: List[int] = []
    for i in range(1, len(signal) - 1):
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            prominence = min(signal[i - 1] - signal[i], signal[i + 1] - signal[i])
            if prominence >= min_prominence and signal[i] <= threshold:
                valleys.append(i)
    return valleys


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and", "or",
    "but", "for", "with", "this", "that", "was", "are", "be", "been", "have",
    "has", "had", "not", "we", "i", "you", "he", "she", "they", "do", "does",
    "so", "if", "as", "by", "from", "also", "just", "then", "than", "about",
    "up", "out", "what", "when", "which", "who", "will", "can", "its", "into",
    "more", "very", "like", "all", "now", "how", "no", "get", "one", "some",
    "there", "their", "them", "our", "your", "his", "her", "my", "me", "re",
    "go", "come", "know", "want", "look", "right", "okay", "yeah", "ve",
    "ll", "don", "didn", "isn", "wasn", "aren", "won", "can", "let", "got",
})


def _tfidf_title(texts: List[str], n: int) -> str:
    if not texts:
        return "Untitled"
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", max_features=500,
                            ngram_range=(1, 2), sublinear_tf=True)
        try:
            mat = vec.fit_transform(texts)
        except ValueError:
            return "Untitled"
        names  = vec.get_feature_names_out()
        scores = np.asarray(mat.todense()).sum(axis=0).ravel()
        keywords = [names[i] for i in scores.argsort()[-n:][::-1]]
    except ImportError:
        words: List[str] = []
        for t in texts:
            words.extend(re.sub(r"[^a-z\s]", "", t.lower()).split())
        counter = Counter(w for w in words if w not in _STOPWORDS and len(w) > 2)
        keywords = [w for w, _ in counter.most_common(n)]
    keywords.sort(key=len)
    return " ".join(k.title() for k in keywords[:n]) if keywords else "Untitled"


#  main class 

class SentenceSegmentation:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        video_path: str = "",
        *,
        window_size: int             = 4,
        smooth_sigma: float          = 2.0,
        valley_alpha: float          = 0.6,
        min_valley_prominence: float = 0.005,
        min_segment_sentences: int   = 2,
        n_title_keywords: int        = 4,
        batch_size: int              = 32,
    ):
        self.video_path            = video_path
        self.window_size           = window_size
        self.smooth_sigma          = smooth_sigma
        self.valley_alpha          = valley_alpha
        self.min_valley_prominence = min_valley_prominence
        self.min_segment_sentences = min_segment_sentences
        self.n_title_keywords      = n_title_keywords
        self.batch_size            = batch_size

        self.chunks: List[Dict[str, Any]] = chunks

        with open(_MODEL_DIR / "vocab.pkl", "rb") as vf:
            self._vocab = pickle.load(vf)

        self._model = Transformer(
            vocab_size=len(self._vocab),
            d_model=384, n_layers=6, n_heads=6,
            d_ff=1536, max_len=_MAX_LEN, pooling="mean",
        ).to(_DEVICE)
        self._model.load_state_dict(
            torch.load(_MODEL_DIR / "best_model.pt", map_location=_DEVICE)
        )
        self._model.eval()

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        all_ids = [_make_ids(t, self._vocab, _MAX_LEN) for t in texts]
        parts: List[torch.Tensor] = []
        for start in range(0, len(all_ids), self.batch_size):
            batch = all_ids[start : start + self.batch_size]
            max_len = max(len(ids) for ids in batch)
            padded  = [ids + [0] * (max_len - len(ids)) for ids in batch]
            ids_t   = torch.tensor(padded, dtype=torch.long, device=_DEVICE)
            with torch.no_grad():
                parts.append(self._model.encode(ids_t))
        return torch.cat(parts, dim=0)

    @staticmethod
    def _block_similarity(embeddings: torch.Tensor, w: int) -> np.ndarray:
        E    = embeddings.cpu().float().numpy()
        N, _ = E.shape
        cum  = np.cumsum(E, axis=0)
        depth = np.zeros(N, dtype=np.float32)

        def block_mean(a: int, b: int) -> np.ndarray:
            raw = cum[b - 1] if a == 0 else cum[b - 1] - cum[a - 1]
            mu  = raw / (b - a)
            return mu / (np.linalg.norm(mu) + 1e-8)

        for i in range(w, N - w):
            depth[i] = float(np.dot(block_mean(max(0, i - w), i),
                                    block_mean(i, min(N, i + w))))
        if N > 2 * w:
            depth[:w]     = depth[w]
            depth[N - w:] = depth[N - w - 1]
        return depth

    def segment(self) -> List[Dict[str, Any]]:
        """
        Run the full pipeline and return merged chapter segments.

        Output:
        [
            {
                "text"      : "...",
                "start"     : 0.0,
                "end"       : 198.67,
                "video_path": "videos/video.mp4",
                "title"     : "Chapter Title"
            },
            ...
        ]
        """
        if not self.chunks:
            return []

        N     = len(self.chunks)
        texts = [c["text"] for c in self.chunks]

        embeddings = self._encode_batch(texts)
        w          = max(1, min(self.window_size, N // 4))
        smoothed   = _smooth(self._block_similarity(embeddings, w), self.smooth_sigma)
        boundaries = _find_valleys(smoothed, self.min_valley_prominence, self.valley_alpha)

        ranges: List[tuple] = []
        prev = 0
        for bp in sorted(set(boundaries)):
            if bp - prev >= self.min_segment_sentences:
                ranges.append((prev, bp - 1))
                prev = bp
        ranges.append((prev, N - 1))

        segments: List[Dict[str, Any]] = []
        for seg_start, seg_end in ranges:
            seg_chunks = self.chunks[seg_start : seg_end + 1]
            seg_texts  = [c["text"] for c in seg_chunks]
            segments.append({
                "text"      : " ".join(seg_texts),
                "start"     : round(seg_chunks[ 0]["start"], 2),
                "end"       : round(seg_chunks[-1]["end"],   2),
                "video_path": seg_chunks[0].get("video_path", self.video_path),
                "title"     : _tfidf_title(seg_texts, self.n_title_keywords),
            })

        return segments

    def save(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.segment(), f, indent=4, ensure_ascii=False)
