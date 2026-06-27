import numpy as np
import os

EMBED_DIM = 50
# one for subject and one for object
SEMANTIC_DIM = EMBED_DIM * 2

class GloVeEmbedder:
    def __init__(
        self,
        glove_path=None,
    ):
        self._vectors = {}
        self._dim = EMBED_DIM
        self._use_hash_fallback = False
        if glove_path and os.path.isfile(glove_path):
            self._load_glove(glove_path)
        else:
            print(f"[semantic] Warning: GloVe file not found at '{glove_path}', using hash fallback")
            self._use_hash_fallback = True

    def _load_glove(self, path):
        print(f"[semantic] Loading GloVe from {path}", flush=True)
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                if vec.shape[0] == self._dim:
                    self._vectors[word] = vec
                    count += 1
        print(f"[semantic] Loaded {count} word vectors (dim={self._dim})")

    def _hash_embed(self, word):
        rng = np.random.default_rng(abs(hash(word)) % (2**32))
        return rng.standard_normal(self._dim).astype(np.float32)

    def embed(self, word):
        if self._use_hash_fallback == True:
            return self._hash_embed(word)
        tokens = word.lower().split()
        vecs = []
        for tok in tokens:
            if tok in self._vectors:
                vecs.append(self._vectors[tok])
            else:
                # try some variations of the token if it's not existed
                for candidate in [tok + "s", tok[:-1], tok[:-2]]:
                    if candidate in self._vectors:
                        vecs.append(self._vectors[candidate])
                        break
                else:
                    vecs.append(self._hash_embed(tok))
        return np.mean(vecs, axis=0).astype(np.float32)

    @property
    def dim(self):
        return self._dim

    def similarity(self, word_a, word_b):
        a = self.embed(word_a)
        b = self.embed(word_b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class SemanticFeatureExtractor:
    dim = SEMANTIC_DIM

    def __init__(self, glove_path=None):
        self.embedder = GloVeEmbedder(glove_path=glove_path)

    def extract(self, subj_label, obj_label):
        sv = self.embedder.embed(subj_label)
        ov = self.embedder.embed(obj_label)
        feat = np.concatenate([sv, ov]).astype(np.float32)
        assert feat.shape == (SEMANTIC_DIM,), (f"Expected {SEMANTIC_DIM}-d, got {feat.shape}")
        return feat

    def extract_batch(self, pairs):
        return np.stack([self.extract(s, o) for s, o in pairs], axis=0)

    def zero_shot_nearest(self, query_subj, query_obj, known_pairs, top_k=5):
        q = self.extract(query_subj, query_obj)
        scored = []
        for s, o in known_pairs:
            k = self.extract(s, o)
            sim = float(np.dot(q, k) / (np.linalg.norm(q) * np.linalg.norm(k) + 1e-8))
            scored.append(((s, o), sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
