import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class IVFInsertResult:
    vector_id: int
    level1_id: int
    level2_id: int
class TwoLevelIVFIndex:
    """2-level IVF index with append-only vectors and append-only cluster postings.Small metadata files are JSON so we can safely recover if something crashes."""

    def __init__(
        self,root_dir: str,dimension: int | None = None,
        max_level1_clusters: int = 500,max_level2_per_level1: int = 20,
        level1_probes:int | None = None,
        level2_probes:int | None = None,
        level1_threshold:float|None =None,
        level2_threshold:float|None= None,
    ) -> None:
        self.root_dir =Path(root_dir)
        self.postings_dir =self.root_dir / "postings"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.postings_dir.mkdir(parents=True, exist_ok=True)

        self.vectors_path =self.root_dir / "vectors.dat"
        self.level1_centroids_path = self.root_dir / "level1_centroids.dat"
        self.level2_centroids_path= self.root_dir / "level2_centroids.dat"
        self.metadata_path =self.root_dir / "metadata.json"
        self.layout_path =self.root_dir / "layout.json"

        self.max_level1_clusters= max_level1_clusters
        self.max_level2_per_level1=max_level2_per_level1
        self._level1_probes =level1_probes
        self._level2_probes =level2_probes
        self._level1_threshold= level1_threshold
        self._level2_threshold= level2_threshold

        self._lock = threading.Lock()
        self._ensure_bootstrap(dimension=dimension)

    def _ensure_bootstrap(self, dimension: int | None) -> None:
        if not self.metadata_path.exists():
            if dimension is None:
                dimension = 384
            manifest = {
                "dimension": int(dimension),
                "next_vector_id": 0,
                "vector_count": 0,
            }
            layout = {
                "level1_to_level2": [],
                "level1_counts": [],
                "level2_counts": [],
            }
            self._write_json(self.metadata_path, manifest)
            self._write_json(self.layout_path, layout)
            self.vectors_path.touch(exist_ok=True)

            self.level1_centroids_path.touch(exist_ok=True)
            self.level2_centroids_path.touch(exist_ok=True)

        if not self.layout_path.exists():
            self._write_json(
                self.layout_path,
                {"level1_to_level2": [],"level1_counts": [],"level2_counts": []
                },
            )

        self._metadata = self._read_json(self.metadata_path)
        if dimension is not None and self._metadata["vector_count"] == 0:
            self._metadata["dimension"] = int(dimension)
            self._write_json(self.metadata_path, self._metadata)

    def _read_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True)
        self._replace(tmp, path)

    def _replace(self, src: Path, dst: Path) -> None:
        for attempt in range(10):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                time.sleep(0.01 * (attempt + 1))
        os.replace(src, dst)

    @property
    def dimension(self) -> int:
        return int(self._metadata["dimension"])

    @property
    def vector_count(self) -> int:
        return int(self._metadata["vector_count"])

    def _load_centroids(self, path: Path) -> np.ndarray:
        if not path.exists() or path.stat().st_size == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        return raw.reshape((-1, self.dimension))

    def _save_centroids(self, path: Path, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        tmp = path.with_suffix(path.suffix + ".tmp")
        values.tofile(tmp)
        self._replace(tmp, path)

    def _get_thresholds(self) -> tuple[float, float]:
        if self._level1_threshold is not None and self._level2_threshold is not None:
            return self._level1_threshold, self._level2_threshold
        n = max(10, self.vector_count)
        scale = math.log10(n / 10)
        l1 = max(0.52, 0.72-0.04 * scale)
        l2 = max(0.68, 0.85- 0.04 * scale)
        return l1, l2

    def _get_probes(self, n_l1: int, layout: dict) -> tuple[int, int]:
        if self._level1_probes is not None and self._level2_probes is not None:
            return (min(self._level1_probes, n_l1),self._level2_probes,)
        n_l2_total = sum(len(v) for v in layout["level1_to_level2"])
        avg_l2 = n_l2_total / max(1, n_l1)
        l1_probes = max(1, min(n_l1, math.ceil(math.sqrt(n_l1))))
        l2_probes = max(1, min(int(avg_l2), math.ceil(math.sqrt(avg_l2))))
        return l1_probes, l2_probes

    def _normalize(self,x:np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            norm =np.linalg.norm(x)
            if norm ==0:
                return x.astype(np.float32)
            return (x/norm).astype(np.float32)
        else:
            if x.size ==0:
                return x.astype(np.float32)
            norms =np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms == 0] =1.0
            return (x/norms).astype(np.float32)

    def _append_vector(self, vec: np.ndarray) -> int:
        vec_id =int(self._metadata["next_vector_id"])
        with open(self.vectors_path, "ab") as f:
            np.asarray(vec, dtype=np.float32).tofile(f)
        self._metadata["next_vector_id"] = vec_id + 1
        self._metadata["vector_count"] = int(self._metadata["vector_count"]) + 1
        self._write_json(self.metadata_path, self._metadata)
        return vec_id

    def _append_posting(self, level2_id: int, vec_id: int) -> None:
        posting_path = self.postings_dir/f"c_{level2_id}.dat"
        with open(posting_path,"ab") as f:
            np.asarray([vec_id], dtype=np.uint32).tofile(f)

    def _get_cluster_vector_ids(self, level2_id: int) -> np.ndarray:
        posting_path = self.postings_dir /f"c_{level2_id}.dat"
        if not posting_path.exists() or posting_path.stat().st_size == 0:
            return np.empty((0,),dtype=np.uint32)
        return np.fromfile(posting_path,dtype=np.uint32)

    def insert(self, vector: np.ndarray) -> IVFInsertResult:
        vec = np.asarray(vector,dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Vector must be 1D")

        with self._lock:
            if self.vector_count == 0 and vec.shape[0] != self.dimension:
                self._metadata["dimension"] = int(vec.shape[0])
                self._write_json(self.metadata_path, self._metadata)
            elif vec.shape[0] != self.dimension:
                raise ValueError(f"Vector dimension {vec.shape[0]} != expected {self.dimension}")

            vec = self._normalize(vec)

            level1 =self._load_centroids(self.level1_centroids_path)
            level2 =self._load_centroids(self.level2_centroids_path)
            layout = self._read_json(self.layout_path)

            if level1.shape[0] == 0:
                vec_id = self._append_vector(vec)
                level1 =np.vstack([level1, vec.reshape(1, -1)])
                level2 =np.vstack([level2, vec.reshape(1, -1)])
                layout["level1_to_level2"]= [[0]] 
                layout["level1_counts"] =[1]
                layout["level2_counts"] = [1]

                self._save_centroids(self.level1_centroids_path, level1)
                self._save_centroids(self.level2_centroids_path, level2)
                
                self._append_posting(0, vec_id)

                self._write_json(self.layout_path, layout)
                return IVFInsertResult(vector_id=vec_id, level1_id=0, level2_id=0)

            level1_norm = self._normalize(level1)
            level1_scores = level1_norm @ vec
            best_level1 = int(np.argmax(level1_scores))
            best_level1_score = float(level1_scores[best_level1])

            l1_thresh, l2_thresh = self._get_thresholds()
            can_create_l1 = len(layout["level1_to_level2"]) < self.max_level1_clusters
            if best_level1_score < l1_thresh and can_create_l1:
                new_level1_id = int(level1.shape[0])
                level1 = np.vstack([level1, vec.reshape(1, -1)])
                level2 = np.vstack([level2, vec.reshape(1, -1)])
                new_level2_id = int(level2.shape[0] - 1)

                layout["level1_to_level2"].append([new_level2_id])
                layout["level1_counts"].append(0)
                layout["level2_counts"].append(0)

                best_level1 = new_level1_id

            level2_ids = layout["level1_to_level2"][best_level1]
            if len(level2_ids) == 0:
                level2 = np.vstack([level2, vec.reshape(1, -1)])
                new_level2_id = int(level2.shape[0] - 1)
                layout["level1_to_level2"][best_level1].append(new_level2_id)
                layout["level2_counts"].append(0)
                chosen_level2 = new_level2_id
            else:
                local_level2 = level2[np.asarray(level2_ids, dtype=np.int64)]
                local_level2_norm = self._normalize(local_level2)
                level2_scores = local_level2_norm @ vec
                local_best = int(np.argmax(level2_scores))
                best_level2_score = float(level2_scores[local_best])
                chosen_level2 = int(level2_ids[local_best])

                can_create_l2 = len(level2_ids) < self.max_level2_per_level1
                if best_level2_score < l2_thresh and can_create_l2:
                    level2 = np.vstack([level2, vec.reshape(1, -1)])
                    chosen_level2 = int(level2.shape[0] - 1)
                    layout["level1_to_level2"][best_level1].append(chosen_level2)
                    layout["level2_counts"].append(0)

            vec_id = self._append_vector(vec)
            self._append_posting(chosen_level2, vec_id)

            l1_count = int(layout["level1_counts"][best_level1])
            l1_old = level1[best_level1].copy()
            l1_new = (l1_old * l1_count + vec) / (l1_count + 1)
            level1[best_level1] = l1_new
            layout["level1_counts"][best_level1] = l1_count + 1

            l2_count = int(layout["level2_counts"][chosen_level2])
            l2_old = level2[chosen_level2].copy()
            l2_new = (l2_old * l2_count + vec) / (l2_count + 1)
            level2[chosen_level2] = l2_new
            layout["level2_counts"][chosen_level2] = l2_count + 1

            self._save_centroids(self.level1_centroids_path, level1)
            self._save_centroids(self.level2_centroids_path, level2)
            self._write_json(self.layout_path, layout)

            return IVFInsertResult(vector_id=vec_id, level1_id=best_level1, level2_id=chosen_level2)

    def _load_vectors_mmap(self):
        if self.vector_count == 0:
            return None
        return np.memmap(
            self.vectors_path,
            dtype=np.float32,
            mode="r",
            shape=(self.vector_count, self.dimension),
        )

    def query(self, vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []

        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Query vector must be 1D")
        if self.vector_count == 0:
            return []
        if vec.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension {vec.shape[0]} != expected {self.dimension}")

        vec = self._normalize(vec)
        level1 = self._load_centroids(self.level1_centroids_path)
        level2 = self._load_centroids(self.level2_centroids_path)
        layout = self._read_json(self.layout_path)

        if level1.shape[0] == 0:
            return []

        level1_norm = self._normalize(level1)
        level2_norm = self._normalize(level2)

        l1_probes, l2_probes = self._get_probes(level1.shape[0], layout)
        level1_scores = level1_norm @ vec
        n_l1 = min(l1_probes, level1.shape[0])
        if n_l1 < level1.shape[0]:
            top_l1 = np.argpartition(level1_scores, -n_l1)[-n_l1:]
            top_l1 = top_l1[np.argsort(level1_scores[top_l1])[::-1]]
        else:
            top_l1 = np.argsort(level1_scores)[::-1]

        candidate_ids = []
        for l1_id in top_l1.tolist():
            level2_ids = layout["level1_to_level2"][int(l1_id)]
            if not level2_ids:
                continue
            l2_arr = np.asarray(level2_ids, dtype=np.int64)
            l2_scores = level2_norm[l2_arr] @ vec
            n_l2 = min(l2_probes, len(level2_ids))
            if n_l2 < len(level2_ids):
                top_l2_local = np.argpartition(l2_scores, -n_l2)[-n_l2:]
                top_l2_local = top_l2_local[np.argsort(l2_scores[top_l2_local])[::-1]]
            else:
                top_l2_local = np.argsort(l2_scores)[::-1]

            for local_idx in top_l2_local.tolist():
                g_l2_id = int(level2_ids[int(local_idx)])
                ids = self._get_cluster_vector_ids(g_l2_id)
                if ids.size > 0:
                    candidate_ids.append(ids)

        if not candidate_ids:
            return []

        merged = np.concatenate(candidate_ids).astype(np.int64)
        if merged.size == 0:
            return []

        vectors = self._load_vectors_mmap()
        if vectors is None:
            return []
        sims = vectors[merged] @ vec

        n =min(top_k, merged.shape[0])
        if n < merged.shape[0]:
            idx = np.argpartition(sims, -n)[-n:]
            idx = idx[np.argsort(sims[idx])[::-1]]
        else:
            idx = np.argsort(sims)[::-1]

        result = []
        for i in idx.tolist():
            vec_id = int(merged[i])
            sim = float(sims[i])
            result.append((vec_id, sim))

        return result