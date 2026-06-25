import heapq
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np


class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) index.

    Parameters
    ----------
    M : int
        Max neighbours per node at layers > 0.Layer 0 uses 2*M.
        Higher M -> better recall, more memory and slower inserts.
        Typical values: 8-64.  Default: 20.
    ef_construction : int
        Beam width during graph construction. Higher -> better graph quality
        and recall, slower inserts.  Must be >= M.  Default: 200.
    """

    def __init__(
        self,
        root_dir: str,
        dimension: int | None = None,
        M: int = 20,
        ef_construction: int = 200,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.M  = M
        self.M0 = 2 * M                   
        self.ef_construction = max(ef_construction, M)
        self._mL = 1.0 / math.log(M)

        self.vectors_path  = self.root_dir / "vectors.dat"
        self.graph_path    = self.root_dir / "graph.npz"
        self.metadata = self.root_dir / "manifest.json"

        self._lock  = threading.Lock()
        self._dirty = False

        self._vec_buf: list[np.ndarray] = []

        self._neighbors: list[list[set[int]]] = []
        self._levels:    list[int]             = []
        self._entry_point: int = -1
        self._max_layer:   int = -1

        self._load_or_bootstrap(dimension)

    def _load_or_bootstrap(self, dimension: int | None) -> None:
        if self.metadata.exists():
            self._metadata = json.loads(self.metadata.read_text())
            given_dim = int(dimension) if dimension is not None else None
            stored_dim = self._metadata["dimension"]
            if given_dim is not None and self._metadata["vector_count"] > 0 and given_dim != stored_dim:
                raise ValueError(
                    f"Dimension mismatch: index has {stored_dim}, caller passed {given_dim}"
                )
            if self.graph_path.exists():
                self._load_graph()
            n = len(self._levels)
            if n > 0 and self.vectors_path.exists():
                raw = np.fromfile(str(self.vectors_path), dtype=np.float32)
                mat = raw.reshape(n, self._metadata["dimension"])
                self._vec_buf = [mat[i].copy() for i in range(n)]
            # Keep metadata in sync with the graph
            self._metadata["vector_count"] = n
        else:
            dim = int(dimension) if dimension is not None else 384
            self._metadata = {
                "dimension":       dim,
                "vector_count":    0,
                "M":               self.M,
                "M0":              self.M0,
                "ef_construction": self.ef_construction,
            }
            self._save_metadata()

    def _load_graph(self) -> None:
        data = np.load(str(self.graph_path), allow_pickle=False)
        levels      = data["levels"].tolist()
        entry_point = int(data["entry_point"][0])

        neighbors: list[list[set[int]]] = [
            [set() for _ in range(lv + 1)] for lv in levels
        ]

        currLevel = 0
        while f"node_ids_{currLevel}" in data:
            node_ids = data[f"node_ids_{currLevel}"].tolist()
            adj      = data[f"adj_{currLevel}"]
            for row, node_id in enumerate(node_ids):
                neighbors[node_id][currLevel] = {int(x) for x in adj[row] if x >= 0}
            currLevel += 1

        self._levels      = levels
        self._neighbors   = neighbors
        self._entry_point = entry_point
        self._max_layer   = max(levels) if levels else -1

    def save(self) -> None:
        """Atomically write all vectors and the graph to disk."""
        with self._lock:
            if self._vec_buf:
                arr = np.stack(self._vec_buf).astype(np.float32)
                tmp_vec = self.root_dir / "vectors.tmp.dat"
                arr.tofile(str(tmp_vec))
                self._atomic_replace(tmp_vec, self.vectors_path)

            self._save_graph_locked()

            # 3. Update metadata
            self._metadata["vector_count"] = len(self._vec_buf)
            self._save_metadata()
            self._dirty = False

    def _save_graph_locked(self) -> None:
        if not self._levels:
            return

        data: dict[str, np.ndarray] = {
            "levels":      np.array(self._levels, dtype=np.int8),
            "entry_point": np.array([self._entry_point], dtype=np.int32),
        }

        for currLevel in range(self._max_layer + 1):
            M_eff    = self.M0 if currLevel == 0 else self.M
            node_ids = [i for i, lv in enumerate(self._levels) if lv >= currLevel]
            if not node_ids:
                continue
            adj = np.full((len(node_ids), M_eff), -1, dtype=np.int32)
            for row, node_id in enumerate(node_ids):
                nbs = list(self._neighbors[node_id][currLevel])[:M_eff]
                adj[row, :len(nbs)] = nbs
            data[f"node_ids_{currLevel}"] = np.array(node_ids, dtype=np.int32)
            data[f"adj_{currLevel}"]      = adj

        # np.savez_compressed appends .npz if the path doesn't end in it,
        # so name the temp file with .npz to avoid a double-extension.
        tmp = self.root_dir / "graph.tmp.npz"
        np.savez_compressed(str(tmp), **data)  # type: ignore[arg-type]
        self._atomic_replace(tmp, self.graph_path)

    def _save_metadata(self) -> None:
        tmp = self.metadata.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._metadata, ensure_ascii=True))
        self._atomic_replace(tmp, self.metadata)

    def _atomic_replace(self, src: Path, dst: Path) -> None:
        for attempt in range(10):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                time.sleep(0.01 * (attempt + 1))
        os.replace(src, dst)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._dirty:
            self.save()

    @property
    def dimension(self) -> int:
        return int(self._metadata["dimension"])

    @property
    def vector_count(self) -> int:
        return len(self._vec_buf)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        return (x / norm).astype(np.float32) if norm > 0 else x.astype(np.float32)

    def _random_level(self) -> int:
        return int(-math.log(np.random.random()) * self._mL)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """
        Beam search on a single layer.
        Returns up to ``ef`` (cosine_distance, node_id) pairs, best first.
        """
        visited = set(entry_points)

        # candidates: min-heap of (dist, id) — pop the closest first
        # results : max-heap of (-dist, id) — pop the furthest when full
        candidates: list[tuple[float, int]] = []
        results:    list[tuple[float, int]] = []

        for ep in entry_points:
            d = 1.0 - float(self._vec_buf[ep] @ query)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst = -results[0][0] if results else float("inf")
            if c_dist > worst and len(results) >= ef:
                break

            for neighbor in self._neighbors[c_id][layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = 1.0 - float(self._vec_buf[neighbor] @ query)
                if d < -results[0][0] or len(results) < ef:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted((-nd, node_id) for nd, node_id in results)

    def _select_neighbors(
        self,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """
        HNSW heuristic neighbour selection.
        A candidate is kept only if it is closer to the base node than to every
        already-selected neighbour.This drops redundant directional edges and
        preserves long-range bridge links.Pruned candidates back-fill remaining
        slots so degree stays near M.
        """
        selected: list[tuple[float, int]] = []
        pruned:   list[tuple[float, int]] = []

        for dist, cand in candidates:
            if len(selected) >= M:
                break
            cand_vec = self._vec_buf[cand]
            keep = True
            for _, s in selected:
                if (1.0 - float(cand_vec @ self._vec_buf[s])) < dist:
                    keep = False
                    break
            if keep:
                selected.append((dist, cand))
            else:
                pruned.append((dist, cand))

        for item in pruned:
            if len(selected) >= M:
                break
            selected.append(item)

        return [node_id for _, node_id in selected]

    def insert(self, vector: np.ndarray) -> int:
        """
        Insert one vector.  Returns its assigned integer ID.
        """
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Vector must be 1-D")

        with self._lock:
            if self._vec_buf and vec.shape[0] != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, got {vec.shape[0]}"
                )
            if not self._vec_buf:
                self._metadata["dimension"] = int(vec.shape[0])

            vec = self._normalize(vec)
            node_id = len(self._vec_buf)
            self._vec_buf.append(vec)

            level = self._random_level()
            self._levels.append(level)
            self._neighbors.append([set() for _ in range(level + 1)])

            self._link_node(node_id, vec, level)
            self._dirty = True
            return node_id

    def _link_node(self, node_id: int, vec: np.ndarray, level: int) -> None:
        """Wire a newly appended node into the in-RAM graph."""
        if self._entry_point == -1:
            self._entry_point = node_id
            self._max_layer   = level
            return

        ep = [self._entry_point]

        # Greedy descent through layers above the new node's level
        for currLevel in range(self._max_layer, level, -1):
            results = self._search_layer(vec, ep, ef=1, layer=currLevel)
            ep = [results[0][1]]

        # Bidirectional wiring from the node's top layer down to 0
        for currLevel in range(min(level, self._max_layer), -1, -1):
            M_cap = self.M0 if currLevel == 0 else self.M
            candidates = self._search_layer(
                vec, ep, ef=self.ef_construction, layer=currLevel
            )
            chosen = self._select_neighbors(candidates, self.M)

            self._neighbors[node_id][currLevel] = set(chosen)

            for neighbor in chosen:
                self._neighbors[neighbor][currLevel].add(node_id)
                if len(self._neighbors[neighbor][currLevel]) > M_cap:
                    neighbor_vec = self._vec_buf[neighbor]
                    scored = sorted(
                        (1.0 - float(self._vec_buf[n] @ neighbor_vec), n)
                        for n in self._neighbors[neighbor][currLevel]
                    )
                    kept = self._select_neighbors(scored, M_cap)
                    self._neighbors[neighbor][currLevel] = set(kept)

            ep = [candidates[0][1]] if candidates else ep

        if level > self._max_layer:
            self._entry_point = node_id
            self._max_layer   = level

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        ef: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Return the ``top_k`` approximate nearest neighbours.
        Parameters
        ----------
        ef : int or None
            Search beam width at layer 0.  Larger values improve recall at the
            cost of latency.
        Returns
        -------
        list of(vector_id, cosine_similarity) sorted by closest first
        """
        if top_k <= 0 or not self._vec_buf or self._entry_point == -1:
            return []
        if ef is None:
            ef = max(top_k, 50)

        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Query vector must be 1-D")
        if vec.shape[0] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {vec.shape[0]}"
            )

        vec = self._normalize(vec)
        ep  = [self._entry_point]

        for currLevel in range(self._max_layer, 0, -1):
            results = self._search_layer(vec, ep, ef=1,layer=currLevel)
            ep = [results[0][1]]

        candidates = self._search_layer(vec, ep, ef=ef,layer=0)

        return [(node_id, 1.0 - dist) for dist, node_id in candidates[:top_k]]
