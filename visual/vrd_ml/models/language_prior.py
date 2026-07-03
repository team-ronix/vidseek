from collections import defaultdict
from visual.vrd_ml.features.semantic import SemanticFeatureExtractor


class LanguagePrior:
    def __init__(
        self,
        predicate_classes,
        glove_path=None,
        smoothing=1.0,
        min_pair_count=2,
        zero_shot_top_k=5,
        zero_shot_min_sim=0.5,
    ):
        self.predicate_classes = predicate_classes
        self.n_pred = max(len(predicate_classes), 1)
        self.smoothing = smoothing
        self.min_pair_count = min_pair_count
        self.zero_shot_top_k = zero_shot_top_k
        self.zero_shot_min_sim = zero_shot_min_sim
        self.glove_path = glove_path
        self.pair_counts = defaultdict(lambda: defaultdict(int))
        self.subj_counts = defaultdict(lambda: defaultdict(int))
        self.obj_counts = defaultdict(lambda: defaultdict(int))
        self.global_counts = defaultdict(int)
        self.pair_totals = defaultdict(int)
        self.subj_totals = defaultdict(int)
        self.obj_totals = defaultdict(int)
        self.global_total = 0
        self._known_pairs = []
        self._is_fitted = False
        self._semantic = SemanticFeatureExtractor(glove_path=glove_path)

    def fit(self, dataset):
        for ann in dataset:
            for rel in ann.relationships:
                s, p, o = rel.subject.label, rel.predicate, rel.object_.label
                self.pair_counts[(s, o)][p] += 1
                self.subj_counts[s][p] += 1
                self.obj_counts[o][p] += 1
                self.global_counts[p] += 1
                self.pair_totals[(s, o)] += 1
                self.subj_totals[s] += 1
                self.obj_totals[o] += 1
                self.global_total += 1
        self._known_pairs = list(self.pair_counts.keys())
        self._is_fitted = True
        return self


    def score(self, subj_label, obj_label, predicate):
        if self._is_fitted != True:

            return 1.0
        key = (subj_label, obj_label)
        pair_total = self.pair_totals.get(key, 0)
        # if we've seen this pair enough times, use pair-level statistics
        if pair_total >= self.min_pair_count:
            count = self.pair_counts[key].get(predicate, 0)
            return self._smoothed(count, pair_total)
        # otherwise use subject/object marginals
        s_total = self.subj_totals.get(subj_label, 0)
        o_total = self.obj_totals.get(obj_label, 0)
        if s_total > 0 or o_total > 0:
            probs = []
            if s_total > 0:
                probs.append(self._smoothed(self.subj_counts[subj_label].get(predicate, 0), s_total))
            if o_total > 0:
                probs.append(self._smoothed(self.obj_counts[obj_label].get(predicate, 0), o_total))
            if pair_total > 0:
                probs.append(self._smoothed(self.pair_counts[key].get(predicate, 0), pair_total))
            return sum(probs) / len(probs)
        # zero-shot: find similar known pairs and use their statistics
        neighbor_prob = self._zero_shot_score(subj_label, obj_label, predicate)
        if neighbor_prob != None:
            return neighbor_prob
        return self._smoothed(self.global_counts.get(predicate, 0), self.global_total)

    def _smoothed(self, count, total):
        return (count + self.smoothing) / (total + self.smoothing * self.n_pred)

    def _zero_shot_score(self, subj_label, obj_label, predicate):
        if not self._known_pairs:
            return None
        neighbors = self._semantic.zero_shot_nearest(
            subj_label, obj_label, self._known_pairs, top_k=self.zero_shot_top_k
        )
        # only use neighbors with enough similarity
        neighbors = [(pair, sim) for pair, sim in neighbors if sim >= self.zero_shot_min_sim]
        if not neighbors:
            return None
        # weighted average of neighbor probabilities
        weighted_count, weight_total = 0.0, 0.0
        for (s, o), sim in neighbors:
            total = self.pair_totals[(s, o)]
            if total == 0:
                continue
            p = self.pair_counts[(s, o)].get(predicate, 0) / total
            weighted_count += sim * p
            weight_total += sim
        if weight_total == 0:
            return None
        raw = weighted_count / weight_total
        return max(raw, 1.0 / (self.n_pred * 10))

    def state_dict(self):
        return {
            "predicate_classes": self.predicate_classes,
            "smoothing": self.smoothing,
            "min_pair_count": self.min_pair_count,
            "zero_shot_top_k": self.zero_shot_top_k,
            "zero_shot_min_sim": self.zero_shot_min_sim,
            "pair_counts": {k: dict(v) for k, v in self.pair_counts.items()},
            "subj_counts": {k: dict(v) for k, v in self.subj_counts.items()},
            "obj_counts": {k: dict(v) for k, v in self.obj_counts.items()},
            "global_counts": dict(self.global_counts),
            "pair_totals": dict(self.pair_totals),
            "subj_totals": dict(self.subj_totals),
            "obj_totals": dict(self.obj_totals),
            "global_total": self.global_total,
            "known_pairs": self._known_pairs,
        }

    @classmethod
    def from_state_dict(cls, state, glove_path=None):
        obj = cls(
            predicate_classes=state["predicate_classes"],
            glove_path=glove_path,
            smoothing=state["smoothing"],
            min_pair_count=state["min_pair_count"],
            zero_shot_top_k=state["zero_shot_top_k"],
            zero_shot_min_sim=state["zero_shot_min_sim"],
        )
        obj.pair_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in state["pair_counts"].items()},
        )
        obj.subj_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in state["subj_counts"].items()},
        )
        obj.obj_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in state["obj_counts"].items()},
        )
        obj.global_counts = defaultdict(int, state["global_counts"])
        obj.pair_totals = defaultdict(int, state["pair_totals"])
        obj.subj_totals = defaultdict(int, state["subj_totals"])
        obj.obj_totals = defaultdict(int, state["obj_totals"])
        obj.global_total = state["global_total"]
        obj._known_pairs = state["known_pairs"]
        obj._is_fitted = True
        return obj
