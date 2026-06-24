from typing import List, Tuple, Optional
import numpy as np
from visual.vrd_ml.vrd_dataset import DetectedObject, RelationshipTriplet


def _pair_distance(subj: DetectedObject, obj: DetectedObject) -> float:
    sx = (subj.bbox.x1 + subj.bbox.x2) / 2.0
    sy = (subj.bbox.y1 + subj.bbox.y2) / 2.0
    ox = (obj.bbox.x1 + obj.bbox.x2) / 2.0
    oy = (obj.bbox.y1 + obj.bbox.y2) / 2.0
    return float(((sx - ox) ** 2 + (sy - oy) ** 2) ** 0.5)


def enumerate_pairs(
    objects: List[DetectedObject],
    min_score: float = 0.0,
    max_pairs: Optional[int] = None,
    exclude_same: bool = True,
    min_iou: float = 0.0,
    require_overlap: bool = False,
    verbose: bool = False,
) -> List[Tuple[DetectedObject, DetectedObject]]:
    # filter by minimum confidence
    filtered = [o for o in objects if o.score >= min_score]
    pairs: List[Tuple[DetectedObject, DetectedObject]] = []
    for i, subj in enumerate(filtered):
        for j, obj in enumerate(filtered):
            if exclude_same and i == j:
                continue
            iou = subj.bbox.iou(obj.bbox)
            if require_overlap and iou == 0.0:
                continue
            if iou < min_iou:
                continue
            pairs.append((subj, obj))

    # always sort by combined detection score descending - meaningful
    # whenever scores actually vary, harmless (stable) when they don't.
    pairs.sort(key=lambda p: -(p[0].score * p[1].score))

    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs

    scores = [o.score for o in filtered]
    scores_are_uniform = bool(scores) and (max(scores) - min(scores)) < 1e-9

    if scores_are_uniform:
        if verbose:
            print(
                f"[pair_sampler] all {len(filtered)} object scores are "
                f"identical ({scores[0] if scores else 'n/a'}) - score-based "
                f"ranking carries no information here. Falling back to "
                f"spatial-proximity ranking before truncating "
                f"{len(pairs)} candidate pairs down to max_pairs={max_pairs}. "
                f"If these are ground-truth boxes, consider max_pairs=None "
                f"to evaluate every pair instead."
            )
        pairs.sort(key=lambda p: _pair_distance(p[0], p[1]))

    return pairs[:max_pairs]


def _obj_key(o: DetectedObject) -> Tuple:
    return (o.label, o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2)


def sample_training_pairs(
    objects: List[DetectedObject],
    gt_relations: list[RelationshipTriplet],
    neg_ratio: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[DetectedObject, DetectedObject]], List[int]]:
    if rng is None:
        rng = np.random.default_rng()

    # build positive set
    pos_pairs: List[Tuple[DetectedObject, DetectedObject]] = []
    pos_labels: List[int] = []

    # use stable (label, x1, y1, x2, y2) key instead of id() so that
    # deduplication is correct even when DetectedObject instances are rebuilt.
    gt_set = set()
    for rel in gt_relations:
        pos_pairs.append((rel.subject, rel.object_))
        pos_labels.append(rel.predicate_idx)
        gt_set.add((_obj_key(rel.subject), _obj_key(rel.object_)))

    # build negative pool - all pairs not in gt_set
    neg_pool: List[Tuple[DetectedObject, DetectedObject]] = []
    for i, subj in enumerate(objects):
        for j, obj in enumerate(objects):
            if i == j:
                continue
            if (_obj_key(subj), _obj_key(obj)) in gt_set:
                continue
            neg_pool.append((subj, obj))

    # sample negatives
    n_neg = min(int(len(pos_pairs) * neg_ratio), len(neg_pool))
    if n_neg > 0 and neg_pool:
        indices = rng.choice(len(neg_pool), size=n_neg, replace=False)
        neg_pairs = [neg_pool[k] for k in indices]
        neg_labels = [-1] * n_neg
    else:
        neg_pairs = []
        neg_labels = []

    all_pairs = pos_pairs + neg_pairs
    all_labels = pos_labels + neg_labels
    return all_pairs, all_labels
