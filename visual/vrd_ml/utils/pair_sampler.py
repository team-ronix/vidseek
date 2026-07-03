import numpy as np


def _pair_distance(subj, obj):
    sx = (subj.bbox.x1 + subj.bbox.x2) / 2.0
    sy = (subj.bbox.y1 + subj.bbox.y2) / 2.0
    ox = (obj.bbox.x1 + obj.bbox.x2) / 2.0
    oy = (obj.bbox.y1 + obj.bbox.y2) / 2.0
    return float(((sx - ox) ** 2 + (sy - oy) ** 2) ** 0.5)


def enumerate_pairs(
    objects,
    min_score=0.0,
    max_pairs=None,
    exclude_same=True,
    min_iou=0.0,
    require_overlap=False,
    verbose=False,
):
    filtered = [o for o in objects if o.score >= min_score]
    pairs = []
    for i, subj in enumerate(filtered):
        for j, obj in enumerate(filtered):
            if exclude_same == True and i == j:
                continue
            iou = subj.bbox.iou(obj.bbox)
            if require_overlap == True and iou == 0.0:
                continue
            if iou < min_iou:
                continue
            pairs.append((subj, obj))
    pairs.sort(key=lambda p: -(p[0].score * p[1].score))
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    scores = [o.score for o in filtered]
    scores_are_uniform = bool(scores) and (max(scores) - min(scores)) < 1e-9
    if scores_are_uniform == True:
        if verbose == True:
            print(
                f"[pair_sampler] All object scores are identical. Using spatial ranking "
                f"to reduce {len(pairs)} pairs to {max_pairs}."
            )
        pairs.sort(key=lambda p: _pair_distance(p[0], p[1]))
    return pairs[:max_pairs]


def _obj_key(o):
    return (o.label, o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2)


def sample_training_pairs(
    objects,
    gt_relations,
    neg_ratio=3.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    pos_pairs = []
    pos_labels = []
    gt_set = set()
    for rel in gt_relations:
        pos_pairs.append((rel.subject, rel.object_))
        pos_labels.append(rel.predicate_idx)
        gt_set.add((_obj_key(rel.subject), _obj_key(rel.object_)))
    neg_pool = []
    for i, subj in enumerate(objects):
        for j, obj in enumerate(objects):
            if i == j:
                continue
            if (_obj_key(subj), _obj_key(obj)) in gt_set:
                continue
            neg_pool.append((subj, obj))
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
