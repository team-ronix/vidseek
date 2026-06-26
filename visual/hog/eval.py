from collections import defaultdict
from tqdm import tqdm
import numpy as np
from visual.hog.utils import calculate_iou
import pickle
import os



def _voc_ap_11pt(scores, tp_flags, n_gt) -> float:
    if n_gt == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    tp_cum = np.cumsum(np.array(tp_flags)[order])
    fp_cum = np.cumsum(1 - np.array(tp_flags)[order])
    rec = tp_cum / (n_gt + 1e-6)
    prec = tp_cum / (tp_cum + fp_cum + 1e-6)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return float(ap)


def evaluate_hog_detection_ap(
    detector,
    dataset,
    max_imgs=None,
    iou_match_thresh: float = 0.5,
    nms_iou_thresh: float = 0.3,
    score_thresh: float = 0.05,
    split_name: str = "Test",
    pyramid_lambda=None,
    use_context: bool = True,
    checkpoint_path=None,
    checkpoint_every: int = 50,
):
    active = [c for c in detector.classes if detector.cls_comps.get(c)]
    n = len(dataset) if max_imgs is None else min(max_imgs, len(dataset))
    det_scores: dict = defaultdict(list)
    det_tp: dict = defaultdict(list)
    n_gt: dict = defaultdict(int)
    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        det_scores = defaultdict(list, ckpt["det_scores"])
        det_tp = defaultdict(list, ckpt["det_tp"])
        n_gt = defaultdict(int, ckpt["n_gt"])
        start_idx = ckpt["last_idx"] + 1
        print(f"[Checkpoint] Resuming from image {start_idx} ({checkpoint_path})")
    processed = 0
    for idx in tqdm(range(start_idx, n), desc=f"Eval {split_name}", initial=start_idx, total=n):
        img = dataset.get_image(idx)
        if img is None:
            processed += 1
            continue
        gt_boxes, gt_labels = dataset.get_annotation(idx)
        gt_by_class: dict = defaultdict(list)
        for box, lbl in zip(gt_boxes, gt_labels):
            if lbl in active:
                gt_by_class[lbl].append(list(box))
                n_gt[lbl] += 1
        boxes, scores, labels = detector.detect(
            img,
            threshold=score_thresh,
            overlap_threshold=nms_iou_thresh,
            pyramid_lambda=pyramid_lambda,
            use_context=use_context,
        )
        dets_by_class: dict = defaultdict(list)
        for box, score, lbl in zip(boxes, scores, labels):
            if lbl in active:
                dets_by_class[lbl].append((score, list(box)))
        for cls in active:
            dets_by_class[cls].sort(key=lambda t: t[0], reverse=True)
            gt_matched = [False] * len(gt_by_class[cls])
            for score, det_box in dets_by_class[cls]:
                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_by_class[cls]):
                    if gt_matched[j]:
                        continue
                    iou = calculate_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_match_thresh and best_j >= 0:
                    gt_matched[best_j] = True
                    det_tp[cls].append(1)
                else:
                    det_tp[cls].append(0)
                det_scores[cls].append(score)
        del img, gt_boxes, gt_labels, gt_by_class, boxes, scores, labels
        processed += 1
        if checkpoint_path and processed % checkpoint_every == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "det_scores": dict(det_scores),
                    "det_tp": dict(det_tp),
                    "n_gt": dict(n_gt),
                    "last_idx": idx,
                }, f)
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    ap_per_class = {
        cls: _voc_ap_11pt(det_scores[cls], det_tp[cls], n_gt[cls])
        for cls in active
    }
    valid = [v for v in ap_per_class.values() if not np.isnan(v)]
    mean_ap = float(np.mean(valid)) if valid else 0.0
    return ap_per_class, mean_ap