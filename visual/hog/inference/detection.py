import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from visual.hog.datastructures.bbox_regressor import BBoxRegressor


def _score_level(feat_map, comp, step):
    ch, cw = comp.cell_h, comp.cell_w
    _, _, feat_depth = feat_map.shape
    windows = sliding_window_view(feat_map, (ch, cw, feat_depth))[:, :, 0]
    windows = windows[::step, ::step]
    n_ys, n_xs = windows.shape[:2]
    X = windows.reshape(n_ys * n_xs, ch * cw * feat_depth).astype(np.float32, copy=False)
    if comp.cal is None:
        scores_all = comp.svm.decision_function(X)
        # Use (Eq. 30 normalisation) to convert SVM decision values to pseudo-probabilities in [0, 1] using sigmoid.
        scores_all = 1.0 / (1.0 + np.exp(-2.0 * scores_all))
    else:
        scores_all = comp.cal.predict_proba(X)[:, 1]
    return scores_all, X, n_xs, n_ys


def _boxes_from_hits(hit_flat, n_xs, step, level_scale, comp_ch, comp_cw, cell_size):
    iy = hit_flat // n_xs
    ix = hit_flat % n_xs
    y_cells = iy * step
    x_cells = ix * step
    cell = cell_size
    scale = level_scale
    x0s = (x_cells * cell * scale).astype(int)
    y0s = (y_cells * cell * scale).astype(int)
    x1s = ((x_cells + comp_cw) * cell * scale).astype(int)
    y1s = ((y_cells + comp_ch) * cell * scale).astype(int)
    return list(zip(x0s.tolist(), y0s.tolist(), x1s.tolist(), y1s.tolist()))


def multiscale_detect_one_comp(detector, pyramid, comp, internal_threshold: float = 0.05):
    detections, det_scores = [], []
    bbox_regressor = comp.bbox_regressor if comp.bbox_regressor.fitted else None
    ch = comp.cell_h
    cw = comp.cell_w
    step = detector.pyramid_step
    cell_size = comp.cell_size
    for level in pyramid:
        feat_map = level.feature_map
        H_cells, W_cells, _ = feat_map.shape
        if H_cells < ch or W_cells < cw:
            continue
        scores_all, X, n_xs, n_ys = _score_level(feat_map, comp, step)
        hit_flat = np.where(scores_all > internal_threshold)[0]
        if hit_flat.size == 0:
            del scores_all, X
            continue

        boxes = _boxes_from_hits(hit_flat, n_xs, step, level.scale, ch, cw, cell_size)
        hit_scores = scores_all[hit_flat].tolist()
        if bbox_regressor is not None:
            hit_feats = X[hit_flat]
            all_deltas = bbox_regressor.predict(hit_feats)
            boxes = BBoxRegressor.apply_deltas_batch(boxes, all_deltas)
        detections.extend(boxes)
        det_scores.extend(hit_scores)
        del scores_all, X
    return detections, det_scores


def non_max_suppression(boxes, scores, overlap_threshold: float):
    if not boxes:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    areas = (x1 - x0) * (y1 - y0)
    order = scores.argsort()[::-1]
    kept_boxes, kept_scores = [], []
    while order.size > 0:
        kept_boxes.append(boxes[order[0]])
        kept_scores.append(scores[order[0]])
        xx0 = np.maximum(x0[order[0]], x0[order[1:]])
        yy0 = np.maximum(y0[order[0]], y0[order[1:]])
        xx1 = np.minimum(x1[order[0]], x1[order[1:]])
        yy1 = np.minimum(y1[order[0]], y1[order[1:]])
        w = np.maximum(0, xx1 - xx0)
        h = np.maximum(0, yy1 - yy0)
        inter_area = w * h
        union_area = areas[order[0]] + areas[order[1:]] - inter_area
        iou = inter_area / (union_area + 1e-6)
        inds_to_keep = np.where(iou <= overlap_threshold)[0]
        order = order[inds_to_keep + 1]
    del x0, y0, x1, y1, areas, order
    return kept_boxes, kept_scores


def detect(
    detector,
    image,
    overlap_threshold: float = 0.3,
    pyramid_lambda=None,
):
    all_boxes, all_scores, all_labels = [], [], []
    if not detector.trained_flag:
        print("Model must be trained before making predictions.")
        return all_boxes, all_scores, all_labels
    pyramid = detector.hog_descriptor.compute_feature_pyramid(image, pyramid_lambda)
    tasks = [(cls, comp) for cls, comps in detector.cls_comps.items() for comp in comps]
    def _score_comp(task):
        cls, comp = task
        dets, sc = multiscale_detect_one_comp(detector, pyramid, comp)
        return cls, dets, sc
    n_workers = min(len(tasks), 2)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        comp_results = list(executor.map(_score_comp, tasks))
    cls_dets_map = defaultdict(lambda: ([], []))
    for cls, dets, sc in comp_results:
        if dets:
            cls_dets_map[cls][0].extend(dets)
            cls_dets_map[cls][1].extend(sc)

    for cls, (cls_dets, cls_scores) in cls_dets_map.items():
        filtered_dets, filtered_scores = non_max_suppression(
            cls_dets, cls_scores, overlap_threshold
        )
        all_boxes.extend(filtered_dets)
        all_scores.extend(filtered_scores)
        all_labels.extend([cls] * len(filtered_dets))

    if all_boxes:
        order = np.argsort(all_scores)[::-1]
        all_boxes = [all_boxes[i] for i in order]
        all_scores = [all_scores[i] for i in order]
        all_labels = [all_labels[i] for i in order]
    del pyramid
    return all_boxes, all_scores, all_labels