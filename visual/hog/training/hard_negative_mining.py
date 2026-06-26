import gc
import heapq
import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from visual.hog.datastructures.voc_dataset import VOCDataset
from collections import defaultdict

def _gt_overlap_mask(
    feat_map_shape,
    ch: int,
    cw: int,
    cell: float,
    scale: float,
    gt_boxes_for_class: list,
    min_iou: float,
) -> np.ndarray:
    H_cells, W_cells, _ = feat_map_shape
    n_y = H_cells - ch + 1
    n_x = W_cells - cw + 1
    y_idx = np.arange(n_y, dtype=np.float32)
    x_idx = np.arange(n_x, dtype=np.float32)
    win_x0 = (x_idx * cell * scale)[np.newaxis, :]
    win_y0 = (y_idx * cell * scale)[:, np.newaxis]
    win_x1 = win_x0 + cw * cell * scale
    win_y1 = win_y0 + ch * cell * scale
    area_win = (win_x1 - win_x0) * (win_y1 - win_y0)
    overlap = np.zeros((n_y, n_x), dtype=bool)
    for gt_box in gt_boxes_for_class:
        gx0, gy0, gx1, gy1 = gt_box
        ix0 = np.maximum(win_x0, gx0)
        iy0 = np.maximum(win_y0, gy0)
        ix1 = np.minimum(win_x1, gx1)
        iy1 = np.minimum(win_y1, gy1)
        inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
        area_gt = (gx1 - gx0) * (gy1 - gy0)
        union = area_win + area_gt - inter
        iou = inter / (union + 1e-6)
        overlap |= (iou > min_iou)
    return overlap


def _score_windows_chunked(
    feat_map: np.ndarray,
    valid_ys: np.ndarray,
    valid_xs: np.ndarray,
    comp,
    ch: int,
    cw: int,
    chunk_size: int = 2048,
) -> np.ndarray:
    _, _, feat_depth = feat_map.shape
    feat_dim = ch * cw * feat_depth
    n_valid = valid_ys.size
    scrs = np.empty(n_valid, dtype=np.float32)
    win_views = sliding_window_view(feat_map, (ch, cw, feat_depth))[:, :, 0]
    for start in range(0, n_valid, chunk_size):
        end = min(start + chunk_size, n_valid)
        ys = valid_ys[start:end]
        xs = valid_xs[start:end]
        chunk = win_views[ys, xs].reshape(end - start, feat_dim).astype(np.float32)
        scrs[start:end] = comp.svm.decision_function(chunk)
        del chunk
    del win_views
    return scrs


def _extract_feats_at_indices(
    feat_map: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    ch: int,
    cw: int,
) -> np.ndarray:
    _, _, feat_depth = feat_map.shape
    win_views = sliding_window_view(feat_map, (ch, cw, feat_depth))[:, :, 0]
    feats = win_views[ys, xs].reshape(
        len(ys), ch * cw * feat_depth).astype(np.float32)
    del win_views
    return feats


def flush_initial_negatives(detector) -> None:
    n_flushed = 0
    for comps in detector.cls_comps.values():
        for comp in comps:
            n_flushed += comp.X_bg.shape[0]
            comp.X_bg = np.array([])
    print(f"flush_initial_negatives: cleared {n_flushed} pre-SVM random "
          f"negatives from cache (epoch 1 cache reset).")


def clear_easy_negatives(detector) -> None:
    for cls in detector.classes:
        for comp in detector.cls_comps[cls]:
            comp.clear_easy_negatives(detector.hard_neg_threshold)


def mine_hard_negatives_background(
    detector,
    dataset: VOCDataset,
    max_images: int | None = None,
    score_chunk_size: int = 2048,
) -> None:
    print("  -> Mining hard background negatives")
    n = len(dataset) if max_images is None else min(len(dataset), max_images)
    indices = np.random.permutation(n)
    cell = detector.hog_descriptor.cell_size
    counter = 0
    for index in tqdm(indices, desc="Mining Negatives"):
        all_full = all(
            comp.X_bg.shape[0] >= int(comp.X_pos.shape[0] * detector.bg_multiplier)
            for comps in detector.cls_comps.values()
            for comp in comps
        )
        if all_full:
            break
        img = dataset.get_image(index)
        if img is None:
            continue
        boxes, lbls = dataset.get_annotation(index)
        p = detector.hog_descriptor.compute_feature_pyramid(img)
        del img
        for cls, comps in detector.cls_comps.items():
            gt_boxes_cls = [
                box for box, lbl in zip(boxes, lbls) if lbl == cls
            ]
            for comp in comps:
                n_pos = comp.X_pos.shape[0]
                n_bg_target = int(n_pos * detector.bg_multiplier)
                n_needed = n_bg_target - comp.X_bg.shape[0]
                if n_needed <= 0:
                    continue
                take_at_most = min(n_needed, detector.max_hard_per_image)
                ch, cw = comp.cell_h, comp.cell_w
                candidates: list[tuple[float, int, int, int, int]] = []
                candidate_seq = 0
                for level_idx, level in enumerate(p):
                    feat_map = level.feature_map
                    H_cells, W_cells, feat_depth = feat_map.shape
                    scale = level.scale
                    if H_cells < ch or W_cells < cw:
                        continue
                    if gt_boxes_cls:
                        gt_overlap = _gt_overlap_mask(
                            feat_map.shape, ch, cw, cell, scale,
                            gt_boxes_cls, detector.min_iou_between_gt_and_latent,
                        )
                    else:
                        n_y = H_cells - ch + 1
                        n_x = W_cells - cw + 1
                        gt_overlap = np.zeros((n_y, n_x), dtype=bool)
                    valid_ys, valid_xs = np.where(~gt_overlap)
                    del gt_overlap
                    if valid_ys.size == 0:
                        continue
                    scrs = _score_windows_chunked(
                        feat_map, valid_ys, valid_xs, comp, ch, cw,
                        chunk_size=score_chunk_size,
                    )
                    hard_mask = scrs > detector.hard_neg_threshold
                    if not hard_mask.any():
                        del scrs, valid_ys, valid_xs
                        continue
                    hard_idx = np.where(hard_mask)[0]
                    hard_scores = scrs[hard_idx]
                    hard_ys = valid_ys[hard_idx]
                    hard_xs = valid_xs[hard_idx]
                    del scrs, valid_ys, valid_xs, hard_mask, hard_idx
                    for s, iy, ix in zip(hard_scores.tolist(), hard_ys.tolist(), hard_xs.tolist()):
                        item = (s, candidate_seq, level_idx, int(iy), int(ix))
                        candidate_seq += 1
                        if len(candidates) < take_at_most:
                            heapq.heappush(candidates, item)
                        elif s > candidates[0][0]:
                            heapq.heapreplace(candidates, item)
                    del hard_scores, hard_ys, hard_xs
                if not candidates:
                    continue
                by_level: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
                for sc, seq, level_idx, iy, ix in candidates:
                    by_level[level_idx].append((sc, iy, ix))
                del candidates
                new_feats: list[np.ndarray] = []
                for level_idx, items in by_level.items():
                    feat_map = p[level_idx].feature_map
                    ys_arr = np.array([it[1] for it in items], dtype=np.intp)
                    xs_arr = np.array([it[2] for it in items], dtype=np.intp)
                    feats = _extract_feats_at_indices(feat_map, ys_arr, xs_arr, ch, cw)
                    new_feats.append(feats)
                    del feats, ys_arr, xs_arr
                if new_feats:
                    block = np.vstack(new_feats, dtype=np.float16)
                    if comp.X_bg.shape[0] == 0:
                        comp.X_bg = block
                    else:
                        comp.X_bg = np.vstack([comp.X_bg, block])
                    del block
                del new_feats, by_level
        del p, boxes, lbls
        counter += 1
        if counter % 50 == 0:
            gc.collect()
    for comps in detector.cls_comps.values():
        for comp in comps:
            print(f"{comp.cls_name}-{comp.id}: {comp.X_bg.shape[0]} hard bg negatives")


def resample_other_class_positives(
    detector,
    pos_patches: dict[str, list[np.ndarray]],
) -> None:
    n_other_classes = len(detector.classes) - 1
    per_class_other_ratio = (detector.other_classes_total_ratio / max(1, n_other_classes))
    for cls in detector.classes:
        for comp in detector.cls_comps[cls]:
            n_other_per_cls = max(1, int(comp.X_pos.shape[0] * per_class_other_ratio))
            tot_resam = 0
            other_feats = []
            for other_cls in detector.classes:
                if other_cls == cls:
                    continue
                other_patches = pos_patches[other_cls]
                n_sample = min(len(other_patches), n_other_per_cls)
                if n_sample == 0:
                    continue
                indices = np.random.choice(
                    len(other_patches), size=n_sample, replace=False)
                for i in indices:
                    resized = cv2.resize(other_patches[i], (comp.pixel_w, comp.pixel_h))
                    feat = detector._extract_custom_hog(resized)
                    if (comp.svm.decision_function(feat.reshape(1, -1))[0] > detector.hard_neg_threshold):
                        other_feats.append(feat)
                        tot_resam += 1
                    del resized, feat
            comp.X_pos_other_classes = (np.array(other_feats) if other_feats else np.array([]))
            print(f" Refreshed other-class positives: '{cls}' comp {comp.id} - {tot_resam} samples retained.")


def mine_hard_negatives_pyramid(
    detector,
    train_ds: VOCDataset,
    max_images: int | None,
    pos_patches: dict[str, list[np.ndarray]],
    score_chunk_size: int = 2048,
    do_flush_initial: bool = False,
) -> None:
    if do_flush_initial:
        flush_initial_negatives(detector)
    else:
        clear_easy_negatives(detector)
    mine_hard_negatives_background(
        detector, train_ds, max_images,
        score_chunk_size=score_chunk_size,
    )
    resample_other_class_positives(detector, pos_patches)
    for cls, comps in detector.cls_comps.items():
        for comp in comps:
            tot_hard = (comp.X_bg.shape[0] + comp.X_pos_other_classes.shape[0])
            print(f"[{cls}] comp {comp.id}: {tot_hard} hard negatives in cache")