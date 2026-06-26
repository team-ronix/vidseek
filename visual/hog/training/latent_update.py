import gc
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from visual.hog.datastructures.voc_dataset import VOCDataset


def _compute_window_iou_mask(
    feat_map_shape,
    ch: int,
    cw: int,
    cell: float,
    scale: float,
    gt_box,
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
    gx0, gy0, gx1, gy1 = gt_box
    ix0 = np.maximum(win_x0, gx0)
    iy0 = np.maximum(win_y0, gy0)
    ix1 = np.minimum(win_x1, gx1)
    iy1 = np.minimum(win_y1, gy1)
    inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
    area_win = (win_x1 - win_x0) * (win_y1 - win_y0)
    area_gt = (gx1 - gx0) * (gy1 - gy0)
    union = area_win + area_gt - inter
    iou = inter / (union + 1e-6)
    return iou >= min_iou, iou



def update_positive_latents(
    detector,
    dataset: VOCDataset,
    max_images: int | None = None,
) -> None:
    print(" -> Updating positive latent assignments via pyramid search")
    n = len(dataset) if max_images is None else min(len(dataset), max_images)
    indices = np.arange(n)
    for comps in detector.cls_comps.values():
        for comp in comps:
            comp.X_pos = np.array([])
            comp.bbr_X = np.array([])
            comp.bbr_y = np.array([])
    cell = detector.hog_descriptor.cell_size
    min_iou = detector.min_iou_between_gt_and_latent
    fb_iou = max(0.1, min_iou / 2.0)
    acc_X_pos: dict = {}
    acc_bbr_X: dict = {}
    acc_bbr_y: dict = {}
    for cls, comps in detector.cls_comps.items():
        for comp in comps:
            key = (cls, comp.id)
            acc_X_pos[key] = []
            acc_bbr_X[key] = []
            acc_bbr_y[key] = []
    counter = 0
    for idx in tqdm(indices, desc="Latent Update"):
        img = dataset.get_image(idx)
        if img is None:
            continue
        boxes, lbls = dataset.get_annotation(idx)
        p = detector.hog_descriptor.compute_feature_pyramid(img)
        for gt_box, label in zip(boxes, lbls):
            if label not in detector.classes:
                continue
            gt_cx = (gt_box[0] + gt_box[2]) / 2.0
            gt_cy = (gt_box[1] + gt_box[3]) / 2.0
            gt_w = max(1e-5, gt_box[2] - gt_box[0])
            gt_h = max(1e-5, gt_box[3] - gt_box[1])
            for comp in detector.cls_comps[label]:
                key = (label, comp.id)
                ch, cw = comp.cell_h, comp.cell_w
                comp_best_score = -float("inf")
                comp_best_feat = None
                comp_best_bbr_target = None
                fb_best_iou = -1.0
                fb_best_score = -float("inf")
                fb_best_feat = None
                fb_best_bbr = None
                for level in p:
                    feat_map = level.feature_map
                    H_cells, W_cells, feat_depth = feat_map.shape
                    scale = level.scale
                    if H_cells < ch or W_cells < cw:
                        continue
                    iou_mask, iou_vals = _compute_window_iou_mask(
                        feat_map.shape, ch, cw, cell, scale,
                        gt_box, min_iou
                    )
                    max_iou_here = float(iou_vals.max()) if iou_vals.size > 0 else 0.0
                    if max_iou_here >= fb_iou and max_iou_here > fb_best_iou:
                        fb_y, fb_x = np.unravel_index(int(np.argmax(iou_vals)), iou_vals.shape)
                        fb_feat_depth = feat_depth
                        windows_full_fb = sliding_window_view(feat_map, (ch, cw, fb_feat_depth))[:, :, 0]
                        fb_feat_raw = windows_full_fb[fb_y, fb_x].reshape(ch * cw * fb_feat_depth).astype(np.float32)
                        fb_score = float(comp.svm.decision_function(fb_feat_raw.reshape(1, -1))[0])
                        win_x0_px = fb_x * cell * scale
                        win_y0_px = fb_y * cell * scale
                        win_x1_px = win_x0_px + cw * cell * scale
                        win_y1_px = win_y0_px + ch * cell * scale
                        win_cx = (win_x0_px + win_x1_px) / 2.0
                        win_cy = (win_y0_px + win_y1_px) / 2.0
                        win_w = max(1e-5, win_x1_px - win_x0_px)
                        win_h = max(1e-5, win_y1_px - win_y0_px)
                        fb_bbr = [
                            (gt_cx - win_cx) / win_w,
                            (gt_cy - win_cy) / win_h,
                            np.log(gt_w / win_w),
                            np.log(gt_h / win_h),
                        ]
                        fb_best_iou = max_iou_here
                        fb_best_score = fb_score
                        fb_best_feat = fb_feat_raw
                        fb_best_bbr = fb_bbr
                        del windows_full_fb, fb_feat_raw
                    valid_ys, valid_xs = np.where(iou_mask)
                    if valid_ys.size == 0:
                        continue
                    windows_full = sliding_window_view(
                        feat_map, (ch, cw, feat_depth))[:, :, 0]
                    valid_windows = windows_full[valid_ys, valid_xs]
                    X_valid = valid_windows.reshape(
                        valid_ys.size, ch * cw * feat_depth).astype(np.float32)
                    batch_scores = comp.svm.decision_function(X_valid)
                    best_idx = int(np.argmax(batch_scores))
                    if batch_scores[best_idx] > comp_best_score:
                        comp_best_score = float(batch_scores[best_idx])
                        comp_best_feat = X_valid[best_idx].copy()
                        wy = int(valid_ys[best_idx])
                        wx = int(valid_xs[best_idx])
                        win_x0_px = wx * cell * scale
                        win_y0_px = wy * cell * scale
                        win_x1_px = win_x0_px + cw * cell * scale
                        win_y1_px = win_y0_px + ch * cell * scale
                        win_cx = (win_x0_px + win_x1_px) / 2.0
                        win_cy = (win_y0_px + win_y1_px) / 2.0
                        win_w = max(1e-5, win_x1_px - win_x0_px)
                        win_h = max(1e-5, win_y1_px - win_y0_px)
                        comp_best_bbr_target = [
                            (gt_cx - win_cx) / win_w,
                            (gt_cy - win_cy) / win_h,
                            np.log(gt_w / win_w),
                            np.log(gt_h / win_h),
                        ]
                    del X_valid, batch_scores, valid_windows, windows_full
                if comp_best_feat is not None:
                    acc_X_pos[key].append(comp_best_feat.astype(np.float16))
                    acc_bbr_X[key].append(comp_best_feat)
                    acc_bbr_y[key].append(np.array(comp_best_bbr_target, dtype=np.float32))
                elif fb_best_feat is not None:
                    acc_X_pos[key].append(fb_best_feat.astype(np.float16))
                    acc_bbr_X[key].append(fb_best_feat)
                    acc_bbr_y[key].append(np.array(fb_best_bbr, dtype=np.float32))
        del img, p, boxes, lbls
        counter += 1
        if counter % 50 == 0:
            gc.collect()
    for cls, comps in detector.cls_comps.items():
        for comp in comps:
            key = (cls, comp.id)
            rows_pos = acc_X_pos[key]
            rows_bbr = acc_bbr_X[key]
            rows_y = acc_bbr_y[key]
            comp.X_pos = (np.vstack(rows_pos) if rows_pos else np.empty((0,), dtype=np.float16))
            comp.bbr_X = (np.vstack(rows_bbr) if rows_bbr else np.empty((0,), dtype=np.float32))
            comp.bbr_y = (np.vstack(rows_y) if rows_y else np.empty((0, 4), dtype=np.float32))


def fit_bbox_regs(detector) -> None:
    print("-> Fitting bounding-box regressors on latent offsets")
    for cls, comps in detector.cls_comps.items():
        for comp in comps:
            if comp.bbr_X.shape[0] > 0:
                comp.bbox_reg.fit(comp.bbr_X, comp.bbr_y)
                comp.bbox_reg.fitted = True
            else:
                print(f"Warning: No BBR targets for '{cls}' component {comp.id}.")