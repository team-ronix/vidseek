from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2


def load_image(src) -> np.ndarray | None:
    if isinstance(src, (str, Path)):
        return cv2.imread(str(src))
    return src


def calculate_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


def apply_deltas_batch(boxes: list, all_deltas: np.ndarray, clip_to: Optional[Tuple[int, int]] = None) -> list:
    n = len(boxes)
    if n == 0:
        return boxes
    b = np.array(boxes, dtype=np.float64)
    p_w = b[:, 2] - b[:, 0]
    p_h = b[:, 3] - b[:, 1]
    p_cx = b[:, 0] + 0.5 * p_w
    p_cy = b[:, 1] + 0.5 * p_h
    tx = all_deltas[:, 0]
    ty = all_deltas[:, 1]
    tw = np.clip(all_deltas[:, 2], -4.0, 4.0)
    th = np.clip(all_deltas[:, 3], -4.0, 4.0)
    gt_cx = tx * p_w + p_cx
    gt_cy = ty * p_h + p_cy
    gt_w = np.exp(tw) * p_w
    gt_h = np.exp(th) * p_h
    x0s = np.round(gt_cx - 0.5 * gt_w).astype(int)
    y0s = np.round(gt_cy - 0.5 * gt_h).astype(int)
    x1s = np.round(gt_cx + 0.5 * gt_w).astype(int)
    y1s = np.round(gt_cy + 0.5 * gt_h).astype(int)
    if clip_to is not None:
        img_w, img_h = clip_to
        x0s = np.clip(x0s, 0, img_w - 1)
        y0s = np.clip(y0s, 0, img_h - 1)
        x1s = np.clip(x1s, 0, img_w)
        y1s = np.clip(y1s, 0, img_h)
    valid = (x1s > x0s) & (y1s > y0s)
    result = []
    for i in range(n):
        if valid[i]:
            result.append((int(x0s[i]), int(y0s[i]), int(x1s[i]), int(y1s[i])))
        else:
            result.append(boxes[i])
    return result