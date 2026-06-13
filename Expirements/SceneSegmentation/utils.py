import numpy as np


def bgr_to_hsv(img):
    bgr = img.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    stack = np.stack([r, g, b], axis=-1)
    cmax = stack.max(axis=-1)
    cmin = stack.min(axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)

    mask_r = (cmax == r) & (delta != 0)

    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)


    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    s = delta / np.maximum(cmax, 1e-12)
    h /= 2
    s *= 255

    cmax *= 255
    return np.stack([h.astype(np.uint8), s.astype(np.uint8), cmax.astype(np.uint8)], axis=-1)


def calc_hist(hsv_channel, n_bins, value_range):
    ch = hsv_channel.ravel()
    return np.histogram(ch, bins=n_bins, range=value_range)[0].astype(np.float32)


def intersection(gt, pred):
    return max(min(gt[1], pred[1]) - max(gt[0], pred[0]), 0)

def iou(gt, pred):
    i = intersection(gt, pred)
    return i / ((pred[1] - pred[0]) + (gt[1] - gt[0]) - i + 1e-32)

def precision_interval(gt, pred):
    return intersection(gt, pred) / ((pred[1] - pred[0]) + 1e-32)

def recall_interval(gt, pred):
    return intersection(gt, pred) / ((gt[1] - gt[0]) + 1e-32)

def f1_interval(gt, pred):
    p = precision_interval(gt, pred)
    r = recall_interval(gt, pred)
    return 2 * p * r / (p + r + 1e-32)


def calculate_interval_metric(gt_data, pred_data, metric_name):
    metric_handlers = {
        'precision': precision_interval,
        'recall': recall_interval,
        'f1': f1_interval,
        'iou': iou
    }
    metric_handler = metric_handlers.get(metric_name, iou)
    result = []
    for i, gt_intervals in enumerate(gt_data):
        for gt in gt_intervals:
            vals=[]
            for pred in pred_data[i]:
                v = metric_handler(gt, pred)
                vals.append(v)
            result.append(max(vals))
    return sum(result) / len(result) if result else 0