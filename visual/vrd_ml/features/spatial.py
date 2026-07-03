import numpy as np
from visual.vrd_ml.vrd_dataset import BBox

SPATIAL_DIM = 14

def compute_spatial_features(subj_box, obj_box, img_w, img_h):
    eps = 1e-6
    W, H = float(img_w), float(img_h)
    scx, scy = subj_box.center
    ocx, ocy = obj_box.center
    dx = (ocx - scx) / W
    dy = (ocy - scy) / H
    area_ratio = np.log((obj_box.area + eps) / (subj_box.area + eps))
    iou = subj_box.iou(obj_box)
    w_ratio = (obj_box.width + eps) / (subj_box.width + eps)
    h_ratio = (obj_box.height + eps) / (subj_box.height + eps)
    sub_norm = np.array([
        subj_box.x1 / W, subj_box.y1 / H,
        subj_box.x2 / W, subj_box.y2 / H,
    ])
    obj_norm = np.array([
        obj_box.x1 / W, obj_box.y1 / H,
        obj_box.x2 / W, obj_box.y2 / H,
    ])
    feat = np.concatenate([
        [dx, dy, area_ratio, iou, w_ratio, h_ratio],
        sub_norm,
        obj_norm,
    ]).astype(np.float32)
    return feat


def compute_union_box(subj_box, obj_box):
    return subj_box.union(obj_box)


class SpatialFeatureExtractor:
    dim = SPATIAL_DIM

    def extract(self, subj_box, obj_box, img_w, img_h):
        return compute_spatial_features(subj_box, obj_box, img_w, img_h)

    def extract_batch(self, pairs, img_w, img_h):
        return np.stack([compute_spatial_features(s, o, img_w, img_h) for s, o in pairs], axis=0)
