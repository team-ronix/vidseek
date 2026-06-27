import torch, numpy as np
from collections import defaultdict
from visual.faster_rcnn.voc_dataset import VOC_CLASSES


def _iou_one(box, boxes):
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    ab = (box[2] - box[0]) * (box[3] - box[1])
    bs = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (ab + bs - inter + 1e-8)


class VOCEvaluator:
    def __init__(self, num_classes=20, iou_thresh=0.5):
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        self.dets = defaultdict(list)
        self.gts = defaultdict(dict)

    def update(self, img_id, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, gt_difficult=None):
        for c in range(1, self.num_classes+1):
            m = gt_labels == c
            boxes = gt_boxes[m] if m.any() else torch.zeros((0, 4))
            if gt_difficult != None and m.any():
                diff = gt_difficult[m]
            else:
                diff = torch.zeros(boxes.shape[0], dtype=torch.bool)
            self.gts[c][img_id] = (boxes, diff)
        for i in range(len(pred_scores)):
            c = pred_labels[i].item()
            self.dets[c].append((img_id, pred_scores[i].item(), pred_boxes[i]))

    def compute_map(self):
        aps = {}
        for c in range(1, self.num_classes + 1):
            aps[VOC_CLASSES[c-1]] = self._ap(c)
        mAP = np.mean(list(aps.values()))
        return mAP, aps

    def _ap(self, c):
        dets = sorted(self.dets.get(c, []), key=lambda x: -x[1])
        n_gt = sum(
            (~diff).sum().item()
            for boxes, diff in self.gts[c].values()
        )
        if not dets or n_gt == 0:
            return 0.0
        matched = defaultdict(set)
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for i, (img_id, sc, box) in enumerate(dets):
            entry = self.gts[c].get(img_id)
            if entry is None:
                fp[i] = 1
                continue
            gt, diff = entry
            if len(gt) == 0:
                fp[i] = 1
                continue
            ious = _iou_one(box, gt)
            best_iou, best_j = ious.max(0)
            best_j = best_j.item()
            if best_iou >= self.iou_thresh and best_j not in matched[img_id]:
                if diff[best_j] == True:
                    pass
                else:
                    tp[i] = 1
                    matched[img_id].add(best_j)
            else:
                fp[i] = 1
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec = tp_c / (n_gt + 1e-8)
        prec = tp_c / (tp_c + fp_c + 1e-8)
        return sum(
            prec[rec >= t].max() if (rec >= t).any() else 0
            for t in np.arange(0, 1.1, 0.1)
        ) / 11.0
