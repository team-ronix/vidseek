import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models import vgg16 as _vgg16
from visual.faster_rcnn.utils.box_utils import bbox_transform, iou_matrix

class RoIPool(nn.Module):
    def __init__(self, out_size=7, spatial_scale=1/16):
        super().__init__()
        self.out_size = out_size
        self.spatial_scale = spatial_scale

    def forward(self, feat, rois):
        assert feat.shape[0] == 1, "RoIPool only supports batch_size=1"
        N = rois.shape[0]
        C = feat.shape[1]
        out = torch.zeros(N, C, self.out_size, self.out_size, device=feat.device, dtype=feat.dtype)
        for i in range(N):
            x1 = (rois[i,0]*self.spatial_scale).floor().long().clamp(min=0)
            y1 = (rois[i,1]*self.spatial_scale).floor().long().clamp(min=0)
            x2 = (rois[i,2]*self.spatial_scale).ceil().long().clamp(min=x1+1)
            y2 = (rois[i,3]*self.spatial_scale).ceil().long().clamp(min=y1+1)
            x2 = x2.clamp(max=feat.shape[3])
            y2 = y2.clamp(max=feat.shape[2])
            roi_f = feat[0, :, y1:y2, x1:x2]
            if roi_f.numel() == 0: continue
            out[i] = F.adaptive_max_pool2d(roi_f.unsqueeze(0), self.out_size)[0]
        return out

class RoIAlign(nn.Module):
    def __init__(self, out_size=7, spatial_scale=1/16):
        super().__init__()
        self.out_size = out_size
        self.spatial_scale = spatial_scale

    def forward(self, feat, rois):
        if rois.numel() == 0:
            C = feat.shape[1]
            return torch.zeros((0, C, self.out_size, self.out_size), device=feat.device, dtype=feat.dtype)

        batch_idx = torch.zeros((rois.shape[0], 1), device=rois.device, dtype=rois.dtype)
        boxes = torch.cat([batch_idx, rois], dim=1)
        return roi_align(
            feat,
            boxes,
            output_size=self.out_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1,
            aligned=True,
        )



class DetectionHead(nn.Module):
    def __init__(
        self, num_classes, roi_size=7, in_ch=512, fc_dim=4096,
        batch=128, pos_frac=0.25, pos_iou=0.5,
        neg_iou_hi=0.5, neg_iou_lo=0.1, lam=1.0, roi_op="align", pretrained=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.batch = batch
        self.pos_frac = pos_frac
        self.pos_iou = pos_iou
        self.neg_iou_hi = neg_iou_hi
        self.neg_iou_lo = neg_iou_lo
        self.lam = lam
        self.pretrained = pretrained
        if roi_op == "align":
            self.roi = RoIAlign(roi_size, 1/16)
        else:
            self.roi = RoIPool(roi_size, 1/16)
        flat = in_ch * roi_size * roi_size
        self.fc6 = nn.Linear(flat, fc_dim)
        self.fc7 = nn.Linear(fc_dim, fc_dim)
        self.cls_score = nn.Linear(fc_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(fc_dim, 4*(num_classes + 1))
        self.register_buffer('reg_std', torch.tensor([0.1, 0.1, 0.2, 0.2]))
        self._init_weights()

    def _init_weights(self):
        if self.pretrained:
            clf = _vgg16(weights="IMAGENET1K_V1").classifier
            self.fc6.weight.data.copy_(clf[0].weight.data)
            self.fc6.bias.data.copy_(clf[0].bias.data)
            self.fc7.weight.data.copy_(clf[3].weight.data)
            self.fc7.bias.data.copy_(clf[3].bias.data)
        else:
            nn.init.normal_(self.fc6.weight, 0, 0.01)
            nn.init.constant_(self.fc6.bias, 0)
            nn.init.normal_(self.fc7.weight, 0, 0.01)
            nn.init.constant_(self.fc7.bias, 0)
        nn.init.normal_(self.cls_score.weight, 0, 0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, 0, 0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, feat, proposals, gt_boxes=None, gt_labels=None):
        if self.training and gt_boxes is not None:
            proposals, labels, reg_tgt = self._sample(proposals, gt_boxes, gt_labels)
            
        if proposals.shape[0] == 0:
            C = self.num_classes
            device = feat.device
            empty_cls = torch.zeros((0, C + 1), device=device)
            empty_reg = torch.zeros((0, 4 * (C + 1)), device=device)
            if not self.training or gt_boxes is None:
                return empty_cls, empty_reg, proposals
            z = feat.sum() * 0.0
            # return zero losses as tensors (not integer zero) to avoid issues with autograd
            # and use .backward() for backpropagation to compute gradients without errors
            return z, z

        pooled = self.roi(feat, proposals)
        flat = pooled.flatten(start_dim=1)
        x = F.relu(self.fc6(flat), inplace=True)
        x = F.relu(self.fc7(x), inplace=True)
        cls_sc = self.cls_score(x)
        bbox = self.bbox_pred(x)

        if not self.training or gt_boxes is None:
            std_rep = self.reg_std.repeat(self.num_classes + 1)
            return cls_sc, bbox * std_rep, proposals

        cls_loss = F.cross_entropy(cls_sc, labels)
        # ignore background and focus on positive samples for regression loss as label of background is 0
        pos_mask = labels > 0
        reg_loss = cls_sc.sum() * 0.0
        if pos_mask.sum() > 0:
            pc = labels[pos_mask]
            idx = (pc * 4).unsqueeze(1) + torch.arange(4, device=cls_sc.device)
            pred = bbox[pos_mask].gather(1, idx)
            reg_loss = self.lam * F.smooth_l1_loss(pred, reg_tgt[pos_mask])

        return cls_loss, reg_loss

    def _sample(self, proposals, gt_boxes, gt_labels):
        if gt_boxes.shape[0] == 0:
            n = proposals.shape[0]
            labels = torch.zeros(n, dtype=torch.long, device=proposals.device)
            reg_tgt = torch.zeros((n, 4), dtype=proposals.dtype, device=proposals.device)
            ni = torch.randperm(n)[:self.batch]
            return proposals[ni], labels[ni], reg_tgt[ni]
        
        proposals = torch.cat([proposals, gt_boxes], 0)
        ious = iou_matrix(proposals, gt_boxes)
        max_iou, best_gt = ious.max(1)
        labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
        fg = max_iou >= self.pos_iou
        labels[fg] = gt_labels[best_gt[fg]]
        bg = (max_iou >= self.neg_iou_lo) & (max_iou < self.neg_iou_hi)
        labels[~fg & ~bg] = -1
        num_pos = int(self.batch * self.pos_frac)
        pi = fg.nonzero(as_tuple=True)[0]
        ni = bg.nonzero(as_tuple=True)[0]
        if len(pi) > num_pos:
            labels[pi[torch.randperm(len(pi))[num_pos:]]] = -1
        num_neg = self.batch - min(len(pi), num_pos)
        if len(ni) > num_neg:
            labels[ni[torch.randperm(len(ni))[num_neg:]]] = -1
        valid = labels >= 0
        proposals = proposals[valid]
        labels = labels[valid]
        best_gt = best_gt[valid]
        reg_tgt = bbox_transform(proposals, gt_boxes[best_gt])
        reg_tgt = reg_tgt / self.reg_std
        return proposals, labels, reg_tgt
