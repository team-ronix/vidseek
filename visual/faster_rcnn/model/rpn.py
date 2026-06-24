import torch
import torch.nn as nn
import torch.nn.functional as F
from visual.faster_rcnn.model.anchors import AnchorGenerator
from visual.faster_rcnn.utils.box_utils import (
    bbox_transform, bbox_transform_inv,
    clip_boxes, filter_boxes, iou_matrix, nms,
)


class RPN(nn.Module):
    def __init__(
        self,
        in_ch=512, mid_ch=512, feat_stride=16,
        anchor_scales=(128, 256, 512), anchor_ratios=(0.5, 1.0, 2.0),
        rpn_batch=256, pos_iou=0.7, neg_iou=0.3, pos_frac=0.5,
        pre_nms_train=6000, post_nms_train=2000,
        pre_nms_test=6000, post_nms_test=300,
        nms_thresh=0.7, min_size=16, lam=10.0,
    ):
        super().__init__()
        self.feat_stride = feat_stride
        self.rpn_batch = rpn_batch
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.pos_frac = pos_frac
        self.pre_nms_train = pre_nms_train
        self.post_nms_train = post_nms_train
        self.pre_nms_test = pre_nms_test
        self.post_nms_test = post_nms_test
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.lam = lam

        k = len(anchor_scales) * len(anchor_ratios)
        self.k = k
        self.conv = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.cls_layer = nn.Conv2d(mid_ch, 2*k, 1)
        self.reg_layer = nn.Conv2d(mid_ch, 4*k, 1)
        self.anchor_gen = AnchorGenerator(anchor_scales, anchor_ratios, feat_stride)
        self._init_weights()

    def _init_weights(self):
        for l in [self.conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(l.weight, 0, 0.01)
            nn.init.constant_(l.bias, 0)


    def forward(self, feat, img_shape, gt_boxes=None):
        B, C, fh, fw = feat.shape
        anchors = self.anchor_gen.generate(fh, fw).to(feat.device)
        h = F.relu(self.conv(feat), inplace=True)
        cls_logits = self.cls_layer(h)
        reg_deltas = self.reg_layer(h)

        cls_t = cls_logits.permute(0,2,3,1).contiguous().view(-1, 2)
        reg_t = reg_deltas.permute(0,2,3,1).contiguous().view(-1, 4)
        scores = F.softmax(cls_t, 1)[:, 1]   # objectness

        # decode & clip
        props = bbox_transform_inv(anchors, reg_t.detach())
        props = clip_boxes(props, img_shape)

        # filter tiny + NMS
        pre_k = self.pre_nms_train if self.training else self.pre_nms_test
        post_k = self.post_nms_train if self.training else self.post_nms_test

        keep = filter_boxes(props, self.min_size)
        props, sc = props[keep], scores[keep]
        order = sc.argsort(descending=True)[:pre_k]
        props, sc = props[order], sc[order]
        keep = nms(props, sc, self.nms_thresh)[:post_k]
        props= props[keep]

        if not self.training or gt_boxes is None:
            return props, None, None

        cls_loss, reg_loss = self._loss(anchors, cls_t, reg_t, gt_boxes, img_shape)
        return props, cls_loss, reg_loss


    def _loss(self, anchors, cls_t, reg_t, gt_boxes, img_shape):
        H, W = img_shape
        labels = torch.full((len(anchors),), -1, dtype=torch.long, device=anchors.device)

        inside = (
            (anchors[:,0]>=0)&(anchors[:,1]>=0)&
            (anchors[:,2]<=W)&(anchors[:,3]<=H)
        )

        if len(gt_boxes) == 0:
            labels[inside] = 0
        else:
            ious = iou_matrix(anchors[inside], gt_boxes)
            max_iou, gt_idx = ious.max(1)
            idx_in = inside.nonzero(as_tuple=True)[0]

            labels[idx_in[max_iou < self.neg_iou]] = 0
            labels[idx_in[max_iou >= self.pos_iou]] = 1
            if ious.numel() > 0:
                labels[idx_in[ious.argmax(0)]] = 1   # best anchor per GT

        # sample 256
        num_pos = int(self.rpn_batch * self.pos_frac)
        pos_i = (labels==1).nonzero(as_tuple=True)[0]
        neg_i = (labels==0).nonzero(as_tuple=True)[0]
        if len(pos_i) > num_pos:
            labels[pos_i[torch.randperm(len(pos_i))[num_pos:]]] = -1
        num_neg = self.rpn_batch - min(len(pos_i), num_pos)
        if len(neg_i) > num_neg:
            labels[neg_i[torch.randperm(len(neg_i))[num_neg:]]] = -1

        valid = labels >= 0
        Ncls = valid.sum().clamp(min=1)
        cls_loss = F.cross_entropy(cls_t[valid], labels[valid], reduction='sum') / Ncls.float()

        pos_mask = labels == 1
        Nreg = len(anchors) // self.k
        if pos_mask.sum() > 0 and len(gt_boxes) > 0:
            pos_a = anchors[pos_mask]
            _, best = iou_matrix(pos_a, gt_boxes).max(1)
            tgt = bbox_transform(pos_a, gt_boxes[best])
            reg_loss = self.lam * F.smooth_l1_loss(reg_t[pos_mask], tgt, reduction='sum') / Nreg
        else:
            reg_loss = reg_t.sum() * 0.0

        return cls_loss, reg_loss
