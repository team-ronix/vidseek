import torch
import torch.nn as nn
import torch.nn.functional as F
from visual.faster_rcnn.model.backbone import VGG16Backbone
from visual.faster_rcnn.model.rpn import RPN
from visual.faster_rcnn.model.detection_head import DetectionHead
from visual.faster_rcnn.utils.box_utils import bbox_trans_inv, clip_boxes, nms


class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes=20,
        pretrained=True,
        feat_stride=16,
        anchor_scales=(128, 256, 512),
        anchor_ratios=(0.5, 1.0, 2.0),
        score_thresh=0.05,
        nms_thresh=0.3,
        max_dets=100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_dets = max_dets
        self.backbone = VGG16Backbone(pretrained=pretrained)
        self.rpn = RPN(
            in_ch=512, mid_ch=512,
            feat_stride=feat_stride,
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
        )
        self.det_head = DetectionHead(num_classes=num_classes, pretrained=pretrained)

    def forward(self, images, img_shapes, gt_boxes=None, gt_labels=None):
        img_shape = img_shapes[0]
        gt_b = gt_boxes[0] if gt_boxes is not None else None
        gt_l = gt_labels[0] if gt_labels is not None else None
        feat = self.backbone(images)
        props, rpn_cls, rpn_reg = self.rpn(feat, img_shape, gt_b)
        if self.training:
            det_cls, det_reg = self.det_head(feat, props, gt_b, gt_l)
            total = rpn_cls + rpn_reg + det_cls + det_reg
            return dict(
                rpn_cls_loss=rpn_cls,
                rpn_reg_loss=rpn_reg,
                det_cls_loss=det_cls,
                det_reg_loss=det_reg,
                total_loss=total
            )
        else:
            cls_sc, bbox_d, props = self.det_head(feat, props)
            return self._post(cls_sc, bbox_d, props, img_shape)


    def _post(self, cls_sc, bbox_d, props, img_shape):
        probs = F.softmax(cls_sc, 1)
        all_bxs, all_scs, all_lbls = [], [], []
        for c in range(1, self.num_classes + 1):
            sc = probs[:, c]
            d = bbox_d[:, c*4:(c+1)*4]
            bx = clip_boxes(bbox_trans_inv(props, d), img_shape)
            keep = sc >= self.score_thresh
            if keep.sum() == 0: continue
            bx, sc = bx[keep], sc[keep]
            keep2 = nms(bx, sc, self.nms_thresh)
            all_bxs.append(bx[keep2])
            all_scs.append(sc[keep2])
            all_lbls.append(torch.full((keep2.numel(),), c, dtype=torch.long, device=sc.device))
        if all_bxs:
            bxs = torch.cat(all_bxs)
            scrs = torch.cat(all_scs)
            lbls = torch.cat(all_lbls)
            if len(scrs) > self.max_dets:
                idx = scrs.topk(self.max_dets).indices
                bxs, scrs, lbls = bxs[idx], scrs[idx], lbls[idx]
        else:
            bxs = torch.zeros((0,4))
            scrs = torch.zeros(0)
            lbls = torch.zeros(0, dtype=torch.long)
        return [dict(boxes=bxs, scores=scrs, labels=lbls)]
