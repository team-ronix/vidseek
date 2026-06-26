import torch
import numpy as np

class AnchorGenerator:
    def __init__(
        self,
        scales=(128, 256, 512),
        ratios=(0.5, 1.0, 2.0),
        feat_stride=16,
    ):
        self.scales = np.array(scales, dtype=np.float32)
        self.ratios = np.array(ratios, dtype=np.float32)
        self.feat_stride = feat_stride
        self.base_anchors = self._make_base_anchors()

    def _make_base_anchors(self):
        anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                w = scale / np.sqrt(ratio)
                h = scale * np.sqrt(ratio)
                anchors.append([-w/2, -h/2, w/2, h/2])
        return np.array(anchors, dtype=np.float32)

    def generate(self, feat_h, feat_w):
        sh_x= np.arange(feat_w) * self.feat_stride + self.feat_stride // 2
        sh_y= np.arange(feat_h) * self.feat_stride + self.feat_stride // 2
        sx, sy = np.meshgrid(sh_x,sh_y)
        shifts = np.stack([sx.ravel(), sy.ravel(), sx.ravel(), sy.ravel()], axis=1).astype(np.float32)
        all_anchors = shifts[:, None, :] + self.base_anchors[None, :, :]
        return torch.from_numpy(all_anchors.reshape(-1, 4))
