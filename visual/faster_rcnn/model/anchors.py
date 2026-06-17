import torch
import numpy as np

class AnchorGenerator:
    def __init__(
        self,
        scales=(128, 256, 512),
        ratios=(0.5, 1.0, 2.0),
        feat_stride=16,
    ):
        # feat_stride is the downsampling factor of the feature map relative to the input image produced by the backbone.
        # although vgg16 reduces the spatial resolution by a factor of 32 after its five max-pooling layers (2^5 = 32), 
        # faster r-cnn uses the feature map before the final pooling layer (uses four max pooling layers), resulting in an effective stride of 16 (2^4) pixels.
        # that means moving one cell in the feature map corresponds to moving 16 pixels in the original image.
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
        shift_x = np.arange(feat_w) * self.feat_stride + self.feat_stride // 2
        shift_y = np.arange(feat_h) * self.feat_stride + self.feat_stride // 2
        sx, sy = np.meshgrid(shift_x, shift_y)
        shifts = np.stack([sx.ravel(), sy.ravel(), sx.ravel(), sy.ravel()], axis=1).astype(np.float32)
        all_anchors = shifts[:, None, :] + self.base_anchors[None, :, :]
        return torch.from_numpy(all_anchors.reshape(-1, 4))
