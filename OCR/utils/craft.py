import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)

        features = list(vgg.features.children())

        self.block1 = nn.Sequential(*features[:7]) 
        self.block2 = nn.Sequential(*features[7:14])
        self.block3 = nn.Sequential(*features[14:24])
        self.block4 = nn.Sequential(*features[24:34])
        self.block5 = nn.Sequential(*features[34:44])

        self.block6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s1 = self.block1(x) 
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        s5 = self.block5(s4)
        s6 = self.block6(s5)
        return s1, s2, s3, s4, s5, s6
    

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CRAFT(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = VGG16()

        self.upconv1 = UpConvBlock(512 + 512, 256)
        self.upconv2 = UpConvBlock(256 + 512, 128)
        self.upconv3 = UpConvBlock(128 + 256, 64)
        self.upconv4 = UpConvBlock(64 + 128, 32)

        self.last_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

        self._initialize_decoder_weights()

    def _initialize_decoder_weights(self):
        for module in [self.upconv1, self.upconv2, self.upconv3, self.upconv4, self.conv_cls]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1, s2, s3, s4, s5, s6 = self.backbone(x)

        y = torch.cat([s6, s5], dim=1)
        y = F.interpolate(y, size=s4.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv1(y)

        y = torch.cat([y, s4], dim=1)
        y = F.interpolate(y, size=s3.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv2(y)

        y = torch.cat([y, s3], dim=1)
        y = F.interpolate(y, size=s2.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv3(y)

        y = torch.cat([y, s2], dim=1)
        y = F.interpolate(y, size=s1.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv4(y)

        y = self.last_conv(y)

        return y.permute(0, 2, 3, 1)
    

def generate_gaussian(size=64):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    gauss = np.exp(-(xx**2 + yy**2) / (2 * 0.5**2))
    return gauss.astype(np.float32)


def warp_gaussian(dst_quad, img_h, img_w, gaussian_size=64):
    src = np.array([[0,0],[64,0],[64,64],[0,64]], dtype=np.float32)
    dst = dst_quad.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    gaussian = generate_gaussian(gaussian_size)
    return cv2.warpPerspective(gaussian, M, (img_w, img_h))

def make_heatmaps(charBB, img_h, img_w):
    region = np.zeros((img_h, img_w), dtype=np.float32)
    affinity = np.zeros((img_h, img_w), dtype=np.float32)

    boxes = charBB.transpose(2, 1, 0)

    for i, box in enumerate(boxes):
        region = np.maximum(region, warp_gaussian(box, img_h, img_w))

        if i + 1 < len(boxes):
            next_box = boxes[i + 1]
            aff_quad = np.array([
                (box[0] + box[1]) / 2,
                (next_box[0] + next_box[1]) / 2,
                (next_box[2] + next_box[3]) / 2,
                (box[2] + box[3]) / 2,
            ])
            affinity = np.maximum(affinity, warp_gaussian(aff_quad, img_h, img_w))

    return region, affinity

