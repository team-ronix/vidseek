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
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
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

        self.upconv1 = UpConvBlock(1024 + 512, 256)
        self.upconv2 = UpConvBlock(256 + 512, 128)
        self.upconv3 = UpConvBlock(128 + 256, 64)
        self.upconv4 = UpConvBlock(64 + 128, 32)

        self.conv_cls = nn.Sequential(
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

        y = self.conv_cls(y)

        return y.permute(0, 2, 3, 1)


def generate_gaussian_kernel(size=64):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    gauss = np.exp(-(xx**2 + yy**2) / (2 * 0.5**2))
    return gauss.astype(np.float32)


def create_region_heatmap(dst_quad, img_h, img_w, gaussian_size=64):
    src = np.array([[0, 0], [64, 0], [64, 64], [0, 64]], dtype=np.float32)
    dst = dst_quad.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    gaussian = generate_gaussian_kernel(gaussian_size)
    return cv2.warpPerspective(gaussian, M, (img_w, img_h))


def make_heatmaps(charBB, img_h, img_w):
    region = np.zeros((img_h, img_w), dtype=np.float32)
    affinity = np.zeros((img_h, img_w), dtype=np.float32)

    boxes = charBB.transpose(2, 1, 0)

    for i, box in enumerate(boxes):
        region = np.maximum(region, create_region_heatmap(box, img_h, img_w))

        if i + 1 < len(boxes):
            next_box = boxes[i + 1]
            aff_quad = np.array([
                (box[0] + box[1]) / 2,
                (next_box[0] + next_box[1]) / 2,
                (next_box[2] + next_box[3]) / 2,
                (box[2] + box[3]) / 2,
            ])
            affinity = np.maximum(affinity, create_region_heatmap(aff_quad, img_h, img_w))

    return region, affinity


def get_word_boxes(region, affinity, region_thresh=0.4, affinity_thresh=0.3, padding=8):
    combined = np.clip(region + affinity, 0, 1)
    binary = ((region > region_thresh) | (combined > affinity_thresh)).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(binary, connectivity=4)

    boxes = []
    for label in range(1, n_labels):
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        rect = cv2.minAreaRect(contours[0])
        center, (w, h), angle = rect
        rect = (center, (w + 2 * padding, h + 2 * padding), angle)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return boxes


def extract_word_images_craft(frame, model, device, img_size=768, region_thresh=0.4, affinity_thresh=0.3, pad=4):
    H, W = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = img_size / max(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h))

    t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t)

    region = out[0,:,:,0].cpu().numpy()
    affinity = out[0,:,:,1].cpu().numpy()

    boxes = get_word_boxes(region, affinity, region_thresh, affinity_thresh, pad)

    hm_h, hm_w = region.shape
    sx = W / hm_w
    sy = H / hm_h

    word_images = []
    for box in boxes:
        scaled = (box * np.array([sx, sy])).astype(np.int32)
        x1 = max(int(scaled[:,0].min()), 0)
        y1 = max(int(scaled[:,1].min()), 0)
        x2 = min(int(scaled[:,0].max()), W)
        y2 = min(int(scaled[:,1].max()), H)
        if x2 > x1 and y2 > y1:
            word_images.append(frame[y1:y2, x1:x2])

    return word_images
