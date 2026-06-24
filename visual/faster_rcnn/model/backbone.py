import torch.nn as nn
import torchvision.models as tvm

class VGG16Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
            features = list(vgg.features.children())
            # blocks 1 to 4 with their max pooling layers, block5 without its max pooling layer
            self.block1 = nn.Sequential(*features[0:5])
            self.block2 = nn.Sequential(*features[5:10])
            self.block3 = nn.Sequential(*features[10:17])
            self.block4 = nn.Sequential(*features[17:24])
            self.block5 = nn.Sequential(*features[24:30])
        else:
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.block5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
                # removing the last max pool layer
            )
            self._init_weights()
            
            
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # the width and height of the feature map are reduced by a factor of 16 compared to the input image because of the 4 max pooling layers (2^4 = 16)
        return x
