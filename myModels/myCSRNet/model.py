import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        # 使用预训练的VGG-16模型的前10层作为前端
        self.frontend = nn.Sequential(*list(vgg16(pretrained=True).features.children())[:23])

        # 后端使用扩张卷积层
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # 上采样将输出28x28 -> 224x224
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 前端特征提取
        x = self.frontend(x)

        # 后端扩张卷积
        x = self.backend(x)

        # upsample
        x = self.upsample(x)
        
        # 返回生成的密度图
        return x