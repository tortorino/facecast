"""The generator network (SRResNet-style). The same architecture as in facecast.ipynb."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    """Turns a low-resolution face (N,3,h,w) in the 0 - 1 range into a `scale` times larger one."""

    def __init__(self, scale=8, num_blocks=16, num_feats=64):
        super().__init__()
        self.scale = scale
        # Head: color channels -> feature channels
        self.head = nn.Conv2d(3, num_feats, 3, padding=1)
        # Body: residual blocks at the low resolution
        blocks = [ResBlock(num_feats) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks, nn.Conv2d(num_feats, num_feats, 3, padding=1))
        # Upsampling: every stage doubles the resolution
        up = []
        for _ in range(int(math.log2(scale))):
            up += [nn.Conv2d(num_feats, 4 * num_feats, 3, padding=1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*up)
        # Tail: feature channels -> color channels
        self.tail = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feats, 3, 3, padding=1),
        )

    def forward(self, lr):
        # The network predicts the difference to the bicubic upscale of the input
        base = F.interpolate(lr.float(), scale_factor=self.scale, mode="bicubic", align_corners=False)
        x = self.head(lr - 0.5)
        x = x + self.body(x)
        x = self.tail(self.up(x))
        return x.float() + base
