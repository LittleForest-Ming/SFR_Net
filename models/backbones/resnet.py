from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = 'resnet34', pretrained: bool = True):
        super().__init__()
        if name not in {'resnet18', 'resnet34'}:
            raise ValueError(f'Unsupported backbone: {name}')
        channels = [64, 128, 256, 512]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(ResidualBlock(64, channels[0]), ResidualBlock(channels[0], channels[0]))
        self.layer2 = nn.Sequential(ResidualBlock(channels[0], channels[1], stride=2), ResidualBlock(channels[1], channels[1]))
        self.layer3 = nn.Sequential(ResidualBlock(channels[1], channels[2], stride=2), ResidualBlock(channels[2], channels[2]))
        self.layer4 = nn.Sequential(ResidualBlock(channels[2], channels[3], stride=2), ResidualBlock(channels[3], channels[3]))
        self.out_channels = channels

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
