# -*- coding: utf-8 -*-
"""模型定义：自定义轻量 CNN（SimpleCNN）与 CIFAR 版 ResNet-20。

- SimpleCNN：VGG 风格，卷积-批归一化-ReLU 堆叠 + 最大池化 + Dropout，
  参数约 0.81M，适合在 CPU 上训练。
- ResNet20：He et al. 针对 CIFAR-10 提出的标准残差网络（BasicBlock，
  三层各 3 个残差块，通道 16/32/64），参数约 0.27M。
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------- SimpleCNN

class ConvBlock(nn.Sequential):
    """两个 3x3 卷积 + BN + ReLU（保持分辨率不变）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    """三层卷积块 + 全连接分类头的轻量 CNN，用于 CIFAR-10。"""

    def __init__(self, num_classes=10, drop1=0.1, drop2=0.2, drop_fc=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2),               # 32 -> 16
            nn.Dropout(drop1),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),               # 16 -> 8
            nn.Dropout(drop1),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),               # 8 -> 4
            nn.Dropout(drop2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------- ResNet-20

class BasicBlock(nn.Module):
    """标准残差块：3x3-BN-ReLU-3x3-BN + 恒等/投影捷径。"""

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet20(nn.Module):
    """CIFAR-10 版 ResNet-20：首层 3x3 卷积（不下采样），三层残差阶段。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_ch, n_blocks, stride):
        layers = [BasicBlock(self.in_channels, out_ch, stride)]
        self.in_channels = out_ch
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return self.fc(torch.flatten(out, 1))


# ------------------------------------------------------------------ factory

MODEL_REGISTRY = {"simplecnn": SimpleCNN, "resnet20": ResNet20}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """按名称构造模型。"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name}，可选：{list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes)
