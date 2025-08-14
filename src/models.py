# src/models.py
# Backbones with GroupNorm and feature/head split for feature-space DP & Mixup.

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------
# Norm utils
# ---------------------------


def replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    """
    Recursively replace all BatchNorm2d with GroupNorm(num_groups).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=num_groups, num_channels=child.num_features, affine=True
            )
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups=num_groups)
    return module


# ---------------------------
# ResNet18 backbone (GN) with feature/head split
# ---------------------------


class ResNet18GN(nn.Module):
    """
    ResNet-18 with GroupNorm and explicit feature extractor (up to GAP) + classifier head.
    pre-logits feature dimension = 512
    """

    def __init__(self, num_classes: int = 10, gn_groups: int = 8):
        super().__init__()
        base = models.resnet18(weights=None)
        replace_bn_with_gn(base, num_groups=gn_groups)

        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feat_dim = base.fc.in_features  # 512
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # init head
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.features(x)  # (B, 512, 1, 1)
        z = torch.flatten(x, 1)  # pre-logits: (B, 512)
        logits = self.classifier(z)
        if return_features:
            return z, logits
        return logits


# ---------------------------
# MobileNetV2 backbone (GN) with feature/head split
# ---------------------------


class MobileNetV2GN(nn.Module):
    """
    MobileNetV2 with GroupNorm and explicit feature extractor (up to GAP) + classifier head.
    pre-logits feature dimension = 1280 (for width_mult=1.0)
    """

    def __init__(
        self, num_classes: int = 10, gn_groups: int = 8, width_mult: float = 1.0
    ):
        super().__init__()
        base = models.mobilenet_v2(weights=None, width_mult=width_mult)
        replace_bn_with_gn(base, num_groups=gn_groups)

        self.features = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # infer feature dim from classifier input
        feat_dim = (
            base.classifier[1].in_features
            if isinstance(base.classifier, nn.Sequential)
            else 1280
        )
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.features(x)  # (B, C, 1, 1)
        z = torch.flatten(x, 1)  # pre-logits: (B, C)
        logits = self.classifier(z)
        if return_features:
            return z, logits
        return logits


# ---------------------------
# Factory
# ---------------------------


def build_model(
    name: str = "resnet18_gn",
    num_classes: int = 10,
    gn_groups: int = 8,
    **kwargs,
) -> nn.Module:
    """
    name: one of {"resnet18", "resnet18_gn", "mobilenetv2", "mobilenetv2_gn"}
    """
    key = name.lower()
    if key in ("resnet18", "resnet18_gn"):
        return ResNet18GN(num_classes=num_classes, gn_groups=gn_groups)
    elif key in ("mobilenetv2", "mobilenetv2_gn"):
        width_mult = float(kwargs.get("width_mult", 1.0))
        return MobileNetV2GN(
            num_classes=num_classes, gn_groups=gn_groups, width_mult=width_mult
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


# ---------------------------
# Simple self-test
# ---------------------------

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    m1 = build_model("resnet18_gn", num_classes=10)
    z1, y1 = m1(x, return_features=True)
    print("ResNet18GN:", z1.shape, y1.shape)  # (2, 512), (2, 10)

    m2 = build_model("mobilenetv2_gn", num_classes=10)
    z2, y2 = m2(x, return_features=True)
    print("MobileNetV2GN:", z2.shape, y2.shape)  # (2, 1280), (2, 10)
