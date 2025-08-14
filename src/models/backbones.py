# src/models/backbones.py

from torchvision.models import resnet18, mobilenet_v2
import torch.nn as nn


def get_resnet18(num_classes=10):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_mobilenetv2(num_classes=10):
    model = mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
