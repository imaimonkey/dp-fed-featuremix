# src/featuremix.py

import torch
import torch.nn.functional as F
import numpy as np


def feature_mixup(features, labels, alpha=1.0):
    """Feature-space Mixup 구현"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = features.size(0)
    index = torch.randperm(batch_size).to(features.device)

    mixed_features = lam * features + (1 - lam) * features[index, :]
    labels_a, labels_b = labels, labels[index]

    return mixed_features, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
