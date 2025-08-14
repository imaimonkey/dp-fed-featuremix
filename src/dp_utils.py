# src/dp_utils.py

import torch


def clip_gradients(model, max_norm):
    """Gradient clipping for DP"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def add_dp_noise(model, noise_multiplier, max_norm, device):
    """Add DP noise after gradient clipping"""
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0, std=noise_multiplier * max_norm, size=p.grad.data.shape
            ).to(device)
            p.grad.data.add_(noise)
