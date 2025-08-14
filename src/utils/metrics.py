# src/utils/metrics.py

import torch


def evaluate(model, device, dataloader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    acc = 100.0 * correct / len(dataloader.dataset)
    return acc
