# experiments/baseline_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os

from src.models.backbones import get_resnet18, get_mobilenetv2
from src.utils.metrics import evaluate


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    acc = 100.0 * correct / len(train_loader.dataset)
    return total_loss / len(train_loader), acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_set = datasets.CIFAR10(
        root="data/", train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root="data/", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.model == "resnet18":
        model = get_resnet18().to(device)
    elif args.model == "mobilenetv2":
        model = get_mobilenetv2().to(device)
    else:
        raise ValueError("Unsupported model")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_acc = evaluate(model, device, test_loader)

        print(
            f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )

    # 모델 저장
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{args.model}_baseline.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=["resnet18", "mobilenetv2"]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
