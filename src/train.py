# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.utils.data_loader import get_dataloaders
from src.featuremix import feature_mixup, mixup_criterion
from src.dp_utils import clip_gradients, add_dp_noise
from src.utils.logger import CSVLogger
import argparse
import os


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Model
    if args.model == "resnet18":
        model = models.resnet18(num_classes=10)
    elif args.model == "mobilenetv2":
        model = models.mobilenet_v2(num_classes=10)
    model = model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Logger
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = os.path.join("experiments", "results", f"{args.exp_name}.csv")
    txt_path = os.path.join("experiments", "results", f"{args.exp_name}.txt")
    logger = CSVLogger(csv_path)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            feats = model.forward(inputs)

            if args.use_mixup:
                mixed_feats, y_a, y_b, lam = feature_mixup(
                    feats, targets, alpha=args.mixup_alpha
                )
                outputs = model.fc(mixed_feats)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = feats
                loss = criterion(outputs, targets)

            loss.backward()

            if args.use_dp:
                clip_gradients(model, args.dp_clip)
                add_dp_noise(model, args.dp_noise_multiplier, args.dp_clip, device)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        logger.log_epoch(epoch + 1, avg_loss)

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    logger.log_test_accuracy(acc)

    # Write to .txt summary
    with open(txt_path, "w") as f:
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Use Mixup: {args.use_mixup}\n")
        f.write(f"Use DP: {args.use_dp}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Test Accuracy: {acc:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=["resnet18", "mobilenetv2"]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--dp_clip", type=float, default=1.0)
    parser.add_argument("--dp_noise_multiplier", type=float, default=1.0)
    parser.add_argument("--exp_name", type=str, default="default_exp")
    args = parser.parse_args()

    train(args)
