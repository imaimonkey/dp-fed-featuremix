# experiments/dp_train.py

from src.train import train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=["resnet18", "mobilenetv2"]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)

    # ✅ DP만 적용
    parser.add_argument("--use_mixup", action="store_false")
    parser.add_argument("--use_dp", action="store_true")
    parser.add_argument("--dp_clip", type=float, default=1.0)
    parser.add_argument("--dp_noise_multiplier", type=float, default=1.0)

    args = parser.parse_args()
    train(args)
