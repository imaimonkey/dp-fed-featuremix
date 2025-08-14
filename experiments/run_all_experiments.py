import os
import itertools
import subprocess

os.environ["PYTHONPATH"] = os.path.abspath(".")

# 가상환경 안 잡힐 때 적용
python_path = "./venv-dpfed/Scripts/python.exe"


models = ["resnet18", "mobilenetv2"]
use_mixup = [False, True]
use_dp = [False, True]

base_args = {
    "batch_size": 128,
    "lr": 5e-4,
    "epochs": 30,
    "mixup_alpha": 0.4,
    "dp_clip": 1.0,
    "dp_noise_multiplier": 0.3,
}

combinations = list(itertools.product(models, use_mixup, use_dp))

for model, mix, dp in combinations:
    exp_name = f"{model}"
    if mix:
        exp_name += "_mixup"
    if dp:
        exp_name += "_dp"

    cmd = [
        python_path,
        "src/train.py",
        f"--model={model}",
        f"--batch_size={base_args['batch_size']}",
        f"--lr={base_args['lr']}",
        f"--epochs={base_args['epochs']}",
        f"--mixup_alpha={base_args['mixup_alpha']}",
        f"--dp_clip={base_args['dp_clip']}",
        f"--dp_noise_multiplier={base_args['dp_noise_multiplier']}",
        f"--exp_name={exp_name}",
    ]

    if mix:
        cmd.append("--use_mixup")
    if dp:
        cmd.append("--use_dp")

    print(f"\n▶ Running experiment: {exp_name}")
    subprocess.run(cmd)
