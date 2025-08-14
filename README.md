# DP-Fed-FeatureMix

Privacy-preserving federated learning framework combining **Differential Privacy (DP)** and **Feature-space Mixup**.  
Mitigates **non-IID** data issues, preserves model utility, and reduces communication costs **without sharing raw data**.

원본 데이터를 중앙 서버나 다른 클라이언트와 직접 공유하지 않고, **차등 개인정보보호(DP)**와 **피처 공간 Mixup**을 결합한 프라이버시 보존 연합학습(FL) 프레임워크.  

**non-IID 데이터 문제를 완화**하고, **모델 성능을 유지**하며, **통신 비용을 절감**

---

## Features
- **Feature-space Mixup**: Mixup applied at the pre-logits feature level.
- **Differential Privacy at feature level**: Gaussian noise + per-example clipping.
- **Non-IID robustness**: Handles Dirichlet α=0.1~0.5 data splits.
- **Communication efficiency**: No raw data or gradient sharing.
- **Comprehensive evaluation**: Utility, privacy, robustness, fairness.

---

## Repository Structure
```
dp-fed-featuremix/
│── data/                  # Dataset folder (ignored in .gitignore)
│── src/
│   ├── models/             # Model architectures (ResNet-18, MobileNetV2)
│   ├── utils/              # Data loading, metrics, DP accounting
│   ├── train.py            # Training entry point
│   ├── featuremix.py       # Feature-space Mixup implementation
│   └── dp_utils.py         # DP noise injection & clipping
│── experiments/            # Experiment configs & logs
│── requirements.txt
│── README.md
└── .gitignore
```

---

## Installation
```bash
# Clone this repository
git clone https://github.com/imaimonkey/dp-fed-featuremix.git
cd dp-fed-featuremix

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train Federated Model with DP-FeatureMix
```bash
python src/train.py \
    --dataset CIFAR10 \
    --model resnet18 \
    --non_iid_alpha 0.1 \
    --epsilon 4 \
    --rounds 200 \
    --local_epochs 2 \
    --batch_size 64
```

### 2. Run Baselines
```bash
# FedAvg
python src/train.py --method fedavg ...

# DP-FedAvg
python src/train.py --method dp-fedavg ...

# FedMix
python src/train.py --method fedmix ...
```

---

## Evaluation
- **Dataset**: CIFAR-10 train (non-IID split) / CIFAR-10 test / CIFAR-10-C (robustness)
- **Metrics**:
  - Utility: Top-1 Accuracy, Macro-F1
  - Privacy: ε, δ, Membership Inference AUC
  - Robustness: CIFAR-10-C Acc, ECE, NLL
  - Fairness: Bottom 10% client accuracy
  - System: Communication cost, rounds-to-target

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation
If you use this code in your research, please cite:
```
@misc{dp-fed-featuremix,
  author = {imaimonkey},
  title = {DP-Fed-FeatureMix: Differential Privacy and Feature-space Mixup for Federated Learning},
  year = {2025},
  howpublished = {\url{https://github.com/imaimonkey/dp-fed-featuremix}}
}
```
