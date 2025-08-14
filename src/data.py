# src/data.py
# CIFAR-10 loader + Dirichlet non-IID partition + DataLoader helpers

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------
# Reproducibility helpers
# ---------------------------


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Datasets & transforms
# ---------------------------


def get_cifar10_datasets(
    data_root: str = "./data",
    aug_train: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Returns CIFAR-10 train/test datasets with common transforms.
    Train: RandomCrop+Flip(optional), Normalize
    Test : Normalize only
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2430, 0.2610]

    train_tfms = (
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if aug_train
        else []
    )
    train_tfms += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    test_tfms = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.Compose(train_tfms),
    )
    test = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.Compose(test_tfms),
    )
    return train, test


# ---------------------------
# Dirichlet non-IID partition
# ---------------------------


@dataclass
class FLPartition:
    client_indices: List[List[int]]  # indices per client (over training set)
    alpha: float
    seed: int


def dirichlet_label_skew_partition(
    labels: torch.Tensor, n_clients: int, alpha: float, seed: int = 0
) -> List[List[int]]:
    """
    Split indices by label using Dirichlet(alpha) per class → realistic label-skew.
    - Smaller alpha → more skew (non-IID 강함)
    - Deterministic given seed
    """
    set_seed(seed)

    num_classes = int(labels.max().item()) + 1
    idx_by_class = [torch.where(labels == c)[0].tolist() for c in range(num_classes)]
    for lst in idx_by_class:
        random.shuffle(lst)

    # init empty buckets
    buckets: List[List[int]] = [[] for _ in range(n_clients)]

    for c in range(num_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue

        # draw proportions once per class
        dist = torch.distributions.Dirichlet(torch.full((n_clients,), alpha))
        props = dist.sample().tolist()

        # length per client (round to int)
        counts = [int(round(p * len(idxs))) for p in props]
        # fix rounding drift
        diff = len(idxs) - sum(counts)
        for i in range(abs(diff)):
            counts[i % n_clients] += 1 if diff > 0 else -1

        start = 0
        for cid, k in enumerate(counts):
            if k <= 0:
                continue
            buckets[cid].extend(idxs[start : start + k])
            start += k

    # shuffle client indices
    for cid in range(n_clients):
        random.shuffle(buckets[cid])
    return buckets


def make_fl_partition_cifar10(
    train_dataset: datasets.CIFAR10,
    n_clients: int = 100,
    alpha: float = 0.1,
    seed: int = 0,
) -> FLPartition:
    """Return FLPartition with client index lists."""
    labels = torch.tensor(train_dataset.targets)
    client_indices = dirichlet_label_skew_partition(
        labels, n_clients=n_clients, alpha=alpha, seed=seed
    )
    return FLPartition(client_indices=client_indices, alpha=alpha, seed=seed)


# ---------------------------
# DataLoader helpers
# ---------------------------


def make_client_loader(
    train_dataset: datasets.CIFAR10,
    indices: List[int],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    subset = Subset(train_dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_test_loader(
    test_dataset: datasets.CIFAR10,
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------
# Convenience: end-to-end builder
# ---------------------------


def build_federated_cifar10(
    data_root: str = "./data",
    n_clients: int = 100,
    alpha: float = 0.1,
    seed: int = 0,
    aug_train: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10, FLPartition]:
    """
    Download CIFAR-10, create train/test datasets, and a fixed Dirichlet partition.
    """
    train_ds, test_ds = get_cifar10_datasets(data_root=data_root, aug_train=aug_train)
    partition = make_fl_partition_cifar10(
        train_ds, n_clients=n_clients, alpha=alpha, seed=seed
    )
    return train_ds, test_ds, partition


# ---------------------------
# Small self-test (optional)
# ---------------------------

if __name__ == "__main__":
    tr, te, part = build_federated_cifar10(n_clients=10, alpha=0.1, seed=0)
    sizes = [len(idx) for idx in part.client_indices]
    print(
        f"[OK] clients={len(sizes)} | avg per client={sum(sizes)/len(sizes):.1f} | alpha={part.alpha} | seed={part.seed}"
    )
