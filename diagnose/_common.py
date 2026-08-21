"""
Shared loading utilities for the failure-diagnosis scripts.

Reuses the exact embedding layout and EC-first-digit phase construction
from run.py, so diagnostics operate on the same data the benchmark used.

Embedding layout produced by extract.py:
    data/{dataset}/{encoder}/train_embeddings.pt   float [N, dim]
    data/{dataset}/{encoder}/train_labels.pt       long  [N]
    data/{dataset}/{encoder}/test_embeddings.pt
    data/{dataset}/{encoder}/test_labels.pt
    data/{dataset}/ec_numbers.json                 list[str], len == num_classes
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ALL_ENCODERS = ["esm2_650m", "esm2_150m", "gearnet_edge", "saprot", "protbert"]
ALL_DATASETS = ["enzyme_reaction", "ec_number"]
ALL_SEEDS = [0, 42, 7]
DATA_ROOT = Path("./data")


def load_embeddings(dataset, encoder, split="train", data_root=DATA_ROOT):
    """Return (emb, lab) as numpy arrays for one dataset/encoder/split."""
    cache = Path(data_root) / dataset / encoder
    emb = torch.load(cache / f"{split}_embeddings.pt", weights_only=True)
    lab = torch.load(cache / f"{split}_labels.pt", weights_only=True)
    return emb.numpy().astype(np.float64), lab.numpy().astype(np.int64)


def load_ec_phases(dataset, data_root=DATA_ROOT):
    """
    Rebuild the canonical EC-first-digit phases used in run.py.
    Returns (phases, num_classes) where phases is a list of lists of class ids.
    """
    data_dir = Path(data_root) / dataset
    with open(data_dir / "ec_numbers.json") as f:
        labels_list = json.load(f)
    groups = defaultdict(list)
    for idx, ec in enumerate(labels_list):
        groups[ec.split(".")[0]].append(idx)
    phases = [groups[k] for k in sorted(groups, key=int)]
    return phases, len(labels_list)


def shuffle_phases(phases, seed):
    """Match run.py's per-seed phase shuffling."""
    rng = random.Random(seed + 42)
    order = list(range(len(phases)))
    rng.shuffle(order)
    return [phases[i] for i in order]


def class_means(emb, lab, num_classes=None):
    """Per-class mean (prototype). Returns (means [C, d], counts [C])."""
    if num_classes is None:
        num_classes = int(lab.max()) + 1
    d = emb.shape[1]
    means = np.zeros((num_classes, d))
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mask = lab == c
        counts[c] = mask.sum()
        if counts[c] > 0:
            means[c] = emb[mask].mean(axis=0)
    return means, counts


def pooled_within_scatter(emb, lab, class_ids):
    """
    Pooled within-class scatter matrix S = sum_c sum_i (x-mu_c)(x-mu_c)^T
    over the given class_ids, plus the total sample count used.
    """
    d = emb.shape[1]
    S = np.zeros((d, d))
    n_total = 0
    for c in class_ids:
        mask = lab == c
        n_c = mask.sum()
        if n_c < 2:
            continue
        X = emb[mask]
        Xc = X - X.mean(axis=0, keepdims=True)
        S += Xc.T @ Xc
        n_total += n_c
    return S, n_total
