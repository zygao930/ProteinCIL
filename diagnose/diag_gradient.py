"""
Diagnostic 1 -- Hypothesis A:
Gradient-based stability mechanisms lose their signal under frozen encoders.

Vision CIL methods (EWC, RER, DER++, ...) rely on informative gradients:
EWC needs a Fisher matrix that identifies important parameters; RER needs
gradient-conflict signals between tasks. When the encoder is frozen, gradients
flow only through a small MLP head, so we test whether the two signals those
methods depend on actually carry information here.

We measure, for each encoder/dataset:
  (1) Inter-phase gradient conflict: cosine similarity between the head-gradient
      of phase t and phase t' (t != t'), evaluated at a shared init. If gradients
      barely conflict (cosine ~ 0 or positive), conflict-based methods (RER,
      PCGrad-style) have nothing to act on.
  (2) Fisher concentration: fraction of total Fisher information carried by the
      top-1% of head parameters. If Fisher is extremely concentrated / degenerate,
      EWC's per-parameter importance is uninformative for protecting old classes.

Output: diagnose/results/grad_{encoder}_{dataset}_seed{seed}.json
Aggregate later across seeds/encoders for the paper table.

Run:
  python diagnose/diag_gradient.py                       # all encoders/datasets
  python diagnose/diag_gradient.py --encoder esm2_650m --dataset ec_number
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import (ALL_ENCODERS, ALL_DATASETS, ALL_SEEDS,
                     load_embeddings, load_ec_phases, shuffle_phases)

HIDDEN = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_head(d, num_classes):
    return nn.Sequential(
        nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, num_classes)
    ).to(DEVICE)


def phase_gradient(head, emb, lab, class_ids, num_classes):
    """Mean cross-entropy gradient over samples of the given classes, flattened."""
    idx = np.isin(lab, list(class_ids))
    if idx.sum() == 0:
        return None
    X = torch.tensor(emb[idx], dtype=torch.float32, device=DEVICE)
    y = torch.tensor(lab[idx], dtype=torch.long, device=DEVICE)
    head.zero_grad()
    logits = head(X)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in head.parameters()])
    return g.detach().cpu().numpy()


def fisher_diagonal(head, emb, lab, class_ids):
    """
    Diagonal Fisher approx: E[grad(log p(y|x))^2] over the classes' samples.

    Uses per-sample gradients computed in one batched backward via the linear
    layers' structure, instead of a Python loop over samples. For a 2-layer MLP
    head we accumulate the squared gradients of the two Linear weights/biases,
    which dominate the Fisher and are what EWC actually regularizes.
    """
    idx = np.isin(lab, list(class_ids))
    Xall = emb[idx]
    yall = lab[idx]
    n_use = min(256, len(yall))
    sel = np.random.choice(len(yall), n_use, replace=False)
    X = torch.tensor(Xall[sel], dtype=torch.float32, device=DEVICE)
    y = torch.tensor(yall[sel], dtype=torch.long, device=DEVICE)

    # forward keeping activations we need for per-sample grads
    lin1, act, lin2 = head[0], head[1], head[2]
    h_pre = X @ lin1.weight.t() + lin1.bias          # [n, H]
    h = act(h_pre)                                    # [n, H]
    logits = h @ lin2.weight.t() + lin2.bias          # [n, C]
    logp = F.log_softmax(logits, dim=1)

    # d log p_y / d logits = onehot(y) - softmax(logits)   -> [n, C]
    p = logp.exp()
    g_logits = -p
    g_logits[torch.arange(len(y)), y] += 1.0          # [n, C]

    # grads w.r.t. lin2 (weight [C,H], bias [C]) per sample, squared then summed
    # dL/dW2 = g_logits^T outer h  ->  per-sample: g_logits[:,c] * h[:,k]
    fisher_W2 = (g_logits.pow(2).t() @ h.pow(2))      # [C, H]
    fisher_b2 = g_logits.pow(2).sum(0)                # [C]

    # backprop g_logits to hidden: g_h = g_logits @ W2, then through ReLU
    g_h = g_logits @ lin2.weight                      # [n, H]
    g_h = g_h * (h_pre > 0).float()                   # ReLU grad
    # grads w.r.t. lin1 (weight [H,d], bias [H])
    fisher_W1 = (g_h.pow(2).t() @ X.pow(2))           # [H, d]
    fisher_b1 = g_h.pow(2).sum(0)                     # [H]

    fisher = torch.cat([
        fisher_W1.flatten(), fisher_b1.flatten(),
        fisher_W2.flatten(), fisher_b2.flatten(),
    ]).detach().cpu().numpy() / n_use
    return fisher


def run_one(encoder, dataset, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    emb, lab = load_embeddings(dataset, encoder, "train")
    phases, num_classes = load_ec_phases(dataset)
    phases = shuffle_phases(phases, seed)
    d = emb.shape[1]

    head = make_head(d, num_classes)

    # (1) inter-phase gradient conflict at shared init
    grads = []
    for p in phases:
        g = phase_gradient(head, emb, lab, set(p), num_classes)
        if g is not None:
            grads.append(g)
    cosines = []
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            gi, gj = grads[i], grads[j]
            denom = (np.linalg.norm(gi) * np.linalg.norm(gj)) + 1e-12
            cosines.append(float(gi @ gj / denom))
    cosines = np.array(cosines)

    # (2) Fisher concentration over the first (base) phase
    fisher = fisher_diagonal(head, emb, lab, set(phases[0]))
    fisher_sorted = np.sort(fisher)[::-1]
    total = fisher_sorted.sum() + 1e-12
    top1pct = int(max(1, 0.01 * len(fisher_sorted)))
    conc_top1 = float(fisher_sorted[:top1pct].sum() / total)
    # effective number of parameters (participation ratio) of the Fisher diagonal
    part_ratio = float((fisher.sum() ** 2) / ((fisher ** 2).sum() + 1e-12))

    result = {
        "encoder": encoder, "dataset": dataset, "seed": seed,
        "dim": d, "num_classes": num_classes, "n_phases": len(phases),
        "grad_conflict_mean_cosine": float(cosines.mean()),
        "grad_conflict_frac_negative": float((cosines < 0).mean()),
        "grad_conflict_min_cosine": float(cosines.min()),
        "fisher_top1pct_concentration": conc_top1,
        "fisher_participation_ratio": part_ratio,
        "fisher_n_params": int(len(fisher)),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", default=None)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--seed", action="append", type=int, default=None)
    args = ap.parse_args()

    encoders = args.encoder or ALL_ENCODERS
    datasets = args.dataset or ALL_DATASETS
    seeds = args.seed or ALL_SEEDS

    out_dir = Path("diagnose/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for enc in encoders:
        for ds in datasets:
            for sd in seeds:
                try:
                    res = run_one(enc, ds, sd)
                except FileNotFoundError as e:
                    print(f"SKIP {enc}/{ds}/seed{sd}: {e}")
                    continue
                fp = out_dir / f"grad_{enc}_{ds}_seed{sd}.json"
                with open(fp, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[{enc}/{ds}/s{sd}] "
                      f"conflict_cos={res['grad_conflict_mean_cosine']:+.3f} "
                      f"frac_neg={res['grad_conflict_frac_negative']:.2f} "
                      f"fisher_conc(top1%)={res['fisher_top1pct_concentration']:.2f} "
                      f"-> {fp.name}")


if __name__ == "__main__":
    main()