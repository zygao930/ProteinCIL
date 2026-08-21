"""
Naive fine-tuning baseline (incremental lower bound).

Trains on new-phase data only, with no mechanism to preserve old classes.
"""

import torch.nn as nn
from methods._common import make_clf, train_loop, evaluate, print_results


def run_naive(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
              emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
              hidden=1024, **kwargs):
    clf = make_clf(emb_dim, num_classes, hidden=hidden, device=device)
    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")
        train_loop(clf, train_emb[tr], train_lab[tr], device, lr, epochs, batch)
        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(evaluate(clf, test_emb[te], test_lab[te], device) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results("Naive", R)
    return R
