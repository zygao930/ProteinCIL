"""
Joint training baseline (offline upper bound).

Retrains on all data seen so far at each phase.
"""

import torch
import torch.nn as nn
from methods._common import make_clf, train_loop, evaluate, print_results


def run_joint(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
              emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
              hidden=1024, **kwargs):
    clf = make_clf(emb_dim, num_classes, hidden=hidden, device=device)
    R = []
    all_emb, all_lab = [], []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        all_emb.append(train_emb[tr])
        all_lab.append(train_lab[tr])
        combined_emb = torch.cat(all_emb)
        combined_lab = torch.cat(all_lab)
        print(f"Phase {pid}: {len(pc)} classes, total {len(combined_lab)} samples")
        for m in clf.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        train_loop(clf, combined_emb, combined_lab, device, lr, epochs, batch)
        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(evaluate(clf, test_emb[te], test_lab[te], device) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results("Joint", R)
    return R
