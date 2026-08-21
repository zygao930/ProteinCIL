"""
Elastic Weight Consolidation (EWC).

Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017.

Uses a Fisher-weighted L2 penalty to regularize parameter drift from previous tasks.
"""

import torch
import torch.nn.functional as F
from methods._common import make_clf, train_loop, evaluate, print_results


def run_ewc(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
            emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
            ewc_lambda=2000.0, hidden=1024, **kwargs):
    clf = make_clf(emb_dim, num_classes, hidden=hidden, device=device)
    fisher_list, params_list = [], []

    def compute_fisher(emb, lab):
        clf.eval()
        fisher = {n: torch.zeros_like(p) for n, p in clf.named_parameters() if p.requires_grad}
        emb, lab = emb.to(device), lab.to(device)
        n = len(lab)
        for s in range(0, n, batch):
            b_emb, b_lab = emb[s:s+batch], lab[s:s+batch]
            clf.zero_grad()
            loss = F.cross_entropy(clf(b_emb), b_lab)
            loss.backward()
            for na, p in clf.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[na] += p.grad.data.pow(2) * len(b_lab)
        for na in fisher:
            fisher[na] /= n
        return fisher

    def ewc_penalty():
        pen = torch.tensor(0.0, device=device)
        for fisher, old_p in zip(fisher_list, params_list):
            for n, p in clf.named_parameters():
                if n in fisher:
                    pen = pen + (fisher[n] * (p - old_p[n]).pow(2)).sum()
        return pen

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")
        train_loop(clf, train_emb[tr], train_lab[tr], device, lr, epochs, batch,
                   extra_loss_fn=lambda: (ewc_lambda / 2.0) * ewc_penalty())
        fisher = compute_fisher(train_emb[tr], train_lab[tr])
        fisher_list.append(fisher)
        params_list.append({n: p.data.clone() for n, p in clf.named_parameters() if p.requires_grad})
        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(evaluate(clf, test_emb[te], test_lab[te], device) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results(f"EWC (lambda={ewc_lambda})", R)
    return R
