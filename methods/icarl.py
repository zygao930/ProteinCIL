"""
iCaRL: Incremental Classifier and Representation Learning.

Rebuffi et al., "iCaRL: Incremental classifier and representation learning", CVPR 2017.

Combines exemplar replay with knowledge distillation from the previous model.
"""

import copy
import torch
import torch.nn.functional as F
from methods._common import make_clf, evaluate, print_results, ReplayBuffer


def run_icarl(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
              emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
              buffer_size=2000, hidden=1024, **kwargs):
    clf = make_clf(emb_dim, num_classes, hidden=hidden, device=device)
    old_clf = None
    buf = ReplayBuffer(buffer_size)
    seen_classes = []

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        seen_classes.extend(pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")

        # Combine current data with buffer
        cur_emb = train_emb[tr].to(device)
        cur_lab = train_lab[tr].to(device)
        if len(buf) > 0:
            b_emb, b_lab = buf.emb.to(device), buf.lab.to(device)
            all_emb = torch.cat([cur_emb, b_emb])
            all_lab = torch.cat([cur_lab, b_lab])
        else:
            all_emb, all_lab = cur_emb, cur_lab

        # Train with distillation
        clf.train()
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        n = len(all_lab)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s+batch]
                logits = clf(all_emb[idx])
                loss = F.cross_entropy(logits, all_lab[idx])
                # Knowledge distillation from old model
                if old_clf is not None:
                    with torch.no_grad():
                        old_logits = old_clf(all_emb[idx])
                    old_classes = list(set(seen_classes) - set(pc))
                    if old_classes:
                        old_probs = torch.sigmoid(old_logits[:, old_classes])
                        new_log_probs = F.logsigmoid(logits[:, old_classes])
                        dist_loss = -torch.mean(old_probs * new_log_probs +
                                                (1 - old_probs) * torch.log(1 - torch.sigmoid(logits[:, old_classes]) + 1e-8))
                        loss = loss + dist_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

        old_clf = copy.deepcopy(clf)
        buf.update(train_emb[tr], train_lab[tr])

        # Evaluate using linear classifier (consistent with other methods)
        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(evaluate(clf, test_emb[te], test_lab[te], device) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results(f"iCaRL (buf={buffer_size})", R)
    return R
