"""
DER++: Dark Experience Replay++.

Buzzega et al., "Dark experience for general continual learning: a strong, simple baseline",
NeurIPS 2020.

Stores logits alongside exemplars and adds an MSE consistency loss on buffered logits.
"""

import torch
import torch.nn.functional as F
from methods._common import make_clf, evaluate, print_results, ReplayBuffer


def run_derpp(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
              emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
              buffer_size=500, alpha=0.5, beta=0.5, hidden=1024, **kwargs):
    clf = make_clf(emb_dim, num_classes, hidden=hidden, device=device)
    buf = ReplayBuffer(buffer_size)

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")

        clf.train()
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        emb = train_emb[tr].to(device)
        lab = train_lab[tr].to(device)
        n = len(lab)

        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s+batch]
                logits = clf(emb[idx])
                loss = F.cross_entropy(logits, lab[idx])

                if len(buf) > 0:
                    buf_sample = buf.sample(batch, device)
                    r_emb, r_lab = buf_sample[0], buf_sample[1]
                    r_logits = clf(r_emb)
                    loss = loss + alpha * F.cross_entropy(r_logits, r_lab)
                    if len(buf_sample) > 2:
                        r_old_logits = buf_sample[2]
                        loss = loss + beta * F.mse_loss(r_logits, r_old_logits)

                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            cur_logits = clf(emb).cpu()
        buf.update(train_emb[tr], train_lab[tr], logits=cur_logits)

        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(evaluate(clf, test_emb[te], test_lab[te], device) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results(f"DER++ (buf={buffer_size}, alpha={alpha}, beta={beta})", R)
    return R
