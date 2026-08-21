"""
Experience Replay baseline.

Chaudhry et al., "On tiny episodic memories in continual learning", 2019.

Maintains a fixed-size replay buffer with reservoir sampling and mixes
replayed samples with current-phase data during training.
"""

import torch
import torch.nn.functional as F
from methods._common import make_clf, evaluate, print_results, ReplayBuffer


def run_replay(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
               emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
               buffer_size=500, replay_alpha=0.5, hidden=1024, **kwargs):
    clf = make_clf(emb_dim, num_classes, hidden=hidden, device=device)
    buf = ReplayBuffer(buffer_size)

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")
        clf.train()
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        emb, lab = train_emb[tr].to(device), train_lab[tr].to(device)
        n = len(lab)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s+batch]
                loss = F.cross_entropy(clf(emb[idx]), lab[idx])
                if len(buf) > 0:
                    r_emb, r_lab = buf.sample(batch, device)
                    loss = loss + replay_alpha * F.cross_entropy(clf(r_emb), r_lab)
                opt.zero_grad()
                loss.backward()
                opt.step()
        buf.update(train_emb[tr], train_lab[tr])
        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(evaluate(clf, test_emb[te], test_lab[te], device) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results(f"Replay (buf={buffer_size}, alpha={replay_alpha})", R)
    return R
