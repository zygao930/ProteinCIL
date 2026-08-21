"""
RER: Relational Experience Replay.

Wang et al., "Relational experience replay: Continual learning by adaptively tuning
task-wise relationship", IEEE Trans. Multimedia 2024.

Uses gradient cosine similarity between current and buffered losses to adaptively
weight the replay contribution.
"""

import torch
import torch.nn.functional as F
from methods._common import make_clf, evaluate, print_results, ReplayBuffer


def run_rer(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
            emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
            buffer_size=500, hidden=1024, **kwargs):
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

                if len(buf) > 0:
                    r_emb, r_lab = buf.sample(batch, device)

                    loss_cur = F.cross_entropy(clf(emb[idx]), lab[idx])
                    loss_buf = F.cross_entropy(clf(r_emb), r_lab)

                    opt.zero_grad()
                    loss_cur.backward(retain_graph=True)
                    g1 = torch.cat([p.grad.flatten() for p in clf.parameters() if p.grad is not None]).clone()

                    opt.zero_grad()
                    loss_buf.backward()
                    g2 = torch.cat([p.grad.flatten() for p in clf.parameters() if p.grad is not None]).clone()

                    cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
                    adaptive_alpha = max(0.0, min(1.0, (1.0 - cos_sim) / 2.0))

                    opt.zero_grad()
                    combined = g1 + adaptive_alpha * g2
                    offset = 0
                    for p in clf.parameters():
                        numel = p.numel()
                        p.grad = combined[offset:offset+numel].reshape(p.shape)
                        offset += numel
                    opt.step()
                else:
                    loss_cur = F.cross_entropy(clf(emb[idx]), lab[idx])
                    opt.zero_grad()
                    loss_cur.backward()
                    opt.step()

        buf.update(train_emb[tr], train_lab[tr])

        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(evaluate(clf, test_emb[te], test_lab[te], device) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results(f"RER (buf={buffer_size})", R)
    return R
