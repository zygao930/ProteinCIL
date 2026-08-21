"""
EASE: Expandable Subspace Ensemble for Pre-trained Model-based
Class-Incremental Learning.

Zhou et al., CVPR 2024.

Trains a low-rank adapter per phase and classifies via nearest-prototype
in the ensemble of adapted embedding spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from methods._common import print_results


class LowRankAdapter(nn.Module):
    def __init__(self, emb_dim, rank, init_data=None):
        super().__init__()
        self.down = nn.Linear(emb_dim, rank, bias=False)
        self.up = nn.Linear(rank, emb_dim, bias=False)
        if init_data is not None:
            with torch.no_grad():
                centered = init_data - init_data.mean(0)
                _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                k = min(rank, Vt.shape[0])
                self.down.weight[:k] = Vt[:k]
                self.up.weight[:, :k] = Vt[:k].t()
        else:
            nn.init.kaiming_uniform_(self.down.weight)
            nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.relu(self.down(x)))


def run_ease(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
             emb_dim, num_classes, device, subspace_dim=256,
             lr=1e-3, epochs=50, batch=64, **kwargs):

    adapters = []
    prototypes = {}
    seen_classes = []

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        seen_classes.extend(pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")

        emb = train_emb[tr].to(device).float()
        lab = train_lab[tr].to(device)

        for c in pc:
            mask = (lab == c)
            if mask.any():
                prototypes[c] = emb[mask].mean(0)

        adapter_t = LowRankAdapter(emb_dim, subspace_dim, init_data=emb).to(device)
        temp_clf = nn.Linear(emb_dim, num_classes).to(device)

        for a in adapters:
            for p in a.parameters():
                p.requires_grad = False

        opt = torch.optim.Adam(
            list(adapter_t.parameters()) + list(temp_clf.parameters()), lr=lr)
        n = len(lab)

        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                x, y = emb[idx], lab[idx]
                loss = F.cross_entropy(temp_clf(x + adapter_t(x)), y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        adapter_t.eval()
        adapters.append(adapter_t)
        del temp_clf

        # Precompute adapted prototypes for all adapters (once per phase)
        proto_stack = torch.stack([prototypes[c] for c in seen_classes]).to(device)
        adapted_protos_list = []
        with torch.no_grad():
            for a in adapters:
                adapted_protos_list.append((proto_stack + a(proto_stack)).cpu())
        n_adapters = len(adapters)

        # Precompute adapted test embeddings per adapter (once per phase)
        all_te = list(range(len(test_lab)))
        t_emb_full = test_emb.float()
        adapted_test_list = []
        with torch.no_grad():
            for a in adapters:
                chunks = []
                for start in range(0, len(t_emb_full), 2048):
                    end = min(start + 2048, len(t_emb_full))
                    x = t_emb_full[start:end].to(device)
                    chunks.append((x + a(x)).cpu())
                adapted_test_list.append(torch.cat(chunks))

        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            if not te:
                accs.append(0.0)
                continue

            t_lab_j = test_lab[te]
            n_test = len(te)
            ensemble_dists = torch.zeros(n_test, len(seen_classes))

            for si in range(n_adapters):
                at = adapted_test_list[si][te]       # (n_test, d)
                ap = adapted_protos_list[si]          # (C, d)
                # Expanded quadratic: ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2
                at_sq = (at ** 2).sum(1, keepdim=True)        # (n_test, 1)
                ap_sq = (ap ** 2).sum(1, keepdim=True).t()    # (1, C)
                ensemble_dists += at_sq - 2 * (at @ ap.t()) + ap_sq

            ensemble_dists /= n_adapters
            pred_idx = ensemble_dists.argmin(1)
            pred_classes = torch.tensor([seen_classes[i] for i in pred_idx.tolist()])
            acc = (pred_classes == t_lab_j).float().mean().item()
            accs.append(acc)

        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")

    print_results(f"EASE (k={subspace_dim})", R)
    return R
