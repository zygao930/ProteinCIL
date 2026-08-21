"""
RanPAC: Random Projections and Pre-trained Models for Continual Learning.

McDonnell et al., NeurIPS 2023.

Projects embeddings through a fixed random matrix with ReLU activation,
then solves ridge regression incrementally via recursive least squares.
"""

import math
import torch
import torch.nn.functional as F
from methods._common import print_results


def run_ranpac(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
               emb_dim, num_classes, device, proj_dim=4096, ridge_lambda=1.0,
               **kwargs):

    torch.manual_seed(42)
    W_proj = torch.randn(emb_dim, proj_dim, device=device) / math.sqrt(emb_dim)

    A = ridge_lambda * torch.eye(proj_dim, device=device, dtype=torch.float32)
    B = torch.zeros(proj_dim, num_classes, device=device, dtype=torch.float32)
    seen_classes = []
    chunk_size = 2048

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        seen_classes.extend(pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")

        emb = train_emb[tr].to(device).float()
        lab = train_lab[tr]

        for start in range(0, len(emb), chunk_size):
            end = min(start + chunk_size, len(emb))
            z_chunk = F.relu(emb[start:end] @ W_proj)
            lab_chunk = lab[start:end]
            Y_chunk = torch.zeros(end - start, num_classes, device=device, dtype=torch.float32)
            for i, l in enumerate(lab_chunk.tolist()):
                Y_chunk[i, l] = 1.0
            A = A + z_chunk.t() @ z_chunk
            B = B + z_chunk.t() @ Y_chunk

        W_clf = torch.linalg.solve(A, B)

        seen_mask = torch.full((num_classes,), float('-inf'), device=device)
        for c in seen_classes:
            seen_mask[c] = 0.0

        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            if not te:
                accs.append(0.0)
                continue
            t_emb = test_emb[te].to(device).float()
            t_lab = test_lab[te].to(device)
            correct = 0
            total = 0
            for start in range(0, len(t_emb), chunk_size):
                end = min(start + chunk_size, len(t_emb))
                t_z = F.relu(t_emb[start:end] @ W_proj)
                logits = t_z @ W_clf + seen_mask.unsqueeze(0)
                pred = logits.argmax(1)
                correct += (pred == t_lab[start:end]).sum().item()
                total += end - start
            accs.append(correct / total if total > 0 else 0.0)

        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")

    print_results(f"RanPAC (D={proj_dim}, lambda={ridge_lambda})", R)
    return R
