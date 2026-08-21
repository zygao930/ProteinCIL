"""
FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free
Continual Learning.

Goswami et al., NeurIPS 2023.

Builds class prototypes (means) and a shared covariance matrix updated
incrementally. Classification uses Mahalanobis distance to prototypes.
"""

import torch
from methods._common import print_results


def run_fecam_common(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
                     emb_dim, shrinkage=1e-4, device="cpu", **kwargs):
    class_means = {}
    sum_cov = torch.zeros(emb_dim, emb_dim, device=device)
    num_classes_seen = 0
    cov_inv = None

    def update_prototypes(emb, lab):
        nonlocal sum_cov, num_classes_seen, cov_inv
        emb = emb.to(device)
        for c in torch.unique(lab).tolist():
            mask = lab == c
            c_emb = emb[mask]
            mean = c_emb.mean(dim=0)
            class_means[c] = mean
            if c_emb.shape[0] > 1:
                centered = c_emb - mean
                cov = (centered.T @ centered) / (c_emb.shape[0] - 1)
            else:
                cov = torch.eye(emb_dim, device=device)
            sum_cov += cov
            num_classes_seen += 1
        avg_cov = sum_cov / num_classes_seen + shrinkage * torch.eye(emb_dim, device=device)
        cov_inv = torch.linalg.inv(avg_cov)

    def eval_fecam_common(emb, lab, eval_batch=64):
        seen = sorted(class_means.keys())
        if not seen:
            return 0.0
        means_cpu = torch.stack([class_means[c] for c in seen]).cpu()
        cov_inv_cpu = cov_inv.cpu()
        correct, total = 0, 0
        for s in range(0, len(emb), eval_batch):
            batch_emb = emb[s:s+eval_batch].cpu()
            batch_lab = lab[s:s+eval_batch]
            diff = batch_emb.unsqueeze(1) - means_cpu.unsqueeze(0)
            dists = (diff @ cov_inv_cpu * diff).sum(dim=2)
            pred_idx = dists.argmin(dim=1)
            pred_class = torch.tensor([seen[i] for i in pred_idx])
            correct += (pred_class == batch_lab).sum().item()
            total += len(batch_lab)
        return correct / total if total > 0 else 0.0

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")
        update_prototypes(train_emb[tr], train_lab[tr])
        accs = []
        for j in range(pid + 1):
            te = get_indices(test_lab, phases[j])
            accs.append(eval_fecam_common(test_emb[te], test_lab[te]) if te else 0.0)
        R.append(accs)
        print(f"  accs: {[f'{a:.3f}' for a in accs]}")
    print_results(f"FeCAM-Common (shrinkage={shrinkage})", R)
    return R
