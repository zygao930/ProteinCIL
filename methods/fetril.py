"""
FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning.

Petit et al., "FeTrIL: Feature translation for exemplar-free class-incremental learning",
WACV 2023.

Generates pseudo-features for old classes by translating new-class embeddings
using the difference between old and new class means, then retrains a classifier
on combined real and pseudo features.
"""

import torch
import torch.nn as nn
from methods._common import make_clf, train_loop, evaluate, print_results


def run_fetril(train_emb, train_lab, test_emb, test_lab, phases, get_indices,
               emb_dim, num_classes, device, lr=1e-3, epochs=50, batch=64,
               hidden=1024, **kwargs):
    clf = make_clf(emb_dim, num_classes, hidden=hidden, device=device)
    class_means = {}

    R = []
    for pid, pc in enumerate(phases):
        tr = get_indices(train_lab, pc)
        print(f"Phase {pid}: {len(pc)} classes, {len(tr)} samples")

        # Compute means for new classes
        for c in pc:
            mask = train_lab[tr] == c
            if mask.any():
                class_means[c] = train_emb[tr][mask].mean(dim=0)

        # Generate pseudo features for old classes
        pseudo_emb_list = []
        pseudo_lab_list = []
        if pid > 0:
            old_classes = [c for c in class_means if c not in pc]

            # Precompute new class embeddings once
            new_class_data = {}
            for new_c in pc:
                new_mask = train_lab[tr] == new_c
                if new_mask.any():
                    new_class_data[new_c] = train_emb[tr][new_mask]
            available_new = list(new_class_data.keys())

            # Adaptive limits based on dataset scale
            MAX_NEW_CLASSES_SAMPLE = min(len(available_new), max(5, len(pc) // 4))
            MAX_PSEUDO_PER_OLD = max(50, len(tr) // max(len(old_classes), 1))
            MAX_TOTAL_PSEUDO = min(max(2000, len(tr)), 50000)

            total_pseudo = 0
            for old_c in old_classes:
                if total_pseudo >= MAX_TOTAL_PSEUDO:
                    break
                old_mean = class_means[old_c]

                # Sample a subset of new classes if too many
                if len(available_new) > MAX_NEW_CLASSES_SAMPLE:
                    sampled_new = [available_new[i] for i in torch.randperm(len(available_new))[:MAX_NEW_CLASSES_SAMPLE].tolist()]
                else:
                    sampled_new = available_new

                old_pseudo = []
                for new_c in sampled_new:
                    new_emb = new_class_data[new_c]
                    shift = old_mean - class_means[new_c]
                    old_pseudo.append(new_emb + shift)

                if old_pseudo:
                    combined = torch.cat(old_pseudo)
                    if len(combined) > MAX_PSEUDO_PER_OLD:
                        idx = torch.randperm(len(combined))[:MAX_PSEUDO_PER_OLD]
                        combined = combined[idx]
                    remaining = MAX_TOTAL_PSEUDO - total_pseudo
                    if len(combined) > remaining:
                        combined = combined[:remaining]
                    pseudo_emb_list.append(combined)
                    pseudo_lab_list.append(torch.full((len(combined),), old_c, dtype=torch.long))
                    total_pseudo += len(combined)

        # Combine real + pseudo data
        all_emb_parts = [train_emb[tr]]
        all_lab_parts = [train_lab[tr]]
        if pseudo_emb_list:
            all_emb_parts.extend(pseudo_emb_list)
            all_lab_parts.extend(pseudo_lab_list)
        combined_emb = torch.cat(all_emb_parts)
        combined_lab = torch.cat(all_lab_parts)

        # Re init and train classifier on combined data
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
    print_results("FeTrIL", R)
    return R
