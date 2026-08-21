"""
Diagnostic 2 -- Hypothesis B:
No fixed strategy generalizes across encoders because the five frozen encoders
induce fundamentally different embedding geometries.

Each baseline implicitly assumes a geometry: RanPAC assumes a random projection
preserves distances; FeCAM assumes a single shared covariance describes all
classes; EASE assumes classes are separable in a learned subspace. If the
geometry itself varies wildly across encoders, no single fixed assumption can
hold everywhere -- which is exactly what Table 1's per-encoder collapses show.

We compute encoder-agnostic geometry descriptors on the training embeddings:
  - dim: raw embedding dimensionality
  - r95 / r99: intrinsic rank (PCs explaining 95% / 99% of pooled within-class
    variance) -- the quantity RanPAC's projection budget and FeCAM's covariance
    both implicitly depend on
  - effective_rank: exp(entropy of normalized eigenvalues), a smooth rank proxy
  - cov_condition_number: lambda_max / lambda_min(+eps) of the pooled within-class
    covariance -- large values mean covariance inversion (FeCAM / Mahalanobis) is
    ill-posed
  - between_over_within: ratio of between-class to within-class scatter trace
    (Fisher-style separability)
  - mean_pairwise_proto_dist / std: scale + spread of class prototypes

Output: diagnose/results/geom_{encoder}_{dataset}_seed{seed}.json

Run:
  python diagnose/diag_geometry.py
  python diagnose/diag_geometry.py --encoder gearnet_edge --dataset ec_number
"""

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (ALL_ENCODERS, ALL_DATASETS, ALL_SEEDS,
                     load_embeddings, class_means, pooled_within_scatter)


def intrinsic_rank(eigvals, thresh):
    """Smallest r whose top-r eigenvalues explain >= thresh of the total."""
    ev = np.sort(eigvals)[::-1]
    ev = np.clip(ev, 0, None)
    csum = np.cumsum(ev) / (ev.sum() + 1e-12)
    return int(np.searchsorted(csum, thresh) + 1)


def effective_rank(eigvals):
    """exp(Shannon entropy of the normalized eigenvalue spectrum)."""
    ev = np.clip(eigvals, 0, None)
    p = ev / (ev.sum() + 1e-12)
    p = p[p > 0]
    ent = -(p * np.log(p)).sum()
    return float(np.exp(ent))


def run_one(encoder, dataset, seed):
    np.random.seed(seed)
    emb, lab = load_embeddings(dataset, encoder, "train")
    d = emb.shape[1]
    num_classes = int(lab.max()) + 1

    # pooled within-class covariance over all classes
    S, n_total = pooled_within_scatter(emb, lab, range(num_classes))
    cov = S / max(1, n_total - num_classes)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)

    lam_max = float(eigvals.max())
    lam_min_pos = float(eigvals[eigvals > 1e-12].min()) if (eigvals > 1e-12).any() else 1e-12
    cond = lam_max / (lam_min_pos + 1e-12)

    r95 = intrinsic_rank(eigvals, 0.95)
    r99 = intrinsic_rank(eigvals, 0.99)
    eff_rank = effective_rank(eigvals)

    # between vs within separability
    means, counts = class_means(emb, lab, num_classes)
    valid = counts > 0
    global_mean = emb.mean(axis=0, keepdims=True)
    between = 0.0
    for c in np.where(valid)[0]:
        diff = (means[c] - global_mean[0])
        between += counts[c] * float(diff @ diff)
    within_trace = float(np.trace(S))
    between_over_within = between / (within_trace + 1e-12)

    # prototype pairwise distances (subsample for large C)
    idx = np.where(valid)[0]
    if len(idx) > 400:
        idx = np.random.choice(idx, 400, replace=False)
    M = means[idx]
    dists = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
    triu = dists[np.triu_indices(len(idx), k=1)]

    result = {
        "encoder": encoder, "dataset": dataset, "seed": seed,
        "dim": d, "num_classes": num_classes,
        "n_samples": int(len(lab)),
        "samples_per_class": float(len(lab) / num_classes),
        "r95": r95, "r99": r99,
        "r95_over_dim": float(r95 / d),
        "effective_rank": eff_rank,
        "cov_condition_number": cond,
        "cov_lambda_max": lam_max,
        "cov_lambda_min_pos": lam_min_pos,
        "between_over_within": between_over_within,
        "mean_pairwise_proto_dist": float(triu.mean()),
        "std_pairwise_proto_dist": float(triu.std()),
        "proto_dist_cv": float(triu.std() / (triu.mean() + 1e-12)),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", default=None)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--seed", action="append", type=int, default=None)
    args = ap.parse_args()

    encoders = args.encoder or ALL_ENCODERS
    datasets = args.dataset or ALL_DATASETS
    seeds = args.seed or ALL_SEEDS

    out_dir = Path("diagnose/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for enc in encoders:
        for ds in datasets:
            for sd in seeds:
                try:
                    res = run_one(enc, ds, sd)
                except FileNotFoundError as e:
                    print(f"SKIP {enc}/{ds}/seed{sd}: {e}")
                    continue
                fp = out_dir / f"geom_{enc}_{ds}_seed{sd}.json"
                with open(fp, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[{enc}/{ds}/s{sd}] "
                      f"dim={res['dim']} r95={res['r95']} "
                      f"eff_rank={res['effective_rank']:.1f} "
                      f"cond={res['cov_condition_number']:.1e} "
                      f"B/W={res['between_over_within']:.3f} -> {fp.name}")


if __name__ == "__main__":
    main()
