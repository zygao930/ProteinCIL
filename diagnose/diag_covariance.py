"""
Hypothesis C: shared-covariance prototype methods (FeCAM) collapse at scale.
Part 1 (estimation): shrinkage alpha / samples-per-dim show estimation is fine.
Part 2 (capacity): shared vs grouped vs per-class Mahalanobis accuracy shows the
real failure is the single shared covariance's capacity, not estimation.

Run: python diagnose/diag_covariance.py [--encoder E] [--dataset D] [--seed S] [--groups K]
Output: diagnose/results/cov_{encoder}_{dataset}_seed{seed}.json (per_phase inside).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import (ALL_ENCODERS, ALL_DATASETS, ALL_SEEDS,
                     load_embeddings, load_ec_phases, shuffle_phases,
                     pooled_within_scatter)

SHRINK = 1e-4          # matches FeCAM-Common's shrinkage in methods.py
N_GROUPS_DEFAULT = 8   # covariance groups for the "grouped" capacity model
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_t(x):
    return torch.as_tensor(x, dtype=torch.float32, device=_DEVICE)


# ---------------------------------------------------------------- estimation --

def intrinsic_rank(cov, thresh=0.95):
    ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    csum = np.cumsum(ev) / (ev.sum() + 1e-12)
    return int(np.searchsorted(csum, thresh) + 1)


def oas_shrinkage(cov, n):
    """OAS-style isotropic shrinkage intensity. Returns (alpha, cov_shrunk)."""
    d = cov.shape[0]
    mu = np.trace(cov) / d
    target = mu * np.eye(d)
    tr_cov2 = np.sum(cov * cov)
    tr_cov_sq = np.trace(cov) ** 2
    num = (1 - 2.0 / d) * tr_cov2 + tr_cov_sq
    den = (n + 1 - 2.0 / d) * (tr_cov2 - tr_cov_sq / d)
    alpha = 1.0 if den <= 0 else float(min(1.0, num / den))
    cov_shrunk = (1 - alpha) * cov + alpha * target
    return alpha, cov_shrunk


def normalized_cov(S, n, num_classes_seen):
    return S / max(1, n - num_classes_seen)


# ------------------------------------------------------------------ capacity --

def regularized_cov(S, n, d, extra=SHRINK):
    cov = S / max(1, n)
    return cov + extra * np.eye(d)


def class_stats(emb, lab, class_ids):
    """Return {c: (mean [d], diag_scatter [d], count)} for the given classes.
    Stores only diagonal scatter (d,) not full d x d, so memory is O(C*d) not
    O(C*d^2) -- essential at C~5000, d~3000."""
    stats = {}
    for c in class_ids:
        mask = lab == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        X = emb[mask]
        mu = X.mean(axis=0)
        Xc = X - mu
        diag_scatter = (Xc * Xc).sum(axis=0)       # [d], per-dim scatter only
        stats[c] = (mu, diag_scatter, n_c)
    return stats




def kmeans_groups(prototypes, k, seed, iters=25):
    """Tiny k-means over prototypes -> group id per prototype row."""
    rng = np.random.default_rng(seed)
    n = len(prototypes)
    k = min(k, n)
    centers = prototypes[rng.choice(n, k, replace=False)].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((prototypes[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        new = d2.argmin(1)
        if np.array_equal(new, assign):
            break
        assign = new
        for g in range(k):
            m = assign == g
            if m.any():
                centers[g] = prototypes[m].mean(0)
    return assign


def mahalanobis_acc_diag(test_emb, test_lab, means, inv_vars, logdets):
    """Fast path for diagonal per-class covariances. inv_vars: [C,d] inverse
    variances; logdets: [C] sum(log var). Fully vectorized, no inversion."""
    order = np.arange(len(means))
    M = _to_t(means)                    # [C, d]
    IV = _to_t(inv_vars)                # [C, d]
    LD = _to_t(logdets)                 # [C]
    IVM = IV * M                        # [C, d] precomputed
    term3_all = (IV * M * M).sum(1)     # [C]
    N = len(test_lab)
    if N == 0:
        return 0.0
    C = M.shape[0]
    cs = 512
    TB = 2048
    correct = 0
    for ts in range(0, N, TB):
        Xb = _to_t(test_emb[ts:ts+TB])           # [b, d]
        yb = torch.as_tensor(test_lab[ts:ts+TB], device=_DEVICE)
        x2b = Xb * Xb                            # [b, d]
        best = torch.full((len(Xb),), float("inf"), device=_DEVICE)
        pred = torch.full((len(Xb),), -1, dtype=torch.long, device=_DEVICE)
        for s in range(0, C, cs):
            term1 = x2b @ IV[s:s+cs].T           # [b, k]
            term2 = -2.0 * (Xb @ IVM[s:s+cs].T)  # [b, k]
            dist = term1 + term2 + term3_all[s:s+cs].unsqueeze(0) + LD[s:s+cs].unsqueeze(0)
            gmin, garg = dist.min(dim=1)
            upd = gmin < best
            best[upd] = gmin[upd]
            pred[upd] = torch.as_tensor(order[s:s+cs], device=_DEVICE)[garg[upd]]
        correct += int((pred == yb).sum().item())
    return correct / N


def _to_t(x):
    return torch.as_tensor(x, dtype=torch.float32, device=_DEVICE)


def mahalanobis_acc(test_emb, test_lab, means, precisions, group_of, class_order,
                    logdets=None):
    """Min-Mahalanobis, vectorized per shared-precision group via Cholesky
    whitening (Mahalanobis -> squared Euclidean). log|Sigma| added per group."""
    if logdets is None:
        logdets = {g: 0.0 for g in precisions}

    order = np.array(class_order)
    group_of = np.array(group_of)
    means_t = _to_t(means)                         # [C, d]
    N = len(test_lab)
    if N == 0:
        return 0.0

    # precompute per-group whitening of the class means (small: k x d)
    Ws, Mw_g, order_g = {}, {}, {}
    for g in precisions:
        cls_idx = np.where(group_of == g)[0]
        if len(cls_idx) == 0:
            continue
        P = _to_t(precisions[g])
        try:
            W = torch.linalg.cholesky(P, upper=True)
        except Exception:
            evals, evecs = torch.linalg.eigh(P)
            evals = torch.clamp(evals, min=0)
            W = (evecs * evals.sqrt()) @ evecs.T
        Ws[g] = W
        Mw_g[g] = means_t[cls_idx] @ W.T           # [k, d]
        order_g[g] = order[cls_idx]

    correct = 0
    TB = 2048                                      # test batch to bound memory
    for s in range(0, N, TB):
        Xb = _to_t(test_emb[s:s+TB])               # [b, d]
        yb = torch.as_tensor(test_lab[s:s+TB], device=_DEVICE)
        best = torch.full((len(Xb),), float("inf"), device=_DEVICE)
        pred = torch.full((len(Xb),), -1, dtype=torch.long, device=_DEVICE)
        for g in Ws:
            Xw = Xb @ Ws[g].T                       # [b, d]
            Mw = Mw_g[g]                            # [k, d]
            x2 = (Xw * Xw).sum(1, keepdim=True)
            m2 = (Mw * Mw).sum(1)
            dist = x2 + m2.unsqueeze(0) - 2.0 * (Xw @ Mw.T) + float(logdets[g])
            gmin, garg = dist.min(dim=1)
            upd = gmin < best
            best[upd] = gmin[upd]
            pred[upd] = torch.as_tensor(order_g[g], device=_DEVICE)[garg[upd]]
        correct += int((pred == yb).sum().item())
    return correct / N


# ---------------------------------------------------------------------- main --

def run_one(encoder, dataset, seed, n_groups):
    np.random.seed(seed)
    tr_emb, tr_lab = load_embeddings(dataset, encoder, "train")
    te_emb, te_lab = load_embeddings(dataset, encoder, "test")
    phases, num_classes = load_ec_phases(dataset)
    phases = shuffle_phases(phases, seed)
    d = tr_emb.shape[1]

    seen = []
    S_running = np.zeros((d, d))
    n_running = 0
    per_phase = []

    for t, p in enumerate(phases):
        seen.extend(p)
        stats = class_stats(tr_emb, tr_lab, seen)
        order = sorted(stats.keys())
        if len(order) < 2:
            continue
        n_seen_cls = len(order)

        # -------- PART 1: estimation quality of the shared covariance --------
        S_p, n_p = pooled_within_scatter(tr_emb, tr_lab, p)
        S_running = S_running + S_p
        n_running += n_p
        cov = normalized_cov(S_running, n_running, n_seen_cls)
        r95 = intrinsic_rank(cov, 0.95)
        alpha, cov_shrunk = oas_shrinkage(cov, n_running)
        sign, logabsdet = np.linalg.slogdet(cov_shrunk)
        logdet_per_dim = float(logabsdet / d) if sign > 0 else float("nan")

        # -------- PART 2: capacity of shared vs grouped vs per-class ----------
        means = np.stack([stats[c][0] for c in order])
        te_mask = np.isin(te_lab, order)
        Xte, yte = te_emb[te_mask], te_lab[te_mask]

        if len(yte) == 0:
            acc_shared = acc_grouped = acc_perclass = float("nan")
        else:
            # diagonal per-class variances from stored diag scatter
            var_c = np.zeros((n_seen_cls, d))
            for i, c in enumerate(order):
                diag_scatter, nc = stats[c][1], stats[c][2]
                var_c[i] = diag_scatter / max(1, nc - 1)

            # shared: one diagonal covariance = mean of per-class diagonal vars
            var_shared = var_c.mean(0) + SHRINK
            iv_shared = np.tile(1.0 / var_shared, (n_seen_cls, 1))
            ld_shared = np.tile(np.log(var_shared).sum(), n_seen_cls)
            acc_shared = mahalanobis_acc_diag(Xte, yte, means, iv_shared, ld_shared)

            # grouped: one diagonal covariance per prototype k-means group
            assign = kmeans_groups(means, n_groups, seed)
            iv_grp = np.zeros((n_seen_cls, d))
            ld_grp = np.zeros(n_seen_cls)
            for g in np.unique(assign):
                idx = np.where(assign == g)[0]
                var_g = var_c[idx].mean(0) + SHRINK
                iv_grp[idx] = 1.0 / var_g
                ld_grp[idx] = np.log(var_g).sum()
            acc_grouped = mahalanobis_acc_diag(Xte, yte, means, iv_grp, ld_grp)

            # perclass: each class its own diagonal covariance (capacity ceiling)
            var_pc = var_c + SHRINK
            iv_pc = 1.0 / var_pc
            ld_pc = np.log(var_pc).sum(1)
            acc_perclass = mahalanobis_acc_diag(Xte, yte, means, iv_pc, ld_pc)

        per_phase.append({
            "phase": t,
            "seen_classes": n_seen_cls,
            "seen_samples": int(n_running),
            # estimation
            "samples_per_dim": float(n_running / d),
            "samples_per_class": float(n_running / max(1, n_seen_cls)),
            "r95": r95,
            "shrinkage_alpha": alpha,
            "logdet_per_dim": logdet_per_dim,
            # capacity
            "acc_shared": acc_shared,
            "acc_grouped": acc_grouped,
            "acc_perclass": acc_perclass,
            "gap_grouped_minus_shared": (acc_grouped - acc_shared)
                if len(yte) else float("nan"),
            "gap_perclass_minus_shared": (acc_perclass - acc_shared)
                if len(yte) else float("nan"),
        })

    last = per_phase[-1]
    result = {
        "encoder": encoder, "dataset": dataset, "seed": seed,
        "dim": d, "num_classes": num_classes, "n_phases": len(phases),
        "n_groups": n_groups, "per_phase": per_phase,
        # estimation headline
        "final_samples_per_dim": last["samples_per_dim"],
        "final_shrinkage_alpha": last["shrinkage_alpha"],
        "max_shrinkage_alpha": max(pp["shrinkage_alpha"] for pp in per_phase),
        # capacity headline
        "final_acc_shared": last["acc_shared"],
        "final_acc_grouped": last["acc_grouped"],
        "final_acc_perclass": last["acc_perclass"],
        "final_gap_grouped": last["gap_grouped_minus_shared"],
        "final_gap_perclass": last["gap_perclass_minus_shared"],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", default=None)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--seed", action="append", type=int, default=None)
    ap.add_argument("--groups", type=int, default=N_GROUPS_DEFAULT)
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
                    res = run_one(enc, ds, sd, args.groups)
                except FileNotFoundError as e:
                    print(f"SKIP {enc}/{ds}/seed{sd}: {e}")
                    continue
                fp = out_dir / f"cov_{enc}_{ds}_seed{sd}.json"
                with open(fp, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[{enc}/{ds}/s{sd}] "
                      f"alpha={res['final_shrinkage_alpha']:.3f} "
                      f"| shared={res['final_acc_shared']:.3f} "
                      f"grouped={res['final_acc_grouped']:.3f} "
                      f"perclass={res['final_acc_perclass']:.3f} "
                      f"gap(grp-shr)={res['final_gap_grouped']:+.3f} -> {fp.name}")


if __name__ == "__main__":
    main()