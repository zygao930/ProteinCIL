"""Per-encoder failure cases: worst cross-EC-family misclassifications under
a shared-covariance prototype classifier. Output: diagnose/results/confusion_*.json"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from _common import (ALL_ENCODERS, ALL_DATASETS, ALL_SEEDS,
                     load_embeddings, load_ec_phases)

SHRINK = 1e-4
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EC_FAMILY_NAME = {
    "1": "oxidoreductases", "2": "transferases", "3": "hydrolases",
    "4": "lyases", "5": "isomerases", "6": "ligases", "7": "translocases",
}


def load_ec_strings(dataset):
    with open(Path("./data") / dataset / "ec_numbers.json") as f:
        return json.load(f)


def load_test_ids(dataset, n_expected):
    """Return IDs aligned with test_embeddings.pt when canonical data exists."""
    fp = Path("./data") / dataset / "canonical.json"
    if not fp.exists():
        return [str(i) for i in range(n_expected)]
    with open(fp) as f:
        samples = json.load(f).get("test", [])
    if len(samples) != n_expected:
        return [str(i) for i in range(n_expected)]
    return [str(sample.get("id", i)) for i, sample in enumerate(samples)]


def family_of(ec_str):
    return ec_str.split(".")[0]


def fit_shared_prototype(tr_emb, tr_lab, num_classes, d):
    """Prototypes + one shared shrinkage covariance, precision matrix.
    Vectorized: subtract each sample's class mean, then one pooled X^T X."""
    means = np.zeros((num_classes, d))
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mask = tr_lab == c
        n_c = int(mask.sum())
        counts[c] = n_c
        if n_c > 0:
            means[c] = tr_emb[mask].mean(0)
    # center every sample by its class mean, then a single pooled scatter
    centered = tr_emb - means[tr_lab]
    valid = counts[tr_lab] > 1
    Xc = centered[valid]
    denom = max(1, len(Xc) - int((counts > 1).sum()))
    avg_cov = (Xc.T @ Xc) / denom + SHRINK * np.eye(d)
    P = np.linalg.inv(avg_cov)
    return means, P


def predict(te_emb, means, P, batch=2048, return_whitener=False):
    """Mahalanobis nearest-prototype prediction via whitening."""
    Pt = torch.as_tensor(P, dtype=torch.float32, device=_DEVICE)
    try:
        W = torch.linalg.cholesky(Pt, upper=True)
    except Exception:
        ev, evec = torch.linalg.eigh(Pt)
        W = (evec * torch.clamp(ev, min=0).sqrt()) @ evec.T
    M = torch.as_tensor(means, dtype=torch.float32, device=_DEVICE)
    Mw = M @ W.T
    m2 = (Mw * Mw).sum(1)
    preds = np.empty(len(te_emb), dtype=np.int64)
    for s in range(0, len(te_emb), batch):
        Xb = torch.as_tensor(te_emb[s:s+batch], dtype=torch.float32, device=_DEVICE)
        Xw = Xb @ W.T
        x2 = (Xw * Xw).sum(1, keepdim=True)
        dist = x2 + m2.unsqueeze(0) - 2.0 * (Xw @ Mw.T)
        preds[s:s+batch] = dist.argmin(1).cpu().numpy()
    if return_whitener:
        # X @ right_transform is the same whitening operation used above by
        # Xb @ W.T. Saving it lets visualize_grouping.py use classifier-aligned
        # coordinates without recomputing the covariance inverse.
        return preds, W.T.detach().cpu().numpy()
    return preds


def run_one(encoder, dataset, seed):
    np.random.seed(seed)
    tr_emb, tr_lab = load_embeddings(dataset, encoder, "train")
    te_emb, te_lab = load_embeddings(dataset, encoder, "test")
    _, num_classes = load_ec_phases(dataset)
    ec_strings = load_ec_strings(dataset)
    d = tr_emb.shape[1]

    means, P = fit_shared_prototype(tr_emb, tr_lab, num_classes, d)
    preds, right_transform = predict(
        te_emb, means, P, return_whitener=True)
    test_ids = load_test_ids(dataset, len(te_lab))

    whitener_dir = Path('diagnose/results')
    whitener_dir.mkdir(parents=True, exist_ok=True)
    whitener_path = whitener_dir / f'whitener_{encoder}_{dataset}_v1.npz'
    np.savez(
        whitener_path,
        means=np.asarray(means, dtype=np.float32),
        right_transform=np.asarray(right_transform, dtype=np.float32),
        train_shape=np.asarray(tr_emb.shape, dtype=np.int64),
    )

    acc = float((preds == te_lab).mean())

    # cross-family confusion counts
    fam = [family_of(s) for s in ec_strings]
    fam_conf = defaultdict(int)                     # (true_fam, pred_fam) -> count
    fam_example = {}                                # (true_fam,pred_fam)->(true_c,pred_c)
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    for true_c, pred_c in zip(te_lab, preds):
        per_class_total[int(true_c)] += 1
        if true_c == pred_c:
            per_class_correct[int(true_c)] += 1
            continue
        tf, pf = fam[true_c], fam[pred_c]
        if tf != pf:
            fam_conf[(tf, pf)] += 1
            fam_example.setdefault((tf, pf), (ec_strings[true_c], ec_strings[pred_c]))

    # worst cross-family confusions
    worst_fam = sorted(fam_conf.items(), key=lambda kv: -kv[1])[:6]
    worst_fam_out = []
    for (tf, pf), cnt in worst_fam:
        ex_t, ex_p = fam_example[(tf, pf)]
        worst_fam_out.append({
            "true_family": f"{tf} ({EC_FAMILY_NAME.get(tf,'?')})",
            "pred_family": f"{pf} ({EC_FAMILY_NAME.get(pf,'?')})",
            "misclassified_samples": cnt,
            "example": f"{ex_t} -> {ex_p}",
        })

    # worst single classes by recall (min 3 test samples)
    recalls = []
    for c, tot in per_class_total.items():
        if tot >= 3:
            recalls.append((c, per_class_correct[c] / tot, tot))
    recalls.sort(key=lambda x: x[1])
    worst_classes = [{
        "class_ec": ec_strings[c],
        "family": f"{family_of(ec_strings[c])} ({EC_FAMILY_NAME.get(family_of(ec_strings[c]),'?')})",
        "recall": round(r, 3),
        "test_samples": tot,
    } for c, r, tot in recalls[:8]]

    return {
        "encoder": encoder, "dataset": dataset, "seed": seed,
        "shared_cov_accuracy": round(acc, 4),
        "worst_cross_family_confusions": worst_fam_out,
        "worst_classes_by_recall": worst_classes,
        # These arrays are aligned by position with test_embeddings.pt. They
        # let downstream plots highlight only the individual proteins that the
        # diagnostic classifier actually misclassified.
        "sample_predictions": {
            "test_index": list(range(len(te_lab))),
            "sample_id": test_ids,
            "y_true": np.asarray(te_lab, dtype=np.int64).tolist(),
            "y_pred": np.asarray(preds, dtype=np.int64).tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", default=None)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--seed", action="append", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if the output json already exists")
    args = ap.parse_args()

    encoders = args.encoder or ALL_ENCODERS
    datasets = args.dataset or ALL_DATASETS
    seeds = args.seed or [0]

    out_dir = Path("diagnose/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for enc in encoders:
        for ds in datasets:
            for sd in seeds:
                fp = out_dir / f"confusion_{enc}_{ds}_seed{sd}.json"
                if fp.exists() and not args.overwrite:
                    print(f"SKIP (exists) {fp.name}")
                    continue
                try:
                    res = run_one(enc, ds, sd)
                except FileNotFoundError as e:
                    print(f"SKIP {enc}/{ds}/seed{sd}: {e}")
                    continue
                with open(fp, "w") as f:
                    json.dump(res, f, indent=2)
                top = res["worst_cross_family_confusions"]
                top1 = top[0] if top else {}
                print(f"[{enc}/{ds}/s{sd}] acc={res['shared_cov_accuracy']:.3f} "
                      f"| worst: {top1.get('true_family','-')} -> "
                      f"{top1.get('pred_family','-')} "
                      f"({top1.get('misclassified_samples','-')} samples) -> {fp.name}")


if __name__ == "__main__":
    main()
