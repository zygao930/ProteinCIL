import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from methods import set_seed, METHODS, compute_metrics

# ========================== CONFIG ==========================

ALL_ENCODERS = ["esm2_650m", "esm2_150m", "gearnet_edge", "saprot", "protbert"]
ALL_DATASETS = ["enzyme_reaction", "ec_number"]
ALL_SEEDS = [0, 42, 7]

HIDDEN_DIM = 1024
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 64
BUFFER_SIZE = 2000
EWC_LAMBDA = 2000.0
SHUFFLE_PHASES = True

DATASET_OVERRIDES = {
    "ec_number": {"epochs": 100, "batch": 256},
}

METHOD_PARAMS = {
    "ewc":          {"ewc_lambda": EWC_LAMBDA},
    "replay":       {"replay_alpha": 0.5},
    "fecam_common": {"shrinkage": 1e-4},
    "icarl":        {},
    "derpp":        {"alpha": 0.5, "beta": 0.5},
    "fetril":       {},
    "rer":          {},
    "ease": {"subspace_dim": 256},
    "ranpac": {"proj_dim": 12800, "ridge_lambda": 1.0},
}

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# ========================== CLI =============================

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", action="append",
                    help="Encoder(s) to run (repeatable). Default: all")
parser.add_argument("--dataset", action="append",
                    help="Dataset(s) to run (repeatable). Default: all")
parser.add_argument("--seed", type=int, action="append",
                    help="Seed(s) to run (repeatable). Default: all")
parser.add_argument("--method", action="append",
                    help="Method(s) to run (repeatable). Default: all enabled in METHODS")
args = parser.parse_args()

ENCODERS = args.encoder or ALL_ENCODERS
DATASETS = args.dataset or ALL_DATASETS
SEEDS = args.seed or ALL_SEEDS
RUN_METHODS = args.method

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================


def result_file(encoder, ds, method, seed):
    return RESULTS_DIR / f"{encoder}_{ds}_{method}_seed{seed}.json"


def get_hidden_dim(emb_dim):
    if HIDDEN_DIM is not None:
        return HIDDEN_DIM
    return max(256, emb_dim // 2)


def load_dataset(name, encoder):
    data_dir = Path("./data") / name
    cache = data_dir / encoder

    if (cache / "train_embeddings.pt").exists():
        train_emb = torch.load(cache / "train_embeddings.pt", weights_only=True)
        train_lab = torch.load(cache / "train_labels.pt", weights_only=True)
        test_emb = torch.load(cache / "test_embeddings.pt", weights_only=True)
        test_lab = torch.load(cache / "test_labels.pt", weights_only=True)
    elif (cache / "train_emb.pt").exists():
        train_emb = torch.load(cache / "train_emb.pt", weights_only=True)
        train_lab = torch.load(cache / "train_labels.pt", weights_only=True)
        test_emb = torch.load(cache / "test_emb.pt", weights_only=True)
        test_lab = torch.load(cache / "test_labels.pt", weights_only=True)
    else:
        raise FileNotFoundError(f"No embeddings found in {cache}")

    # Build canonical phases
    data_dir = Path("./data") / name
    if name in ("enzyme_reaction", "ec_number"):
        with open(data_dir / "ec_numbers.json") as f:
            labels_list = json.load(f)
        groups = defaultdict(list)
        for idx, ec in enumerate(labels_list):
            groups[ec.split(".")[0]].append(idx)
        phases = [groups[k] for k in sorted(groups, key=int)]
        nc = len(labels_list)

    def get_indices(labels, classes):
        s = set(classes)
        return [i for i, l in enumerate(labels.tolist()) if l in s]

    return train_emb, train_lab, test_emb, test_lab, phases, get_indices, nc


def shuffle_phases(phases, seed):
    """Return a shuffled copy of phases for the given seed."""
    rng = random.Random(seed + 42)
    order = list(range(len(phases)))
    rng.shuffle(order)
    return [phases[i] for i in order]


# ========================== MAIN ==========================

for encoder in ENCODERS:
    for ds in DATASETS:
        try:
            train_emb, train_lab, test_emb, test_lab, phases_canonical, get_indices, num_classes = load_dataset(ds, encoder)
        except Exception as e:
            print(f"\n=== SKIP {encoder}/{ds}: {e} ===\n")
            continue

        overrides = DATASET_OVERRIDES.get(ds, {})
        ep = overrides.get("epochs", DEFAULT_EPOCHS)
        bs = overrides.get("batch", DEFAULT_BATCH)
        lr = overrides.get("lr", DEFAULT_LR)
        emb_dim = train_emb.shape[1]
        hidden = get_hidden_dim(emb_dim)

        # Filter methods
        methods_to_run = {k: v for k, v in METHODS.items()
                          if RUN_METHODS is None or k in RUN_METHODS}

        for seed in SEEDS:
            set_seed(seed)
            phases = shuffle_phases(phases_canonical, seed) if SHUFFLE_PHASES else phases_canonical
            phase_sizes = [len(p) for p in phases]

            print(f"\n{'='*60}")
            print(f"ENCODER: {encoder} | DATASET: {ds} | {num_classes} classes | {len(phases)} phases | seed={seed}")
            print(f"Train: {train_emb.shape}, Test: {test_emb.shape}")
            print(f"{'='*60}")

            for method_name, run_fn in methods_to_run.items():
                rf = result_file(encoder, ds, method_name, seed)
                if rf.exists():
                    print(f"\n--- {method_name} --- SKIPPED (result exists: {rf})")
                    continue

                set_seed(seed)
                print(f"\n--- {method_name} ---")

                # Build kwargs
                run_kwargs = dict(
                    emb_dim=emb_dim,
                    num_classes=num_classes,
                    device=DEVICE,
                    epochs=ep,
                    batch=bs,
                    lr=lr,
                    buffer_size=BUFFER_SIZE,
                    hidden=hidden,
                )
                run_kwargs.update(METHOD_PARAMS.get(method_name, {}))

                try:
                    R = run_fn(train_emb, train_lab, test_emb, test_lab,
                               phases, get_indices, **run_kwargs)
                    avg_acc, fgt, bwt = compute_metrics(R)
                    result = {
                        "encoder": encoder, "dataset": ds, "method": method_name,
                        "seed": seed,
                        "avg_acc": avg_acc, "forgetting": fgt, "bwt": bwt,
                        "R": R,
                        "config": {
                            "hidden": hidden, "lr": lr, "epochs": ep, "batch": bs,
                            "buffer_size": BUFFER_SIZE,
                            "shuffle_phases": SHUFFLE_PHASES,
                            "phase_sizes": phase_sizes,
                            **METHOD_PARAMS.get(method_name, {}),
                        },
                    }
                    with open(rf, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"  -> Saved to {rf}")
                except Exception as e:
                    print(f"  -> FAILED: {e}")
                    import traceback
                    traceback.print_exc()