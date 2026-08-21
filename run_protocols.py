# run_protocols.py
# Usage:

# CUDA_VISIBLE_DEVICES=0 python run_protocols.py \
#   --method ewc --method replay --method icarl --method derpp --method ease\
#   --method fetril --method rer --method ranpac --method fecam_common \
#   --protocol random --protocol finer_ec --protocol mixed \
#   --dataset enzyme_reaction --dataset ec_number \
#   --seed 0 --seed 7 --seed 42 \
#   --results_dir ./results_protocols
#
# This script does not modify run.py or methods.py.
# It reuses the baseline implementations from methods.py
# and only changes the phase construction protocol.

import argparse
import json
import random
import torch
from pathlib import Path
from collections import defaultdict, Counter

import methods as M


# ========================== CONFIG ==========================

ALL_ENCODERS = ["esm2_650m", "esm2_150m", "gearnet_edge", "saprot", "protbert"]
ALL_DATASETS = ["enzyme_reaction", "ec_number"]
ALL_SEEDS = [0, 42, 7]

HIDDEN_DIM = 1024
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 64
BUFFER_SIZE = 2000

DATASET_OVERRIDES = {
    "ec_number": {"epochs": 100, "batch": 256},
}

METHOD_REGISTRY = {
    "naive": M.run_naive,
    "joint": M.run_joint,
    "ewc": M.run_ewc,
    "replay": M.run_replay,
    "fecam_common": M.run_fecam_common,
    "icarl": M.run_icarl,
    "derpp": M.run_derpp,
    "fetril": M.run_fetril,
    "rer": M.run_rer,
    "ranpac": M.run_ranpac,
    "ease": M.run_ease,
}

METHOD_PARAMS = {
    "ewc":          {"ewc_lambda": 2000.0},
    "replay":       {"replay_alpha": 0.5},
    "fecam_common": {"shrinkage": 1e-4},
    "icarl":        {},
    "derpp":        {"alpha": 0.5, "beta": 0.5},
    "fetril":       {},
    "rer":          {},
    "ranpac": {"proj_dim": 12800, "ridge_lambda": 1.0},
    "ease": {"subspace_dim": 256},
}


# ========================== CLI =============================

parser = argparse.ArgumentParser()

parser.add_argument("--encoder", action="append",
                    help="Encoder(s) to run. Default: all")

parser.add_argument("--dataset", action="append",
                    choices=ALL_DATASETS,
                    help="Dataset(s) to run. Default: enzyme_reaction and ec_number")

parser.add_argument("--seed", type=int, action="append",
                    help="Seed(s) to run. Default: 0, 42, 7")

parser.add_argument("--method", action="append",
                    choices=list(METHOD_REGISTRY.keys()),
                    help="Method(s) to run. Default: fecam_common and derpp")

parser.add_argument("--protocol", action="append",
                    choices=["ec_first", "random", "finer_ec", "mixed"],
                    help="Phase construction protocol(s). Default: random, finer_ec, mixed")

parser.add_argument("--results_dir", type=str, default="./results_protocols",
                    help="Directory for protocol results")

parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing result files")

parser.add_argument("--no_shuffle_order", action="store_true",
                    help="Disable seed-based phase-order shuffling after phase construction")

args = parser.parse_args()

ENCODERS = args.encoder or ALL_ENCODERS
DATASETS = args.dataset or ALL_DATASETS
SEEDS = args.seed or ALL_SEEDS
RUN_METHODS = args.method or ["fecam_common", "derpp"]
PROTOCOLS = args.protocol or ["random", "finer_ec", "mixed"]

RESULTS_DIR = Path(args.results_dir)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== HELPERS ==========================

def result_file(encoder, ds, method, protocol, seed):
    return RESULTS_DIR / f"{encoder}_{ds}_{method}_{protocol}_seed{seed}.json"


def get_hidden_dim(emb_dim):
    if HIDDEN_DIM is not None:
        return HIDDEN_DIM
    return max(256, emb_dim // 2)


def sort_ec_top_keys(keys):
    def key_fn(x):
        try:
            return int(x)
        except Exception:
            return x
    return sorted(keys, key=key_fn)


def build_ec_first_phases(labels_list):
    groups = defaultdict(list)
    for idx, ec in enumerate(labels_list):
        top = str(ec).split(".")[0]
        groups[top].append(idx)
    return [groups[k] for k in sort_ec_top_keys(groups.keys())]


def build_phases_from_ec(labels_list, protocol, seed=0, n_phases=7):
    """
    Build class-incremental phases from EC labels.

    ec_first:
        Top-level EC category defines each phase.

    random:
        Randomly assigns classes to phases while preserving the phase sizes
        of the EC-first protocol.

    finer_ec:
        Groups classes by finer EC prefix, then greedily packs these buckets
        into seven phases. This keeps finer functional buckets mostly intact.

    mixed:
        Constructs phases by mixing classes from different top-level EC groups,
        reducing alignment between phase boundaries and EC hierarchy.
    """
    rng = random.Random(seed + 123)
    nc = len(labels_list)

    ec_first = build_ec_first_phases(labels_list)
    phase_sizes = [len(p) for p in ec_first]
    n_phases = len(phase_sizes)

    if protocol == "ec_first":
        return ec_first

    if protocol == "random":
        classes = list(range(nc))
        rng.shuffle(classes)
        phases = []
        start = 0
        for sz in phase_sizes:
            phases.append(classes[start:start + sz])
            start += sz
        return phases

    if protocol == "finer_ec":
        buckets = defaultdict(list)
        for idx, ec in enumerate(labels_list):
            parts = str(ec).split(".")
            if len(parts) >= 2:
                key = ".".join(parts[:2])
            else:
                key = parts[0]
            buckets[key].append(idx)

        bucket_items = list(buckets.values())
        rng.shuffle(bucket_items)

        phases = [[] for _ in range(n_phases)]
        sizes = [0 for _ in range(n_phases)]

        # Greedy bin packing by current fill ratio.
        for bucket in sorted(bucket_items, key=len, reverse=True):
            pid = min(
                range(n_phases),
                key=lambda i: sizes[i] / max(1, phase_sizes[i])
            )
            phases[pid].extend(bucket)
            sizes[pid] += len(bucket)

        return phases

    if protocol == "mixed":
        top_groups = defaultdict(list)
        for idx, ec in enumerate(labels_list):
            top = str(ec).split(".")[0]
            top_groups[top].append(idx)

        for k in top_groups:
            rng.shuffle(top_groups[k])

        keys = sort_ec_top_keys(top_groups.keys())
        phases = [[] for _ in range(n_phases)]

        # Fill each phase to the EC-first phase size, cycling through top-level groups.
        for pid, target_size in enumerate(phase_sizes):
            cursor = pid
            while len(phases[pid]) < target_size:
                nonempty = [k for k in keys if len(top_groups[k]) > 0]
                if not nonempty:
                    break
                k = nonempty[cursor % len(nonempty)]
                phases[pid].append(top_groups[k].pop())
                cursor += 1

        return phases

    raise ValueError(f"Unknown protocol: {protocol}")


def shuffle_phases(phases, seed):
    rng = random.Random(seed + 42)
    order = list(range(len(phases)))
    rng.shuffle(order)
    return [phases[i] for i in order]


# Cache only the most recently used raw dataset.
# The loop visits the same encoder/dataset many times across protocols and seeds,
# so this avoids repeated torch.load/json.load without keeping all encoders in RAM.
RAW_DATASET_CACHE = {}
GET_INDICES_CACHE = {}


def load_raw_dataset(name, encoder):
    key = (name, encoder)
    if key in RAW_DATASET_CACHE:
        return RAW_DATASET_CACHE[key]

    RAW_DATASET_CACHE.clear()
    GET_INDICES_CACHE.clear()

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

    ec_file = data_dir / "ec_numbers.json"
    if not ec_file.exists():
        raise FileNotFoundError(f"Missing EC label file: {ec_file}")

    with open(ec_file) as f:
        labels_list = json.load(f)

    RAW_DATASET_CACHE[key] = (train_emb, train_lab, test_emb, test_lab, labels_list)
    return RAW_DATASET_CACHE[key]


def make_get_indices(train_lab, test_lab):
    # Convert labels to Python lists once. The returned indices preserve the
    # original dataset order exactly, matching the old implementation.
    train_list = [int(y) for y in train_lab.tolist()]
    test_list = [int(y) for y in test_lab.tolist()]
    memo = {}

    def get_indices(labels, classes):
        classes_key = tuple(sorted(int(c) for c in classes))

        if labels is train_lab:
            side = "train"
            labs = train_list
        elif labels is test_lab:
            side = "test"
            labs = test_list
        else:
            # Fallback for unexpected label tensors. This keeps the old behavior
            # while still memoizing repeated calls on the same labels object.
            side = f"labels_{id(labels)}"
            labs = [int(y) for y in labels.tolist()]

        key = (side, classes_key)
        if key not in memo:
            cls_set = set(classes_key)
            memo[key] = [i for i, y in enumerate(labs) if y in cls_set]

        # Return a copy so callers cannot accidentally mutate the cached list.
        return list(memo[key])

    return get_indices


def get_cached_get_indices(name, encoder, train_lab, test_lab):
    key = (name, encoder)
    if key not in GET_INDICES_CACHE:
        GET_INDICES_CACHE[key] = make_get_indices(train_lab, test_lab)
    return GET_INDICES_CACHE[key]


def load_dataset(name, encoder, protocol, seed, shuffle_order=True):
    train_emb, train_lab, test_emb, test_lab, labels_list = load_raw_dataset(name, encoder)

    phases = build_phases_from_ec(labels_list, protocol=protocol, seed=seed)

    if shuffle_order:
        phases = shuffle_phases(phases, seed)

    num_classes = len(labels_list)
    get_indices = get_cached_get_indices(name, encoder, train_lab, test_lab)

    return train_emb, train_lab, test_emb, test_lab, phases, get_indices, num_classes


# ========================== MAIN ==========================

for encoder in ENCODERS:
    for ds in DATASETS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                try:
                    train_emb, train_lab, test_emb, test_lab, phases, get_indices, num_classes = load_dataset(
                        ds,
                        encoder,
                        protocol=protocol,
                        seed=seed,
                        shuffle_order=not args.no_shuffle_order,
                    )
                except Exception as e:
                    print(f"\n=== SKIP {encoder}/{ds}/{protocol}/seed{seed}: {e} ===\n")
                    continue

                overrides = DATASET_OVERRIDES.get(ds, {})
                ep = overrides.get("epochs", DEFAULT_EPOCHS)
                bs = overrides.get("batch", DEFAULT_BATCH)
                lr = overrides.get("lr", DEFAULT_LR)

                emb_dim = train_emb.shape[1]
                hidden = get_hidden_dim(emb_dim)
                phase_sizes = [len(p) for p in phases]

                M.set_seed(seed)

                print(f"\n{'=' * 80}")
                print(f"ENCODER: {encoder} | DATASET: {ds} | PROTOCOL: {protocol} | seed={seed}")
                print(f"{num_classes} classes | {len(phases)} phases")
                print(f"Train: {train_emb.shape}, Test: {test_emb.shape}")
                print(f"Phase sizes: {phase_sizes}")
                print(f"{'=' * 80}")

                for method_name in RUN_METHODS:
                    run_fn = METHOD_REGISTRY[method_name]
                    rf = result_file(encoder, ds, method_name, protocol, seed)

                    if rf.exists() and not args.overwrite:
                        print(f"\n--- {method_name} --- SKIPPED (exists: {rf})")
                        continue

                    M.set_seed(seed)
                    print(f"\n--- {method_name} ---")

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
                        R = run_fn(
                            train_emb,
                            train_lab,
                            test_emb,
                            test_lab,
                            phases,
                            get_indices,
                            **run_kwargs,
                        )

                        avg_acc, fgt, bwt = M.compute_metrics(R)

                        result = {
                            "encoder": encoder,
                            "dataset": ds,
                            "protocol": protocol,
                            "method": method_name,
                            "seed": seed,
                            "avg_acc": avg_acc,
                            "forgetting": fgt,
                            "bwt": bwt,
                            "R": R,
                            "config": {
                                "hidden": hidden,
                                "lr": lr,
                                "epochs": ep,
                                "batch": bs,
                                "buffer_size": BUFFER_SIZE,
                                "shuffle_order": not args.no_shuffle_order,
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