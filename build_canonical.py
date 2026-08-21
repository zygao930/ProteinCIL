import argparse
import csv
import json
import os
from pathlib import Path


# ========================== CONFIG ==========================
# RAW_ROOT is set via --raw_root (default: ./raw_data).
# Organise your raw data under RAW_ROOT following the layout below.
# ============================================================

RAW_ROOT = None  # set by CLI
RAW = {}         # built by _init_paths()
OUT_DIR = Path("data")


def _init_paths(raw_root):
    global RAW_ROOT, RAW
    RAW_ROOT = Path(raw_root)
    RAW = {
        "ec_number": {
            "clean_dir": RAW_ROOT / "CLEAN/app/data",               # split70.csv, split30.csv
            "pdb_dir": RAW_ROOT / "ec_number_pdb",                   # {uid}.pdb files
            "threedi_fasta": RAW_ROOT / "ec_number_3di/3di.fasta",
        },
        "enzyme_reaction": {
            "torchdrug_dir": RAW_ROOT / "enzyme_reaction_raw",       # torchdrug download dir
            "records_dir": RAW_ROOT / "enzyme_reaction_records",      # dumped JSON records
            "pdb_dir": RAW_ROOT / "enzyme_reaction_raw/EnzymeCommission",
            "threedi_fasta": RAW_ROOT / "enzyme_reaction_3di/3di.fasta",
        },
    }

# ============================================================


def load_threedi(fasta_path):
    """Load 3Di fasta into {header: sequence} dict."""
    seqs = {}
    header = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:].split()[0]
                seqs[header] = ""
            elif header:
                seqs[header] += line
    return seqs


def build_ec_number():
    cfg = RAW["ec_number"]
    out = OUT_DIR / "ec_number"
    out.mkdir(parents=True, exist_ok=True)

    # Load available resources
    pdb_ids = set(f.replace(".pdb", "") for f in os.listdir(cfg["pdb_dir"]))
    threedi = load_threedi(cfg["threedi_fasta"])
    threedi_ids = set(threedi.keys())

    # Read CLEAN CSVs
    def read_split(filename):
        records = []
        with open(Path(cfg["clean_dir"]) / filename) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                uid, ec, seq = row[0].strip(), row[1].split(";")[0].strip(), row[2].strip()
                if uid and ec and seq:
                    records.append({"id": uid, "ec": ec, "sequence": seq})
        return records

    train_raw = read_split("split70.csv")
    test_raw = read_split("split30.csv")

    # Filter: keep only samples with PDB + 3Di
    excluded = open(out / "excluded.log", "w")
    excluded.write("split\tid\treason\n")

    def accept(rec, split):
        uid = rec["id"]
        if uid not in pdb_ids:
            excluded.write(f"{split}\t{uid}\tno_pdb\n")
            return False
        if uid not in threedi_ids:
            excluded.write(f"{split}\t{uid}\tno_3di\n")
            return False
        return True

    train = [r for r in train_raw if accept(r, "train")]
    test = [r for r in test_raw if accept(r, "test")]

    # Attach pdb_path and threedi
    for r in train + test:
        r["pdb_path"] = str(Path(cfg["pdb_dir"]) / f"{r['id']}.pdb")
        r["threedi"] = threedi[r["id"]]

    # Build contiguous labels
    all_ecs = sorted(set(r["ec"] for r in train + test))
    ec2id = {ec: i for i, ec in enumerate(all_ecs)}
    for r in train + test:
        r["label"] = ec2id[r["ec"]]

    excluded.close()

    canonical = {
        "dataset": "ec_number",
        "num_classes": len(all_ecs),
        "label_names": all_ecs,
        "train": train,
        "test": test,
    }
    with open(out / "canonical.json", "w") as f:
        json.dump(canonical, f)

    # Also write standalone label file for run.py compatibility
    with open(out / "ec_numbers.json", "w") as f:
        json.dump(all_ecs, f)

    n_excl = sum(1 for _ in open(out / "excluded.log")) - 1  # minus header
    print(f"ec_number: train={len(train)}, test={len(test)}, "
          f"classes={len(all_ecs)}, excluded={n_excl}")
    return canonical


def build_enzyme_reaction():
    cfg = RAW["enzyme_reaction"]
    out = OUT_DIR / "enzyme_reaction"
    out.mkdir(parents=True, exist_ok=True)

    threedi = load_threedi(cfg["threedi_fasta"])
    # 3Di keys have format "pdbid_chain_xxx", match by prefix
    threedi_by_pdb = {}
    for key, seq in threedi.items():
        prefix = key.rsplit("_", 1)[0]
        threedi_by_pdb[prefix] = seq

    with open(cfg["records_dir"] / "train_records.json") as f:
        train_raw = json.load(f)
    with open(cfg["records_dir"] / "test_records.json") as f:
        test_raw = json.load(f)

    # Build PDB file index once (instead of os.listdir per sample)
    pdb_base = Path(cfg["pdb_dir"])
    pdb_files = {}  # subdir -> list of (filename, full_path)
    for subdir in ["train", "test", "valid"]:
        d = pdb_base / subdir
        if d.exists():
            pdb_files[subdir] = [(f, str(d / f)) for f in os.listdir(d) if f.endswith(".pdb")]

    def find_pdb(pdb_id):
        prefix = pdb_id + "_"
        for subdir_files in pdb_files.values():
            for fname, fpath in subdir_files:
                if fname.startswith(prefix):
                    return fpath
        return None

    excluded = open(out / "excluded.log", "w")
    excluded.write("split\tid\treason\n")

    def accept_er(rec, split):
        pid = rec["pdb_id"]
        # Check 3Di
        tdi = threedi_by_pdb.get(pid)
        if tdi is None:
            excluded.write(f"{split}\t{pid}\tno_3di\n")
            return False
        # Check PDB
        pdb_path = find_pdb(pid)
        if pdb_path is None:
            excluded.write(f"{split}\t{pid}\tno_pdb\n")
            return False
        rec["pdb_path"] = pdb_path
        rec["threedi"] = tdi
        rec["id"] = pid
        rec["sequence"] = rec.get("sequence", rec.get("seq", ""))
        return True

    train = [r for r in train_raw if accept_er(r, "train")]
    test = [r for r in test_raw if accept_er(r, "test")]
    excluded.close()

    # Map argmax labels to real EC numbers
    with open(cfg["records_dir"] / "ec_list.json") as f:
        ec_list = json.load(f)
    for r in train + test:
        r["ec"] = ec_list[r["label"]]

    all_ecs = sorted(set(r["ec"] for r in train + test))
    ec2id = {ec: i for i, ec in enumerate(all_ecs)}
    for r in train + test:
        r["label"] = ec2id[r["ec"]]

    canonical = {
        "dataset": "enzyme_reaction",
        "num_classes": len(all_ecs),
        "label_names": all_ecs,
        "train": train,
        "test": test,
    }
    with open(out / "canonical.json", "w") as f:
        json.dump(canonical, f)
    with open(out / "ec_numbers.json", "w") as f:
        json.dump(all_ecs, f)

    n_excl = sum(1 for _ in open(out / "excluded.log")) - 1
    print(f"enzyme_reaction: train={len(train)}, test={len(test)}, "
          f"classes={len(all_ecs)}, excluded={n_excl}")
    return canonical


BUILDERS = {
    "ec_number": build_ec_number,
    "enzyme_reaction": build_enzyme_reaction,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(BUILDERS.keys()) + ["all"])
    parser.add_argument("--raw_root", type=str, default="./raw_data",
                        help="Root directory containing raw data (default: ./raw_data)")
    args = parser.parse_args()

    _init_paths(args.raw_root)

    if args.dataset == "all":
        for name, fn in BUILDERS.items():
            fn()
    else:
        BUILDERS[args.dataset]()