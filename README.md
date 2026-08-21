# ProteinCIL: Class-Incremental Learning on Frozen Protein Representations

A benchmark for evaluating class-incremental learning (CIL) methods on frozen protein encoder embeddings.

## Repository Structure

```
ProteinCIL/
├── methods/                  # CIL baselines (one file per method)
│   ├── _common.py            # Shared: classifier, training, evaluation, replay buffer
│   ├── naive.py              # Naive fine-tuning (lower bound)
│   ├── joint.py              # Joint training (upper bound)
│   ├── ewc.py                # Elastic Weight Consolidation
│   ├── replay.py             # Experience Replay
│   ├── icarl.py              # iCaRL
│   ├── derpp.py              # DER++
│   ├── fetril.py             # FeTrIL
│   ├── rer.py                # Relational Experience Replay
│   ├── fecam.py              # FeCAM (shared Mahalanobis prototype)
│   ├── ranpac.py             # RanPAC (random projection + ridge)
│   └── ease.py               # EASE (expandable subspace ensemble)
├── diagnose/                 # Diagnostic analysis scripts
│   ├── _common.py            # Shared data loading for diagnostics
│   ├── diag_gradient.py      # Gradient signal analysis (Hypothesis A)
│   ├── diag_geometry.py      # Embedding geometry analysis (Hypothesis B)
│   ├── diag_covariance.py    # Covariance granularity experiments
│   ├── diag_confusion.py     # Cross-family confusion analysis
│   └── diag_forgetting.py    # Forgetting pattern analysis
├── build_canonical.py        # Step 1: Raw data -> canonical.json
├── extract.py                # Step 2: Canonical data -> cached embeddings
├── run.py                    # Step 3: Main benchmark runner (EC-first phases)
├── run_protocols.py          # Step 3b: Protocol robustness experiments
└── requirements.txt
```

## Quick Start


### 1. Prepare raw data

Organise your raw data under a single root directory (e.g. `./raw_data`):

```
raw_data/
├── CLEAN/app/data/                   # EC Number: split70.csv, split30.csv
├── ec_number_pdb/                    # EC Number: AlphaFold PDB files ({uid}.pdb)
├── ec_number_3di/3di.fasta           # EC Number: Foldseek 3Di tokens
├── enzyme_reaction_raw/              # Enzyme Reaction: TorchDrug download
│   └── EnzymeCommission/             #   PDB files (train/, test/, valid/)
├── enzyme_reaction_records/          # Enzyme Reaction: train_records.json, test_records.json, ec_list.json
└── enzyme_reaction_3di/3di.fasta     # Enzyme Reaction: Foldseek 3Di tokens
```

**Data sources:**

| Dataset | Source |
|---|---|
| Enzyme Reaction | TorchDrug EnzymeCommission dataset |
| EC Number | CLEAN split (Enzyme Commission numbers) |
| PDB structures | AlphaFold v4 predicted structures |
| 3Di tokens | Generated with Foldseek from PDB files |

Then build the canonical format:

```bash
python build_canonical.py --dataset enzyme_reaction --raw_root ./raw_data
python build_canonical.py --dataset ec_number --raw_root ./raw_data
```

This creates `data/{dataset}/canonical.json` and `data/{dataset}/ec_numbers.json`.

### 2. Extract embeddings

```bash
python extract.py --dataset enzyme_reaction --encoder esm2_650m
python extract.py --dataset ec_number --encoder esm2_650m
```

For GearNet, set the checkpoint path:

```bash
export GEARNET_CHECKPOINT=/path/to/mc_gearnet_edge.pth
python extract.py --dataset enzyme_reaction --encoder gearnet_edge
```

Available encoders: `esm2_650m`, `esm2_150m`, `protbert`, `saprot`, `gearnet_edge`.

This caches embeddings under `data/{dataset}/{encoder}/`.

### 3. Run benchmark

```bash
# Run all methods on all encoders/datasets/seeds
python run.py

# Run specific methods/encoders
python run.py --method ewc --method replay --encoder esm2_650m --dataset enzyme_reaction --seed 0
```

### 4. Run protocol robustness experiments

```bash
python run_protocols.py \
  --method ewc --method replay --method ranpac --method fecam_common \
  --protocol random --protocol finer_ec --protocol mixed \
  --dataset enzyme_reaction --dataset ec_number
```

### 5. Run diagnostics

```bash
cd diagnose
python diag_gradient.py    # Gradient signal analysis
python diag_geometry.py    # Embedding geometry
python diag_covariance.py  # Covariance granularity
python diag_confusion.py   # Cross-family confusions
```


## Encoders

| Encoder | Type | Embedding dim |
|---|---|---|
| ESM2-650M | Sequence PLM | 1280 |
| ESM2-150M | Sequence PLM | 640 |
| ProtBert | Sequence PLM | 1024 |
| GearNet-Edge | Structure GNN | 3072 |
| SaProt | Seq+Struct fusion | 1280 |

## Datasets

| Dataset | Classes | Train | Test | Phases |
|---|---|---|---|---|
| Enzyme Reaction | 348 | 29,215 | 5,651 | 7 |
| EC Number | 4,788 | 73,259 | 9,112 | 7 |

## Metrics

After each phase, every method is evaluated on all classes seen so far, producing an accuracy matrix R. We report:

**Average Accuracy**: final retained performance across all phases.

**Forgetting**: degradation from the best historical accuracy on each phase.

**Plasticity**: new-class acquisition accuracy at each phase.

