import warnings
warnings.filterwarnings("ignore")
try:
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    pass

import argparse, gc, json, os, time, torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = ["enzyme_reaction", "ec_number"]

BATCH_SIZES = {
    "esm2_650m": 32,
    "esm2_150m": 64,
    "protbert": 32,
    "saprot": 32,
    "gearnet_edge": 32,
}


def _masked_mean_pool(out, attention_mask):
    mask = attention_mask.unsqueeze(-1).clone()
    for i in range(mask.shape[0]):
        mask[i, 0] = 0
        seq_len = attention_mask[i].sum().item()
        if seq_len > 1:
            mask[i, seq_len - 1] = 0
    return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def load_esm(model_name):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    dim = model.config.hidden_size

    @torch.no_grad()
    def encode_batch(samples):
        seqs = [s["sequence"] for s in samples]
        tokens = tokenizer(seqs, return_tensors="pt", truncation=True,
                           max_length=1024, padding=True).to(DEVICE)
        out = model(**tokens).last_hidden_state
        pooled = _masked_mean_pool(out, tokens["attention_mask"])
        return [pooled[i].cpu() for i in range(len(samples))]

    return encode_batch, dim


def load_protbert(model_name="Rostlab/prot_bert"):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    dim = model.config.hidden_size

    @torch.no_grad()
    def encode_batch(samples):
        seqs = [" ".join(list(s["sequence"])) for s in samples]
        tokens = tokenizer(seqs, return_tensors="pt", truncation=True,
                           max_length=1024, padding=True).to(DEVICE)
        out = model(**tokens).last_hidden_state
        pooled = _masked_mean_pool(out, tokens["attention_mask"])
        return [pooled[i].cpu() for i in range(len(samples))]

    return encode_batch, dim


def load_saprot(model_name="westlake-repl/SaProt_650M_AF2"):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    dim = model.config.hidden_size

    @torch.no_grad()
    def encode_batch(samples):
        sa_seqs = []
        for s in samples:
            seq, tdi = s["sequence"], s["threedi"]
            n = min(len(seq), len(tdi))
            sa_seqs.append("".join(aa + st.lower() for aa, st in zip(seq[:n], tdi[:n])))
        tokens = tokenizer(sa_seqs, return_tensors="pt", truncation=True,
                           max_length=2048, padding=True).to(DEVICE)
        out = model(**tokens).last_hidden_state
        pooled = _masked_mean_pool(out, tokens["attention_mask"])
        return [pooled[i].cpu() for i in range(len(samples))]

    return encode_batch, dim


def load_gearnet(checkpoint=None):
    if checkpoint is None:
        checkpoint = os.environ.get("GEARNET_CHECKPOINT", "checkpoints/mc_gearnet_edge.pth")
    from torchdrug import models, layers, data as td_data
    from torchdrug.data import Protein
    from torchdrug.layers import geometry
    from copy import deepcopy
    from concurrent.futures import ThreadPoolExecutor
    import torch.nn.functional as F

    if not hasattr(Protein, 'copy'):
        Protein.copy = lambda self: deepcopy(self)

    graph_construction = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SequentialEdge(max_distance=2),
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
        ],
        edge_feature="gearnet",
    )

    model = models.GearNet(
        input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
        num_relation=7, batch_norm=True, concat_hidden=True,
        short_cut=True, readout="sum", edge_input_dim=59,
    ).to(DEVICE)
    if checkpoint:
        state = torch.load(checkpoint, map_location=DEVICE)
        model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    dim = 3072

    def parse_one(s):
        try:
            return Protein.from_pdb(s["pdb_path"], atom_feature=None, bond_feature=None)
        except Exception as e:
            return e

    @torch.no_grad()
    def encode_batch(samples):
        with ThreadPoolExecutor(max_workers=16) as pool:
            parsed = list(pool.map(parse_one, samples))

        results = [None] * len(samples)
        valid = []
        for i, p in enumerate(parsed):
            if isinstance(p, Exception):
                results[i] = p
            else:
                valid.append((i, p))

        if not valid:
            return results

        try:
            batch = Protein.pack([p for _, p in valid])
            batch = graph_construction(batch)
            batch = batch.to(DEVICE)
            node_input = F.one_hot(batch.residue_type, num_classes=21).float()
            out = model(batch, node_input)
            feats = out["graph_feature"]
            for j, (i, _) in enumerate(valid):
                results[i] = feats[j].cpu()
        except Exception as e:
            for i, p in valid:
                try:
                    g = Protein.pack([p])
                    g = graph_construction(g)
                    g = g.to(DEVICE)
                    node_input = F.one_hot(g.residue_type, num_classes=21).float()
                    out = model(g, node_input)
                    results[i] = out["graph_feature"].squeeze(0).cpu()
                except Exception as e2:
                    results[i] = e2

        return results

    return encode_batch, dim


LOADERS = {
    "esm2_650m":    lambda: load_esm("facebook/esm2_t33_650M_UR50D"),
    "esm2_150m":    lambda: load_esm("facebook/esm2_t30_150M_UR50D"),
    "protbert":     lambda: load_protbert(),
    "saprot":       lambda: load_saprot(),
    "gearnet_edge": lambda: load_gearnet(),
}


def extract(dataset, encoder_name):
    canonical_path = Path("data") / dataset / "canonical.json"
    if not canonical_path.exists():
        print(f"ERROR: {canonical_path} not found. Run build_canonical.py first.")
        return

    with open(canonical_path) as f:
        canonical = json.load(f)

    out_dir = Path("data") / dataset / encoder_name
    train_pt = out_dir / "train_embeddings.pt"
    test_pt = out_dir / "test_embeddings.pt"
    if train_pt.exists() and test_pt.exists():
        t = torch.load(train_pt, weights_only=True)
        if t.shape[0] == len(canonical["train"]) and (t.sum(dim=1) != 0).sum() >= t.shape[0] - 10:
            print(f"  SKIP (already done, {t.shape})")
            return

    print(f"Loading {encoder_name}...")
    encode_batch, dim = LOADERS[encoder_name]()
    bs = BATCH_SIZES[encoder_name]
    print(f"  dim={dim}, batch={bs}")

    out_dir = Path("data") / dataset / encoder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_log = open(out_dir / "failures.log", "w")
    fail_log.write("split\tindex\tid\terror\n")
    n_fail = 0

    for split in ["train", "test"]:
        samples = canonical[split]
        n = len(samples)
        embeddings = torch.zeros(n, dim)
        labels = torch.zeros(n, dtype=torch.long)
        t0 = time.time()

        for batch_start in range(0, n, bs):
            batch_end = min(batch_start + bs, n)
            batch = samples[batch_start:batch_end]

            try:
                results = encode_batch(batch)
            except Exception as e:
                results = [e] * len(batch)

            for j, res in enumerate(results):
                idx = batch_start + j
                labels[idx] = batch[j]["label"]
                if isinstance(res, Exception):
                    fail_log.write(f"{split}\t{idx}\t{batch[j].get('id','?')}\t{res}\n")
                    n_fail += 1
                else:
                    embeddings[idx] = res

            elapsed = time.time() - t0
            speed = batch_end / elapsed if elapsed > 0 else 0
            eta = (n - batch_end) / speed if speed > 0 else 0
            print(f"  {split}: {batch_end}/{n} ({speed:.0f}/s, ETA {eta:.0f}s)", end="\r")

        print(f"  {split}: {n} done." + " " * 30)
        torch.save(embeddings, out_dir / f"{split}_embeddings.pt")
        torch.save(labels, out_dir / f"{split}_labels.pt")

    fail_log.close()
    print(f"  Failures: {n_fail}" + (f"  -> {out_dir}/failures.log" if n_fail else ""))

    del encode_batch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS + ["all"])
    parser.add_argument("--encoder", required=True, choices=list(LOADERS) + ["all"])
    args = parser.parse_args()

    ds_list = DATASETS if args.dataset == "all" else [args.dataset]
    enc_list = list(LOADERS) if args.encoder == "all" else [args.encoder]

    for ds in ds_list:
        for enc in enc_list:
            print(f"\n{'='*50}")
            print(f"{ds} x {enc}")
            print(f"{'='*50}")
            extract(ds, enc)