"""Mine saved R matrices for worst catastrophic-forgetting trajectories.
R[i][j]=acc on phase j after phase i. Output: diagnose/results/forgetting_*.json"""

import argparse
import json
from pathlib import Path

import numpy as np

ALL_ENCODERS = ["esm2_650m", "esm2_150m", "gearnet_edge", "saprot", "protbert"]
ALL_DATASETS = ["enzyme_reaction", "ec_number"]
DEFAULT_METHODS = ["ewc", "replay", "icarl", "derpp", "rer", "ranpac",
                   "fecam_common"]


def analyze_R(R):
    """Given accuracy matrix R (list of lists, lower-triangular filled), return
    the worst-forgotten phase and its trajectory."""
    T = len(R)
    # pad to square for easy indexing (R[i] has length i+1)
    learned = {j: R[j][j] for j in range(T)}
    final = {j: R[T - 1][j] for j in range(T - 1)}   # phase T-1 has no "after"
    drops = {j: learned[j] - final[j] for j in final}
    if not drops:
        return None
    worst_j = max(drops, key=drops.get)

    curve = [R[i][worst_j] for i in range(worst_j, T)]
    # biggest single-step drop along that curve
    step_drops = [curve[k] - curve[k + 1] for k in range(len(curve) - 1)]
    if step_drops:
        worst_step = int(np.argmax(step_drops))
        collapse_at_phase = worst_j + worst_step + 1
        biggest_step_drop = float(step_drops[worst_step])
    else:
        collapse_at_phase = worst_j
        biggest_step_drop = 0.0

    return {
        "worst_phase": int(worst_j),
        "learned_acc": round(float(learned[worst_j]), 3),
        "final_acc": round(float(final[worst_j]), 3),
        "total_drop": round(float(drops[worst_j]), 3),
        "collapse_at_phase": int(collapse_at_phase),
        "biggest_single_step_drop": round(biggest_step_drop, 3),
        "retention_curve": [round(float(x), 3) for x in curve],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--encoder", action="append", default=None)
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_dir", type=str, default="./results")
    args = ap.parse_args()

    datasets = args.dataset or ALL_DATASETS
    encoders = args.encoder or ALL_ENCODERS
    methods = args.method or DEFAULT_METHODS
    rdir = Path(args.results_dir)

    out_dir = Path("diagnose/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        summary = {}
        for enc in encoders:
            summary[enc] = {}
            for m in methods:
                fp = rdir / f"{enc}_{ds}_{m}_seed{args.seed}.json"
                if not fp.exists():
                    continue
                with open(fp) as f:
                    res = json.load(f)
                R = res.get("R")
                if not R:
                    continue
                analysis = analyze_R(R)
                if analysis is None:
                    continue
                analysis["reported_forgetting"] = res.get("forgetting")
                analysis["reported_avg_acc"] = res.get("avg_acc")
                summary[enc][m] = analysis
                print(f"[{enc}/{ds}/{m}] phase {analysis['worst_phase']}: "
                      f"learned {analysis['learned_acc']:.2f} -> "
                      f"final {analysis['final_acc']:.2f} "
                      f"(drop {analysis['total_drop']:.2f}, "
                      f"biggest fall at phase {analysis['collapse_at_phase']})")

        out_fp = out_dir / f"forgetting_{ds}.json"
        with open(out_fp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  -> {out_fp}\n")


if __name__ == "__main__":
    main()