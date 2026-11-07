"""Amendment A1: build the Dirichlet(alpha=0.1) size-matched label-skew partition (separate file; frozen files untouched)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedgrande.fl import partitions as P  # noqa: E402
from fedgrande.fl.features import load_modeling  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402

OUT = ROOT / "data" / "derived" / "phase2b"
REP = ROOT / "results" / "phase2b" / "prepare"


def main():
    cfg = json.load(open(ROOT / "configs" / "phase2b" / "amendment_A1_dirichlet.json"))
    X, y, meta = load_modeling()
    cl, info = P.dirichlet_size_matched(meta, y, cfg["alpha"], cfg["partition_seed"])
    nat = P.natural(meta)
    sizes = np.bincount(nat)
    df = pd.DataFrame({"row_index": meta.row_index, "leakage_group_id": meta.leakage_group_id, f"client_{cfg['partition_name']}": cl})
    df.to_csv(OUT / "partitions_dirichlet_v1.csv", index=False)
    d = P.describe(cl, y, meta, sizes)
    d.to_csv(REP / "partition_dirichlet0.1.csv", index=False)
    cross = int((pd.DataFrame({"g": meta.leakage_group_id, "c": cl}).groupby("g").c.nunique() > 1).sum())
    roles = [np.load(OUT / f"prep_C2_outer{k}.npz")["roles"] for k in range(5)]
    per_fold = [{"fold": k, "clients_without_fit_pos": int(sum(y[(roles[k] == "fit") & (cl == c)].sum() == 0 for c in range(21))),
                 "clients_without_val_rows": int(sum(((roles[k] == "val") & (cl == c)).sum() == 0 for c in range(21))),
                 "clients_single_class_val": int(sum(len(np.unique(y[(roles[k] == "val") & (cl == c)])) == 1 for c in range(21)))}
                for k in range(5)]
    rep = {"config": cfg, **info, "leakage_groups_crossing_clients": cross, "max_abs_size_deviation": int(d.size_deviation.abs().max()),
           "pos_rate_min": float(d.pos_rate.min()), "pos_rate_max": float(d.pos_rate.max()),
           "clients_with_zero_pos": int((d.pos_rows == 0).sum()), "clients_all_pos": int((d.pos_rows == d.rows).sum()),
           "per_fold_edge_cases": per_fold, "file": str((OUT / "partitions_dirichlet_v1.csv").relative_to(ROOT)),
           "sha256": sha256_file(OUT / "partitions_dirichlet_v1.csv"), "meta": run_metadata()}
    write_json(REP / "partition_dirichlet0.1.json", rep)
    print(d[["client", "rows", "target_rows", "pos_rows", "pos_rate", "n_waves_represented"]].to_string(index=False))
    print({k: rep[k] for k in ("leakage_groups_crossing_clients", "max_abs_size_deviation", "pos_rate_min", "pos_rate_max", "clients_with_zero_pos", "clients_all_pos")})
    print(per_fold)


if __name__ == "__main__":
    main()
