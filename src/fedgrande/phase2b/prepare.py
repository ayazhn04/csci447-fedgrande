"""B0/B1 data preparation: fixed client partitions and per-fold preprocessing caches.

Writes data/derived/phase2b/partitions_v1.csv (+ report) and prep_C2_outer{k}.npz (+ JSON summary).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedgrande.fl import partitions as P  # noqa: E402
from fedgrande.fl.features import load_modeling, schema  # noqa: E402
from fedgrande.fl.preprocess import FederatedPreprocessor  # noqa: E402
from fedgrande.paths import CONFIGS, DERIVED, ROOT  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402

OUT = ROOT / "data" / "derived" / "phase2b"
REP = ROOT / "results" / "phase2b" / "prepare"


def fold_roles(inner, k):
    s = inner[(inner.scheme == "C2") & (inner.outer_fold == k)].sort_values("row_index")
    return s.role.to_numpy()


def main():
    cfg = json.load(open(ROOT / "configs" / "phase2b" / "core.json"))
    X, y, meta = load_modeling()
    S = schema(X)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REP, exist_ok=True)

    nat = P.natural(meta)
    iid = P.iid_size_matched(meta, y, seed=cfg["seeds"]["partition_seed"])
    sizes = np.bincount(nat)
    part = meta.copy()
    part["client_natural"], part["client_iid"] = nat, iid
    part.to_csv(OUT / "partitions_v1.csv", index=False)
    d_nat = P.describe(nat, y, meta, sizes)
    d_nat.insert(1, "wave", np.sort(meta.wave.unique()))
    d_iid = P.describe(iid, y, meta, sizes)
    d_nat.to_csv(REP / "partition_natural.csv", index=False)
    d_iid.to_csv(REP / "partition_iid.csv", index=False)
    groups_cross = {name: int((pd.DataFrame({"g": meta.leakage_group_id, "c": cl}).groupby("g").c.nunique() > 1).sum())
                    for name, cl in (("natural", nat), ("iid", iid))}

    inner = pd.read_csv(DERIVED / "splits_inner_v1.csv")
    order = {c: list(X[c].cat.categories) for c in S["categorical"]}
    fold_summ = []
    for k in cfg["outer_split"]["folds"]:
        roles = fold_roles(inner, k)
        fit = roles == "fit"
        pp = FederatedPreprocessor(S["numeric"], S["categorical"], order).fit_from_clients(
            [X[fit & (nat == c)] for c in range(len(sizes))])
        Z = pp.transform(X)
        _, unseen_val = pp.transform(X[roles == "val"], return_unseen=True)
        _, unseen_test = pp.transform(X[roles == "test"], return_unseen=True)
        g = meta.leakage_group_id.to_numpy()
        for a, b in (("fit", "val"), ("fit", "test"), ("val", "test")):
            assert not set(g[roles == a]) & set(g[roles == b]), f"leakage group crosses {a}/{b} in fold {k}"
        f = OUT / f"prep_C2_outer{k}.npz"
        np.savez(f, Z=Z, roles=roles.astype("U4"), feature_names=np.array(pp.feature_names_, dtype="U128"), n_numeric=np.array(len(S["numeric"])),
                 mean=pp.params_["mean"], sd=pp.params_["sd"])
        s = pp.summary()
        fold_summ.append({"fold": k, **{kk: s[kk] for kk in ("encoded_dim", "one_hot_columns", "n_train_rows", "missing_indicator_columns")},
                          "n_fit": int(fit.sum()), "n_val": int((roles == "val").sum()), "n_test": int((roles == "test").sum()),
                          "field_levels_in_training": s["categories_per_column"]["field"],
                          "unseen_or_unk_val": unseen_val, "unseen_or_unk_test": unseen_test,
                          "file": str(f.relative_to(ROOT)), "sha256": sha256_file(f)})
        print(fold_summ[-1]["fold"], s["encoded_dim"], unseen_val, unseen_test)
    rep = {"raw_feature_schema": {"n_raw": S["n_raw"], "n_numeric": S["n_numeric"], "n_categorical": S["n_categorical"],
                                  "numeric": S["numeric"], "categorical": S["categorical"]},
           "partitions": {"file": "data/derived/phase2b/partitions_v1.csv", "sha256": sha256_file(OUT / "partitions_v1.csv"),
                          "leakage_groups_crossing_clients": groups_cross,
                          "iid_max_abs_size_deviation": int(d_iid.size_deviation.abs().max()),
                          "iid_pos_rate_range": [float(d_iid.pos_rate.min()), float(d_iid.pos_rate.max())],
                          "natural_pos_rate_range": [float(d_nat.pos_rate.min()), float(d_nat.pos_rate.max())]},
           "folds": fold_summ, "meta": run_metadata()}
    write_json(REP / "prepare_report.json", rep)
    print(json.dumps(rep["partitions"], indent=1))


if __name__ == "__main__":
    main()
