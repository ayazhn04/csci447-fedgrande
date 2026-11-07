"""Gate C: historical row-wise CV (C1) vs dyad-grouped CV (C2); verification and mirror-leakage count."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedgrande import splits  # noqa: E402
from fedgrande.data import load_xy  # noqa: E402
from fedgrande.paths import DERIVED, RESULTS  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402


def main():
    X, y, full = load_xy()
    d = pd.read_csv(DERIVED / "dyads_v1.csv")
    assert (d["row_index"].to_numpy() == np.arange(len(y))).all()
    groups, mirror = d["dyad_id"].to_numpy(), d["mirror_row"].to_numpy()

    f1 = splits.fold_assignment_rowwise(y)
    f2 = splits.fold_assignment_grouped(y, groups)
    # determinism: regenerate and compare
    assert (f1 == splits.fold_assignment_rowwise(y)).all() and (f2 == splits.fold_assignment_grouped(y, groups)).all()

    v1, v2 = splits.verify(f1, y, groups, mirror), splits.verify(f2, y, groups, mirror)
    out = pd.DataFrame({"row_index": np.arange(len(y)), "dyad_id": groups, "C1_rowwise_fold": f1, "C2_dyad_fold": f2})

    # inner (validation) splits for every outer fold; C2 grouped, C1 row-wise (historical style)
    inner = []
    for scheme, fold in (("C1", f1), ("C2", f2)):
        for k in range(splits.N_SPLITS):
            tr = np.flatnonzero(fold != k)
            if scheme == "C2":
                fit, val = splits.inner_train_val(tr, y, groups)
            else:
                g_row = np.arange(len(y))  # each row its own group == row-wise stratified
                fit, val = splits.inner_train_val(tr, y, g_row)
            role = np.full(len(y), "test", dtype=object)
            role[fit], role[val] = "fit", "val"
            inner.append(pd.DataFrame({"scheme": scheme, "outer_fold": k, "row_index": np.arange(len(y)), "role": role}))
    inner = pd.concat(inner, ignore_index=True)
    # inner-split checks
    inner_checks = []
    for (scheme, k), g in inner.groupby(["scheme", "outer_fold"]):
        r = {role: g.loc[g.role == role, "row_index"].to_numpy() for role in ("fit", "val", "test")}
        fit_val_overlap = len(set(groups[r["fit"]]) & set(groups[r["val"]]))
        inner_checks.append({"scheme": scheme, "outer_fold": int(k), **{f"n_{a}": int(len(b)) for a, b in r.items()},
                             "val_pos_rate": float(y[r["val"]].mean()), "fit_pos_rate": float(y[r["fit"]].mean()),
                             "fit_val_dyad_group_overlap": int(fit_val_overlap),
                             "test_equals_outer_test": bool((np.sort(r["test"]) == np.flatnonzero((f1 if scheme == "C1" else f2) == k)).all())})

    os.makedirs(DERIVED, exist_ok=True)
    p_out, p_in = DERIVED / "splits_outer_v1.csv", DERIVED / "splits_inner_v1.csv"
    out.to_csv(p_out, index=False)
    inner.to_csv(p_in, index=False)
    report = {
        "C1_rowwise": {"method": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)", **v1},
        "C2_dyad_grouped": {"method": "group-level StratifiedKFold(n_splits=5, shuffle=True, random_state=42) over dyad_id groups (dyads_v1), stratified by group label", **v2},
        "alternatives": {name: {"max_abs_pos_rate_deviation": vv["max_abs_pos_rate_deviation"], "test_sizes": [f["n_test"] for f in vv["folds"]],
                                "dyad_group_overlap": vv["total_dyad_group_overlap"], "seedable": seedable}
                         for name, vv, seedable in (
                             ("sklearn StratifiedGroupKFold(shuffle=True, random_state=42)", splits.verify(splits.fold_assignment_sgkf(y, groups, shuffle=True), y, groups, mirror), True),
                             ("sklearn StratifiedGroupKFold(shuffle=False)", splits.verify(splits.fold_assignment_sgkf(y, groups, shuffle=False), y, groups, mirror), False),
                             ("group-level StratifiedKFold (chosen)", v2, True))},
        "inner_validation": {"method": "fold 0 of the group-level StratifiedKFold(5, shuffle=True, random_state=42) on the outer-training rows; "
                                       "C2 groups=dyad_id, C1 groups=row (row-wise)", "checks": inner_checks},
        "historical_fold_hash_check": {"C1_hash": v1["fold_hash"],
                                       "phase1_audit_hash": "ca9d1164c5100d5b310dbb644672d9dad2d0cdaceb38055b0e6f80ee3d1c1755",
                                       "same_as_phase1": v1["fold_hash"] == "ca9d1164c5100d5b310dbb644672d9dad2d0cdaceb38055b0e6f80ee3d1c1755"},
        "outputs": {"outer": str(p_out), "outer_sha256": sha256_file(p_out), "inner": str(p_in), "inner_sha256": sha256_file(p_in)},
        "inputs": {"dyads_csv_sha256": sha256_file(DERIVED / "dyads_v1.csv")},
        "meta": run_metadata(seed=splits.SEED),
    }
    write_json(RESULTS / "gate_c" / "splits_report.json", report)
    for name, v in (("C1", v1), ("C2", v2)):
        print(name, "assigned once:", v["all_rows_assigned_once"], "| group overlap:", v["total_dyad_group_overlap"],
              "| test rows w/ mirror in train:", v["total_test_rows_whose_mirror_is_in_train"],
              "| max pos-rate dev:", round(v["max_abs_pos_rate_deviation"], 5))
        for f in v["folds"]:
            print("   ", {k: f[k] for k in ("fold", "n_train", "n_test", "row_overlap", "dyad_group_overlap", "test_pos", "test_rows_whose_mirror_is_in_train")})
    print("C1 hash equals Phase 1:", report["historical_fold_hash_check"]["same_as_phase1"])
    print(pd.DataFrame(inner_checks).to_string())


if __name__ == "__main__":
    main()
