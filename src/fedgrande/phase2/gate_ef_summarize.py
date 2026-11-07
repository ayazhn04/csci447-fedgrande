"""Gate E/F summary: tabulate measured runs, the C1 leakage contrast, and transparent compute projections.

Measured numbers are read from results/phase2/gate_ef/<run>/metrics.json. Projections are labelled as such and
computed from the measured seconds/epoch with the formulas stored in the output.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from fedgrande.paths import CONFIGS, DERIVED, RESULTS  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402

N_CLIENTS = 21
PROTOCOLS = {"5folds_x_1seed": 5, "5folds_x_3seeds": 15, "3folds_x_3seeds": 9}


def main():
    cfg = json.load(open(CONFIGS / "gate_ef_runs.json"))
    rows = []
    for r in cfg["runs"]:
        p = RESULTS / "gate_ef" / r["run_id"] / "metrics.json"
        if not p.exists():
            rows.append({"run_id": r["run_id"], "status": "missing"})
            continue
        m = json.load(open(p))
        curve = pd.read_csv(RESULTS / "gate_ef" / r["run_id"] / "curve.csv")
        rows.append({"run_id": r["run_id"], "status": "ok", "scheme": m["scheme"], "n_estimators": m["n_estimators"], "depth": m["depth"],
                     "V_effective": m["selected_variables_per_estimator"], "input_features": m["network_input_features_after_preprocessing"],
                     "batch_size": m["effective_batch_size"], "compiled": m["compiled"], "epochs_completed": m["epochs_completed"],
                     "best_epoch": m["best_epoch"], "fit_seconds": round(m["fit_seconds"], 1),
                     "sec_per_epoch_median": round(float(curve.epoch_seconds.iloc[1:].median() if len(curve) > 1 else curve.epoch_seconds.median()), 3),
                     "first_epoch_seconds": round(float(curve.epoch_seconds.iloc[0]), 2),
                     "predict_test_seconds": round(m["predict_test_seconds"], 2), "peak_rss_mb": round(m["peak_rss_mb_process"]),
                     "trainable_parameters": m["trainable_parameters"],
                     "test_auc": round(m["test_at_0.5"]["roc_auc"], 4), "test_macro_f1_0.5": round(m["test_at_0.5"]["macro_f1"], 4),
                     "test_accuracy_0.5": round(m["test_at_0.5"]["accuracy"], 4), "val_auc": round(m["val_at_0.5"]["roc_auc"], 4),
                     "val_threshold": m["val_selected_threshold"], "test_macro_f1_valthr": round(m["test_at_val_threshold"]["macro_f1"], 4)})
    table = pd.DataFrame(rows)
    out = RESULTS / "gate_ef"
    table.to_csv(out / "runs_table.csv", index=False)

    # Leakage contrast inside the single C1 model: test rows whose mirror is in training vs not
    leak = None
    c1 = out / "C1_leak_e256_d5" / "predictions.csv"
    if c1.exists():
        pr = pd.read_csv(c1)
        te = pr[pr.role == "test"]
        d = pd.read_csv(DERIVED / "dyads_v1.csv")
        outer = pd.read_csv(DERIVED / "splits_outer_v1.csv")
        fold = outer.C1_rowwise_fold.to_numpy()
        mir = d.mirror_row.to_numpy()[te.row_index.to_numpy()]
        in_train = np.array([(mm >= 0) and (fold[mm] != 0) for mm in mir])
        auc = lambda s: float(roc_auc_score(s.y, s.p1)) if s.y.nunique() == 2 else None  # noqa: E731
        # mirror probability agreement: for leaked rows, is the model's score close to the label of the mirror?
        leak = {"n_test": int(len(te)), "n_mirror_in_train": int(in_train.sum()), "n_mirror_not_in_train": int((~in_train).sum()),
                "pos_mirror_in_train": int(te.y[in_train].sum()), "pos_mirror_not_in_train": int(te.y[~in_train].sum()),
                "auc_mirror_in_train": auc(te[in_train]), "auc_mirror_not_in_train": auc(te[~in_train]),
                "auc_all_C1_test": auc(te),
                "auc_C2_same_config": float(table.loc[table.run_id == "F_e256_d5", "test_auc"].iloc[0]) if (table.run_id == "F_e256_d5").any() else None,
                "caveat": "single model, single fold; subsets differ in size/composition, the contrast is descriptive, not causal"}

    # Projections from measured seconds/epoch (PROJECTED, not measured)
    proj = []
    ok = table[table.status == "ok"]
    for _, r in ok[(ok.scheme == "C2") & (~ok.compiled)].iterrows():
        spe, fit = float(r.sec_per_epoch_median), float(r.fit_seconds)
        unit = {
            "centralized_fit_s": fit,
            "local_only_all_clients_s": fit,          # same total samples per epoch, ~same epochs (assumption A2)
            "fedavg_R100_E1_s": 100 * spe,            # 100 rounds x 1 local epoch over all clients' data (A3)
            "fedavg_R200_E1_s": 200 * spe,
        }
        for proto, n_runs in PROTOCOLS.items():
            for parts, n_parts in (("nat+iid", 2), ("nat+iid+dirichlet", 3)):
                # per (fold, seed, partition): local-only + FedAvg-GRANDE(R=100); centralized once per (fold, seed)
                total = n_runs * (unit["centralized_fit_s"] + n_parts * (unit["local_only_all_clients_s"] + unit["fedavg_R100_E1_s"]))
                proj.append({"config": f"E{int(r.n_estimators)}_d{int(r.depth)}", "protocol": proto, "partitions": parts,
                             "grande_hours_projected": round(total / 3600, 2), "per_unit_seconds_measured_basis": unit})
    proj_df = pd.DataFrame([{k: v for k, v in p.items() if k != "per_unit_seconds_measured_basis"} for p in proj])
    proj_df.to_csv(out / "projections.csv", index=False)
    report = {
        "measured_runs": rows, "leakage_contrast_C1": leak, "projections": proj,
        "projection_assumptions": {
            "A1": "seconds/epoch measured on C2 outer fold 0 (≈5.36k fit + 1.34k val rows), 6 torch CPU threads, no other load; median over epochs 2..N",
            "A2": "local-only over 21 clients costs about one centralized fit (same total samples per epoch; per-client early stopping assumed to take a similar number of epochs). Small-client batch overhead is ignored.",
            "A3": "FedAvg-GRANDE: R rounds x 1 local epoch over all clients ~= R centralized epochs (client compute is sequential on one machine); aggregation and per-round validation overhead not included",
            "A4": "FedAvg-MLP and logistic-regression baselines are not included (expected to be small relative to GRANDE; to be measured in Phase 2B)",
            "A5": "projection = n_folds*n_seeds * [centralized + n_partitions * (local_only + fedavg)]",
        },
    }
    write_json(out / "gate_ef_summary.json", report)
    print(table.to_string(index=False))
    print(json.dumps(leak, indent=1))
    print(proj_df.to_string(index=False))


if __name__ == "__main__":
    main()
