#!/usr/bin/env python3
"""Build the small artifact snapshot used by submission/CSCI447P016.ipynb.

Reads only frozen experiment outputs (tag experiments-final) and curated tables. Writes submission/artifacts/.
No training and no model evaluation: probabilities are copied from the saved predictions.csv files, and every
copied run is checked against its saved result.json (confusion matrix and ROC-AUC) before writing.

Usage: python -I scripts/build_submission_artifacts.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "artifacts"
TAG = "experiments-final"
CORE = [("central_grande", "none"), ("fedavg_grande", "natural"), ("fedavg_grande", "iid"), ("local_grande", "natural"),
        ("local_grande", "iid"), ("central_mlp", "none"), ("fedavg_mlp", "natural"), ("fedavg_mlp", "iid")]
SEEDS, FOLDS = (42, 447, 2026), range(5)


def col(m, p, s):
    return f"{m}_{'central' if p == 'none' else p}_s{s}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov = {"experiments_tag": TAG, "experiments_commit": git("rev-parse", f"{TAG}^{{commit}}"),
            "built_from_commit": git("rev-parse", "HEAD"), "files": {}}

    # 1. predictions: all test rows (every fold) and the validation rows of the hold-out fold (fold 0)
    blocks, info = {}, []
    for m, p in CORE:
        for f in FOLDS:
            for s in SEEDS:
                d = ROOT / "results/phase2b/runs" / f"{m}__{p}__f{f}__s{s}"
                r = json.load(open(d / "result.json"))
                pr = pd.read_csv(d / "predictions.csv")
                keep = pr[(pr.role == "test") | ((pr.role == "val") & (f == 0))]
                blocks.setdefault((f, "test"), {})[col(m, p, s)] = keep[keep.role == "test"].set_index("row_index")[["y", "p1"]]
                if f == 0:
                    blocks.setdefault((f, "val"), {})[col(m, p, s)] = keep[keep.role == "val"].set_index("row_index")[["y", "p1"]]
                t = r["test_at_val_threshold"]
                info.append({"method": m, "partition": p, "fold": f, "seed": s, "val_threshold": r["val_threshold"],
                             "best_step": r["best_step"], "steps_executed": r["steps_executed"],
                             "saved_test_auc": t["roc_auc"], "saved_test_macro_f1": t["macro_f1"], "saved_test_accuracy": t["accuracy"],
                             "saved_tn": t["confusion_[[tn,fp],[fn,tp]]"][0][0], "saved_fp": t["confusion_[[tn,fp],[fn,tp]]"][0][1],
                             "saved_fn": t["confusion_[[tn,fp],[fn,tp]]"][1][0], "saved_tp": t["confusion_[[tn,fp],[fn,tp]]"][1][1]})
    rows = []
    for (f, role), b in sorted(blocks.items()):
        names = list(b)
        idx = b[names[0]].index
        frame = pd.DataFrame({"fold": f, "role": role, "row_index": idx, "y": b[names[0]].y.to_numpy()})
        for n in names:
            assert (b[n].index == idx).all() and (b[n].y.to_numpy() == frame.y.to_numpy()).all()
            frame[n] = b[n].p1.to_numpy()
        rows.append(frame)
    P = pd.concat(rows, ignore_index=True)
    P.to_csv(OUT / "core_predictions.csv", index=False, float_format="%.9g")
    I = pd.DataFrame(info)
    I.to_csv(OUT / "core_run_info.csv", index=False, float_format="%.10g")

    # check the written (rounded) probabilities reproduce every saved test confusion matrix and AUC
    P2 = pd.read_csv(OUT / "core_predictions.csv")
    for r in I.itertuples():
        t = P2[(P2.fold == r.fold) & (P2.role == "test")]
        pr = t[col(r.method, r.partition, r.seed)].to_numpy()
        cm = confusion_matrix(t.y, (pr >= r.val_threshold).astype(int), labels=[0, 1])
        assert (cm == [[r.saved_tn, r.saved_fp], [r.saved_fn, r.saved_tp]]).all(), r
        assert abs(roc_auc_score(t.y, pr) - r.saved_test_auc) < 1e-12, r
    print("predictions:", P.shape, "runs checked:", len(I))

    # 2. frozen leakage groups and splits (for notebook cross-checks)
    dy = pd.read_csv(ROOT / "data/derived/phase2/dyads_v1.csv")[["row_index", "dyad_id", "dyad_status", "mirror_row"]]
    so = pd.read_csv(ROOT / "data/derived/phase2/splits_outer_v1.csv")
    si = pd.read_csv(ROOT / "data/derived/phase2/splits_inner_v1.csv")
    si0 = si[(si.scheme == "C2") & (si.outer_fold == 0)][["row_index", "role"]].rename(columns={"role": "fold1_role"})
    S = dy.merge(so[["row_index", "C1_rowwise_fold", "C2_dyad_fold"]], on="row_index").merge(si0, on="row_index")
    S = S.rename(columns={"dyad_id": "leakage_group_id", "dyad_status": "group_kind"})
    S.to_csv(OUT / "frozen_groups_and_splits.csv", index=False)

    # 3. preprocessing check value: digest of the frozen encoded matrix of the hold-out fold
    dig = json.load(open(ROOT / "data/derived/phase2b/cache_digests.json"))["prep_C2_outer0.npz"]
    prov["fold1_encoded_matrix_sha256"] = dig["Z"]

    # 4. training curves of the hold-out fold, seed 42
    c = pd.read_csv(ROOT / "results/phase2b/runs/central_grande__none__f0__s42/curve.csv")
    r = pd.read_csv(ROOT / "results/phase2b/runs/fedavg_grande__natural__f0__s42/rounds.csv")
    cur = pd.concat([
        pd.DataFrame({"model": "Centralized GRANDE", "step": c.step, "train_loss": c.train_loss, "val_loss": c.val_loss, "val_auc": c.val_auc}),
        pd.DataFrame({"model": "FedAvg GRANDE", "step": r["round"], "train_loss": r.train_loss_weighted, "val_loss": r.val_loss, "val_auc": r.val_auc})])
    cur.to_csv(OUT / "holdout_training_curves.csv", index=False, float_format="%.10g")

    # 5. heterogeneity, consensus, calibration, communication, payload (from curated tables)
    het = pd.read_csv(ROOT / "results/curated/heterogeneity.csv")
    oof = pd.read_csv(ROOT / "results/phase2b/summary/oof_summary_all.csv")
    oof = oof[oof.method == "fedavg_grande"].set_index("partition")
    H = het[["partition", "fedavg_grande_oof_auc", "fedavg_grande_oof_auc_sd", "fedavg_grande_fold_auc", "consensus_round20",
             "unanimous_frac_round20", "best_round_mean"]].copy()
    H.insert(4, "macro_f1_at_05", [oof.loc[p, "oof_f1_05_mean"] for p in H.partition])
    H.to_csv(OUT / "heterogeneity_summary.csv", index=False, float_format="%.10g")
    pd.read_csv(ROOT / "reports/figures/final_style/data/fig03_consensus_by_round.csv").to_csv(
        OUT / "consensus_rounds.csv", index=False, float_format="%.10g")
    per = pd.read_csv(ROOT / "results/curated/personalization.csv")
    per[["method", "label", "within_client_auc_per_fold_A1", "within_client_auc_per_fold_A1_sd", "log_loss", "log_loss_sd",
         "brier", "ece10", "f1_05", "fold_auc"]].to_csv(OUT / "calibration_summary.csv", index=False, float_format="%.10g")
    co = pd.read_csv(ROOT / "results/curated/compression.csv")
    st = pd.read_csv(ROOT / "results/curated/structure_freezing.csv")
    ad = pd.read_csv(ROOT / "results/curated/adaptive_freezing.csv")
    keep = ["config", "partition", "auc", "auc_sd", "d_auc", "gb_to_best"]
    C = pd.concat([st[st.config == "F0"][keep], co[co.config.isin(["C1", "C2", "C3"])][keep],
                   st[st.config.isin(["F1", "F2", "F3"])][keep], ad[ad.config.isin(["G1", "G2"])][keep]], ignore_index=True)
    f0 = C[C.config == "F0"].set_index("partition").gb_to_best
    C["saving_x"] = [f0[p] / g for p, g in zip(C.partition, C.gb_to_best)]
    C.to_csv(OUT / "communication_summary.csv", index=False, float_format="%.10g")
    ad.to_csv(OUT / "adaptive_freezing_summary.csv", index=False, float_format="%.10g")
    g1 = []
    for p in ("natural", "iid"):
        for f in FOLDS:
            for s in SEEDS:
                r = pd.read_csv(ROOT / "results/phase2f/runs" / f"fedavg_grande_cgsf_global__{p}__f{f}__s{s}/rounds.csv")
                g1.append(r[["round", "cons_mean", "cons_unanimous", "bytes_down", "bytes_up"]].assign(partition=p, fold=f, seed=s))
    pd.concat(g1)[["partition", "fold", "seed", "round", "cons_mean", "cons_unanimous", "bytes_down", "bytes_up"]].to_csv(
        OUT / "g1_round_logs.csv", index=False, float_format="%.17g")
    trig = json.load(open(ROOT / "results/phase2f/summary/phase2f_summary.json"))["G1_trigger"]
    pd.DataFrame(trig["per_run"]).to_csv(OUT / "g1_trigger_rounds.csv", index=False)
    g0 = json.load(open(ROOT / "results/phase2e/gate0.json"))
    pd.DataFrame([{"tensor": t["tensor"], "shape": "x".join(map(str, t["shape"])), "params": t["params"],
                   "share_pct": t["pct_fp32_payload"]} for t in g0["tensors"]]).to_csv(OUT / "grande_tensors.csv", index=False, float_format="%.10g")

    # 6. provenance
    src = {"core_predictions.csv": "results/phase2b/runs/{central_grande,fedavg_grande,local_grande,central_mlp,fedavg_mlp}__*__f*__s*/predictions.csv (120 runs)",
           "core_run_info.csv": "results/phase2b/runs/*/result.json (same 120 runs)",
           "frozen_groups_and_splits.csv": "data/derived/phase2/dyads_v1.csv, splits_outer_v1.csv, splits_inner_v1.csv",
           "holdout_training_curves.csv": "results/phase2b/runs/central_grande__none__f0__s42/curve.csv, fedavg_grande__natural__f0__s42/rounds.csv",
           "heterogeneity_summary.csv": "results/curated/heterogeneity.csv, results/phase2b/summary/oof_summary_all.csv",
           "consensus_rounds.csv": "reports/figures/final_style/data/fig03_consensus_by_round.csv (from results/phase2b/runs/fedavg_grande__*/rounds.csv)",
           "calibration_summary.csv": "results/curated/personalization.csv",
           "communication_summary.csv": "results/curated/{structure_freezing,compression,adaptive_freezing}.csv",
           "adaptive_freezing_summary.csv": "results/curated/adaptive_freezing.csv",
           "g1_trigger_rounds.csv": "results/phase2f/summary/phase2f_summary.json",
           "grande_tensors.csv": "results/phase2e/gate0.json",
           "g1_round_logs.csv": "results/phase2f/runs/fedavg_grande_cgsf_global__*/rounds.csv (30 runs; consensus and bytes per round)"}
    for n, s in src.items():
        prov["files"][n] = {"sources": s, "method": "copied/selected by scripts/build_submission_artifacts.py (no training)",
                            "bytes": (OUT / n).stat().st_size, "sha256": sha(OUT / n)}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    tot = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {len(src) + 1} files, {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
