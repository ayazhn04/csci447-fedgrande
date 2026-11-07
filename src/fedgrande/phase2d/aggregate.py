"""Phase 2D aggregation: C0 (Phase 2B FedAvg-GRANDE, reused) vs C1-C3; tables, frontier plots and report from saved outputs."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedgrande.fl import comm  # noqa: E402
from fedgrande.fl.metrics import binary_metrics  # noqa: E402
from fedgrande.fl.models import build_grande, trainable_state  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.phase2.phase2a_summary import md_table  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402

CONFIGS = {"C0": ("phase2b", "fedavg_grande", 256, "fp32"), "C1": ("phase2d", "fedavg_grande64", 64, "fp32"),
           "C2": ("phase2d", "fedavg_grande_int8", 256, "int8"), "C3": ("phase2d", "fedavg_grande64_int8", 64, "int8")}
LABEL = {"C0": "C0 256 FP32", "C1": "C1 64 FP32", "C2": "C2 256 INT8-up", "C3": "C3 64 INT8-up"}
PARTS, FOLDS, SEEDS = ("natural", "iid"), range(5), (42, 447, 2026)
OUT = ROOT / "results" / "phase2d" / "summary"
FIG = ROOT / "reports" / "figures" / "phase2d"
GB = 1e9


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    shapes = {e: trainable_state(build_grande(502, 42, n_estimators=e, depth=5)) for e in (256, 64)}
    fbe = {e: {"features_by_estimator": build_grande(502, 42, n_estimators=e, depth=5).features_by_estimator} for e in (256, 64)}
    runs, preds, rounds = [], [], []
    for c, (phase, method, E, up) in CONFIGS.items():
        for part in PARTS:
            for f in FOLDS:
                for s in SEEDS:
                    d = ROOT / "results" / phase / "runs" / f"{method}__{part}__f{f}__s{s}"
                    r = json.load(open(d / "result.json"))
                    acc = comm.accounting_v2(shapes[E], fbe[E], n_clients=21, rounds_executed=r["steps_executed"], best_round=r["best_step"], uplink=up)
                    if "v2" in (r.get("communication") or {}):
                        assert r["communication"]["v2"]["cum_total_to_best"] == acc["cum_total_to_best"], "accounting mismatch"
                    rd = pd.read_csv(d / "rounds.csv")
                    best = rd[rd["round"] == r["best_step"]].iloc[0]
                    upto = rd[rd["round"] <= r["best_step"]]
                    rec = {"config": c, "partition": part, "fold": f, "seed": s, "n_estimators": E, "uplink": up,
                           "params": acc["trainable_parameters"], "best_round": r["best_step"], "rounds": r["steps_executed"],
                           "seconds": r["seconds_total"], "sec_per_round": r["seconds_total"] / r["steps_executed"], "peak_rss_mb": r["peak_rss_mb"],
                           "test_auc_fold": r["test_at_0.5"]["roc_auc"], "val_threshold": r["val_threshold"],
                           **{k: acc[k] for k in ("downlink_bytes_per_client_per_round", "uplink_bytes_per_client_per_round", "downlink_bytes_per_round",
                                                  "uplink_bytes_per_round", "total_bytes_per_round", "bytes_once_at_init_buffers", "cum_downlink_to_best",
                                                  "cum_uplink_to_best", "cum_total_to_best", "cum_total_to_final")},
                           "cons_at_best": best.cons_consensus_mean, "unanimous_at_best": best.cons_frac_nodes_unanimous,
                           "cos_mean_to_best": upto.cos_to_aggregate_mean.mean(), "project_commit": r["meta"]["project_commit"]}
                    if up == "int8":
                        for k in [k for k in rd.columns if k.startswith("qerr_rel_l2_")]:
                            rec[f"{k}__mean_to_best"] = float(upto[k].mean())
                            rec[f"{k}__max_to_best"] = float(upto[k].max())
                    runs.append(rec)
                    p = pd.read_csv(d / "predictions.csv")
                    p = p[p.role == "test"].assign(config=c, partition=part, fold=f, seed=s, pred_vt=lambda x: (x.p1 >= r["val_threshold"]).astype(int))
                    preds.append(p)
                    rounds.append(rd[["round", "val_auc"]].assign(config=c, partition=part, fold=f, seed=s,
                                                                  cum_gb=lambda x: (acc["bytes_once_at_init_buffers"] + x["round"] * acc["total_bytes_per_round"]) / GB))
    R = pd.DataFrame(runs)
    R.to_csv(OUT / "runs_table.csv", index=False)
    P = pd.concat(preds)
    oof = []
    for (c, part, s), g in P.groupby(["config", "partition", "seed"]):
        assert g.fold.nunique() == 5 and g.row_index.is_unique
        a, b = binary_metrics(g.y, g.p1), binary_metrics(g.y, g.p1, pred=g.pred_vt)
        oof.append({"config": c, "partition": part, "seed": s, "oof_auc": a["roc_auc"], "oof_f1_05": a["macro_f1"], "oof_f1_valthr": b["macro_f1"],
                    "oof_acc_05": a["accuracy"], "cm_valthr": b["confusion_[[tn,fp],[fn,tp]]"]})
    O = pd.DataFrame(oof)
    O.to_csv(OUT / "oof_by_seed.csv", index=False)
    base = O[O.config == "C0"].set_index(["partition", "seed"]).oof_auc
    O["delta_auc_vs_C0"] = [r.oof_auc - base[(r.partition, r.seed)] for r in O.itertuples()]
    O["auc_retention_pct"] = [100 * r.oof_auc / base[(r.partition, r.seed)] for r in O.itertuples()]

    agg = R.groupby(["config", "partition"]).agg(params=("params", "first"), best_round=("best_round", "mean"), best_round_sd=("best_round", "std"),
                                                rounds=("rounds", "mean"), seconds=("seconds", "mean"), sec_per_round=("sec_per_round", "mean"),
                                                peak_rss_mb=("peak_rss_mb", "max"), up_client_round=("uplink_bytes_per_client_per_round", "first"),
                                                down_client_round=("downlink_bytes_per_client_per_round", "first"),
                                                total_round_mb=("total_bytes_per_round", lambda x: x.iloc[0] / 1e6),
                                                gb_to_best=("cum_total_to_best", lambda x: x.mean() / GB), gb_up_to_best=("cum_uplink_to_best", lambda x: x.mean() / GB),
                                                gb_down_to_best=("cum_downlink_to_best", lambda x: x.mean() / GB), gb_to_final=("cum_total_to_final", lambda x: x.mean() / GB),
                                                cons_at_best=("cons_at_best", "mean"), unanimous_at_best=("unanimous_at_best", "mean"),
                                                cos_mean_to_best=("cos_mean_to_best", "mean")).reset_index()
    so = O.groupby(["config", "partition"]).agg(auc=("oof_auc", "mean"), auc_sd=("oof_auc", "std"), f1_05=("oof_f1_05", "mean"), f1_05_sd=("oof_f1_05", "std"),
                                               f1_vt=("oof_f1_valthr", "mean"), f1_vt_sd=("oof_f1_valthr", "std"), acc=("oof_acc_05", "mean"),
                                               d_auc=("delta_auc_vs_C0", "mean"), d_auc_sd=("delta_auc_vs_C0", "std"),
                                               retention=("auc_retention_pct", "mean"), retention_sd=("auc_retention_pct", "std")).reset_index()
    T = so.merge(agg, on=["config", "partition"])
    c0 = T[T.config == "C0"].set_index("partition")
    T["uplink_compression_per_round_x"] = [c0.loc[r.partition, "up_client_round"] / r.up_client_round for r in T.itertuples()]
    T["total_compression_per_round_x"] = [c0.loc[r.partition, "total_round_mb"] / r.total_round_mb for r in T.itertuples()]
    T["total_compression_to_best_x"] = [c0.loc[r.partition, "gb_to_best"] / r.gb_to_best for r in T.itertuples()]
    T.to_csv(OUT / "main_table.csv", index=False)

    # paired fold/seed test-AUC differences vs C0 (descriptive)
    fw = R.pivot_table(index=["partition", "fold", "seed"], columns="config", values="test_auc_fold")
    pair = {f"{c}-C0": {p: {"mean": float((g[c] - g["C0"]).mean()), "min": float((g[c] - g["C0"]).min()), "max": float((g[c] - g["C0"]).max()),
                            "n_negative": int(((g[c] - g["C0"]) < 0).sum()), "n": int(len(g))} for p, g in fw.groupby(level="partition")}
            for c in ("C1", "C2", "C3")}
    q = R[R.uplink == "int8"]
    qcols = [k for k in q.columns if k.startswith("qerr_")]
    Q = q.groupby(["config", "partition"])[qcols].mean().T
    Q.columns = [f"{a}|{b}" for a, b in Q.columns]
    Q.to_csv(OUT / "quantization_error.csv")

    # figures: frontier and val AUC vs cumulative GB
    fig, ax = plt.subplots(figsize=(6, 4.2))
    mk = {"C0": "o", "C1": "s", "C2": "^", "C3": "D"}
    col = {"natural": "#2a6fdb", "iid": "#e07b39"}
    for r in T.itertuples():
        ax.errorbar(r.gb_to_best, r.auc, yerr=r.auc_sd, marker=mk[r.config], color=col[r.partition], capsize=3, linestyle="none",
                    label=f"{LABEL[r.config]} ({r.partition})")
    ax.set_xscale("log")
    ax.set_xlabel("cumulative tensor bytes to best validation round (GB, log scale)")
    ax.set_ylabel("pooled OOF ROC-AUC (mean ± SD over 3 seeds)")
    ax.set_title("Federated GRANDE accuracy–communication tradeoff", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "frontier.png", dpi=150)
    plt.close(fig)
    RD = pd.concat(rounds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, part in zip(axes, PARTS):
        for c in CONFIGS:
            g = RD[(RD.config == c) & (RD.partition == part)].groupby("round").agg(val=("val_auc", "mean"), gb=("cum_gb", "first"))
            ax.plot(g.gb, g.val, label=LABEL[c], marker=None)
        ax.set_xscale("log")
        ax.set_title(f"{part}: pooled validation AUC vs cumulative GB", fontsize=10)
        ax.set_xlabel("cumulative GB (conceptual tensor bytes)")
    axes[0].set_ylabel("validation ROC-AUC (mean over runs still active)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "val_auc_vs_gb.png", dpi=150)
    plt.close(fig)

    summary = {"main_table": T.round(6).to_dict("records"), "paired_fold_auc_differences": pair, "quantization_error": Q.round(6).to_dict(),
               "project_commits": sorted(R.project_commit.unique()), "total_new_compute_hours": float(R[R.config != "C0"].seconds.sum() / 3600)}
    write_json(OUT / "phase2d_summary.json", summary)
    show = T[["config", "partition", "params", "auc", "auc_sd", "d_auc", "d_auc_sd", "retention", "f1_05", "f1_vt", "best_round", "gb_to_best",
              "gb_up_to_best", "gb_down_to_best", "uplink_compression_per_round_x", "total_compression_per_round_x", "total_compression_to_best_x",
              "seconds", "sec_per_round", "peak_rss_mb", "cons_at_best", "unanimous_at_best", "cos_mean_to_best"]].round(4)
    L = ["# Phase 2D report (generated)\n", "Generated by `src/fedgrande/phase2d/aggregate.py`; protocol `reports/PHASE2D_PROTOCOL.md`. C0 = Phase 2B FedAvg-GRANDE (reused).\n",
         f"- commits in results: {summary['project_commits']}; new compute {summary['total_new_compute_hours']:.2f} h\n",
         "## Main table (pooled OOF per seed; mean ± SD over 3 seeds; communication = conceptual tensor bytes, mean over 15 runs)\n", md_table(show),
         "\n## Per-seed pooled OOF\n", md_table(O.drop(columns=["cm_valthr"]).round(4)),
         "\n## Paired fold test-AUC differences vs C0 (descriptive)\n", "\n".join(f"- {k}: {v}" for k, v in pair.items()),
         "\n## INT8 relative L2 quantization error (mean over runs; per round mean/max, averaged up to the best round)\n", md_table(Q.round(5)),
         "\nFigures: `reports/figures/phase2d/frontier.png`, `reports/figures/phase2d/val_auc_vs_gb.png`\n"]
    (ROOT / "reports" / "phase2d_report.md").write_text("\n".join(L) + "\n")
    print(show.to_string(index=False))
    print(json.dumps(pair, indent=1))
    print(Q.round(5).to_string())


if __name__ == "__main__":
    main()
