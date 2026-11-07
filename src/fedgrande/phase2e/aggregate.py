"""Phase 2E aggregation: F0 (Phase 2B, reused) vs F1-F3; communication by round; figures; comparison with Phase 2D."""
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

CONFIGS = {"F0": ("phase2b", "fedavg_grande"), "F1": ("phase2e", "fedavg_grande_frzidx"),
           "F2": ("phase2e", "fedavg_grande_frzstruct"), "F3": ("phase2e", "fedavg_grande_frzstruct_int8")}
LABEL = {"F0": "F0 full", "F1": "F1 freeze features", "F2": "F2 freeze structure", "F3": "F3 freeze + INT8"}
PARTS, FOLDS, SEEDS = ("natural", "iid"), range(5), (42, 447, 2026)
OUT = ROOT / "results" / "phase2e" / "summary"
FIG = ROOT / "reports" / "figures" / "phase2e"
GB = 1e9


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    m = build_grande(502, 42, n_estimators=256, depth=5)
    st, fbe = trainable_state(m), {"features_by_estimator": m.features_by_estimator}
    runs, preds, rounds = [], [], []
    for c, (phase, method) in CONFIGS.items():
        for part in PARTS:
            for f in FOLDS:
                for s in SEEDS:
                    d = ROOT / "results" / phase / "runs" / f"{method}__{part}__f{f}__s{s}"
                    r = json.load(open(d / "result.json"))
                    rd = pd.read_csv(d / "rounds.csv")
                    if c == "F0":
                        a = comm.accounting_v2(st, fbe, 21, r["steps_executed"], r["best_step"], "fp32")
                        once, pr = a["bytes_once_at_init_buffers"], a["total_bytes_per_round"]
                        per_round = np.full(r["steps_executed"], pr)
                        down = np.full(r["steps_executed"], a["downlink_bytes_per_round"])
                    else:
                        v3 = r["communication"]["v3"]
                        once = v3["bytes_once_at_init_buffers"]
                        per_round = (rd.bytes_down + rd.bytes_up).to_numpy()
                        down = rd.bytes_down.to_numpy()
                        assert int(once + per_round[: r["best_step"]].sum()) == v3["cum_total_to_best"], "accounting inconsistent"
                    cum = once + np.cumsum(per_round)
                    b, e = r["best_step"], r["steps_executed"]
                    rec = {"config": c, "partition": part, "fold": f, "seed": s, "best_round": b, "rounds": e, "seconds": r["seconds_total"],
                           "sec_per_round": r["seconds_total"] / e, "peak_rss_mb": r["peak_rss_mb"], "test_auc_fold": r["test_at_0.5"]["roc_auc"],
                           "bytes_round_pre": int(per_round[0]), "bytes_round_post": int(per_round[-1]) if e > 21 else np.nan,
                           "cum_to_r20": int(cum[min(20, e) - 1]), "cum_to_best": int(cum[b - 1]), "cum_to_final": int(cum[e - 1]),
                           "cum_down_to_best": int(once + down[:b].sum()), "cum_up_to_best": int((per_round - down)[:b].sum()),
                           "best_after_freeze": b > 20, "project_commit": r["meta"]["project_commit"]}
                    if c != "F0" and e > 20:
                        post = rd[rd["round"] > 20]
                        for k in ("split_values", "leaf_classes_array", "estimator_weights", "split_index_array"):
                            rec[f"move_rel_post_mean__{k}"] = float(post[f"move_rel__{k}"].mean())
                        qc = [k for k in rd.columns if k.startswith("qerr_rel_l2_max__")]
                        for k in qc:
                            rec[f"{k}_post_max"] = float(post[k].max())
                    rec["cons_at_best"] = float(rd.loc[rd["round"] == b, "cons_consensus_mean"].iloc[0])
                    runs.append(rec)
                    p = pd.read_csv(d / "predictions.csv")
                    preds.append(p[p.role == "test"].assign(config=c, partition=part, fold=f, seed=s, pred_vt=lambda x: (x.p1 >= r["val_threshold"]).astype(int)))
                    rounds.append(pd.DataFrame({"round": np.arange(1, e + 1), "val_auc": rd.val_auc.to_numpy(), "cum_gb": cum / GB}).assign(config=c, partition=part))
    R = pd.DataFrame(runs)
    R.to_csv(OUT / "runs_table.csv", index=False)
    P = pd.concat(preds)
    oof = []
    for (c, part, s), g in P.groupby(["config", "partition", "seed"]):
        assert g.fold.nunique() == 5 and g.row_index.is_unique
        a, b = binary_metrics(g.y, g.p1), binary_metrics(g.y, g.p1, pred=g.pred_vt)
        oof.append({"config": c, "partition": part, "seed": s, "oof_auc": a["roc_auc"], "oof_f1_05": a["macro_f1"], "oof_f1_valthr": b["macro_f1"],
                    "oof_acc_05": a["accuracy"], "cm_valthr": str(b["confusion_[[tn,fp],[fn,tp]]"])})
    O = pd.DataFrame(oof)
    base = O[O.config == "F0"].set_index(["partition", "seed"]).oof_auc
    O["d_auc"] = [r.oof_auc - base[(r.partition, r.seed)] for r in O.itertuples()]
    O["retention"] = [100 * r.oof_auc / base[(r.partition, r.seed)] for r in O.itertuples()]
    O.to_csv(OUT / "oof_by_seed.csv", index=False)
    so = O.groupby(["config", "partition"]).agg(auc=("oof_auc", "mean"), auc_sd=("oof_auc", "std"), d_auc=("d_auc", "mean"), d_auc_sd=("d_auc", "std"),
                                               retention=("retention", "mean"), f1_05=("oof_f1_05", "mean"), f1_vt=("oof_f1_valthr", "mean"),
                                               f1_vt_sd=("oof_f1_valthr", "std"), acc=("oof_acc_05", "mean")).reset_index()
    ag = R.groupby(["config", "partition"]).agg(best_round=("best_round", "mean"), best_round_sd=("best_round", "std"), rounds=("rounds", "mean"),
                                               best_after_freeze=("best_after_freeze", "mean"), mb_round_pre=("bytes_round_pre", lambda x: x.mean() / 1e6),
                                               mb_round_post=("bytes_round_post", lambda x: x.mean() / 1e6), gb_to_r20=("cum_to_r20", lambda x: x.mean() / GB),
                                               gb_to_best=("cum_to_best", lambda x: x.mean() / GB), gb_to_final=("cum_to_final", lambda x: x.mean() / GB),
                                               gb_up_to_best=("cum_up_to_best", lambda x: x.mean() / GB), gb_down_to_best=("cum_down_to_best", lambda x: x.mean() / GB),
                                               seconds=("seconds", "mean"), sec_per_round=("sec_per_round", "mean"), peak_rss_mb=("peak_rss_mb", "max"),
                                               cons_at_best=("cons_at_best", "mean")).reset_index()
    T = so.merge(ag, on=["config", "partition"])
    f0 = T[T.config == "F0"].set_index("partition")
    T["post_freeze_round_reduction_pct"] = [100 * (1 - r.mb_round_post / f0.loc[r.partition, "mb_round_pre"]) if r.config != "F0" else 0.0 for r in T.itertuples()]
    T["compression_to_best_x"] = [f0.loc[r.partition, "gb_to_best"] / r.gb_to_best for r in T.itertuples()]
    T["compression_to_final_x"] = [f0.loc[r.partition, "gb_to_final"] / r.gb_to_final for r in T.itertuples()]
    T.to_csv(OUT / "main_table.csv", index=False)

    fw = R.pivot_table(index=["partition", "fold", "seed"], columns="config", values="test_auc_fold")
    pair = {f"{c}-F0": {p: {"mean": float((g[c] - g["F0"]).mean()), "min": float((g[c] - g["F0"]).min()), "max": float((g[c] - g["F0"]).max()),
                            "n_negative": int(((g[c] - g["F0"]) < 0).sum()), "n": int(len(g))} for p, g in fw.groupby(level="partition")} for c in ("F1", "F2", "F3")}
    pair["F2-F1"] = {p: {"mean": float((g["F2"] - g["F1"]).mean()), "n_negative": int(((g["F2"] - g["F1"]) < 0).sum()), "n": int(len(g))} for p, g in fw.groupby(level="partition")}
    pair["F3-F2"] = {p: {"mean": float((g["F3"] - g["F2"]).mean()), "n_negative": int(((g["F3"] - g["F2"]) < 0).sum()), "n": int(len(g))} for p, g in fw.groupby(level="partition")}
    post_dyn = R[R.config != "F0"].groupby(["config", "partition"])[[c for c in R.columns if c.startswith(("move_rel_post", "qerr"))]].mean()
    post_dyn.to_csv(OUT / "post_freeze_dynamics.csv")

    # Phase 2D comparison (existing results; not rerun)
    d2 = pd.read_csv(ROOT / "results" / "phase2d" / "summary" / "main_table.csv")
    d2 = d2[d2.config.isin(["C2", "C3"])][["config", "partition", "auc", "auc_sd", "d_auc", "gb_to_best", "total_compression_to_best_x", "seconds"]]
    comp = pd.concat([T[["config", "partition", "auc", "auc_sd", "d_auc", "gb_to_best", "compression_to_best_x", "seconds"]].rename(columns={"compression_to_best_x": "compression_x"}),
                      d2.rename(columns={"total_compression_to_best_x": "compression_x"}).assign(config=lambda x: "2D-" + x.config)])
    comp.to_csv(OUT / "comparison_with_phase2d.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    col = {"natural": "#2a6fdb", "iid": "#e07b39"}
    mk = {"F0": "o", "F1": "s", "F2": "^", "F3": "D", "2D-C2": "v", "2D-C3": "P"}
    for r in comp.itertuples():
        ax.errorbar(r.gb_to_best, r.auc, yerr=r.auc_sd, marker=mk[r.config], color=col[r.partition], capsize=3, linestyle="none",
                    alpha=1.0 if not r.config.startswith("2D") else 0.45, label=f"{LABEL.get(r.config, r.config)} ({r.partition})")
    ax.set_xscale("log")
    ax.set_xlabel("cumulative tensor GB to best validation round (log scale)")
    ax.set_ylabel("pooled OOF ROC-AUC (mean ± SD, 3 seeds)")
    ax.set_title("Structure-aware FedAvg-GRANDE (faded: Phase 2D C2/C3)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "frontier.png", dpi=150)
    plt.close(fig)

    stab = json.load(open(ROOT / "results" / "phase2e" / "stability_summary.json"))
    S = pd.DataFrame(stab["summary"]).set_index("partition")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for p in PARTS:
        xs = ["40", "60", "best"]
        axes[0].errorbar(xs, [S.loc[p, f"feature_stability_{x}_mean"] for x in ("40", "60", "best")],
                         yerr=[S.loc[p, f"feature_stability_{x}_std"] for x in ("40", "60", "best")], marker="o", color=col[p], capsize=3, label=p)
        axes[1].errorbar(xs, [S.loc[p, f"thr_drift_numeric_sd_mean_{x}_mean"] for x in ("40", "60", "best")],
                         yerr=[S.loc[p, f"thr_drift_numeric_sd_mean_{x}_std"] for x in ("40", "60", "best")], marker="o", color=col[p], capsize=3, label=p)
        for t, ls in (("leaf_classes_array", "-"), ("estimator_weights", "--"), ("split_values", ":")):
            axes[2].plot(xs, [S.loc[p, f"rel_drift_{t}_{x}_mean"] for x in ("40", "60", "best")], marker="o", color=col[p], linestyle=ls, label=f"{t} ({p})")
    axes[0].set_title("node feature identity unchanged since round 20", fontsize=9)
    axes[1].set_title("threshold drift since round 20 (numeric nodes, SD units)", fontsize=9)
    axes[2].set_title("relative norm change since round 20", fontsize=9)
    for ax in axes:
        ax.set_xlabel("global model at round")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=6)
    fig.suptitle("DIAGNOSTIC (ordinary FedAvg F0, seed 42, 5 folds): what still changes after round 20?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "post_round20_drift.png", dpi=150)
    plt.close(fig)

    summary = {"main_table": T.round(6).to_dict("records"), "paired_fold_differences": pair, "post_freeze_dynamics": post_dyn.round(6).reset_index().to_dict("records"),
               "comparison_with_phase2d": comp.round(6).to_dict("records"), "stability": stab, "gate0": json.load(open(ROOT / "results" / "phase2e" / "gate0.json")),
               "project_commits": sorted(R.project_commit.unique()), "new_compute_hours": float(R[R.config != "F0"].seconds.sum() / 3600)}
    write_json(OUT / "phase2e_summary.json", summary)
    show = T[["config", "partition", "auc", "auc_sd", "d_auc", "d_auc_sd", "retention", "f1_05", "f1_vt", "best_round", "best_after_freeze", "mb_round_pre",
              "mb_round_post", "post_freeze_round_reduction_pct", "gb_to_r20", "gb_to_best", "gb_to_final", "compression_to_best_x", "compression_to_final_x",
              "seconds", "sec_per_round", "peak_rss_mb", "cons_at_best"]].round(4)
    L = ["# Phase 2E report (generated)\n", "Generated by `src/fedgrande/phase2e/aggregate.py`; protocol `reports/PHASE2E_PROTOCOL.md`; F0 = Phase 2B FedAvg-GRANDE (reused).\n",
         f"- commits: {summary['project_commits']}; new compute {summary['new_compute_hours']:.2f} h\n", "## Gate 0\n",
         md_table(pd.DataFrame(summary["gate0"]["tensors"]).round(3)), f"\npost-freeze per round: {summary['gate0']['post_freeze_per_round']}\n",
         "## Main table\n", md_table(show), "\n## Per-seed pooled OOF\n", md_table(O.round(4)),
         "\n## Paired fold differences\n", "\n".join(f"- {k}: {v}" for k, v in pair.items()),
         "\n## Post-freeze dynamics (mean per-round relative movement of the global tensors after round 20; INT8 error)\n", md_table(post_dyn.round(6).reset_index()),
         "\n## DIAGNOSTIC: what changes after round 20 in F0\n", md_table(S.round(4).reset_index()),
         "\n## Comparison with Phase 2D (existing results)\n", md_table(comp.round(4)),
         "\nFigures: `reports/figures/phase2e/frontier.png`, `reports/figures/phase2e/post_round20_drift.png`\n"]
    (ROOT / "reports" / "phase2e_report.md").write_text("\n".join(L) + "\n")
    print(show.to_string(index=False))
    print(json.dumps(pair, indent=1))
    print(post_dyn.round(5).T.to_string())
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
