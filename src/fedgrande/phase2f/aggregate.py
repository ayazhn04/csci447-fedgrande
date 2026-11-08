"""Phase 2F aggregation: F0, F2 (reused), G1, G2 (+G3 if run); freeze dynamics, communication curves, frontier, report."""
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

CONFIGS = {"F0": ("phase2b", "fedavg_grande"), "F2": ("phase2e", "fedavg_grande_frzstruct"),
           "G1": ("phase2f", "fedavg_grande_cgsf_global"), "G2": ("phase2f", "fedavg_grande_cgsf_node"),
           "G3": ("phase2f", "fedavg_grande_cgsf_node_int8")}
LABEL = {"F0": "F0 full FedAvg", "F2": "F2 fixed round-20 freeze", "G1": "G1 adaptive global", "G2": "G2 progressive node-wise",
         "G3": "G3 progressive + INT8"}
PARTS, FOLDS, SEEDS = ("natural", "iid"), range(5), (42, 447, 2026)
CHECK = (10, 20, 30, 40, 60)
OUT = ROOT / "results" / "phase2f" / "summary"
FIG = ROOT / "reports" / "figures" / "phase2f"
GB = 1e9
COL = {"natural": "#2a6fdb", "iid": "#e07b39"}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    m = build_grande(502, 42, n_estimators=256, depth=5)
    st, fbe = trainable_state(m), {"features_by_estimator": m.features_by_estimator}
    have = {c for c, (ph, me) in CONFIGS.items() if (ROOT / "results" / ph / "runs" / f"{me}__natural__f0__s42" / "result.json").exists()}
    runs, preds, curves, frz = [], [], [], []
    for c in [k for k in CONFIGS if k in have]:
        phase, method = CONFIGS[c]
        for part in PARTS:
            for f in FOLDS:
                for s in SEEDS:
                    d = ROOT / "results" / phase / "runs" / f"{method}__{part}__f{f}__s{s}"
                    r = json.load(open(d / "result.json"))
                    rd = pd.read_csv(d / "rounds.csv")
                    e, b = r["steps_executed"], r["best_step"]
                    if c == "F0":
                        a = comm.accounting_v2(st, fbe, 21, e, b, "fp32")
                        once, per = a["bytes_once_at_init_buffers"], np.full(e, a["total_bytes_per_round"], dtype=float)
                    else:
                        v3 = r["communication"]["v3"]
                        once, per = v3["bytes_once_at_init_buffers"], (rd.bytes_down + rd.bytes_up).to_numpy(float)
                        assert int(once + per[:b].sum()) == v3["cum_total_to_best"], "accounting inconsistent"
                    cum = once + np.cumsum(per)
                    rec = {"config": c, "partition": part, "fold": f, "seed": s, "best_round": b, "rounds": e, "seconds": r["seconds_total"],
                           "peak_rss_mb": r["peak_rss_mb"], "test_auc_fold": r["test_at_0.5"]["roc_auc"], "gb_to_best": cum[b - 1] / GB,
                           "gb_to_final": cum[e - 1] / GB, "mb_round_first": per[0] / 1e6, "mb_round_last": per[-1] / 1e6,
                           "trigger_round": r.get("info", {}).get("trigger_round"), "project_commit": r["meta"]["project_commit"]}
                    if c in ("G1", "G2", "G3"):
                        z = np.load(d / "freeze_rounds.npz")
                        fr, tr = z["feat"].ravel().astype(int), z["full"].ravel().astype(int)
                        for k in CHECK + ("best", "final"):
                            rr = b if k == "best" else e if k == "final" else k
                            rec[f"feat_frozen_frac_{k}"] = float(((fr != -1) & (fr <= rr)).mean())
                            rec[f"full_frozen_frac_{k}"] = float(((tr != -1) & (tr <= rr)).mean())
                        ff, tt = fr[fr != -1], tr[tr != -1]
                        both = (fr != -1) & (tr != -1)
                        rec.update({"feat_freeze_median": float(np.median(ff)) if len(ff) else np.nan,
                                    "feat_freeze_q25": float(np.percentile(ff, 25)) if len(ff) else np.nan,
                                    "feat_freeze_q75": float(np.percentile(ff, 75)) if len(ff) else np.nan,
                                    "full_freeze_median": float(np.median(tt)) if len(tt) else np.nan,
                                    "full_freeze_q25": float(np.percentile(tt, 25)) if len(tt) else np.nan,
                                    "full_freeze_q75": float(np.percentile(tt, 75)) if len(tt) else np.nan,
                                    "never_feat_frozen": float((fr == -1).mean()), "never_full_frozen": float((tr == -1).mean()),
                                    "threshold_lag_median": float(np.median(tr[both] - fr[both])) if both.any() else np.nan})
                        frz.append(pd.DataFrame({"config": c, "partition": part, "feat": fr, "full": tr}).sample(frac=0.2, random_state=0))
                    runs.append(rec)
                    p = pd.read_csv(d / "predictions.csv")
                    preds.append(p[p.role == "test"].assign(config=c, partition=part, fold=f, seed=s, pred_vt=lambda x: (x.p1 >= r["val_threshold"]).astype(int)))
                    cv = pd.DataFrame({"round": np.arange(1, e + 1), "mb": per / 1e6}).assign(config=c, partition=part)
                    if "frac_feat_frozen" in rd:
                        cv["frac_feat"], cv["frac_full"] = rd.frac_feat_frozen.to_numpy(), rd.frac_full_frozen.to_numpy()
                    curves.append(cv)
    R = pd.DataFrame(runs)
    R.to_csv(OUT / "runs_table.csv", index=False)
    P = pd.concat(preds)
    oof = []
    for (c, part, s), g in P.groupby(["config", "partition", "seed"]):
        assert g.fold.nunique() == 5 and g.row_index.is_unique
        a, bb = binary_metrics(g.y, g.p1), binary_metrics(g.y, g.p1, pred=g.pred_vt)
        oof.append({"config": c, "partition": part, "seed": s, "oof_auc": a["roc_auc"], "oof_f1_05": a["macro_f1"], "oof_f1_valthr": bb["macro_f1"],
                    "oof_acc": a["accuracy"], "cm_valthr": str(bb["confusion_[[tn,fp],[fn,tp]]"])})
    O = pd.DataFrame(oof)
    base = O[O.config == "F0"].set_index(["partition", "seed"]).oof_auc
    O["d_auc"] = [x.oof_auc - base[(x.partition, x.seed)] for x in O.itertuples()]
    O["retention"] = [100 * x.oof_auc / base[(x.partition, x.seed)] for x in O.itertuples()]
    O.to_csv(OUT / "oof_by_seed.csv", index=False)
    pf = R.groupby(["config", "partition", "seed"]).test_auc_fold.mean().groupby(["config", "partition"]).agg(["mean", "std"]).rename(columns={"mean": "fold_auc", "std": "fold_auc_sd"})
    so = O.groupby(["config", "partition"]).agg(auc=("oof_auc", "mean"), auc_sd=("oof_auc", "std"), d_auc=("d_auc", "mean"), d_auc_sd=("d_auc", "std"),
                                               retention=("retention", "mean"), f1_05=("oof_f1_05", "mean"), f1_vt=("oof_f1_valthr", "mean"), acc=("oof_acc", "mean"))
    agcols = {"best_round": ("best_round", "mean"), "rounds": ("rounds", "mean"), "gb_to_best": ("gb_to_best", "mean"), "gb_to_best_sd": ("gb_to_best", "std"),
              "gb_to_final": ("gb_to_final", "mean"), "mb_round_last": ("mb_round_last", "mean"), "seconds": ("seconds", "median"),
              "peak_rss_mb": ("peak_rss_mb", "max")}
    extra = [c for c in R.columns if c.startswith(("feat_frozen_frac", "full_frozen_frac", "feat_freeze_", "full_freeze_", "never_", "threshold_lag"))] + ["trigger_round"]
    for c in extra:
        agcols[c] = (c, "mean")
    ag = R.groupby(["config", "partition"]).agg(**agcols)
    T = so.join(pf).join(ag).reset_index()
    f0 = T[T.config == "F0"].set_index("partition")
    T["compression_to_best_x"] = [f0.loc[x.partition, "gb_to_best"] / x.gb_to_best for x in T.itertuples()]
    T.to_csv(OUT / "main_table.csv", index=False)

    fw = R.pivot_table(index=["partition", "fold", "seed"], columns="config", values="test_auc_fold")
    pair = {}
    for a_, b_ in (("G1", "F0"), ("G2", "F0"), ("G2", "F2"), ("G1", "F2"), ("G2", "G1"), ("G3", "G2")):
        if a_ in fw and b_ in fw:
            pair[f"{a_}-{b_}"] = {p: {"mean": float((g[a_] - g[b_]).mean()), "n_negative": int(((g[a_] - g[b_]) < 0).sum()), "n": int(len(g))}
                                  for p, g in fw.groupby(level="partition")}
    gbw = R.pivot_table(index=["partition", "fold", "seed"], columns="config", values="gb_to_best")
    pair_gb = {f"{a_}/{b_}": {p: float((g[a_] / g[b_]).mean()) for p, g in gbw.groupby(level="partition")}
               for a_, b_ in (("G2", "F2"), ("G1", "F2"), ("G2", "G1")) if a_ in gbw and b_ in gbw}
    g1t = R[R.config == "G1"].trigger_round
    trig = {"per_run": R[R.config == "G1"][["partition", "fold", "seed", "trigger_round"]].to_dict("records"),
            "mean": float(g1t.dropna().mean()) if g1t.notna().any() else None, "sd": float(g1t.dropna().std()) if g1t.notna().sum() > 1 else None,
            "never_triggered": int(g1t.isna().sum()), "n": int(len(g1t))} if "G1" in have else None
    g2 = T[T.config == "G2"].set_index("partition")
    f2 = T[T.config == "F2"].set_index("partition")
    g3_elig = {"G2_abs_auc_loss_le_0.005": bool((g2.d_auc >= -0.005).all()),
               "G2_gb_to_best_lt_F2": bool((g2.gb_to_best < f2.gb_to_best).all()),
               "values": {p: {"G2_d_auc": float(g2.loc[p, "d_auc"]), "G2_gb": float(g2.loc[p, "gb_to_best"]), "F2_gb": float(f2.loc[p, "gb_to_best"])} for p in PARTS}} if "G2" in have else None
    if g3_elig:
        g3_elig["eligible"] = g3_elig["G2_abs_auc_loss_le_0.005"] and g3_elig["G2_gb_to_best_lt_F2"]
        g3_elig["rule"] = "G2 pooled-AUC loss vs F0 <= 0.005 (both partitions, mean over seeds) AND G2 mean GB to best < F2 (both partitions)"

    # frontier with earlier phases (existing results)
    d2 = pd.read_csv(ROOT / "results" / "phase2d" / "summary" / "main_table.csv")
    d2 = d2[d2.config.isin(["C2", "C3"])][["config", "partition", "auc", "auc_sd", "gb_to_best"]].assign(config=lambda x: "2D-" + x.config)
    e2 = pd.read_csv(ROOT / "results" / "phase2e" / "summary" / "main_table.csv")
    e2 = e2[e2.config.isin(["F3"])][["config", "partition", "auc", "auc_sd", "gb_to_best"]].assign(config="2E-F3")
    front = pd.concat([T[["config", "partition", "auc", "auc_sd", "gb_to_best"]], d2, e2])
    front.to_csv(OUT / "frontier.csv", index=False)
    mk = {"F0": "o", "F2": "^", "G1": "s", "G2": "*", "G3": "X", "2D-C2": "v", "2D-C3": "P", "2E-F3": "D"}
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for x in front.itertuples():
        ax.errorbar(x.gb_to_best, x.auc, yerr=x.auc_sd, marker=mk[x.config], color=COL[x.partition], capsize=3, linestyle="none",
                    markersize=10 if x.config in ("G2", "G3") else 6, alpha=0.45 if x.config.startswith(("2D", "2E")) else 1,
                    label=f"{LABEL.get(x.config, x.config)} ({x.partition})")
    ax.set_xscale("log")
    ax.set_xlabel("cumulative tensor GB to best validation round (log scale)")
    ax.set_ylabel("pooled OOF ROC-AUC (mean ± SD, 3 seeds)")
    ax.set_title("Communication–performance frontier (faded: Phase 2D/2E)", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "B_frontier.png", dpi=150)
    plt.close(fig)

    C = pd.concat(curves)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for part, ls in (("natural", "-"), ("iid", "--")):
        for c, colr in (("F0", "k"), ("F2", "#7a3db8"), ("G1", "#2ca02c"), ("G2", "#d62728"), ("G3", "#8c564b")):
            g = C[(C.config == c) & (C.partition == part)].groupby("round").mb.mean()
            if len(g):
                axes[0].plot(g.index, g.values, color=colr, linestyle=ls, label=f"{c} {part}")
        g = C[(C.config == "G2") & (C.partition == part)].groupby("round")[["frac_feat", "frac_full"]].mean()
        if len(g):
            axes[1].plot(g.index, g.frac_feat, color=COL[part], linestyle="-", label=f"feature-frozen ({part})")
            axes[1].plot(g.index, g.frac_full, color=COL[part], linestyle=":", label=f"fully frozen ({part})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("communication round")
    axes[0].set_ylabel("MB per round (21 clients, down + up)")
    axes[0].set_title("Communication per round (mean over runs still active)", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("communication round")
    axes[1].set_ylabel("fraction of internal nodes (at round start)")
    axes[1].set_title("G2: progressive structural freezing", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "A_progressive_freezing.png", dpi=150)
    plt.close(fig)

    if frz:
        Fz = pd.concat(frz)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, part in zip(axes, PARTS):
            g = Fz[(Fz.config == "G2") & (Fz.partition == part)]
            bins = np.arange(10, 102, 2)
            ax.hist(g.feat[g.feat != -1], bins=bins, alpha=0.6, label="feature-logit freeze", color="#2a6fdb")
            ax.hist(g.full[g.full != -1], bins=bins, alpha=0.6, label="full (threshold) freeze", color="#e07b39")
            ax.set_title(f"G2 freeze timing ({part}; 20% node sample per run)", fontsize=10)
            ax.set_xlabel("freeze decision round")
        axes[0].set_ylabel("nodes")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "C_freeze_timing.png", dpi=150)
        plt.close(fig)

    summary = {"available": sorted(have), "main_table": T.round(6).to_dict("records"), "paired_fold_auc": pair, "paired_gb_ratio": pair_gb,
               "G1_trigger": trig, "G3_eligibility": g3_elig, "frontier": front.round(6).to_dict("records"),
               "project_commits": sorted(R.project_commit.unique()), "new_compute_hours": float(R[R.config.isin(["G1", "G2", "G3"])].seconds.sum() / 3600)}
    write_json(OUT / "phase2f_summary.json", summary)
    show_cols = ["config", "partition", "auc", "auc_sd", "d_auc", "d_auc_sd", "retention", "fold_auc", "f1_05", "f1_vt", "best_round", "gb_to_best",
                 "compression_to_best_x", "mb_round_last", "seconds", "peak_rss_mb", "trigger_round", "feat_freeze_median", "full_freeze_median",
                 "threshold_lag_median", "never_feat_frozen", "never_full_frozen"] + [f"feat_frozen_frac_{k}" for k in CHECK + ("best", "final")] + \
                [f"full_frozen_frac_{k}" for k in CHECK + ("best", "final")]
    show = T[[c for c in show_cols if c in T.columns]].round(4)
    L = ["# Phase 2F report (generated)\n", "Generated by `src/fedgrande/phase2f/aggregate.py`; protocol `reports/PHASE2F_PROTOCOL.md`. F0 (Phase 2B) and F2 (Phase 2E) reused.\n",
         f"- commits: {summary['project_commits']}; new compute {summary['new_compute_hours']:.2f} h\n",
         "## Main table\n", md_table(show), "\n## Per-seed pooled OOF\n", md_table(O.round(4)),
         "\n## Paired fold test-AUC differences\n", "\n".join(f"- {k}: {v}" for k, v in pair.items()),
         f"\n\nPaired GB-to-best ratios: {pair_gb}\n", f"\n## G1 trigger rounds\n\n{trig}\n", f"\n## G3 eligibility\n\n{g3_elig}\n",
         "\n## Frontier (with Phase 2D/2E)\n", md_table(front.round(4)),
         "\nFigures: `reports/figures/phase2f/A_progressive_freezing.png`, `B_frontier.png`, `C_freeze_timing.png`\n"]
    (ROOT / "reports" / "phase2f_report.md").write_text("\n".join(L) + "\n")
    print(show.T.to_string())
    print(json.dumps({"pair": pair, "gb": pair_gb, "trig": {k: v for k, v in (trig or {}).items() if k != "per_run"}, "g3": g3_elig}, indent=1))


if __name__ == "__main__":
    main()
