"""Aggregate Phase 2B runs into machine-readable summaries (then plots/report are generated from these).

Outputs in results/phase2b/summary/: runs_table.csv, oof_by_seed.csv, oof_summary.csv, gaps.json, per_client_*.csv,
consensus_*.csv, structure_vs_central.csv, communication.csv, runtime.csv, summary.json
"""
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from fedgrande.fl.metrics import binary_metrics  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402

RUNS = ROOT / "results" / "phase2b" / "runs"
OUT = ROOT / "results" / "phase2b" / "summary"


def msd(s):
    s = pd.Series(s, dtype=float)
    return {"mean": float(s.mean()), "sd": float(s.std(ddof=1)) if len(s) > 1 else float("nan"), "n_seeds": int(s.notna().sum())}


def load_runs(stage_parts):
    rows, preds = [], []
    for f in sorted(glob(str(RUNS / "*" / "result.json"))):
        r = json.load(open(f))
        if r["partition"] not in stage_parts:
            continue
        d = os.path.dirname(f)
        p = pd.read_csv(os.path.join(d, "predictions.csv"))
        p["pred_valthr"] = (p.p1 >= r["val_threshold"]).astype(int)
        for k in ("method", "partition", "fold", "seed"):
            p[k] = r[k]
        preds.append(p)
        c = r.get("communication") or {}
        rows.append({"job_id": r["job_id"], "method": r["method"], "partition": r["partition"], "fold": r["fold"], "seed": r["seed"],
                     "test_auc": r["test_at_0.5"]["roc_auc"], "test_f1_05": r["test_at_0.5"]["macro_f1"],
                     "test_f1_valthr": r["test_at_val_threshold"]["macro_f1"], "test_acc_05": r["test_at_0.5"]["accuracy"],
                     "val_auc_best": r["best_val_auc"], "val_threshold": r["val_threshold"], "best_step": r["best_step"],
                     "steps_executed": r["steps_executed"], "seconds": r["seconds_total"], "peak_rss_mb": r["peak_rss_mb"],
                     "trainable_parameters": r["trainable_parameters"], "bytes_per_round": c.get("bytes_per_round_total"),
                     "bytes_to_best": c.get("cumulative_bytes_to_best_round"), "bytes_to_final": c.get("cumulative_bytes_to_final_round"),
                     "project_commit": r["meta"]["project_commit"], "config_sha256": r["config_sha256"],
                     "local_consensus_mean": (r.get("local_model_consensus") or {}).get("consensus_mean")})
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def views(runs, preds, parts):
    """Expand centralized runs (partition 'none') so they appear in each partition's comparison."""
    out_r, out_p = [], []
    for part in parts:
        for df, out in ((runs, out_r), (preds, out_p)):
            sub = df[df.partition.isin(["none", part])].copy()
            sub["view"] = part
            out.append(sub)
    return pd.concat(out_r, ignore_index=True), pd.concat(out_p, ignore_index=True)


def main(parts=("natural", "iid"), tag="core"):
    os.makedirs(OUT, exist_ok=True)
    runs, preds = load_runs(set(parts) | {"none"})
    runs.to_csv(OUT / f"runs_table_{tag}.csv", index=False)
    commits = sorted(runs.project_commit.unique())
    test = preds[preds.role == "test"]

    # pooled OOF per (method, partition, seed)
    oof = []
    for (m, part, s), g in test.groupby(["method", "partition", "seed"]):
        assert g.fold.nunique() == 5 and g.row_index.is_unique, (m, part, s)
        a, b = binary_metrics(g.y, g.p1, 0.5), binary_metrics(g.y, g.p1, pred=g.pred_valthr)
        oof.append({"method": m, "partition": part, "seed": s, "n": a["n"], "oof_auc": a["roc_auc"], "oof_f1_05": a["macro_f1"],
                    "oof_f1_valthr": b["macro_f1"], "oof_acc_05": a["accuracy"], "cm_05": a["confusion_[[tn,fp],[fn,tp]]"],
                    "cm_valthr": b["confusion_[[tn,fp],[fn,tp]]"]})
    oof = pd.DataFrame(oof)
    oof.to_csv(OUT / f"oof_by_seed_{tag}.csv", index=False)
    summ = []
    for (m, part), g in oof.groupby(["method", "partition"]):
        summ.append({"method": m, "partition": part, **{f"{k}_{s}": v for k in ("oof_auc", "oof_f1_05", "oof_f1_valthr", "oof_acc_05")
                                                          for s, v in msd(g[k]).items() if s != "n_seeds"}, "n_seeds": len(g)})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / f"oof_summary_{tag}.csv", index=False)

    # gaps (per seed, pooled OOF), plus descriptive paired per-fold differences
    def seedvals(m, part, col="oof_auc"):
        return oof[(oof.method == m) & (oof.partition == part)].set_index("seed")[col]

    gaps = {}
    for part in parts:
        for col in ("oof_auc", "oof_f1_valthr"):
            for model in ("grande", "mlp"):
                c, f = seedvals(f"central_{model}", "none", col), seedvals(f"fedavg_{model}", part, col)
                gaps[f"central_minus_fedavg_{model}_{part}_{col}"] = msd(c - f)
            c, l = seedvals("central_grande", "none", col), seedvals("local_grande", part, col)
            gaps[f"central_minus_local_grande_{part}_{col}"] = msd(c - l)
            f, l = seedvals("fedavg_grande", part, col), seedvals("local_grande", part, col)
            gaps[f"fedavg_minus_local_grande_{part}_{col}"] = msd(f - l)
            f, mm = seedvals("fedavg_grande", part, col), seedvals("fedavg_mlp", part, col)
            gaps[f"fedavg_grande_minus_fedavg_mlp_{part}_{col}"] = msd(f - mm)
    if {"natural", "iid"} <= set(parts):
        for model in ("grande", "mlp"):
            gaps[f"fedavg_{model}_iid_minus_natural_oof_auc"] = msd(seedvals(f"fedavg_{model}", "iid") - seedvals(f"fedavg_{model}", "natural"))
        gaps["local_grande_iid_minus_natural_oof_auc"] = msd(seedvals("local_grande", "iid") - seedvals("local_grande", "natural"))
    if {"natural", "dirichlet0.1"} <= set(parts):
        for m_ in ("fedavg_grande", "fedavg_mlp", "local_grande"):
            gaps[f"{m_}_dirichlet0.1_minus_natural_oof_auc"] = msd(seedvals(m_, "dirichlet0.1") - seedvals(m_, "natural"))
            gaps[f"{m_}_dirichlet0.1_minus_iid_oof_auc"] = msd(seedvals(m_, "dirichlet0.1") - seedvals(m_, "iid"))
    fold_pairs = []
    rt = runs.set_index(["method", "partition", "fold", "seed"]).test_auc
    for part in parts:
        for fold in range(5):
            for s in sorted(runs.seed.unique()):
                try:
                    fold_pairs.append({"partition": part, "fold": fold, "seed": s,
                                       "gap_grande": rt[("central_grande", "none", fold, s)] - rt[("fedavg_grande", part, fold, s)],
                                       "gap_mlp": rt[("central_mlp", "none", fold, s)] - rt[("fedavg_mlp", part, fold, s)]})
                except KeyError:
                    pass
    fold_pairs = pd.DataFrame(fold_pairs)
    fold_pairs.to_csv(OUT / f"fold_gaps_{tag}.csv", index=False)
    write_json(OUT / f"gaps_{tag}.json", gaps)

    # per-client metrics from pooled OOF per seed (natural waves for every model; own clients for partitioned models)
    pc_rows = []
    for (m, part, s), g in test.groupby(["method", "partition", "seed"]):
        for wv, gw in g.groupby("wave"):
            a, b = binary_metrics(gw.y, gw.p1, 0.5), binary_metrics(gw.y, gw.p1, pred=gw.pred_valthr)
            pc_rows.append({"method": m, "partition": part, "seed": s, "wave": wv, "n": a["n"], "pos": a["pos"], "auc": a["roc_auc"],
                            "f1_05": a["macro_f1"], "f1_valthr": b["macro_f1"]})
    pc = pd.DataFrame(pc_rows)
    pc.to_csv(OUT / f"per_wave_by_seed_{tag}.csv", index=False)
    pcs = []
    for (m, part, s), g in pc.groupby(["method", "partition", "seed"]):
        pcs.append({"method": m, "partition": part, "seed": s, "auc_mean": g.auc.mean(), "auc_median": g.auc.median(), "auc_sd": g.auc.std(),
                    "auc_min": g.auc.min(), "auc_min_wave": int(g.loc[g.auc.idxmin(), "wave"]), "f1_valthr_mean": g.f1_valthr.mean(),
                    "f1_valthr_median": g.f1_valthr.median(), "f1_valthr_sd": g.f1_valthr.std(), "f1_valthr_min": g.f1_valthr.min(),
                    "f1_05_mean": g.f1_05.mean(), "f1_05_median": g.f1_05.median(), "f1_05_sd": g.f1_05.std(), "f1_05_min": g.f1_05.min()})
    pcs = pd.DataFrame(pcs)
    pcs.to_csv(OUT / f"per_wave_dispersion_by_seed_{tag}.csv", index=False)
    pcs_summary = pcs.drop(columns=["seed", "auc_min_wave"]).groupby(["method", "partition"]).agg(["mean", "std"])
    pcs_summary.columns = ["_".join(c) for c in pcs_summary.columns]
    pcs_summary.reset_index().to_csv(OUT / f"per_wave_dispersion_summary_{tag}.csv", index=False)
    pc.groupby(["method", "partition", "wave"]).agg(n=("n", "first"), pos=("pos", "first"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                                                    f1_valthr_mean=("f1_valthr", "mean")).reset_index().to_csv(OUT / f"per_wave_mean_over_seeds_{tag}.csv", index=False)

    # POST-HOC (added after observing the Dirichlet client-identity shortcut, see PHASE2B_PROTOCOL amendment A2):
    # within-client AUC on each partition's OWN clients, using only clients whose pooled-OOF test rows contain both
    # classes. Within a client the label prior is constant, so the client-identity shortcut cannot inflate it.
    pmap = pd.read_csv(ROOT / "data" / "derived" / "phase2b" / "partitions_v1.csv")
    cmap = {"natural": pmap.client_natural.to_numpy(), "iid": pmap.client_iid.to_numpy()}
    dfile = ROOT / "data" / "derived" / "phase2b" / "partitions_dirichlet_v1.csv"
    if dfile.exists():
        cmap["dirichlet0.1"] = pd.read_csv(dfile)["client_dirichlet0.1"].to_numpy()
    wc = []
    for view in parts:
        sub = test[test.partition.isin(["none", view])]
        for (m, part, s), g in sub.groupby(["method", "partition", "seed"]):
            cl = cmap[view][g.row_index.to_numpy()]
            aucs = [binary_metrics(gg.y, gg.p1)["roc_auc"] for _, gg in g.groupby(cl) if gg.y.nunique() == 2]
            wc.append({"view_partition": view, "method": m, "seed": s, "n_clients_both_classes": len(aucs),
                       "within_client_auc_mean": float(np.mean(aucs)), "within_client_auc_median": float(np.median(aucs)),
                       "within_client_auc_min": float(np.min(aucs))})
    wc = pd.DataFrame(wc)
    wc.to_csv(OUT / f"within_client_auc_by_seed_{tag}.csv", index=False)
    wcs = wc.groupby(["view_partition", "method"]).agg(n_clients=("n_clients_both_classes", "first"),
                                                       auc_mean=("within_client_auc_mean", "mean"), auc_mean_sd=("within_client_auc_mean", "std"),
                                                       auc_median=("within_client_auc_median", "mean"), auc_min=("within_client_auc_min", "mean")).reset_index()
    wcs.to_csv(OUT / f"within_client_auc_summary_{tag}.csv", index=False)

    # per-fold test metrics (protocol §10: descriptive per-fold view): mean over the 5 folds per seed, then over seeds.
    # Unlike pooled OOF, this is insensitive to fold-to-fold differences in probability calibration.
    pf = runs.groupby(["method", "partition", "seed"])[["test_auc", "test_f1_05", "test_f1_valthr"]].mean().reset_index()
    pfs = pf.groupby(["method", "partition"]).agg(fold_mean_auc=("test_auc", "mean"), fold_mean_auc_sd=("test_auc", "std"),
                                                  fold_mean_f1_valthr=("test_f1_valthr", "mean"), fold_mean_f1_05=("test_f1_05", "mean")).reset_index()
    pfs.to_csv(OUT / f"per_fold_summary_{tag}.csv", index=False)

    # structural consensus (FedAvg GRANDE rounds) and relation to the gap
    FIXED = (1, 5, 10, 20)
    fixed_rows = []
    cons_rows = []
    for _, r in runs[runs.method == "fedavg_grande"].iterrows():
        rd = pd.read_csv(RUNS / r.job_id / "rounds.csv")
        upto = rd[rd["round"] <= r.best_step]
        for fr in FIXED:
            row = rd[rd["round"] == fr]
            if len(row):
                fixed_rows.append({"partition": r.partition, "fold": r.fold, "seed": r.seed, "round": fr,
                                   **{k: float(row[k].iloc[0]) for k in ("cons_consensus_mean", "cons_frac_nodes_unanimous", "cons_global_agreement",
                                                                          "cons_thr_sd_numeric_nodes_mean", "cos_to_aggregate_mean", "update_norm_mean", "val_auc")}})
        cons_rows.append({"partition": r.partition, "fold": r.fold, "seed": r.seed, "best_round": r.best_step, "rounds": r.steps_executed,
                          **{f"{k}_mean_to_best": float(upto[k].mean()) for k in ("cons_consensus_mean", "cons_frac_nodes_unanimous",
                             "cons_global_agreement", "cons_thr_sd_numeric_nodes_mean", "cos_to_aggregate_mean", "update_norm_mean",
                             "update_disagreement_split_index")},
                          **{f"{k}_at_best": float(upto[k].iloc[-1]) for k in ("cons_consensus_mean", "cons_frac_nodes_unanimous",
                             "cons_global_agreement", "cons_thr_sd_numeric_nodes_mean", "cos_to_aggregate_mean")}})
    cons = pd.DataFrame(cons_rows)
    if len(cons):
        cons = cons.merge(fold_pairs[["partition", "fold", "seed", "gap_grande"]], on=["partition", "fold", "seed"], how="left")
        cons.to_csv(OUT / f"consensus_by_run_{tag}.csv", index=False)
        cons_summary = cons.drop(columns=["fold", "seed"]).groupby("partition").agg(["mean", "std"])
        cons_summary.columns = ["_".join(c) for c in cons_summary.columns]
        cons_summary.reset_index().to_csv(OUT / f"consensus_summary_{tag}.csv", index=False)
        fx = pd.DataFrame(fixed_rows).merge(fold_pairs[["partition", "fold", "seed", "gap_grande"]], on=["partition", "fold", "seed"], how="left")
        fx.to_csv(OUT / f"consensus_fixed_rounds_by_run_{tag}.csv", index=False)
        fxs = fx.drop(columns=["fold", "seed", "gap_grande"]).groupby(["partition", "round"]).agg(["mean", "std"])
        fxs.columns = ["_".join(c) for c in fxs.columns]
        fxs.reset_index().to_csv(OUT / f"consensus_fixed_rounds_summary_{tag}.csv", index=False)
        rel_fixed = {}
        for fr in FIXED:
            ok = fx[fx["round"] == fr][["cons_consensus_mean", "cons_global_agreement", "gap_grande", "partition"]].dropna()
            rel_fixed[f"round_{fr}"] = {k: {"all_runs": float(spearmanr(ok[k], ok.gap_grande).statistic),
                                            "within_partition": {p: float(spearmanr(g[k], g.gap_grande).statistic) for p, g in ok.groupby("partition")}}
                                        for k in ("cons_consensus_mean", "cons_global_agreement")}
        rel = {"fixed_round_consensus_vs_fold_gap": rel_fixed}
        for x in ("cons_consensus_mean_mean_to_best", "cons_global_agreement_mean_to_best", "cos_to_aggregate_mean_mean_to_best"):
            ok = cons[[x, "gap_grande"]].dropna()
            rho = spearmanr(ok[x], ok.gap_grande)
            rel[x] = {"spearman_rho_vs_fold_gap_all_runs": float(rho.statistic), "n_runs": int(len(ok)),
                      "within_partition": {p: float(spearmanr(g[x], g.gap_grande).statistic) for p, g in cons.groupby("partition") if len(g) > 2}}
        write_json(OUT / f"consensus_vs_gap_{tag}.json", {"relation": rel, "note": "descriptive; fold-seed runs are not independent"})
    # local model consensus
    loc = runs[runs.method == "local_grande"].groupby("partition").local_consensus_mean.agg(["mean", "std"]).reset_index()
    loc.to_csv(OUT / f"local_model_consensus_{tag}.csv", index=False)

    # structural agreement FedAvg global vs centralized model (same init/feature subsets per fold and seed)
    sv = []
    for _, r in runs[runs.method.isin(["fedavg_grande", "local_grande"])].iterrows():
        c = RUNS / f"central_grande__none__f{r.fold}__s{r.seed}" / "selections.npz"
        if not c.exists():
            continue
        C, Fd = np.load(c), np.load(RUNS / r.job_id / "selections.npz")
        assert (C["features_by_estimator"] == Fd["features_by_estimator"]).all()
        agree = (Fd["feat"] == C["feat"][0][None]).mean(axis=(1, 2))
        sv.append({"method": r.method, "partition": r.partition, "fold": r.fold, "seed": r.seed,
                   "frac_nodes_same_feature_as_central_mean": float(agree.mean())})
    sv = pd.DataFrame(sv)
    sv.to_csv(OUT / f"structure_vs_central_{tag}.csv", index=False)
    sv_s = sv.groupby(["method", "partition"]).frac_nodes_same_feature_as_central_mean.agg(["mean", "std"]).reset_index() if len(sv) else sv

    comm = runs[runs.method.str.startswith("fedavg")].groupby(["method", "partition"]).agg(
        trainable_parameters=("trainable_parameters", "first"), bytes_per_round=("bytes_per_round", "first"),
        rounds_mean=("steps_executed", "mean"), best_round_mean=("best_step", "mean"), best_round_sd=("best_step", "std"),
        mb_to_best_mean=("bytes_to_best", lambda s: s.mean() / 1e6), mb_to_final_mean=("bytes_to_final", lambda s: s.mean() / 1e6)).reset_index()
    comm.to_csv(OUT / f"communication_{tag}.csv", index=False)
    rt_ = runs.groupby(["method", "partition"]).agg(jobs=("job_id", "size"), seconds_mean=("seconds", "mean"), seconds_sd=("seconds", "std"),
                                                    seconds_total=("seconds", "sum"), steps_mean=("steps_executed", "mean"),
                                                    best_step_mean=("best_step", "mean"), peak_rss_mb_max=("peak_rss_mb", "max")).reset_index()
    rt_.to_csv(OUT / f"runtime_{tag}.csv", index=False)
    write_json(OUT / f"summary_{tag}.json", {"tag": tag, "n_jobs": int(len(runs)), "project_commits_in_results": commits,
                                              "config_sha256": sorted(runs.config_sha256.unique()), "oof_summary": summ.to_dict("records"),
                                              "gaps": gaps, "structure_vs_central": sv_s.to_dict("records") if len(sv_s) else [],
                                              "local_model_consensus": loc.to_dict("records"),
                                              "communication": comm.to_dict("records"), "runtime": rt_.to_dict("records"),
                                              "within_client_auc_posthoc": wcs.to_dict("records"), "per_fold_summary": pfs.to_dict("records"),
                                              "total_compute_hours": float(runs.seconds.sum() / 3600)})
    print(summ.to_string(index=False))
    print(json.dumps({k: v for k, v in gaps.items() if "oof_auc" in k}, indent=1))


if __name__ == "__main__":
    parts = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("natural", "iid")
    main(parts, tag=sys.argv[2] if len(sys.argv) > 2 else "core")
