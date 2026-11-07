"""Phase 2C aggregation and report (all numbers read from results/phase2c/runs/*)."""
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from fedgrande.fl.metrics import binary_metrics  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.phase2.phase2a_summary import md_table  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402

R = ROOT / "results" / "phase2c"
OUT = R / "summary"
M = ("P0", "P1", "P2", "P3")
NAMES = {"P0": "Global FedAvg GRANDE", "P1": "+ client intercept calibration", "P2": "+ personalized leaves",
         "P3": "+ personalized leaves and estimator weights"}
KEYS = ("roc_auc", "macro_f1_05", "macro_f1_valthr", "log_loss", "brier", "ece10")


def ms(s):
    s = pd.Series(s, dtype=float)
    return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    res = [json.load(open(f)) for f in sorted(glob(str(R / "runs" / "*" / "result.json")))]
    assert len(res) == 15, len(res)
    rows = []
    for r in res:
        for m in M:
            t = r["methods"][m]["test"]
            rows.append({"fold": r["fold"], "seed": r["seed"], "method": m, **{k: t[k] for k in KEYS}, "threshold": t["threshold"],
                         "val_auc": r["methods"][m]["val_auc"], "personalization_seconds": r["personalization_seconds"].get(m, 0.0)})
    pf = pd.DataFrame(rows)
    pf.to_csv(OUT / "per_fold_metrics.csv", index=False)
    seedlvl = pf.groupby(["method", "seed"])[list(KEYS) + ["personalization_seconds"]].agg(["mean", "std"])
    seedlvl.columns = ["_".join(c) for c in seedlvl.columns]
    seedlvl = seedlvl.reset_index()
    seedlvl.to_csv(OUT / "per_seed_fold_means.csv", index=False)

    # pooled OOF per seed (secondary) and within-client per seed
    preds = []
    for f in sorted(glob(str(R / "runs" / "*" / "predictions.csv"))):
        d = pd.read_csv(f)
        fold, seed = os.path.basename(os.path.dirname(f)).replace("f", "").split("_s")
        d["fold"], d["seed"] = int(fold), int(seed)
        preds.append(d)
    P = pd.concat(preds)
    thr = pf.set_index(["method", "fold", "seed"]).threshold
    test = P[P.role == "test"].copy()
    for m in M:
        test[f"{m}_pred"] = [int(p >= thr[(m, f, s)]) for p, f, s in zip(test[m], test.fold, test.seed)]
    pooled, wc_rows = [], []
    for s, g in test.groupby("seed"):
        for m in M:
            pooled.append({"seed": s, "method": m, "pooled_oof_auc": binary_metrics(g.y, g[m])["roc_auc"]})
        for c, gc in g.groupby("client"):
            mixed = gc.y.nunique() == 2
            for m in M:
                pc = np.clip(gc[m].to_numpy(), 1e-7, 1 - 1e-7)
                rec = {"seed": s, "client": c, "method": m, "n": len(gc), "pos": int(gc.y.sum()), "mixed": bool(mixed),
                       "log_loss": float(log_loss(gc.y, pc, labels=[0, 1])), "brier": float(brier_score_loss(gc.y, gc[m])),
                       "accuracy_05": float(((gc[m] >= 0.5).astype(int) == gc.y).mean()),
                       "accuracy_valthr": float((gc[f"{m}_pred"] == gc.y).mean())}
                if mixed:
                    rec.update({"auc": binary_metrics(gc.y, gc[m])["roc_auc"], "macro_f1_05": binary_metrics(gc.y, gc[m], 0.5)["macro_f1"],
                                "macro_f1_valthr": binary_metrics(gc.y, gc[m], pred=gc[f"{m}_pred"])["macro_f1"]})
                wc_rows.append(rec)
    pooled = pd.DataFrame(pooled)
    wc = pd.DataFrame(wc_rows)
    wc.to_csv(OUT / "within_client_by_seed.csv", index=False)
    mixed_clients = sorted(wc.loc[wc.mixed, "client"].unique().tolist())
    n_clients = int(wc.client.nunique())
    wmix = wc[wc.mixed].groupby(["method", "seed"])[["auc", "macro_f1_05", "macro_f1_valthr", "log_loss", "brier"]].mean().reset_index()
    wsingle = wc[~wc.mixed].groupby(["method", "seed"])[["log_loss", "brier", "accuracy_05", "accuracy_valthr"]].mean().reset_index()
    per_client_mixed = wc[wc.mixed].groupby(["client", "method"]).agg(n=("n", "first"), pos=("pos", "first"), auc=("auc", "mean"),
                                                                      auc_sd=("auc", "std"), log_loss=("log_loss", "mean"), brier=("brier", "mean")).reset_index()
    per_client_mixed.to_csv(OUT / "within_client_mixed_per_client.csv", index=False)

    # main table (mean ± SD over seeds of per-seed fold means)
    main_rows = []
    for m in M:
        sl = seedlvl[seedlvl.method == m]
        main_rows.append({"method": NAMES[m], "per-fold AUC": ms(sl.roc_auc_mean), "macro-F1@0.5": ms(sl.macro_f1_05_mean),
                          "macro-F1@val-thr": ms(sl.macro_f1_valthr_mean), "log loss": ms(sl.log_loss_mean), "Brier": ms(sl.brier_mean),
                          "ECE": ms(sl.ece10_mean), f"within-client AUC (mixed, n={len(mixed_clients)})": ms(wmix[wmix.method == m].auc),
                          "single-class clients log loss": ms(wsingle[wsingle.method == m].log_loss),
                          "pooled OOF AUC (secondary)": ms(pooled[pooled.method == m].pooled_oof_auc),
                          "personalization s per fold-seed": ms(sl.personalization_seconds_mean)})
    main = pd.DataFrame(main_rows)
    main.to_csv(OUT / "main_table.csv", index=False)

    # paired fold differences (descriptive)
    wide = pf.pivot_table(index=["fold", "seed"], columns="method", values=list(KEYS))
    diffs = {}
    for a, b in (("P1", "P0"), ("P2", "P1"), ("P3", "P1"), ("P3", "P2"), ("P2", "P0"), ("P3", "P0")):
        d = {k: (wide[(k, a)] - wide[(k, b)]) for k in KEYS}
        diffs[f"{a}-{b}"] = {k: {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                                 "n_folds_positive": int((v > 0).sum()), "n": int(len(v))} for k, v in d.items()}
    wd = wc[wc.mixed].pivot_table(index=["client", "seed"], columns="method", values="auc")
    diffs_within = {f"{a}-{b}": {"mean": float((wd[a] - wd[b]).mean()), "min": float((wd[a] - wd[b]).min()), "max": float((wd[a] - wd[b]).max()),
                                 "n_client_seed_positive": int(((wd[a] - wd[b]) > 1e-12).sum()), "n_client_seed_negative": int(((wd[a] - wd[b]) < -1e-12).sum()),
                                 "n": int(len(wd))} for a, b in (("P1", "P0"), ("P2", "P1"), ("P3", "P1"), ("P3", "P2"))}

    clients = pd.concat([pd.read_csv(f).assign(run=os.path.basename(os.path.dirname(f))) for f in sorted(glob(str(R / "runs" / "*" / "clients.csv")))])
    clients.to_csv(OUT / "clients_all_runs.csv", index=False)
    struct = {"P2_structure_unchanged_all": bool(clients.P2_structure_unchanged.all()), "P3_structure_unchanged_all": bool(clients.P3_structure_unchanged.all()),
              "P2_only_intended_all": bool(clients.P2_only_intended_changed.all()), "P3_only_intended_all": bool(clients.P3_only_intended_changed.all()),
              "P2_changed_params": sorted(clients.P2_changed_params.fillna("").unique().tolist()),
              "P3_changed_params": sorted(clients.P3_changed_params.fillna("").unique().tolist()),
              "client_models_checked": int(len(clients) * 2)}
    rules = {v: clients[f"{v}_rule"].value_counts().to_dict() for v in ("P2", "P3")}
    best_ep = {v: clients.groupby(f"{v}_rule")[f"{v}_best_epoch"].describe()[["mean", "min", "max"]].round(2).to_dict("index") for v in ("P2", "P3")}
    runtime = {"per_client_seconds_mean": {v: float(clients[f"{v}_seconds"].mean()) for v in ("P1", "P2", "P3")},
               "per_fold_seed_seconds_mean": float(np.mean([r["seconds_total"] for r in res])),
               "total_phase2c_seconds_runs": float(np.sum([r["seconds_total"] for r in res]))}
    p2b = pd.read_csv(ROOT / "results" / "phase2b" / "summary" / "communication_all.csv")
    comm = {"global_fedavg_grande_dirichlet_from_phase2b": p2b[(p2b.method == "fedavg_grande") & (p2b.partition == "dirichlet0.1")].to_dict("records"),
            "additional_personalization_bytes": int(sum(r["additional_communication_bytes"] for r in res))}
    regen = json.load(open(R / "checkpoint_regeneration.json"))
    summary = {"n_runs": len(res), "n_clients": n_clients, "mixed_clients_in_pooled_test": mixed_clients,
               "n_mixed": len(mixed_clients), "n_single_class": n_clients - len(mixed_clients),
               "main_table": main_rows, "paired_fold_differences": diffs, "within_client_auc_differences": diffs_within,
               "structure_verification": struct, "selection_rules": rules, "best_epochs": best_ep, "runtime": runtime, "communication": comm,
               "checkpoints_all_verified": regen["all_verified"],
               "project_commits": sorted({r["meta"]["project_commit"] for r in res}), "config_sha256": sorted({r["config_sha256"] for r in res})}
    write_json(OUT / "phase2c_summary.json", summary)

    L = ["# Phase 2C report (generated)\n", "Generated by `src/fedgrande/phase2c/aggregate.py` from `results/phase2c/runs/`; protocol `reports/PHASE2C_PROTOCOL.md`.\n",
         f"- runs {len(res)}; commits {summary['project_commits']}; config {summary['config_sha256']}; checkpoints verified {regen['all_verified']}",
         f"- Dirichlet clients {n_clients}: mixed-class in pooled test {len(mixed_clients)} {mixed_clients}, single-class {n_clients - len(mixed_clients)}\n",
         "## Main table (mean ± SD over 3 seeds of per-seed means over 5 folds)\n", md_table(main),
         "\n## Per-seed fold means\n", md_table(seedlvl.round(4)),
         "\n## Within-client, mixed clients (mean over seeds per client)\n", md_table(per_client_mixed.round(4)),
         "\n## Single-class clients (mean over clients, per seed)\n", md_table(wsingle.round(4)),
         "\n## Paired fold differences (descriptive)\n"]
    for k, v in diffs.items():
        L.append(f"- {k}: " + "; ".join(f"{kk} {vv['mean']:+.4f} [{vv['min']:+.4f}, {vv['max']:+.4f}] ({vv['n_folds_positive']}/{vv['n']} >0)" for kk, vv in v.items()))
    L += ["\n## Within-client AUC differences (mixed clients × seeds)\n"] + [f"- {k}: {v}" for k, v in diffs_within.items()]
    L += [f"\n## Structure verification\n\n{struct}\n", f"## Selection rules\n\n{rules}; best epochs {best_ep}\n",
          f"## Runtime\n\n{runtime}\n", f"## Communication\n\n{comm}\n"]
    (ROOT / "reports" / "phase2c_report.md").write_text("\n".join(L) + "\n")
    print(main.to_string(index=False))
    print(json.dumps({"n_mixed": len(mixed_clients), "mixed": mixed_clients, "struct": struct, "runtime": runtime, "within": diffs_within}, indent=1))


if __name__ == "__main__":
    main()
