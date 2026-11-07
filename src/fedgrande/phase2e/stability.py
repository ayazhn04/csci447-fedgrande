"""Phase 2E (DIAGNOSTIC): what still changes after round 20 in ordinary FedAvg-GRANDE (F0)?

Phase 2B did not store intermediate global states, so limited diagnostic F0 reruns (identical code; bitwise-checked
against Phase 2B) save global states at rounds 20/40/60 and the best round. This script:
  1. runs the diagnostic reruns that are missing (seed 42, folds 0-4, natural and IID = 10 runs),
  2. checks each rerun reproduces Phase 2B bitwise (predictions + per-round validation AUC),
  3. computes feature-identity stability, threshold drift and output-parameter drift relative to round 20.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fedgrande.fl import diagnostics as diag  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402

SNAP = ROOT / "data" / "derived" / "phase2e" / "f0_snapshots"
DIAG = ROOT / "results" / "phase2e" / "diagnostic_f0"
P2B = ROOT / "results" / "phase2b" / "runs"
N_NUMERIC = 58


def ensure(part, fold, seed):
    jid = f"fedavg_grande__{part}__f{fold}__s{seed}"
    if not (SNAP / f"{jid}_round20.pt").exists():
        subprocess.run([sys.executable, "-I", str(ROOT / "src/fedgrande/phase2b/run_job.py"), "--method", "fedavg_grande", "--partition", part,
                        "--fold", str(fold), "--seed", str(seed), "--out-root", str(DIAG), "--save-state", str(SNAP / f"{jid}.pt"),
                        "--snapshot-rounds", "20,40,60"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    a, b = pd.read_csv(P2B / jid / "predictions.csv"), pd.read_csv(DIAG / jid / "predictions.csv")
    ra, rb = pd.read_csv(P2B / jid / "rounds.csv"), pd.read_csv(DIAG / jid / "rounds.csv")
    return jid, bool(a.equals(b) and np.array_equal(ra.val_auc, rb.val_auc))


def main():
    rows = []
    for part in ("natural", "iid"):
        for fold in range(5):
            jid, ok = ensure(part, fold, 42)
            best_round = json.load(open(DIAG / jid / "result.json"))["best_step"]
            S = {r: torch.load(SNAP / f"{jid}_round{r}.pt", map_location="cpu") for r in (20, 40, 60) if (SNAP / f"{jid}_round{r}.pt").exists()}
            S["best"] = torch.load(SNAP / f"{jid}.pt", map_location="cpu")
            sel = {k: diag.selections(v["split_index_array"], v["split_values"], v["features_by_estimator"]) for k, v in S.items()}
            f20, t20 = sel[20]
            rec = {"job": jid, "partition": part, "fold": fold, "seed": 42, "reproduces_phase2b": ok, "best_round": best_round}
            for k in [x for x in S if x != 20]:
                fk, tk = sel[k]
                same = fk == f20
                rec[f"feature_stability_{k}"] = float(same.mean())
                num = same & (f20 < N_NUMERIC)
                oh = same & (f20 >= N_NUMERIC)
                d = np.abs(tk - t20)
                rec[f"thr_drift_numeric_sd_mean_{k}"] = float(d[num].mean()) if num.any() else np.nan     # z-scored -> SD units
                rec[f"thr_drift_numeric_sd_median_{k}"] = float(np.median(d[num])) if num.any() else np.nan
                rec[f"thr_drift_onehot_mean_{k}"] = float(d[oh].mean()) if oh.any() else np.nan
                for t in ("leaf_classes_array", "estimator_weights", "split_values", "split_index_array"):
                    rec[f"rel_drift_{t}_{k}"] = float(torch.linalg.vector_norm(S[k][t] - S[20][t]) / torch.linalg.vector_norm(S[20][t]))
            rec["frac_nodes_numeric_at20"] = float((f20 < N_NUMERIC).mean())
            rows.append(rec)
            print(jid, ok, {k: round(v, 4) for k, v in rec.items() if k.startswith(("feature_stability", "rel_drift_leaf", "rel_drift_est", "thr_drift_numeric_sd_mean"))}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "phase2e" / "stability_by_run.csv", index=False)
    summ = df.drop(columns=["job", "fold", "seed", "reproduces_phase2b"]).groupby("partition").agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    write_json(ROOT / "results" / "phase2e" / "stability_summary.json",
               {"label": "DIAGNOSTIC: F0 reruns (seed 42, folds 0-4) with intermediate snapshots; bitwise-checked against Phase 2B",
                "all_reproduce_phase2b": bool(df.reproduces_phase2b.all()), "summary": summ.reset_index().to_dict("records"),
                "n_numeric_features": N_NUMERIC, "threshold_units": "numeric features are z-scored with training-fold SD -> drift in SD units; one-hot drift in raw 0/1 units"})
    print(summ.T.round(4).to_string())


if __name__ == "__main__":
    main()
