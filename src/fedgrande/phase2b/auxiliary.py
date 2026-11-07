"""Declared auxiliary analyses: N (preprocessing check) and J (stock GRANDE 1.0 reference).

Usage: python -I src/fedgrande/phase2b/auxiliary.py prep_check | stock_reference
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fedgrande.fl import train  # noqa: E402
from fedgrande.fl.features import load_modeling  # noqa: E402
from fedgrande.fl.metrics import binary_metrics, select_threshold  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402

AUX = json.load(open(ROOT / "configs" / "phase2b" / "auxiliary.json"))
CORE = json.load(open(ROOT / "configs" / "phase2b" / "core.json"))


def prep_check():
    from GRANDE.preprocessing import GRANDEPreprocessor

    c = AUX["prep_check"]
    out = ROOT / "results" / "phase2b" / "prep_check"
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(CORE["torch"]["threads"])
    X, y, meta = load_modeling()
    P = np.load(ROOT / "data" / "derived" / "phase2b" / f"prep_C2_outer{c['fold']}.npz")
    roles = P["roles"]
    fit, val, test = roles == "fit", roles == "val", roles == "test"
    stock = GRANDEPreprocessor(**c["stock_preprocessor"])
    Zs_fit = stock.fit_transform(X[fit], y[fit].astype(np.float32))
    Zs = np.zeros((len(y), Zs_fit.shape[1]), dtype=np.float32)
    Zs[fit] = Zs_fit
    Zs[val], Zs[test] = stock.transform(X[val]), stock.transform(X[test])
    mats = {"stock_grande_1.0_preprocessing": Zs, "federated_safe_preprocessing": P["Z"]}
    tcfg = {**CORE["training"], "model_kwargs": CORE["model_kwargs"]["grande"]}
    w = train.class_weights(y[fit])
    rows = []
    for name, Z in mats.items():
        for s in c["seeds"]:
            model, info = train.centralized("grande", Z[fit], y[fit], Z[val], y[val], s, tcfg, w)
            p = train.predict(model, torch.as_tensor(Z))
            thr = select_threshold(y[val], p[val])
            rows.append({"preprocessing": name, "seed": s, "encoded_dim": int(Z.shape[1]), "best_epoch": info["best_step"],
                         "epochs": info["steps_executed"], "val_auc": info["best_val_auc"],
                         "test_auc": binary_metrics(y[test], p[test])["roc_auc"], "test_f1_05": binary_metrics(y[test], p[test])["macro_f1"],
                         "test_f1_valthr": binary_metrics(y[test], p[test], thr)["macro_f1"], "val_threshold": thr})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "prep_check_runs.csv", index=False)
    summ = df.groupby("preprocessing")[["val_auc", "test_auc", "test_f1_05", "test_f1_valthr", "best_epoch"]].agg(["mean", "std"])
    summ.columns = ["_".join(x) for x in summ.columns]
    write_json(out / "prep_check.json", {"config": c, "runs": rows, "summary": summ.reset_index().to_dict("records"),
                                          "stock_loo_columns": list(stock.high_cardinality_indices),
                                          "stock_numeric_impute_value": float(stock.mean_train_num), "stock_first_numeric_column": stock.num_columns[0],
                                          "meta": run_metadata(impl="dev")})
    print(summ)


def stock_reference():
    c = AUX["stock_reference"]
    for k in c["folds"]:
        outdir = ROOT / "results" / "phase2b" / "stock_reference" / f"fold{k}"
        if (outdir / "metrics.json").exists():
            continue
        os.makedirs(outdir, exist_ok=True)
        spec = {"run_id": f"stock_ref_e1024_d5_f{k}", "label": "Stock GRANDE 1.0 reference (not the matched centralized comparator)",
                "scheme": "C2", "outer_fold": k, "drop_columns": c["drop_columns"], "estimator_params": c["estimator_params"]}
        json.dump(spec, open(outdir / "spec.json", "w"), indent=2)
        with open(outdir / "stdout.txt", "w") as so:
            rc = subprocess.run([sys.executable, "-I", str(ROOT / "src" / "fedgrande" / "grande_runner.py"), "--spec", str(outdir / "spec.json"),
                                 "--outdir", str(outdir)], stdout=so, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(k, rc, open(outdir / "stdout.txt").read().strip().splitlines()[-1], flush=True)
    rows = []
    for k in c["folds"]:
        m = json.load(open(ROOT / "results" / "phase2b" / "stock_reference" / f"fold{k}" / "metrics.json"))
        rows.append({"fold": k, "test_auc": m["test_at_0.5"]["roc_auc"], "test_f1_05": m["test_at_0.5"]["macro_f1"],
                     "test_f1_valthr": m["test_at_val_threshold"]["macro_f1"], "fit_seconds": m["fit_seconds"], "epochs": m["epochs_completed"]})
    preds = pd.concat([pd.read_csv(ROOT / "results" / "phase2b" / "stock_reference" / f"fold{k}" / "predictions.csv").assign(fold=k) for k in c["folds"]])
    te = preds[preds.role == "test"]
    write_json(ROOT / "results" / "phase2b" / "stock_reference" / "summary.json",
               {"label": "Stock GRANDE 1.0 reference", "per_fold": rows, "oof_test": binary_metrics(te.y, te.p1, 0.5),
                "note": "own preprocessing (LOO target encoding of field, first-numeric-column imputation) and own training loop; not comparable as the matched comparator"})


if __name__ == "__main__":
    {"prep_check": prep_check, "stock_reference": stock_reference}[sys.argv[1]]()
