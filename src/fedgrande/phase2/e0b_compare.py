"""E0b step 3: compare the two training runs (curves, final parameters, predictions, metrics)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from fedgrande.paths import CONFIGS, DERIVED, RESULTS  # noqa: E402
from fedgrande.phase2.e0a_compare import diff  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402


def main():
    cfg = json.load(open(CONFIGS / "e0b_training.json"))
    d = RESULTS / "gate_a" / "e0b"
    R, Dv = np.load(d / "ref_outputs.npz"), np.load(d / "dev_outputs.npz")
    rr, dr = json.load(open(d / "ref_run.json")), json.load(open(d / "dev_run.json"))
    P = np.load(DERIVED / f"e0b_prepared_{cfg['scheme']}_outer{cfg['outer_fold']}.npz")
    cr, cd = pd.DataFrame(rr["curve"]), pd.DataFrame(dr["curve"])
    curves = cr.merge(cd, on="epoch", suffixes=("_ref", "_dev"))
    curves.to_csv(d / "curves_ref_vs_dev.csv", index=False)
    state_keys = sorted(k for k in R.files if k.startswith("state__"))
    rep = {
        "epochs_ref": len(cr), "epochs_dev": len(cd),
        "train_loss_curve": diff(curves.train_loss_ref, curves.train_loss_dev),
        "val_loss_curve": diff(curves.val_loss_ref, curves.val_loss_dev),
        "best_epoch_ref": int(cr.val_loss.idxmin() + 1), "best_epoch_dev": int(cd.val_loss.idxmin() + 1),
        "loss_decreased_ref": bool(cr.train_loss.iloc[-1] < cr.train_loss.iloc[0] and cr.val_loss.min() < cr.val_loss.iloc[0]),
        "loss_decreased_dev": bool(cd.train_loss.iloc[-1] < cd.train_loss.iloc[0] and cd.val_loss.min() < cd.val_loss.iloc[0]),
        "final_state": {k.replace("state__", ""): diff(R[k], Dv[k]) for k in state_keys},
        "p_val": diff(R["p_val"], Dv["p_val"]), "p_test": diff(R["p_test"], Dv["p_test"]),
        "ref_predict_proba_vs_direct_module": diff(R["p_test"], R["p_test_direct"]),
        "test_pred_agreement_at_0.5": float(((R["p_test"] >= .5) == (Dv["p_test"] >= .5)).mean()),
        "metrics": {impl: {"val_auc": float(roc_auc_score(P["y_val"], o["p_val"])), "test_auc": float(roc_auc_score(P["y_test"], o["p_test"])),
                           "test_macro_f1_at_0.5": float(f1_score(P["y_test"], (o["p_test"] >= .5).astype(int), average="macro"))}
                    for impl, o in (("ref", R), ("dev", Dv))},
        "fit_seconds": {"ref": rr["fit_seconds"], "dev": dr["fit_seconds"]},
        "harness_controls_ref": rr["harness_controls"],
        "commits": {"ref": rr["meta"]["grande"], "dev": dr["meta"]["grande"]},
        "note": "Single split, single seed, 64 estimators, 30 epochs: a diagnostic of training-loop equivalence, not a performance or statistical result.",
    }
    rep["identical_training"] = bool(rep["train_loss_curve"]["max_abs"] == 0 and rep["val_loss_curve"]["max_abs"] == 0
                                     and all(v["max_abs"] == 0 for v in rep["final_state"].values()) and rep["p_test"]["max_abs"] == 0)
    write_json(RESULTS / "gate_a" / "e0b_report.json", rep)
    print(json.dumps({k: rep[k] for k in ("epochs_ref", "epochs_dev", "train_loss_curve", "val_loss_curve", "best_epoch_ref", "best_epoch_dev",
                                          "loss_decreased_ref", "loss_decreased_dev", "p_test", "test_pred_agreement_at_0.5", "metrics",
                                          "identical_training", "fit_seconds")}, indent=1))
    print({k: v["max_abs"] for k, v in rep["final_state"].items()})


if __name__ == "__main__":
    main()
