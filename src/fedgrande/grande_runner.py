"""Run one centralized GRANDE 1.0.0 (GRANDEClassifier, db79c860) fit/evaluate job from a JSON run spec.

Usage (fedgrande-dev env): python -I src/fedgrande/grande_runner.py --spec <spec.json> --outdir <dir>

Run spec keys: run_id, label, scheme ("C1"|"C2"), outer_fold, seed, estimator_params (non-default
GRANDEClassifier arguments), threshold_policy. Writes into outdir: resolved_config.json, metrics.json,
predictions.csv (val+test probabilities), curve.csv (per-epoch losses), train_log.txt.
Nothing from the test fold is used for fitting, early stopping or threshold selection.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score  # noqa: E402

from fedgrande.data import load_xy  # noqa: E402
from fedgrande.paths import DERIVED  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402

EPOCH_RE = re.compile(r"Epoch (\d+) \| TrainLoss: ([\d.naninf]+) \| ValLoss: ([\d.naninf]+) \|.*?Time: ([\d.]+)s")


def maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KiB on Linux


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"threshold": float(thr), "roc_auc": float(roc_auc_score(y, p)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(accuracy_score(y, pred)), "confusion_matrix_[[tn,fp],[fn,tp]]": cm.tolist()}


def select_threshold(y_val, p_val) -> float:
    """Validation-only: threshold maximizing validation macro-F1 over a fixed grid (ties -> closest to 0.5)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = np.array([f1_score(y_val, (p_val >= t).astype(int), average="macro") for t in grid])
    best = np.flatnonzero(scores == scores.max())
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    os.makedirs(a.outdir, exist_ok=True)

    import torch
    from GRANDE import GRANDEClassifier

    X, y, _ = load_xy()
    inner = pd.read_csv(DERIVED / "splits_inner_v1.csv")
    sel = inner[(inner.scheme == spec["scheme"]) & (inner.outer_fold == spec["outer_fold"])].sort_values("row_index")
    idx = {r: sel.loc[sel.role == r, "row_index"].to_numpy() for r in ("fit", "val", "test")}
    assert len(np.intersect1d(idx["fit"], idx["test"])) == 0 and len(np.intersect1d(idx["val"], idx["test"])) == 0

    clf = GRANDEClassifier(**spec["estimator_params"])
    rss_before = maxrss_mb()
    log = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        clf.fit(X.iloc[idx["fit"]], y[idx["fit"]], X_val=X.iloc[idx["val"]], y_val=y[idx["val"]])
    fit_s = time.perf_counter() - t0
    rss_after_fit = maxrss_mb()
    t1 = time.perf_counter()
    p_test = clf.predict_proba(X.iloc[idx["test"]])[:, 1]
    pred_s = time.perf_counter() - t1
    p_val = clf.predict_proba(X.iloc[idx["val"]])[:, 1]

    module = clf.module_._orig_mod if hasattr(clf.module_, "_orig_mod") else clf.module_
    n_params = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    for emb in (clf.numeric_embeddings_, clf.category_embeddings_):
        if emb is not None:
            n_params += int(sum(p.numel() for p in emb.parameters() if p.requires_grad))

    thr_val = select_threshold(y[idx["val"]], p_val)
    curve = pd.DataFrame([{"epoch": int(m[0]), "train_loss": float(m[1]), "val_loss": float(m[2]), "epoch_seconds": float(m[3])}
                          for m in EPOCH_RE.findall(log.getvalue())])
    curve.to_csv(os.path.join(a.outdir, "curve.csv"), index=False)
    open(os.path.join(a.outdir, "train_log.txt"), "w").write(log.getvalue())
    preds = pd.concat([pd.DataFrame({"row_index": idx["val"], "role": "val", "y": y[idx["val"]], "p1": p_val}),
                       pd.DataFrame({"row_index": idx["test"], "role": "test", "y": y[idx["test"]], "p1": p_test})])
    preds.to_csv(os.path.join(a.outdir, "predictions.csv"), index=False)

    resolved = clf.get_params()
    defaults = GRANDEClassifier().get_params()
    metrics = {
        "run_id": spec["run_id"], "label": spec.get("label", "GRANDE 1.0.0 baseline (not a paper reproduction)"),
        "scheme": spec["scheme"], "outer_fold": spec["outer_fold"], "seed": spec["estimator_params"].get("random_state", defaults["random_state"]),
        "split_id": f"{spec['scheme']}/outer{spec['outer_fold']}/splits_inner_v1",
        "n_fit": int(len(idx["fit"])), "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
        "test_pos": int(y[idx["test"]].sum()),
        "test_at_0.5": metrics_at(y[idx["test"]], p_test, 0.5),
        "test_at_val_threshold": metrics_at(y[idx["test"]], p_test, thr_val),
        "val_at_0.5": metrics_at(y[idx["val"]], p_val, 0.5),
        "val_selected_threshold": thr_val,
        "best_epoch": clf.best_epoch_, "epochs_completed": clf.n_epochs_, "best_val_score": float(clf.best_val_score_),
        "fit_seconds": fit_s, "predict_test_seconds": pred_s,
        "seconds_per_epoch_mean": float(curve.epoch_seconds.mean()) if len(curve) else None,
        "peak_rss_mb_process": rss_after_fit, "peak_rss_mb_before_fit": rss_before,
        "trainable_parameters": n_params,
        "network_input_features_after_preprocessing": int(module.number_of_variables),
        "selected_variables_per_estimator": int(module.selected_variables),
        "n_estimators": int(module.n_estimators), "depth": int(module.depth),
        "effective_batch_size": int(min(resolved["batch_size"], len(idx["fit"]) // 2)),
        "compiled": bool(clf.compiled_), "torch_threads": torch.get_num_threads(),
        "preprocessing": {"num_columns": len(clf.preprocessor_.num_columns), "binary_cat": len(clf.preprocessor_.binary_indices),
                          "one_hot_cat": len(clf.preprocessor_.low_cardinality_indices),
                          "loo_target_encoded_cat": list(clf.preprocessor_.high_cardinality_indices),
                          "numeric_imputation_value_(GRANDE: mean of first numeric column)": float(getattr(clf.preprocessor_, "mean_train_num", float("nan"))),
                          "first_numeric_column": clf.preprocessor_.num_columns[0] if clf.preprocessor_.num_columns else None},
    }
    write_json(os.path.join(a.outdir, "resolved_config.json"), {
        "spec": spec, "resolved_estimator_params": resolved,
        "non_default_params": {k: v for k, v in resolved.items() if defaults.get(k) != v},
        "inputs": {"splits_inner_sha256": sha256_file(DERIVED / "splits_inner_v1.csv"), "dyads_sha256": sha256_file(DERIVED / "dyads_v1.csv")},
        "meta": run_metadata(impl="dev")})
    write_json(os.path.join(a.outdir, "metrics.json"), metrics)
    t = metrics["test_at_0.5"]
    print(f"[{spec['run_id']}] fit {fit_s:.1f}s epochs {clf.n_epochs_} best {clf.best_epoch_} | test AUC {t['roc_auc']:.4f} "
          f"F1@0.5 {t['macro_f1']:.4f} | params {n_params} | V {module.selected_variables} | RSS {rss_after_fit:.0f} MB")


if __name__ == "__main__":
    main()
