"""E0a: prepare canonical inputs/states (--prepare) and compare the two sides (--compare). numpy only."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from fedgrande.grande_bridge import BUFFERS, PARAMETERS, STATE_MAPPING  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402

TOL = 1e-6


def effective_v(f, sv):
    if sv > 1:
        return int(min(sv, f))
    return int(min(max(min(int(f * sv), 50), 10), f))


def prepare(cfg, workdir):
    os.makedirs(workdir, exist_ok=True)
    for c in cfg["cases"]:
        rng = np.random.default_rng([cfg["input_seed"], c["n_features"], c["depth"]])
        X = rng.standard_normal((c["n_rows"], c["n_features"])).astype(np.float32)
        if c["nan_fraction"] > 0:
            X[rng.random(X.shape) < c["nan_fraction"]] = np.nan
        y = rng.integers(0, 2, c["n_rows"]).astype(np.int64)
        np.savez(os.path.join(workdir, f"input_{c['name']}.npz"), X=X, y=y)
        E, I, L, V = c["n_estimators"], 2 ** c["depth"] - 1, 2 ** c["depth"], effective_v(c["n_features"], c["selected_variables"])
        r = np.random.default_rng([cfg["canonical_state_seed"], c["n_features"], c["depth"]])
        canon = {
            "split_values": r.normal(0, 1, (E, I, V)).astype(np.float32),     # thresholds on the scale of the inputs
            "split_index_array": r.normal(0, 1, (E, I, V)).astype(np.float32),
            "estimator_weights": r.normal(0, 1, (E, L)).astype(np.float32),
            "leaf_classes_array": r.normal(0, 1, (E, L)).astype(np.float32),
            "features_by_estimator": np.stack([r.choice(c["n_features"], V, replace=False) for _ in range(E)]).astype(np.int64),
        }
        np.savez(os.path.join(workdir, f"canonical_state_{c['name']}.npz"), **canon)
    print("prepared", [c["name"] for c in cfg["cases"]])


def diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0.0, np.abs(a - b))
    return {"shape": list(a.shape), "max_abs": float(np.nanmax(d)) if d.size else 0.0, "mean_abs": float(np.nanmean(d)) if d.size else 0.0,
            "nan_mismatch": int((np.isnan(a) ^ np.isnan(b)).sum()), "bitwise_equal": bool(np.array_equal(a, b, equal_nan=True))}


def compare(cfg, workdir):
    out = {"tolerance_max_abs": TOL, "state_mapping_ref_to_dev": STATE_MAPPING, "cases": []}
    all_pass = True
    for c in cfg["cases"]:
        R = np.load(os.path.join(workdir, f"ref_{c['name']}.npz"))
        D = np.load(os.path.join(workdir, f"dev_{c['name']}.npz"))
        rec = {"case": c,
               "state_keys_ref": R["meta_state_keys"].tolist(), "state_keys_dev": D["meta_state_keys"].tolist(),
               "state_keys_identical": R["meta_state_keys"].tolist() == D["meta_state_keys"].tolist(),
               "selected_variables_ref": int(R["meta_selected_variables"]), "selected_variables_dev": int(D["meta_selected_variables"]),
               "trainable_params_ref": int(R["meta_trainable_params"]), "trainable_params_dev": int(D["meta_trainable_params"]),
               "own_init_state": {k: diff(R[f"own_init__{k}"], D[f"own_init__{STATE_MAPPING[k]}"]) for k in PARAMETERS + BUFFERS},
               "comparisons": {}}
        for tag in ("own_init", "canonical"):
            for what in ("eval_logits", "train_logits", "loss"):
                rec["comparisons"][f"{tag}/{what}"] = diff(R[f"{tag}__{what}"], D[f"{tag}__{what}"])
            for p in PARAMETERS:
                rec["comparisons"][f"{tag}/grad/{p}"] = diff(R[f"{tag}__grad__{p}"], D[f"{tag}__grad__{p}"])
            rec["comparisons"][f"{tag}/ref_eval_vs_ref_train"] = diff(R[f"{tag}__eval_logits"], R[f"{tag}__train_logits"])
        crit = rec["comparisons"]["canonical/eval_logits"]
        rec["pass_forward"] = crit["max_abs"] <= TOL and crit["nan_mismatch"] == 0
        rec["pass_all_compared_tensors"] = all(v["max_abs"] <= TOL and v["nan_mismatch"] == 0 for v in rec["comparisons"].values()) \
            and all(v["max_abs"] == 0 for v in rec["own_init_state"].values()) and rec["state_keys_identical"]
        rec["canonical_logit_scale"] = {"ref_abs_max": float(np.abs(R["canonical__eval_logits"]).max()),
                                         "ref_abs_mean": float(np.abs(R["canonical__eval_logits"]).mean())}
        all_pass &= rec["pass_forward"] and rec["pass_all_compared_tensors"]
        out["cases"].append(rec)
    out["E0a_pass"] = bool(all_pass)
    out["meta_ref"] = json.load(open(os.path.join(workdir, "meta_ref.json")))
    out["meta_dev"] = json.load(open(os.path.join(workdir, "meta_dev.json")))
    write_json(os.path.join(os.path.dirname(workdir.rstrip("/")), "e0a_report.json"), out)
    for rec in out["cases"]:
        print(rec["case"]["name"], "| V ref/dev:", rec["selected_variables_ref"], rec["selected_variables_dev"],
              "| params:", rec["trainable_params_ref"], rec["trainable_params_dev"], "| keys identical:", rec["state_keys_identical"])
        print("   own-init state max diffs:", {k: v["max_abs"] for k, v in rec["own_init_state"].items()})
        for k, v in rec["comparisons"].items():
            print(f"   {k:40s} max={v['max_abs']:.3e} mean={v['mean_abs']:.3e} bitwise={v['bitwise_equal']}")
        print("   PASS forward:", rec["pass_forward"], "| PASS all:", rec["pass_all_compared_tensors"], "| logit scale:", rec["canonical_logit_scale"])
    print("E0a PASS:", out["E0a_pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    prepare(cfg, a.workdir) if a.prepare else compare(cfg, a.workdir)


if __name__ == "__main__":
    main()
