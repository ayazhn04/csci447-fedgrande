"""Phase 2C: P0-P3 for one (fold, seed) on the Dirichlet(0.1) partition. Writes results/phase2c/runs/f{fold}_s{seed}/."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from fedgrande.fl import personalize as P  # noqa: E402
from fedgrande.fl.features import load_modeling  # noqa: E402
from fedgrande.fl.metrics import binary_metrics, select_threshold  # noqa: E402
from fedgrande.fl.models import build_grande  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402

PREP = ROOT / "data" / "derived" / "phase2b"
CKPT = ROOT / "data" / "derived" / "phase2c" / "global_ckpt"
METHODS = ("P0", "P1", "P2", "P3")


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def fold_metrics(y, p, thr):
    a, b = binary_metrics(y, p, 0.5), binary_metrics(y, p, thr)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {"roc_auc": a["roc_auc"], "macro_f1_05": a["macro_f1"], "macro_f1_valthr": b["macro_f1"], "accuracy_05": a["accuracy"],
            "log_loss": float(log_loss(y, pc, labels=[0, 1])), "brier": float(brier_score_loss(y, p)), "ece10": ece(y, p),
            "threshold": thr, "cm_valthr": b["confusion_[[tn,fp],[fn,tp]]"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2c" / "personalization.json"))
    ap.add_argument("--out-root", default=str(ROOT / "results" / "phase2c" / "runs"))
    ap.add_argument("--clients", help="smoke only: comma-separated subset")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    torch.set_num_threads(cfg["torch"]["threads"])
    torch.set_float32_matmul_precision(cfg["torch"]["float32_matmul_precision"])
    out = os.path.join(a.out_root, f"f{a.fold}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()

    D = np.load(PREP / f"prep_C2_outer{a.fold}.npz")
    Z, roles = D["Z"], D["roles"]
    _, y, meta = load_modeling()
    client = pd.read_csv(PREP / "partitions_dirichlet_v1.csv")[f"client_{cfg['partition']}"].to_numpy()
    g = meta.leakage_group_id.to_numpy()
    fit, val, test = roles == "fit", roles == "val", roles == "test"
    for A, B in ((fit, val), (fit, test), (val, test)):
        assert not set(g[A]) & set(g[B])

    ck = CKPT / f"fedavg_grande__{cfg['partition']}__f{a.fold}__s{a.seed}.pt"
    glob = build_grande(Z.shape[1], a.seed, n_estimators=256, depth=5, selected_variables=0.8)
    glob.load_state_dict(torch.load(ck, map_location="cpu"))
    glob.eval()
    Zt = torch.as_tensor(Z)

    p = {m: np.full(len(y), np.nan) for m in METHODS}
    vt_all = val | test
    p["P0"][vt_all] = P.predict(glob, Zt[vt_all])              # same row set/order/batching as Phase 2B
    p["P0"][fit] = P.predict(glob, Zt[fit])                    # fit rows: used only by P1 intercept estimation
    # sanity: equals the Phase 2B saved predictions of this model (validation/test rows)
    ref = pd.read_csv(ROOT / "results" / "phase2b" / "runs" / f"fedavg_grande__{cfg['partition']}__f{a.fold}__s{a.seed}" / "predictions.csv")
    # model outputs are float32; Phase 2B CSVs store them as decimal text (~1e-17 rounding), so compare at float32 precision
    assert np.array_equal(ref.p1.to_numpy().astype(np.float32), p["P0"][ref.row_index.to_numpy()].astype(np.float32)), \
        "global checkpoint does not reproduce Phase 2B"

    K = int(client.max() + 1)
    clients = [int(c) for c in a.clients.split(",")] if a.clients else list(range(K))
    crow = []
    for c in range(K):
        r_fit, r_val, r_test = fit & (client == c), val & (client == c), test & (client == c)
        info = {"client": c, "n_fit": int(r_fit.sum()), "fit_pos": int(y[r_fit].sum()), "n_val": int(r_val.sum()), "val_pos": int(y[r_val].sum()),
                "n_test": int(r_test.sum()), "test_pos": int(y[r_test].sum())}
        rows = r_val | r_test
        t0 = time.perf_counter()
        b = P.fit_intercept(P.logit_from_p(p["P0"][r_fit]), y[r_fit], cfg["P1_intercept"]["lam"])
        p["P1"][rows] = P.apply_intercept(p["P0"][rows], b)
        info.update({"P1_intercept": b, "P1_seconds": time.perf_counter() - t0})
        if c not in clients:
            continue
        for v in ("P2", "P3"):
            t0 = time.perf_counter()
            mdl, pi = P.personalize(glob, v, Z[r_fit], y[r_fit], Z[r_val], y[r_val], a.seed, c, cfg["P2_P3_finetune"])
            p[v][rows] = P.predict(mdl, torch.as_tensor(Z[rows]))
            ver = P.verify_structure(glob, mdl, v)
            info.update({f"{v}_{k}": val_ for k, val_ in pi.items()})
            info.update({f"{v}_seconds": time.perf_counter() - t0, f"{v}_only_intended_changed": ver["only_intended_changed"],
                         f"{v}_changed_params": "|".join(ver["changed_params"]),
                         f"{v}_structure_unchanged": all(val_ for k, val_ in ver.items() if k.startswith("unchanged__"))})
        crow.append(info)
    cdf = pd.DataFrame(crow)
    cdf.to_csv(os.path.join(out, "clients.csv"), index=False)

    vt = val | test
    keep = np.isin(client, clients)
    res = {"fold": a.fold, "seed": a.seed, "checkpoint": str(ck.relative_to(ROOT)), "checkpoint_sha256": sha256_file(ck),
           "clients_run": clients, "methods": {}}
    for m in METHODS:
        sel_v, sel_t = val & keep, test & keep
        thr = select_threshold(y[sel_v], p[m][sel_v])
        res["methods"][m] = {"test": fold_metrics(y[sel_t], p[m][sel_t], thr), "val_auc": binary_metrics(y[sel_v], p[m][sel_v])["roc_auc"]}
    res["structure_all_unchanged"] = bool(cdf[[c for c in cdf.columns if c.endswith("structure_unchanged")]].all().all())
    res["only_intended_changed_all"] = bool(cdf[[c for c in cdf.columns if c.endswith("only_intended_changed")]].all().all())
    res["personalization_seconds"] = {"P1": float(cdf.P1_seconds.sum()), "P2": float(cdf.P2_seconds.sum()), "P3": float(cdf.P3_seconds.sum())}
    res["seconds_total"] = time.perf_counter() - t_all
    res["additional_communication_bytes"] = 0
    res["config_sha256"] = sha256_file(a.config)
    res["meta"] = run_metadata(impl="dev")
    pd.DataFrame({"row_index": np.flatnonzero(vt), "role": roles[vt], "y": y[vt], "client": client[vt],
                  **{m: p[m][vt] for m in METHODS}}).to_csv(os.path.join(out, "predictions.csv"), index=False)
    write_json(os.path.join(out, "result.json"), res)
    print(f"[f{a.fold}_s{a.seed}] {res['seconds_total']:.0f}s structure_ok={res['structure_all_unchanged']} intended_only={res['only_intended_changed_all']} "
          f"val AUC " + " ".join(f"{m}={res['methods'][m]['val_auc']:.4f}" for m in METHODS))


if __name__ == "__main__":
    main()
