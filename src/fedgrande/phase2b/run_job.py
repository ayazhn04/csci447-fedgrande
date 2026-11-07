"""Run one Phase 2B job: (method, partition, fold, seed). Writes results/phase2b/runs/<job_id>/.

Usage: python -I src/fedgrande/phase2b/run_job.py --method fedavg_grande --partition natural --fold 0 --seed 42
       [--config configs/phase2b/core.json] [--out-root results/phase2b/runs] [--max-rounds N --max-epochs N] (smoke only)
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fedgrande.fl import comm, diagnostics as diag, train  # noqa: E402
from fedgrande.fl.metrics import binary_metrics, select_threshold  # noqa: E402
from fedgrande.fl.models import buffer_state, n_trainable, trainable_state  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402

PREP = ROOT / "data" / "derived" / "phase2b"


def maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def job_id(method, partition, fold, seed):
    return f"{method}__{partition}__f{fold}__s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--partition", default="none")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2b" / "core.json"))
    ap.add_argument("--out-root", default=str(ROOT / "results" / "phase2b" / "runs"))
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--save-state", help="optional path: torch.save the selected global/central model state_dict (Phase 2C)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    m = cfg["methods"][a.method]
    kind, proc = m["kind"], m["procedure"]
    assert (proc == "centralized") == (a.partition == "none"), "centralized methods use partition 'none'"
    tcfg = dict(cfg["training"])
    if a.max_rounds:
        tcfg["max_rounds"] = a.max_rounds
    if a.max_epochs:
        tcfg["max_epochs"] = a.max_epochs
    tcfg["model_kwargs"] = m.get("model_kwargs", cfg["model_kwargs"][kind])   # Phase 2D: per-method override
    tcfg["uplink"] = m.get("uplink", "fp32")
    torch.set_num_threads(cfg["torch"]["threads"])
    torch.set_float32_matmul_precision(cfg["torch"]["float32_matmul_precision"])

    jid = job_id(a.method, a.partition, a.fold, a.seed)
    out = os.path.join(a.out_root, jid)
    os.makedirs(out, exist_ok=True)
    t_start = time.perf_counter()

    P = np.load(PREP / f"prep_C2_outer{a.fold}.npz", allow_pickle=False)
    Z, roles, n_numeric = P["Z"], P["roles"], int(P["n_numeric"])
    part_file = PREP / ("partitions_dirichlet_v1.csv" if a.partition.startswith("dirichlet") else "partitions_v1.csv")  # amendment A1
    parts = pd.read_csv(part_file)
    from fedgrande.fl.features import load_modeling
    _, y, meta = load_modeling()
    g = meta.leakage_group_id.to_numpy()
    fit, val, test = roles == "fit", roles == "val", roles == "test"
    for A, B in ((fit, val), (fit, test), (val, test)):
        assert not set(g[A]) & set(g[B]), "leakage group crosses roles"
    w = train.class_weights(y[fit])
    client = parts[f"client_{a.partition}"].to_numpy() if a.partition != "none" else np.zeros(len(y), dtype=int)
    K = int(client.max() + 1)

    info, comm_rec, extra = {}, None, {}
    p = np.full(len(y), np.nan, dtype=np.float64)
    if proc == "centralized":
        model, info = train.centralized(kind, Z[fit], y[fit], Z[val], y[val], a.seed, tcfg, w)
        p[val | test] = train.predict(model, torch.as_tensor(Z[val | test]))
        final_models = [model]
    else:
        cd = [{"Xfit": Z[fit & (client == c)], "yfit": y[fit & (client == c)], "Xval": Z[val & (client == c)],
               "yval": y[val & (client == c)]} for c in range(K)]
        if proc == "local_only":
            models, infos = train.local_only(kind, cd, a.seed, tcfg, w)
            for c in range(K):
                rows = (val | test) & (client == c)
                if rows.any():
                    p[rows] = train.predict(models[c], torch.as_tensor(Z[rows]))
            info = {"clients": infos, "best_step": None, "steps_executed": None}
            pd.DataFrame(infos).to_csv(os.path.join(out, "local_clients.csv"), index=False)
            final_models = models
            comm_rec = {"bytes_per_round_total": 0, "note": "no communication"}
        else:
            model, info = train.fedavg(kind, cd, Z[val], y[val], a.seed, tcfg, w, n_numeric=n_numeric)
            p[val | test] = train.predict(model, torch.as_tensor(Z[val | test]))
            once = {"features_by_estimator": info["init_buffers"]["features_by_estimator"]} if kind == "grande" else {}
            comm_rec = comm.accounting(trainable_state(model), once, n_clients=int(sum(np.array(info["client_weights"]) > 0)),
                                       rounds_executed=info["steps_executed"], best_round=info["best_step"])
            comm_rec["v2"] = comm.accounting_v2(trainable_state(model), once, n_clients=int(sum(np.array(info["client_weights"]) > 0)),
                                                rounds_executed=info["steps_executed"], best_round=info["best_step"], uplink=tcfg["uplink"])
            pd.DataFrame(info.pop("rounds")).to_csv(os.path.join(out, "rounds.csv"), index=False)
            info.pop("init_buffers")
            final_models = [model]
    if a.save_state and proc in ("centralized", "fedavg"):
        os.makedirs(os.path.dirname(os.path.abspath(a.save_state)), exist_ok=True)
        torch.save(final_models[0].state_dict(), a.save_state)
    curve = info.pop("curve", None)
    if curve is not None:
        pd.DataFrame(curve).to_csv(os.path.join(out, "curve.csv"), index=False)

    # GRANDE structure of the selected model(s): selected original feature + threshold per node
    if kind == "grande":
        sels = [diag.selections(mm.split_index_array.detach(), mm.split_values.detach(), mm.features_by_estimator) for mm in final_models]
        np.savez_compressed(os.path.join(out, "selections.npz"), feat=np.stack([s[0] for s in sels]).astype(np.int16),
                            thr=np.stack([s[1] for s in sels]).astype(np.float32),
                            features_by_estimator=final_models[0].features_by_estimator.numpy().astype(np.int16))
        if proc == "local_only":
            trained = [c for c, i in enumerate(info["clients"]) if i["rule"] != "untrained_lt2_fit_rows"]
            extra["local_model_consensus"] = diag.consensus(np.stack([sels[c][0] for c in trained]), np.stack([sels[c][1] for c in trained]), None, n_numeric)

    thr = select_threshold(y[val], p[val])
    res = {
        "job_id": jid, "method": a.method, "kind": kind, "procedure": proc, "partition": a.partition, "fold": a.fold, "seed": a.seed,
        "split_id": f"C2/outer{a.fold}/splits_inner_v1", "partition_file": part_file.name, "partition_file_sha256": sha256_file(part_file),
        "prep_file_sha256": sha256_file(PREP / f"prep_C2_outer{a.fold}.npz"), "encoded_dim": int(Z.shape[1]),
        "n_clients": K, "n_fit": int(fit.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()),
        "class_weights": w.tolist(), "val_threshold": thr,
        "val_at_0.5": binary_metrics(y[val], p[val]), "test_at_0.5": binary_metrics(y[test], p[test]),
        "test_at_val_threshold": binary_metrics(y[test], p[test], thr),
        "best_step": info.get("best_step"), "steps_executed": info.get("steps_executed"), "best_val_auc": info.get("best_val_auc"),
        "info": info, "communication": comm_rec, **extra,
        "trainable_parameters": n_trainable(final_models[0]),
        "seconds_total": time.perf_counter() - t_start, "peak_rss_mb": maxrss_mb(),
        "resolved_training_config": tcfg, "config_file": a.config, "config_sha256": sha256_file(a.config),
        "smoke_overrides": {"max_rounds": a.max_rounds, "max_epochs": a.max_epochs},
        "meta": run_metadata(impl="dev", torch_threads=torch.get_num_threads()),
    }
    pd.DataFrame({"row_index": np.flatnonzero(val | test), "role": roles[val | test], "y": y[val | test],
                  "client": client[val | test], "wave": meta.wave.to_numpy()[val | test], "p1": p[val | test]}).to_csv(
        os.path.join(out, "predictions.csv"), index=False)
    write_json(os.path.join(out, "result.json"), res)
    t = res["test_at_0.5"]
    print(f"[{jid}] {res['seconds_total']:.0f}s best {res['best_step']}/{res['steps_executed']} valAUC {res['best_val_auc']} "
          f"testAUC {t['roc_auc']:.4f} F1@.5 {t['macro_f1']:.4f} RSS {res['peak_rss_mb']:.0f}MB")


if __name__ == "__main__":
    main()
