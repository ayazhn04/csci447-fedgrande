"""E0b step 2: train one implementation's own training loop on the shared prepared matrix.

ref (07f7278b): ``GRANDE.GRANDE.GRANDE.fit``. Two harness-level controls, applied at runtime only (no source edit):
  (1) the instance's ``preprocess`` is replaced by an identity that returns the already-prepared float32 matrix,
      so both versions see exactly the same inputs;
  (2) ``torch.compile`` is replaced by identity for this process, because 07f7278b compiles unconditionally
      (mode="reduce-overhead"), whereas db79c860 is run with compile_model=False.
dev (db79c860): ``GRANDE.training.train_grande`` (the loop GRANDEClassifier.fit calls), compile_model=False.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fedgrande.grande_runner import EPOCH_RE, _Tee  # noqa: E402
from fedgrande.paths import CONFIGS, DERIVED, RESULTS  # noqa: E402
from fedgrande.provenance import run_metadata, write_json  # noqa: E402


@torch.no_grad()
def proba(model, X, bs=256):
    model.eval()
    return np.concatenate([F.softmax(model(torch.from_numpy(X[i:i + bs])), dim=-1)[:, 1].numpy() for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["ref", "dev"])
    a = ap.parse_args()
    cfg = json.load(open(CONFIGS / "e0b_training.json"))
    D = np.load(DERIVED / f"e0b_prepared_{cfg['scheme']}_outer{cfg['outer_fold']}.npz")
    hp = dict(cfg["matched_hyperparameters"])
    outdir = RESULTS / "gate_a" / "e0b"
    os.makedirs(outdir, exist_ok=True)
    log = io.StringIO()
    harness_controls = []

    if a.impl == "ref":
        import importlib

        ref_mod = importlib.import_module("GRANDE.GRANDE")

        torch.compile = lambda model, *args, **kwargs: model  # control (2)
        harness_controls.append("torch.compile replaced by identity in this process (07f7278b compiles unconditionally)")
        torch.set_float32_matmul_precision(cfg["float32_matmul_precision"])  # ref also sets 'high' at import
        params = {**hp, **cfg["ref_only_hyperparameters"], "random_seed": cfg["seed"]}
        est = ref_mod.GRANDE(params=params)
        num_idx, enc_idx = D["num_columns_indices"].tolist(), D["encoded_columns_indices"].tolist()

        def identity_preprocess(X, y=None, is_train=False, **kwargs):  # control (1)
            if is_train:
                est.num_columns_indices, est.encoded_columns_indices = num_idx, enc_idx
                est.mean, est.std = float(np.mean(y)), float(np.std(y))
            X = np.asarray(X, dtype=np.float32)
            return (X, np.asarray(y, dtype=np.float32)) if y is not None else X

        est.preprocess = identity_preprocess
        harness_controls.append("instance.preprocess replaced by identity on the shared prepared matrix")
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            est.fit(D["X_fit"], D["y_fit"], X_val=D["X_val"], y_val=D["y_val"])
        fit_s = time.perf_counter() - t0
        model = est.model
        p_val, p_test = est.predict_proba(D["X_val"])[:, 1], est.predict_proba(D["X_test"])[:, 1]
        p_test_direct = proba(model, D["X_test"])
    else:
        from GRANDE.training import train_grande

        torch.set_float32_matmul_precision(cfg["float32_matmul_precision"])
        hp_dev = {**hp, "random_seed": cfg["seed"]}
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)):
            res = train_grande(hp=hp_dev, objective="binary", X_train=D["X_fit"], y_train=D["y_fit"], X_val=D["X_val"],
                               y_val=D["y_val"], num_columns_indices=D["num_columns_indices"].tolist(),
                               encoded_columns_indices=D["encoded_columns_indices"].tolist(), y_mean=float(np.mean(D["y_fit"])),
                               y_std=float(np.std(D["y_fit"])), device="cpu", compile_model=cfg["compile"],
                               start_time=time.monotonic(), time_limit=None)
        fit_s = time.perf_counter() - t0
        model = res.model
        p_val, p_test = proba(model, D["X_val"]), proba(model, D["X_test"])
        p_test_direct = p_test

    curve = [{"epoch": int(m[0]), "train_loss": float(m[1]), "val_loss": float(m[2])} for m in EPOCH_RE.findall(log.getvalue())]
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(outdir / f"{a.impl}_outputs.npz", p_val=p_val, p_test=p_test, p_test_direct=p_test_direct,
             **{f"state__{k}": v for k, v in state.items()})
    open(outdir / f"{a.impl}_train_log.txt", "w").write(log.getvalue())
    write_json(outdir / f"{a.impl}_run.json", {"impl": a.impl, "fit_seconds": fit_s, "curve": curve,
                                                 "harness_controls": harness_controls, "config": cfg,
                                                 "torch_threads": torch.get_num_threads(), "meta": run_metadata(impl=a.impl)})
    print(f"[{a.impl}] fit {fit_s:.1f}s, epochs logged {len(curve)}")


if __name__ == "__main__":
    main()
