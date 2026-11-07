"""E0b step 1 (fedgrande-dev): build ONE prepared float32 matrix shared by both implementations."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedgrande.data import load_xy  # noqa: E402
from fedgrande.paths import CONFIGS, DERIVED, RESULTS  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402


def main():
    from GRANDE.preprocessing import GRANDEPreprocessor

    cfg = json.load(open(CONFIGS / "e0b_training.json"))
    X, y, _ = load_xy()
    inner = pd.read_csv(DERIVED / "splits_inner_v1.csv")
    sel = inner[(inner.scheme == cfg["scheme"]) & (inner.outer_fold == cfg["outer_fold"])].sort_values("row_index")
    idx = {r: sel.loc[sel.role == r, "row_index"].to_numpy() for r in ("fit", "val", "test")}
    pp_args = {k: v for k, v in cfg["preprocessing"].items() if k != "implementation"}
    pre = GRANDEPreprocessor(**pp_args)
    Xf = pre.fit_transform(X.iloc[idx["fit"]], y[idx["fit"]].astype(np.float32))  # fitted on fit rows only
    Xv, Xt = pre.transform(X.iloc[idx["val"]]), pre.transform(X.iloc[idx["test"]])
    out = DERIVED / f"e0b_prepared_{cfg['scheme']}_outer{cfg['outer_fold']}.npz"
    np.savez(out, X_fit=Xf.astype(np.float32), y_fit=y[idx["fit"]].astype(np.float32), X_val=Xv.astype(np.float32),
             y_val=y[idx["val"]].astype(np.float32), X_test=Xt.astype(np.float32), y_test=y[idx["test"]].astype(np.float32),
             idx_fit=idx["fit"], idx_val=idx["val"], idx_test=idx["test"],
             num_columns_indices=np.array(pre.num_columns_indices), encoded_columns_indices=np.array(pre.encoded_columns_indices))
    write_json(RESULTS / "gate_a" / "e0b" / "prepared_matrix.json", {
        "file": str(out), "sha256": sha256_file(out), "shape_fit": Xf.shape, "shape_val": Xv.shape, "shape_test": Xt.shape,
        "nan_count": int(np.isnan(Xf).sum() + np.isnan(Xv).sum() + np.isnan(Xt).sum()), "preprocessing": cfg["preprocessing"],
        "meta": run_metadata(impl="dev")})
    print("prepared", Xf.shape, Xv.shape, Xt.shape, "nan:", int(np.isnan(Xf).sum()))


if __name__ == "__main__":
    main()
