"""E0a, one side: run the GRANDE network of one commit on fixed inputs and fixed parameter tensors.

Usage: python e0a_forward.py --impl ref|dev --config configs/phase2/e0a_forward.json --workdir results/phase2/gate_a/e0a
The canonical inputs/states are created once by e0a_compare.py --prepare (numpy only), then each side writes
<impl>_<case>.npz with its own-init state, outputs (eval and train mode, dropout 0) and gradients of a fixed loss.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fedgrande.grande_bridge import BUFFERS, PARAMETERS, STATE_MAPPING, build_module, count_trainable  # noqa: E402
from fedgrande.provenance import run_metadata, write_json  # noqa: E402


def run_case(impl, case, fixed, cfg, workdir):
    torch.set_float32_matmul_precision(cfg["float32_matmul_precision"])
    inp = np.load(os.path.join(workdir, f"input_{case['name']}.npz"))
    canon = np.load(os.path.join(workdir, f"canonical_state_{case['name']}.npz"))
    X = torch.from_numpy(inp["X"])
    y = torch.from_numpy(inp["y"]).long()

    m = build_module(impl, number_of_variables=case["n_features"], depth=case["depth"], n_estimators=case["n_estimators"],
                     selected_variables=case["selected_variables"], missing_values=case["missing_values"],
                     random_seed=cfg["module_seed"], dropout=fixed["dropout"], data_subset_fraction=fixed["data_subset_fraction"],
                     bootstrap=fixed["bootstrap"], batch_size=fixed["batch_size"], objective=fixed["objective"],
                     number_of_classes=fixed["number_of_classes"], device=fixed["device"])
    own = {k: v.detach().cpu().numpy().copy() for k, v in m.state_dict().items()}
    res = {f"own_init__{k}": v for k, v in own.items()}
    res["meta_state_keys"] = np.array(sorted(own))
    res["meta_selected_variables"] = np.array(int(m.selected_variables))
    res["meta_trainable_params"] = np.array(count_trainable(m))

    def forward_all(tag):
        m.eval()
        with torch.no_grad():
            res[f"{tag}__eval_logits"] = m(X).numpy()
        m.train()  # dropout is 0, data_subset_fraction 1: training-mode forward must equal eval-mode forward
        m.zero_grad(set_to_none=True)
        out = m(X)
        res[f"{tag}__train_logits"] = out.detach().numpy()
        loss = F.cross_entropy(out, y, reduction="sum")
        loss.backward()
        for p in PARAMETERS:
            g = getattr(m, p).grad
            res[f"{tag}__grad__{p}"] = g.numpy().copy() if g is not None else np.full(1, np.nan)
        res[f"{tag}__loss"] = np.array(float(loss.item()))
        m.eval()

    forward_all("own_init")
    # load the canonical state through the explicit mapping (structural buffers are kept and compared separately)
    sd = m.state_dict()
    for ref_name, dev_name in STATE_MAPPING.items():
        name = ref_name if impl == "ref" else dev_name
        if name in canon.files:
            t = torch.from_numpy(canon[name])
            assert sd[name].shape == t.shape and sd[name].dtype == t.dtype, (name, sd[name].shape, t.shape, sd[name].dtype, t.dtype)
            sd[name] = t
    m.load_state_dict(sd, strict=True)
    forward_all("canonical")
    np.savez(os.path.join(workdir, f"{impl}_{case['name']}.npz"), **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["ref", "dev"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for case in cfg["cases"]:
        run_case(a.impl, case, cfg["fixed"], cfg, a.workdir)
        print(f"[{a.impl}] {case['name']} done")
    write_json(os.path.join(a.workdir, f"meta_{a.impl}.json"), run_metadata(impl=a.impl, torch_threads_used=torch.get_num_threads()))


if __name__ == "__main__":
    main()
