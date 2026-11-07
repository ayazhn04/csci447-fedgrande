"""Phase 2E: structure-freezing schedule, optimizer exclusion, communication accounting, determinism."""
import unittest

import _common  # noqa: F401
import numpy as np
import torch

from fedgrande.fl import comm, compress, train
from fedgrande.fl.models import build_grande, buffer_state, trainable_state

S1, S2 = ["split_index_array"], ["split_index_array", "split_values"]


def toy(n_clients=3, n=48, d=12, seed=0):
    r = np.random.default_rng(seed)
    cd = []
    for c in range(n_clients):
        X = r.standard_normal((n, d)).astype(np.float32)
        y = (X[:, 0] + 0.2 * c > 0).astype(np.int64)
        cd.append({"Xfit": X, "yfit": y, "Xval": X[:12], "yval": y[:12]})
    Xv, yv = np.concatenate([c["Xval"] for c in cd]), np.concatenate([c["yval"] for c in cd])
    w = train.class_weights(np.concatenate([c["yfit"] for c in cd]))
    return cd, Xv, yv, w


def cfg(freeze=None, rounds=4, snaps=None):
    c = {"max_rounds": rounds, "local_epochs": 1, "batch_size": 256, "patience": 100, "warmup": 100,
         "model_kwargs": {"n_estimators": 8, "depth": 3}, "uplink": "fp32"}
    if freeze:
        c["freeze"] = freeze
    if snaps:
        c["snapshot_rounds"] = snaps
    return c


class TestFreeze(unittest.TestCase):
    def run_fz(self, params, up="fp32", seed=5):
        cd, Xv, yv, w = toy()
        return train.fedavg("grande", cd, Xv, yv, seed, cfg({"after_round": 2, "params": params, "uplink_after": up}, rounds=4, snaps=[2, 4]), w, n_numeric=12)

    def test_f1_freezes_only_split_index(self):
        g, info = self.run_fz(S1)
        s2, s4 = info["snapshots"][2], info["snapshots"][4]
        self.assertTrue(torch.equal(s2["split_index_array"], s4["split_index_array"]))
        for k in ("split_values", "leaf_classes_array", "estimator_weights"):
            self.assertFalse(torch.equal(s2[k], s4[k]), k)

    def test_f2_f3_freeze_structure_and_update_outputs(self):
        for up in ("fp32", "int8"):
            g, info = self.run_fz(S2, up)
            s2, s4 = info["snapshots"][2], info["snapshots"][4]
            for k in S2:
                self.assertTrue(torch.equal(s2[k], s4[k]), (up, k))
            for k in ("leaf_classes_array", "estimator_weights"):
                self.assertFalse(torch.equal(s2[k], s4[k]), (up, k))
            r = info["rounds"]
            self.assertEqual(r[1]["active_groups"].split("|"), list(trainable_state(g)))       # round 2: all active
            self.assertEqual(set(r[2]["active_groups"].split("|")), {"leaf_classes_array", "estimator_weights"})
            self.assertEqual(r[2]["move_rel__split_values"], 0.0)

    def test_freeze_exactly_after_round(self):
        g, info = self.run_fz(S2)
        mv = [r["move_rel__split_index_array"] for r in info["rounds"]]
        self.assertTrue(mv[0] > 0 and mv[1] > 0 and mv[2] == 0 and mv[3] == 0)

    def test_frozen_no_grad_and_optimizer_exclusion(self):
        from fedgrande.fl.models import grande_optimizer_subset
        m = build_grande(12, 1, n_estimators=8, depth=3)
        opt = grande_optimizer_subset(m, ["leaf_classes_array", "estimator_weights"])
        ids = {id(p) for gr in opt.param_groups for p in gr["params"]}
        self.assertEqual(ids, {id(m.leaf_classes_array), id(m.estimator_weights)})
        self.assertEqual([gr["lr"] for gr in opt.param_groups], [0.005, 0.01])

    def test_deterministic_and_f0_path_unchanged(self):
        a, _ = self.run_fz(S2, seed=9)
        b, _ = self.run_fz(S2, seed=9)
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]))
        cd, Xv, yv, w = toy()
        g0, _ = train.fedavg("grande", cd, Xv, yv, 9, cfg(rounds=2), w, n_numeric=12)
        gf, _ = train.fedavg("grande", cd, Xv, yv, 9, cfg({"after_round": 2, "params": S2, "uplink_after": "fp32"}, rounds=2), w, n_numeric=12)
        for k, v in g0.state_dict().items():                  # identical until the freeze round
            self.assertTrue(torch.equal(v, gf.state_dict()[k]), k)

    def test_buffers_identical(self):
        g, _ = self.run_fz(S2)
        for k, v in buffer_state(build_grande(12, 5, n_estimators=8, depth=3)).items():
            self.assertTrue(torch.equal(v, buffer_state(g)[k]))


class TestAccounting(unittest.TestCase):
    def test_per_round_bytes(self):
        cd, Xv, yv, w = toy()
        g, info = train.fedavg("grande", cd, Xv, yv, 5, cfg({"after_round": 2, "params": S2, "uplink_after": "int8"}, rounds=4), w, n_numeric=12)
        st = trainable_state(g)
        full = comm.tensor_bytes(st)
        act = {k: st[k] for k in ("leaf_classes_array", "estimator_weights")}
        r = info["rounds"]
        self.assertEqual(r[0]["bytes_down"], 3 * full)
        self.assertEqual(r[0]["bytes_up"], 3 * full)
        self.assertEqual(r[2]["bytes_down"], 3 * full)                       # round 3 = after_round+1: structure broadcast once
        self.assertEqual(r[2]["bytes_up"], 3 * compress.payload_bytes(act))
        self.assertEqual(r[3]["bytes_down"], 3 * comm.tensor_bytes(act))     # afterwards: active tensors only
        tot = sum(x["bytes_down"] + x["bytes_up"] for x in r)
        self.assertEqual(tot, sum(x["bytes_down"] for x in r) + sum(x["bytes_up"] for x in r))


if __name__ == "__main__":
    unittest.main()
