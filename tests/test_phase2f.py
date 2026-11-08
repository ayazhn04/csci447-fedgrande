"""Phase 2F (CGSF): consensus, node-slice masking under Adam, gates, aggregation, accounting, determinism, leakage."""
import unittest

import _common  # noqa: F401
import numpy as np
import torch

from fedgrande.fl import adaptive as A
from fedgrande.fl import train
from fedgrande.fl.models import build_grande, grande_optimizer, trainable_state


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


def cfg(ad, rounds=8):
    return {"max_rounds": rounds, "local_epochs": 1, "batch_size": 256, "patience": 100, "warmup": 100,
            "model_kwargs": {"n_estimators": 8, "depth": 3}, "adaptive": ad}


NODE = {"mode": "nodewise", "warmup": 2, "consecutive": 2, "c_node": 0.95, "d_max": 0.01, "s_max": 0.05, "eps": 1e-6}
GLOB = {"mode": "global", "warmup": 2, "consecutive": 2, "c_mean": 0.0, "c_unan": 0.0}


class TestConsensus(unittest.TestCase):
    def test_known_case(self):
        F = np.array([[[1, 2]], [[1, 3]], [[1, 2]], [[5, 2]]])
        np.testing.assert_allclose(A.node_consensus(F), [[0.75, 0.75]])
        np.testing.assert_allclose(A.node_consensus(np.zeros((5, 2, 3))), np.ones((2, 3)))


class TestMasking(unittest.TestCase):
    def test_adam_cannot_move_masked_slice_and_neighbours_move(self):
        m = build_grande(12, 1, n_estimators=4, depth=3)
        opt = grande_optimizer(m)
        X, y = torch.randn(64, 12), torch.randint(0, 2, (64,))
        w = torch.tensor([0.5, 0.5])
        fm = torch.zeros(4, 7, dtype=torch.bool)
        fm[1, 3] = True
        tm = fm.clone()
        ref = {k: v.clone() for k, v in trainable_state(m).items()}
        A.masked_epoch(m, opt, X, y, w, torch.Generator().manual_seed(0), 16, fm, tm, ref)
        A.masked_epoch(m, opt, X, y, w, torch.Generator().manual_seed(1), 16, fm, tm, ref)   # Adam state carried over too
        self.assertTrue(torch.equal(m.split_index_array.detach()[1, 3], ref["split_index_array"][1, 3]))
        self.assertTrue(torch.equal(m.split_values.detach()[1, 3], ref["split_values"][1, 3]))
        self.assertFalse(torch.equal(m.split_index_array.detach()[1, 2], ref["split_index_array"][1, 2]))
        self.assertFalse(torch.equal(m.split_values.detach()[0, 3], ref["split_values"][0, 3]))


class TestAdaptiveRuns(unittest.TestCase):
    def test_nodewise_gates_and_invariants(self):
        cd, Xv, yv, w = toy()
        g, info = A.fedavg_adaptive(cd, Xv, yv, 3, cfg(NODE, rounds=10), w, 12, feature_sd=np.ones(12))
        fr, tr = info["feat_freeze_round"], info["full_freeze_round"]
        self.assertTrue(((fr == -1) | (fr > NODE["warmup"] + NODE["consecutive"] - 1)).all())      # nothing before warm-up + streak
        self.assertTrue(((tr == -1) | ((fr != -1) & (tr > fr))).all())                               # threshold strictly after feature
        self.assertGreater(int((fr != -1).sum()), 0)
        R = info["rounds"]
        for i, rec in enumerate(R):
            self.assertEqual(rec["bytes_down"], rec["bytes_down_sia"] + rec["bytes_down_sv"] + rec["bytes_down_leaf"] + rec["bytes_down_ew"])
        # frozen nodes stop contributing bytes after one final broadcast
        nf_prev = R[0]["frac_feat_frozen"]
        self.assertLessEqual(R[-1]["bytes_up_sia"], R[0]["bytes_up_sia"])
        self.assertEqual(R[0]["bytes_up_sia"], 3 * 4 * 8 * 7 * 10)

    def test_frozen_slices_unchanged_over_rounds(self):
        cd, Xv, yv, w = toy()
        c = cfg(NODE, rounds=10)
        g, info = A.fedavg_adaptive(cd, Xv, yv, 3, c, w, 12, feature_sd=np.ones(12))
        # rerun with snapshots of early state: nodes frozen by round k equal at end
        g5, info5 = A.fedavg_adaptive(cd, Xv, yv, 3, {**c, "max_rounds": 6}, w, 12, feature_sd=np.ones(12))
        fz = torch.as_tensor(info5["feat_freeze_round"] != -1)
        fz6 = torch.as_tensor((info5["feat_freeze_round"] != -1) & (info5["feat_freeze_round"] < 6))
        self.assertTrue(torch.equal(g.split_index_array.detach()[fz6], g5.split_index_array.detach()[fz6]))

    def test_global_trigger_and_accounting(self):
        cd, Xv, yv, w = toy()
        g, info = A.fedavg_adaptive(cd, Xv, yv, 3, cfg(GLOB, rounds=7), w, 12, feature_sd=np.ones(12))
        self.assertEqual(info["trigger_round"], 4)                  # rounds 3,4 satisfy (warm-up 2, streak 2)
        R = info["rounds"]
        full = R[0]["bytes_down"]
        self.assertEqual(R[4]["bytes_down"], full)                  # round 5: final broadcast of frozen structure
        self.assertEqual(R[4]["bytes_up_sia"] + R[4]["bytes_up_sv"], 0)
        self.assertEqual(R[5]["bytes_down_sia"] + R[5]["bytes_down_sv"], 0)
        self.assertTrue((info["full_freeze_round"] == 4).all())

    def test_determinism_and_prefreeze_identity_with_fedavg(self):
        cd, Xv, yv, w = toy()
        a, ia = A.fedavg_adaptive(cd, Xv, yv, 9, cfg(NODE, rounds=6), w, 12, feature_sd=np.ones(12))
        b, ib = A.fedavg_adaptive(cd, Xv, yv, 9, cfg(NODE, rounds=6), w, 12, feature_sd=np.ones(12))
        np.testing.assert_array_equal(ia["feat_freeze_round"], ib["feat_freeze_round"])
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]))
        # rounds 1..warmup+consecutive-1 cannot freeze -> identical to ordinary FedAvg
        k = NODE["warmup"] + NODE["consecutive"]
        c0 = {**cfg(NODE, rounds=k)}
        c0.pop("adaptive")
        f0, _ = train.fedavg("grande", cd, Xv, yv, 9, c0, w, n_numeric=12)
        ad, _ = A.fedavg_adaptive(cd, Xv, yv, 9, cfg(NODE, rounds=k), w, 12, feature_sd=np.ones(12))
        for kk, v in f0.state_dict().items():
            self.assertTrue(torch.equal(v, ad.state_dict()[kk]), kk)

    def test_validation_data_does_not_affect_freezing(self):
        cd, Xv, yv, w = toy()
        _, ia = A.fedavg_adaptive(cd, Xv, yv, 4, cfg(NODE, rounds=8), w, 12, feature_sd=np.ones(12))
        _, ib = A.fedavg_adaptive(cd, Xv * 3, 1 - yv, 4, cfg(NODE, rounds=8), w, 12, feature_sd=np.ones(12))
        np.testing.assert_array_equal(ia["feat_freeze_round"], ib["feat_freeze_round"])
        np.testing.assert_array_equal(ia["full_freeze_round"], ib["full_freeze_round"])

    def test_threshold_gate_boundary(self):
        cd, Xv, yv, w = toy()
        strict = {**NODE, "d_max": -1.0}                             # impossible threshold gate -> no full freezes
        _, i1 = A.fedavg_adaptive(cd, Xv, yv, 3, cfg(strict, rounds=8), w, 12, feature_sd=np.ones(12))
        self.assertTrue((i1["full_freeze_round"] == -1).all())
        loose = {**NODE, "d_max": 1e9, "s_max": 1e9}                 # always-true gate -> full freeze exactly `consecutive` rounds after feature freeze
        _, i2 = A.fedavg_adaptive(cd, Xv, yv, 3, cfg(loose, rounds=10), w, 12, feature_sd=np.ones(12))
        fr, tr = i2["feat_freeze_round"], i2["full_freeze_round"]
        both = (fr != -1) & (tr != -1)
        self.assertTrue(both.any())
        np.testing.assert_array_equal(tr[both], fr[both] + loose["consecutive"])

    def test_aggregation_mixed_mask(self):
        s1 = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        s2 = {"a": torch.tensor([[3.0, 6.0], [5.0, 8.0]])}
        new = train.weighted_average([s1, s2], [1, 1])
        mask = torch.tensor([True, False])
        before = torch.tensor([[9.0, 9.0], [0.0, 0.0]])
        new["a"][mask] = before[mask]
        torch.testing.assert_close(new["a"], torch.tensor([[9.0, 9.0], [4.0, 6.0]]))


if __name__ == "__main__":
    unittest.main()
