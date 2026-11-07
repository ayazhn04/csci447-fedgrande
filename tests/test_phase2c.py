"""Phase 2C: intercept calibration, restricted personalization, structure preservation, evaluation conventions."""
import json
import unittest

import _common  # noqa: F401
import numpy as np
import pandas as pd
import torch

from fedgrande.fl import personalize as P
from fedgrande.fl.metrics import binary_metrics
from fedgrande.fl.models import build_grande
from fedgrande.paths import ROOT

CFG = {"lr": {"leaf_classes_array": 0.01, "estimator_weights": 0.005}, "betas": [0.9, 0.95], "max_epochs": 4, "patience": 5,
       "min_delta": 1e-4, "batch_size": 256}


def toy(n=60, d=12, seed=0, pos_frac=0.4):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, d)).astype(np.float32)
    y = (r.random(n) < pos_frac).astype(np.int64)
    X[y == 1, 0] += 1.5
    return X, y


class TestIntercept(unittest.TestCase):
    def test_finite_for_single_class_clients(self):
        z = np.zeros(200)
        b_neg = P.fit_intercept(z, np.zeros(200), lam=1.0)
        b_pos = P.fit_intercept(z, np.ones(200), lam=1.0)
        self.assertTrue(np.isfinite(b_neg) and np.isfinite(b_pos))
        self.assertLess(b_neg, -3)
        self.assertGreater(b_pos, 3)

    def test_matches_known_solution_and_stationarity(self):
        z = np.linspace(-2, 2, 101)
        y = (np.arange(101) % 3 == 0).astype(float)
        b = P.fit_intercept(z, y, lam=1.0)
        p = 1 / (1 + np.exp(-(z + b)))
        self.assertAlmostEqual(float(np.sum(p - y) + 1.0 * b), 0.0, places=8)

    def test_uses_only_given_rows_and_preserves_ranking(self):
        p0 = np.array([0.1, 0.3, 0.2, 0.9])
        b = 1.7
        p1 = P.apply_intercept(p0, b)
        np.testing.assert_array_equal(np.argsort(p0), np.argsort(p1))


class TestRestrictedPersonalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = build_grande(12, 3, n_estimators=8, depth=3)
        X, y = toy()
        cls.Xf, cls.yf, cls.Xv, cls.yv = X[:40], y[:40], X[40:], y[40:]

    def test_p2_changes_only_leaves(self):
        m, info = P.personalize(self.g, "P2", self.Xf, self.yf, self.Xv, self.yv, 1, 0, {**CFG, "patience": 100, "min_delta": -1})
        v = P.verify_structure(self.g, m, "P2")
        self.assertTrue(v["only_intended_changed"])
        self.assertEqual(v["changed_params"], ["leaf_classes_array"])

    def test_p3_changes_only_leaves_and_weights(self):
        m, _ = P.personalize(self.g, "P3", self.Xf, self.yf, self.Xv, self.yv, 1, 0, {**CFG, "patience": 100, "min_delta": -1})
        v = P.verify_structure(self.g, m, "P3")
        self.assertTrue(v["only_intended_changed"])
        self.assertEqual(set(v["changed_params"]), {"leaf_classes_array", "estimator_weights"})
        for k in ("split_index_array", "split_values", "features_by_estimator", "path_identifier_list", "internal_node_index_list"):
            self.assertTrue(torch.equal(m.state_dict()[k], self.g.state_dict()[k]), k)

    def test_global_model_not_mutated_and_deterministic(self):
        before = {k: v.clone() for k, v in self.g.state_dict().items()}
        a, _ = P.personalize(self.g, "P3", self.Xf, self.yf, self.Xv, self.yv, 5, 2, CFG)
        b, _ = P.personalize(self.g, "P3", self.Xf, self.yf, self.Xv, self.yv, 5, 2, CFG)
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]), k)
        for k, v in self.g.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)

    def test_single_class_validation_fallback(self):
        yv = np.zeros_like(self.yv)
        _, info = P.personalize(self.g, "P2", self.Xf, self.yf, self.Xv, yv, 1, 0, CFG)
        self.assertEqual(info["rule"], "val_loss")
        _, info2 = P.personalize(self.g, "P2", self.Xf, self.yf, self.Xv, self.yv, 1, 0, CFG)
        self.assertEqual(info2["rule"], "val_auc")

    def test_validation_and_test_rows_do_not_affect_updates(self):
        cfg = {**CFG, "patience": 100, "min_delta": -1}           # no early selection: compare the trained trajectories
        a, _ = P.personalize(self.g, "P3", self.Xf, self.yf, self.Xv, self.yv, 9, 1, cfg)
        b, _ = P.personalize(self.g, "P3", self.Xf, self.yf, self.Xv * 3 + 1, 1 - self.yv, 9, 1, cfg)
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]), k)


class TestEvaluationConventions(unittest.TestCase):
    def test_single_class_auc_undefined(self):
        self.assertTrue(np.isnan(binary_metrics(np.zeros(10), np.linspace(0, 1, 10))["roc_auc"]))


RUNS = ROOT / "results" / "phase2c" / "runs"
CK = ROOT / "data" / "derived" / "phase2c" / "global_ckpt" / "fedavg_grande__dirichlet0.1__f0__s42.pt"


@unittest.skipUnless(CK.exists(), "checkpoint not generated yet")
class TestCheckpointAndOutputs(unittest.TestCase):
    def test_checkpoint_loads_and_reproduces_phase2b(self):
        D = np.load(ROOT / "data" / "derived" / "phase2b" / "prep_C2_outer0.npz")
        g = build_grande(D["Z"].shape[1], 42, n_estimators=256, depth=5)
        g.load_state_dict(torch.load(CK, map_location="cpu"))
        ref = pd.read_csv(ROOT / "results" / "phase2b" / "runs" / "fedavg_grande__dirichlet0.1__f0__s42" / "predictions.csv")
        roles = D["roles"]
        vt = (roles == "val") | (roles == "test")
        p = P.predict(g, torch.as_tensor(D["Z"][vt]))
        np.testing.assert_array_equal(p.astype(np.float32), ref.p1.to_numpy().astype(np.float32))  # outputs are float32

    @unittest.skipUnless((RUNS / "f0_s42" / "result.json").exists(), "final run not available")
    def test_saved_outputs_consistent(self):
        r = json.load(open(RUNS / "f0_s42" / "result.json"))
        self.assertEqual(r["additional_communication_bytes"], 0)
        self.assertTrue(r["structure_all_unchanged"] and r["only_intended_changed_all"])
        pr = pd.read_csv(RUNS / "f0_s42" / "predictions.csv")
        roles = np.load(ROOT / "data" / "derived" / "phase2b" / "prep_C2_outer0.npz")["roles"]
        self.assertTrue(set(pr.role) <= {"val", "test"})
        self.assertTrue((roles[pr.row_index.to_numpy()] == pr.role.to_numpy()).all())
        part = pd.read_csv(ROOT / "data" / "derived" / "phase2b" / "partitions_dirichlet_v1.csv")
        np.testing.assert_array_equal(part["client_dirichlet0.1"].to_numpy()[pr.row_index.to_numpy()], pr.client.to_numpy())


if __name__ == "__main__":
    unittest.main()
