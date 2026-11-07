"""Phase 2B invariants: features, federated-safe preprocessing, partitions, FedAvg mechanics, diagnostics, comms."""
import inspect
import unittest

import _common  # noqa: F401
import numpy as np
import pandas as pd
import torch

from fedgrande.fl import comm, diagnostics as diag, partitions as P, train
from fedgrande.fl.features import load_modeling, schema
from fedgrande.fl.models import build_grande, build_mlp, buffer_state, trainable_state
from fedgrande.fl.preprocess import FederatedPreprocessor, aggregate, client_statistics
from fedgrande.paths import ROOT

PREP = ROOT / "data" / "derived" / "phase2b"


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.meta = load_modeling()
        cls.S = schema(cls.X)
        cls.order = {c: list(cls.X[c].cat.categories) for c in cls.S["categorical"]}
        cls.parts = pd.read_csv(PREP / "partitions_v1.csv")


class TestFeatures(Base):
    def test_wave_and_labels_absent(self):
        for c in ("wave", "match", "decision", "decision_o"):
            self.assertNotIn(c, self.X.columns)
        self.assertEqual(self.S["n_raw"], 119)
        self.assertEqual((self.S["n_numeric"], self.S["n_categorical"]), (58, 61))


class TestPreprocessing(Base):
    def test_no_target_in_api(self):
        for f in (client_statistics, aggregate, FederatedPreprocessor.fit_from_clients, FederatedPreprocessor.fit_pooled,
                  FederatedPreprocessor.transform):
            self.assertFalse({"y", "target", "labels"} & set(inspect.signature(f).parameters))

    def test_aggregated_equals_pooled_training_only(self):
        roles = np.load(PREP / "prep_C2_outer0.npz")["roles"]
        fit = roles == "fit"
        nat = self.parts.client_natural.to_numpy()
        fed = FederatedPreprocessor(self.S["numeric"], self.S["categorical"], self.order).fit_from_clients([self.X[fit & (nat == c)] for c in range(21)])
        A = self.X.loc[fit, self.S["numeric"]].to_numpy(float)
        np.testing.assert_allclose(fed.params_["mean"], np.nanmean(A, 0), rtol=0, atol=1e-9)
        sd = np.nanstd(A, 0)
        np.testing.assert_allclose(fed.params_["sd"], np.where(sd < 1e-6, 1.0, sd), rtol=1e-9, atol=1e-9)
        cache = np.load(PREP / "prep_C2_outer0.npz")
        np.testing.assert_allclose(cache["mean"], fed.params_["mean"], atol=1e-12)

    def test_fit_uses_training_rows_only(self):
        roles = np.load(PREP / "prep_C2_outer0.npz")["roles"]
        fit = roles == "fit"
        X2 = self.X.copy()
        X2.loc[~fit, self.S["numeric"]] = 1e6                     # corrupt validation/test rows
        a = FederatedPreprocessor(self.S["numeric"], self.S["categorical"], self.order).fit_pooled(self.X[fit])
        b = FederatedPreprocessor(self.S["numeric"], self.S["categorical"], self.order).fit_pooled(X2[fit])
        np.testing.assert_array_equal(a.params_["mean"], b.params_["mean"])
        Z = np.load(PREP / "prep_C2_outer0.npz")["Z"]
        np.testing.assert_allclose(Z[fit], a.transform(self.X[fit]), atol=1e-5)

    def test_partition_invariant_schema(self):
        roles = np.load(PREP / "prep_C2_outer1.npz")["roles"]
        fit = roles == "fit"
        fps = []
        for col in ("client_natural", "client_iid"):
            cl = self.parts[col].to_numpy()
            fps.append(FederatedPreprocessor(self.S["numeric"], self.S["categorical"], self.order).fit_from_clients([self.X[fit & (cl == c)] for c in range(21)]))
        self.assertEqual(fps[0].feature_names_, fps[1].feature_names_)
        np.testing.assert_allclose(fps[0].transform(self.X), fps[1].transform(self.X), atol=1e-5)

    def test_unseen_and_missing_map_to_unk(self):
        sub = self.X.iloc[:300]
        pp = FederatedPreprocessor(self.S["numeric"], self.S["categorical"], self.order).fit_pooled(sub)
        row = self.X.iloc[[0]].copy()
        unseen = next(v for v in self.order["field"] if v not in set(sub["field"].astype(str)))
        row["field"] = pd.Categorical([unseen], categories=self.order["field"])
        z, u = pp.transform(row, return_unseen=True)
        s, e = pp.blocks_["field"]
        self.assertEqual(u.get("field"), 1)
        self.assertEqual(float(z[0, s:e].sum()), 0.0)
        self.assertFalse(np.isnan(z).any())


class TestPartitions(Base):
    def test_natural_is_wave(self):
        waves = np.sort(self.meta.wave.unique())
        np.testing.assert_array_equal(waves[self.parts.client_natural.to_numpy()], self.meta.wave.to_numpy())

    def test_groups_never_cross_clients(self):
        for col in ("client_natural", "client_iid"):
            self.assertEqual(int(self.parts.groupby("leakage_group_id")[col].nunique().max()), 1)

    def test_iid_sizes_and_determinism(self):
        iid = P.iid_size_matched(self.meta, self.y, seed=7)
        np.testing.assert_array_equal(iid, self.parts.client_iid.to_numpy())
        np.testing.assert_array_equal(iid, P.iid_size_matched(self.meta, self.y, seed=7))
        dev = np.bincount(iid) - np.bincount(self.parts.client_natural.to_numpy())
        self.assertLessEqual(int(np.abs(dev).max()), 1)
        rates = pd.Series(self.y).groupby(iid).mean()
        self.assertLess(float(rates.max() - rates.min()), 0.02)

    def test_no_group_overlap_between_roles(self):
        g = self.meta.leakage_group_id.to_numpy()
        for k in range(5):
            roles = np.load(PREP / f"prep_C2_outer{k}.npz")["roles"]
            for a, b in (("fit", "test"), ("fit", "val"), ("val", "test")):
                self.assertFalse(set(g[roles == a]) & set(g[roles == b]), (k, a, b))


def toy_clients(n_clients=3, n=40, d=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_clients):
        X = rng.standard_normal((n, d)).astype(np.float32)
        y = (X[:, 0] + 0.3 * c > 0).astype(np.int64)
        out.append({"Xfit": X, "yfit": y, "Xval": X[:10], "yval": y[:10]})
    return out


CFG = {"max_rounds": 1, "local_epochs": 1, "batch_size": 256, "patience": 10, "warmup": 10, "model_kwargs": {}}


class TestFedAvg(unittest.TestCase):
    def test_weighted_average(self):
        s = [{"a": torch.tensor([1.0, 2.0])}, {"a": torch.tensor([3.0, 6.0])}, {"a": torch.tensor([100.0, 100.0])}]
        out = train.weighted_average(s, [1, 3, 0])
        torch.testing.assert_close(out["a"], torch.tensor([2.5, 5.0]))

    def test_identical_init_and_buffers(self):
        a, b = build_grande(30, 42, n_estimators=8, depth=3), build_grande(30, 42, n_estimators=8, depth=3)
        for k, v in a.state_dict().items():
            self.assertTrue(torch.equal(v, b.state_dict()[k]), k)

    def test_one_round_changes_params_and_keeps_buffers(self):
        cd = toy_clients()
        Xv = np.concatenate([c["Xval"] for c in cd])
        yv = np.concatenate([c["yval"] for c in cd])
        w = train.class_weights(np.concatenate([c["yfit"] for c in cd]))
        init = build_grande(12, 5, n_estimators=8, depth=3)
        cfg = {**CFG, "model_kwargs": {"n_estimators": 8, "depth": 3}}
        g, info = train.fedavg("grande", cd, Xv, yv, 5, cfg, w, n_numeric=12)
        changed = any(not torch.equal(p, trainable_state(init)[k]) for k, p in trainable_state(g).items())
        self.assertTrue(changed)
        for k, v in buffer_state(init).items():
            self.assertTrue(torch.equal(v, buffer_state(g)[k]), k)

    def test_validation_not_used_for_updates(self):
        cd = toy_clients()
        w = train.class_weights(np.concatenate([c["yfit"] for c in cd]))
        Xv = np.concatenate([c["Xval"] for c in cd])
        yv = np.concatenate([c["yval"] for c in cd])
        g1, _ = train.fedavg("mlp", cd, Xv, yv, 3, CFG, w)
        g2, _ = train.fedavg("mlp", cd, Xv * -5.0, 1 - yv, 3, CFG, w)      # different validation data/labels
        for k, v in trainable_state(g1).items():
            self.assertTrue(torch.equal(v, trainable_state(g2)[k]), k)

    def test_fedavg_one_client_equals_centralized_epoch(self):
        cd = toy_clients(n_clients=1)
        w = train.class_weights(cd[0]["yfit"])
        g, _ = train.fedavg("mlp", cd, cd[0]["Xval"], cd[0]["yval"], 11, CFG, w)
        m = build_mlp(12, 11)
        from fedgrande.fl.models import mlp_optimizer
        train.train_one_epoch(m, mlp_optimizer(m), torch.as_tensor(cd[0]["Xfit"]), torch.as_tensor(cd[0]["yfit"]), w,
                              train.gen_for(11, 1, 0), 256)
        for k, v in trainable_state(g).items():
            torch.testing.assert_close(v, trainable_state(m)[k], rtol=0, atol=1e-6)


class TestDiagnosticsAndComm(unittest.TestCase):
    def test_selections_map_through_feature_subsets(self):
        sia = torch.zeros(2, 1, 3)
        sia[0, 0, 2] = 5.0
        sia[1, 0, 0] = 5.0
        sv = torch.tensor([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]])
        fbe = torch.tensor([[7, 8, 9], [4, 5, 6]])
        feat, thr = diag.selections(sia, sv, fbe)
        np.testing.assert_array_equal(feat, [[9], [4]])
        np.testing.assert_allclose(thr, [[0.3], [0.4]])

    def test_consensus_known_case(self):
        F = np.array([[[1, 2]], [[1, 3]], [[1, 2]], [[5, 2]]])           # 4 clients, 1 estimator, 2 nodes
        T = np.array([[[0.0, 1.0]], [[0.2, 9.0]], [[0.4, 1.0]], [[9.0, 1.0]]])
        out = diag.consensus(F, T, np.array([[1, 2]]), n_numeric=10)
        self.assertAlmostEqual(out["consensus_mean"], 0.75)
        self.assertEqual(out["frac_nodes_unanimous"], 0.0)
        self.assertAlmostEqual(out["global_agreement"], 6 / 8)
        self.assertAlmostEqual(out["thr_sd_numeric_nodes_mean"], (np.std([0, .2, .4]) + 0.0) / 2)

    def test_comm_bytes(self):
        st = {"a": torch.zeros(10, dtype=torch.float32), "b": torch.zeros(2, 3, dtype=torch.float32)}
        acc = comm.accounting(st, {"f": torch.zeros(4, dtype=torch.int64)}, n_clients=3, rounds_executed=5, best_round=2)
        self.assertEqual(acc["payload_bytes_per_model"], 64)
        self.assertEqual(acc["bytes_per_round_total"], 2 * 3 * 64)
        self.assertEqual(acc["bytes_once_at_init_buffers"], 3 * 32)
        self.assertEqual(acc["cumulative_bytes_to_final_round"], 96 + 5 * 384)
        self.assertEqual(acc["cumulative_bytes_to_best_round"], 96 + 2 * 384)


if __name__ == "__main__":
    unittest.main()


class TestDirichletAmendment(Base):
    def test_dirichlet_partition_groups_and_sizes(self):
        cl, info = P.dirichlet_size_matched(self.meta, self.y, 0.1, 7)
        saved = pd.read_csv(PREP / "partitions_dirichlet_v1.csv")
        np.testing.assert_array_equal(cl, saved["client_dirichlet0.1"].to_numpy())
        self.assertEqual(int(pd.DataFrame({"g": self.meta.leakage_group_id, "c": cl}).groupby("g").c.nunique().max()), 1)
        dev = np.bincount(cl, minlength=21) - np.bincount(self.parts.client_natural.to_numpy())
        self.assertLessEqual(int(np.abs(dev).max()), 1)
        self.assertEqual(int(np.bincount(cl, weights=self.y).sum()), int(self.y.sum()))

    def test_waterfill_respects_caps(self):
        out = P._waterfill(10, np.array([0.9, 0.05, 0.05]), np.array([3.0, 10.0, 10.0]))
        self.assertEqual(int(out.sum()), 10)
        self.assertLessEqual(int(out[0]), 3)
