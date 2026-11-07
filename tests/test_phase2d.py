"""Phase 2D: INT8 update codec, federated integration, no unintended state change, communication accounting."""
import unittest

import _common  # noqa: F401
import numpy as np
import torch

from fedgrande.fl import comm, compress, train
from fedgrande.fl.models import build_grande, buffer_state, trainable_state


class TestCodec(unittest.TestCase):
    def check(self, d):
        q, s = compress.quantize(d)
        dh = compress.dequantize(q, s)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(dh.dtype, torch.float32)
        self.assertTrue(torch.isfinite(s))
        self.assertLessEqual(int(q.abs().max()), 127)
        err = (dh - d).abs()
        self.assertLessEqual(float(err.max()), float(s) / 2 + 1e-6 * max(1.0, float(d.abs().max())))   # uniform-quantization bound
        return q, s, dh

    def test_zero(self):
        q, s, dh = self.check(torch.zeros(5, 3))
        self.assertEqual(float(s), 0.0)
        self.assertTrue(torch.equal(dh, torch.zeros(5, 3)))

    def test_signs_constant_random_extreme(self):
        self.check(torch.tensor([-3.0, -1.0, 0.0, 0.5, 2.0]))
        q, s, dh = self.check(torch.full((4, 4), -2.5))
        self.assertTrue(torch.allclose(dh, torch.full((4, 4), -2.5)))
        g = torch.Generator().manual_seed(0)
        d = torch.randn(1000, generator=g)
        q, s, dh = self.check(d)
        rel = float(torch.linalg.vector_norm(dh - d) / torch.linalg.vector_norm(d))
        self.assertLess(rel, 0.02)
        self.check(torch.tensor([1e30, -1e30, 1.0]))
        self.check(torch.tensor([1e-30, -3e-30]))
        self.assertEqual(int(compress.quantize(torch.tensor([5.0, -5.0]))[0].abs().max()), 127)

    def test_error_metrics_known(self):
        d = torch.linspace(-1, 1, 255)
        q, s = compress.quantize(d)
        dh = compress.dequantize(q, s)
        self.assertAlmostEqual(float(s), 1 / 127, places=7)
        self.assertLessEqual(float((dh - d).abs().max()), 0.5 / 127 + 1e-7)
        self.assertLess(float((dh - d).abs().mean()), 0.5 / 127)


class TestIntegration(unittest.TestCase):
    def test_fedavg_with_quantized_updates_close_to_exact(self):
        g = build_grande(12, 1, n_estimators=8, depth=3)
        glob = trainable_state(g)
        before = {k: v.clone() for k, v in glob.items()}
        rng = torch.Generator().manual_seed(3)
        locals_ = [{k: v + 0.01 * torch.randn(v.shape, generator=rng) for k, v in glob.items()} for _ in range(3)]
        snap = [{k: v.clone() for k, v in s.items()} for s in locals_]
        w = [10, 20, 30]
        exact = train.weighted_average(locals_, w)
        recon = [compress.roundtrip_update(s, glob)[0] for s in locals_]
        approx = train.weighted_average(recon, w)
        for k in exact:
            amax = max(float((s[k] - glob[k]).abs().max()) for s in locals_)
            self.assertLessEqual(float((approx[k] - exact[k]).abs().max()), amax / 127 / 2 + 1e-6)
            self.assertFalse(torch.equal(approx[k], exact[k]))
        for s, sn in zip(locals_, snap):                       # inputs untouched
            for k in s:
                self.assertTrue(torch.equal(s[k], sn[k]))
        for k in glob:
            self.assertTrue(torch.equal(glob[k], before[k]))

    def test_int8_fedavg_round_keeps_buffers_and_reports_error(self):
        rng = np.random.default_rng(0)
        cd = []
        for c in range(3):
            X = rng.standard_normal((40, 12)).astype(np.float32)
            y = (X[:, 0] > 0).astype(np.int64)
            cd.append({"Xfit": X, "yfit": y, "Xval": X[:10], "yval": y[:10]})
        Xv, yv = np.concatenate([c["Xval"] for c in cd]), np.concatenate([c["yval"] for c in cd])
        w = train.class_weights(np.concatenate([c["yfit"] for c in cd]))
        cfg = {"max_rounds": 2, "local_epochs": 1, "batch_size": 256, "patience": 10, "warmup": 10,
               "model_kwargs": {"n_estimators": 8, "depth": 3}, "uplink": "int8"}
        init = build_grande(12, 4, n_estimators=8, depth=3)
        g, info = train.fedavg("grande", cd, Xv, yv, 4, cfg, w, n_numeric=12)
        for k, v in buffer_state(init).items():
            self.assertTrue(torch.equal(v, buffer_state(g)[k]))
        r = info["rounds"][0]
        self.assertIn("qerr_rel_l2_mean_all", r)
        self.assertLess(r["qerr_rel_l2_max_all"], 0.05)
        g32, _ = train.fedavg("grande", cd, Xv, yv, 4, {**cfg, "uplink": "fp32"}, w, n_numeric=12)
        diff = max(float((a - b).abs().max()) for a, b in zip(trainable_state(g).values(), trainable_state(g32).values()))
        self.assertGreater(diff, 0.0)


class TestCommunication(unittest.TestCase):
    def test_v2_accounting(self):
        st = {"a": torch.zeros(100), "b": torch.zeros(10, 10)}
        acc32 = comm.accounting_v2(st, {"f": torch.zeros(5, dtype=torch.int64)}, n_clients=2, rounds_executed=3, best_round=2, uplink="fp32")
        acc8 = comm.accounting_v2(st, {"f": torch.zeros(5, dtype=torch.int64)}, n_clients=2, rounds_executed=3, best_round=2, uplink="int8")
        self.assertEqual(acc32["uplink_bytes_per_client_per_round"], 800)
        self.assertEqual(acc8["uplink_bytes_per_client_per_round"], 200 + 2 * 4)
        self.assertEqual(acc8["downlink_bytes_per_client_per_round"], 800)
        self.assertEqual(acc8["cum_total_to_best"], 2 * 40 + 2 * (2 * 800 + 2 * 208))
        self.assertEqual(acc32["cum_total_to_final"], 80 + 3 * (2 * 800 * 2))

    def test_grande_parameter_counts(self):
        for e, p in ((256, 809_984), (64, 202_496)):
            self.assertEqual(sum(t.numel() for t in trainable_state(build_grande(502, 42, n_estimators=e, depth=5)).values()), p)


if __name__ == "__main__":
    unittest.main()
