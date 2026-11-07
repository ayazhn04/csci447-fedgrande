"""Split determinism and no-leakage invariants."""
import unittest

import _common  # noqa: F401
import numpy as np
import pandas as pd

from fedgrande import splits
from fedgrande.data import load_xy
from fedgrande.paths import DERIVED

PHASE1_HASH = "ca9d1164c5100d5b310dbb644672d9dad2d0cdaceb38055b0e6f80ee3d1c1755"


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.y, _ = load_xy()
        d = pd.read_csv(DERIVED / "dyads_v1.csv")
        cls.g, cls.m = d.dyad_id.to_numpy(), d.mirror_row.to_numpy()
        cls.outer = pd.read_csv(DERIVED / "splits_outer_v1.csv")
        cls.inner = pd.read_csv(DERIVED / "splits_inner_v1.csv")

    def test_rowwise_reproduces_checkpoint1_split(self):
        self.assertEqual(splits.fold_hash(splits.fold_assignment_rowwise(self.y)), PHASE1_HASH)

    def test_deterministic_and_saved(self):
        f2 = splits.fold_assignment_grouped(self.y, self.g)
        np.testing.assert_array_equal(f2, splits.fold_assignment_grouped(self.y, self.g))
        np.testing.assert_array_equal(f2, self.outer.C2_dyad_fold.to_numpy())
        np.testing.assert_array_equal(splits.fold_assignment_rowwise(self.y), self.outer.C1_rowwise_fold.to_numpy())

    def test_grouped_split_has_no_dyad_leakage(self):
        v = splits.verify(self.outer.C2_dyad_fold.to_numpy(), self.y, self.g, self.m)
        self.assertTrue(v["all_rows_assigned_once"])
        self.assertEqual(v["total_dyad_group_overlap"], 0)
        self.assertEqual(v["total_test_rows_whose_mirror_is_in_train"], 0)
        for f in v["folds"]:
            self.assertEqual(f["row_overlap"], 0)
            self.assertTrue(f["train_plus_test_equals_n"])
        self.assertLess(v["max_abs_pos_rate_deviation"], 0.005)

    def test_rowwise_split_does_leak(self):
        v = splits.verify(self.outer.C1_rowwise_fold.to_numpy(), self.y, self.g, self.m)
        self.assertGreater(v["total_test_rows_whose_mirror_is_in_train"], 0)

    def test_inner_validation_is_grouped_and_disjoint(self):
        for k, grp in self.inner[self.inner.scheme == "C2"].groupby("outer_fold"):
            r = {role: grp.loc[grp.role == role, "row_index"].to_numpy() for role in ("fit", "val", "test")}
            self.assertEqual(len(r["fit"]) + len(r["val"]) + len(r["test"]), len(self.y))
            self.assertFalse(set(self.g[r["fit"]]) & set(self.g[r["val"]]))
            self.assertFalse(set(self.g[r["fit"]]) & set(self.g[r["test"]]))
            self.assertFalse(set(self.g[r["val"]]) & set(self.g[r["test"]]))
            np.testing.assert_array_equal(np.sort(r["test"]), np.flatnonzero(self.outer.C2_dyad_fold.to_numpy() == k))


if __name__ == "__main__":
    unittest.main()
