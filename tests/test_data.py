"""Offline loader: checksum, modeling view, dtypes."""
import unittest

import _common  # noqa: F401
import pandas as pd

from fedgrande.data import load_xy


class TestData(unittest.TestCase):
    def test_modeling_view(self):
        X, y, full = load_xy(verify=True)
        self.assertEqual(X.shape, (8378, 120))
        for c in ("match", "decision", "decision_o"):
            self.assertNotIn(c, X.columns)
        self.assertEqual(int(y.sum()), 1380)
        self.assertEqual(sum(isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns), 61)
        self.assertEqual(int(X.isna().sum().sum()), 18372)


if __name__ == "__main__":
    unittest.main()
