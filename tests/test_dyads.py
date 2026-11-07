"""Dyad reconstruction invariants (real data + a synthetic toy case)."""
import unittest

import _common  # noqa: F401
import numpy as np
import pandas as pd

from fedgrande import dyads
from fedgrande.data import load_full
from fedgrande.paths import DERIVED


class TestDyadsRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = load_full()
        cls.rows, cls.rep = dyads.reconstruct(cls.full)

    def test_label_columns_not_used(self):
        used = {c for p in dyads.MIRROR_PAIRS for c in p} | set(dyads.SYMMETRIC) | set(dyads.HARD)
        self.assertFalse(used & {"match", "decision", "decision_o"})

    def test_every_row_has_a_group(self):
        self.assertEqual(len(self.rows), 8378)
        self.assertTrue((self.rows.dyad_id >= 0).all())

    def test_pairs_are_symmetric_and_consistent(self):
        p = self.rows[self.rows.dyad_size == 2]
        m = p.mirror_row.to_numpy()
        self.assertTrue((self.rows.mirror_row.to_numpy()[m] == p.row_index.to_numpy()).all())
        self.assertTrue((self.rows.wave.to_numpy()[m] == p.wave.to_numpy()).all())
        self.assertTrue((self.rows.gender.to_numpy()[m] != p.gender.to_numpy()).all())
        y = self.full["match"].astype(str).to_numpy()
        self.assertTrue((y[m] == y[p.row_index.to_numpy()]).all())

    def test_groups_never_cross_waves(self):
        self.assertEqual(int(self.rows.groupby("dyad_id").wave.nunique().max()), 1)

    def test_status_counts_consistent(self):
        st = self.rows.dyad_status.value_counts()
        self.assertEqual(int(st.sum()), 8378)
        self.assertEqual(int((self.rows.dyad_size == 1).sum()), int(st.get("singleton", 0)))
        self.assertTrue((self.rows.loc[self.rows.dyad_status == "ambiguous_group", "dyad_size"] > 2).all())

    def test_deterministic_and_matches_saved_file(self):
        rows2, _ = dyads.reconstruct(self.full)
        pd.testing.assert_frame_equal(self.rows, rows2)
        saved = pd.read_csv(DERIVED / "dyads_v1.csv")
        np.testing.assert_array_equal(saved.dyad_id.to_numpy(), self.rows.dyad_id.to_numpy())
        np.testing.assert_array_equal(saved.mirror_row.to_numpy(), self.rows.mirror_row.to_numpy())

    def test_every_exact_candidate_is_in_same_group(self):
        S, P, Y = dyads._codes(self.full)
        wave, g = self.full.wave.to_numpy(), self.full.gender.astype(str).to_numpy()
        gid = self.rows.dyad_id.to_numpy()
        for w in np.unique(wave):
            idx = np.flatnonzero(wave == w)
            D = dyads.mismatch_matrix(self.full, idx, idx, S, P, Y)
            D = np.where(g[idx][:, None] == g[idx][None, :], 1, D)
            a, b = np.nonzero(D == 0)
            self.assertTrue((gid[idx[a]] == gid[idx[b]]).all())


class TestDyadsToy(unittest.TestCase):
    def _row(self, wave, gender, me, partner, ratings_given, ratings_received, sym=(1.0, 2.0, 0.5)):
        cols = {}
        (age, race, prefs), (age_o, race_o, prefs_o) = me, partner
        cols.update({"wave": wave, "gender": gender, "age": age, "race": race, "age_o": age_o, "race_o": race_o,
                     "samerace": sym[0], "d_age": sym[1], "interests_correlate": sym[2]})
        for k, (s, p) in enumerate(dyads.MIRROR_PAIRS[2:8]):
            cols[s], cols[p] = prefs[k], prefs_o[k]
        for k, (s, p) in enumerate(dyads.MIRROR_PAIRS[8:]):
            cols[s], cols[p] = ratings_given[k], ratings_received[k]
        cols.update({"match": "0", "decision": "0", "decision_o": "0"})
        return cols

    def test_pair_ambiguity_and_singleton(self):
        A = (21.0, "x", [10.0] * 6)
        B = (23.0, "y", [20.0, 10, 10, 20, 20, 20])
        C = (25.0, "z", [5.0] * 6)
        r_ab = self._row(1, "female", A, B, [7.0] * 6, [6.0] * 6)
        r_ba = self._row(1, "male", B, A, [6.0] * 6, [7.0] * 6)
        r_c = self._row(1, "male", C, (np.nan, None, [np.nan] * 6), [3.0] * 6, [np.nan] * 6, sym=(0.0, 4.0, 0.1))
        df = pd.DataFrame([r_ab, r_ba, r_c])
        rows, rep = dyads.reconstruct(df)
        self.assertEqual(rows.dyad_status.tolist(), ["pair_mutual", "pair_mutual", "singleton"])
        self.assertEqual(rows.mirror_row.tolist(), [1, 0, -1])
        # a second, indistinguishable copy of B's row makes the group ambiguous and keeps all three together
        rows2, _ = dyads.reconstruct(pd.DataFrame([r_ab, r_ba, dict(r_ba)]))
        self.assertEqual(set(rows2.dyad_status), {"ambiguous_group"})
        self.assertEqual(rows2.dyad_id.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
