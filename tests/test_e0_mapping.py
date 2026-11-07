"""E0 state mapping: the mapped names are exactly the GRANDE module state, and the effective V rule holds.

Runs against whichever pinned GRANDE commit is importable in the active environment.
"""
import importlib.metadata as md
import unittest

import _common  # noqa: F401

from fedgrande.grande_bridge import BUFFERS, PARAMETERS, STATE_MAPPING, build_module, count_trainable
from fedgrande.phase2.e0a_compare import effective_v

IMPL = {"0.2.0": "ref", "1.0.0": "dev"}.get(md.version("GRANDE"))


@unittest.skipIf(IMPL is None, "no pinned GRANDE installed")
class TestMapping(unittest.TestCase):
    def test_state_keys_and_shapes(self):
        m = build_module(IMPL, number_of_variables=120, depth=5, n_estimators=8, selected_variables=0.8,
                         missing_values=False, random_seed=42)
        self.assertEqual(set(m.state_dict()), set(STATE_MAPPING))
        self.assertEqual({n for n, _ in m.named_parameters()}, set(PARAMETERS))
        self.assertEqual({n for n, _ in m.named_buffers()}, set(BUFFERS))
        self.assertEqual(tuple(m.split_values.shape), (8, 31, 50))
        self.assertEqual(count_trainable(m), 8 * 31 * 50 * 2 + 8 * 32 * 2)

    def test_selected_variables_rule(self):
        for f, sv in ((120, 0.8), (40, 0.8), (12, 0.8), (241, 0.8), (30, 7)):
            m = build_module(IMPL, number_of_variables=f, depth=3, n_estimators=2, selected_variables=sv,
                             missing_values=False, random_seed=0)
            self.assertEqual(int(m.selected_variables), effective_v(f, sv), (f, sv))


if __name__ == "__main__":
    unittest.main()
