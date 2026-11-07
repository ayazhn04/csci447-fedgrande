"""Client partitions over the whole dataset (fixed, independent of model seeds and outer folds).

Every partition assigns whole leakage groups to clients. Within an outer fold, a client's training/validation/test
rows are its rows intersected with the fold's fit/val/test sets.

natural: client = wave (21 clients).
iid:     21 synthetic clients whose target sizes are the natural wave sizes; leakage groups are dealt so that each
         client's positive-row and negative-row counts track largest-remainder targets proportional to its size.
(A Dirichlet label-skew partition is deliberately not part of the frozen core protocol; see PHASE2B_PROTOCOL.md.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    raw = total * weights / weights.sum()
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    order = np.argsort(-(raw - base), kind="stable")
    base[order[:rem]] += 1
    return base


def _groups(meta: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    g = pd.DataFrame({"gid": meta.leakage_group_id.to_numpy(), "y": y}).groupby("gid")["y"].agg(n="size", pos="sum")
    g["neg"] = g["n"] - g["pos"]
    return g


def natural(meta: pd.DataFrame) -> np.ndarray:
    waves = np.sort(meta.wave.unique())
    return pd.Series(np.arange(len(waves)), index=waves).loc[meta.wave.to_numpy()].to_numpy()


def _deal(g: pd.DataFrame, target_pos: np.ndarray, target_neg: np.ndarray, rng) -> dict:
    """Assign groups to clients following positive/negative row quotas (deterministic given rng)."""
    k = len(target_pos)
    rem_pos, rem_neg = target_pos.astype(float).copy(), target_neg.astype(float).copy()
    assign = {}
    mixed = g[(g.pos > 0) & (g.neg > 0)].sort_values("n", ascending=False)
    for gid, r in mixed.iterrows():
        c = int(np.argmax(rem_pos + rem_neg))
        assign[gid] = c
        rem_pos[c] -= r.pos
        rem_neg[c] -= r.neg
    for kind, rem, col in (("pos", rem_pos, "pos"), ("neg", rem_neg, "neg")):
        pure = g[(g[col] > 0) & (g["n"] == g[col])]
        ids = pure.index.to_numpy()
        ids = ids[rng.permutation(len(ids))]
        # larger groups first (stable w.r.t. the shuffled order) so atomic sizes are placed while slack exists
        ids = ids[np.argsort(-pure.loc[ids, "n"].to_numpy(), kind="stable")]
        for gid in ids:
            c = int(np.argmax(rem))                         # ties -> lowest client index
            assign[gid] = c
            rem[c] -= pure.loc[gid, "n"]
    assert len(assign) == len(g)
    return assign


def iid_size_matched(meta: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    nat = natural(meta)
    sizes = np.bincount(nat)
    g = _groups(meta, y)
    t_pos = _largest_remainder(int(y.sum()), sizes.astype(float))
    t_neg = sizes - t_pos
    assign = _deal(g, t_pos, t_neg, np.random.default_rng(seed))
    return pd.Series(assign).loc[meta.leakage_group_id.to_numpy()].to_numpy().astype(int)


def describe(client: np.ndarray, y: np.ndarray, meta: pd.DataFrame, target_sizes=None) -> pd.DataFrame:
    df = pd.DataFrame({"client": client, "y": y, "gid": meta.leakage_group_id.to_numpy(), "kind": meta.group_kind.to_numpy(),
                       "wave": meta.wave.to_numpy()})
    rows = []
    for c, d in df.groupby("client"):
        rows.append({"client": int(c), "rows": len(d), "pos_rows": int(d.y.sum()), "pos_rate": round(float(d.y.mean()), 4),
                     "leakage_groups": int(d.gid.nunique()), "dyads": int(d.loc[d.kind == "dyad", "gid"].nunique()),
                     "pos_dyads": int(d.loc[(d.kind == "dyad") & (d.y == 1), "gid"].nunique()),
                     "n_waves_represented": int(d.wave.nunique())})
    out = pd.DataFrame(rows)
    if target_sizes is not None:
        out["target_rows"] = np.asarray(target_sizes)[out.client.to_numpy()]
        out["size_deviation"] = out.rows - out.target_rows
    return out


# ---------------------------------------------------------------------------------------------------------------
# Amendment A1 (after the B2 core matrix): Dirichlet label-skew stress test, size-matched to the natural waves.
# ---------------------------------------------------------------------------------------------------------------
def _waterfill(total: int, p: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Integer allocation of `total` proportional to p with per-client caps (excess redistributed by p)."""
    alloc = np.zeros(len(p))
    free = np.ones(len(p), dtype=bool)
    remaining = float(total)
    while remaining > 1e-9 and free.any():
        share = remaining * p * free / (p * free).sum()
        take = np.minimum(share, cap - alloc)
        alloc += take
        remaining -= take.sum()
        free &= alloc < cap - 1e-9
    base = np.floor(alloc).astype(int)
    rem = total - base.sum()
    order = np.argsort(-(alloc - base) + 1e9 * (base >= cap), kind="stable")
    base[order[:rem]] += 1
    return base


def dirichlet_size_matched(meta: pd.DataFrame, y: np.ndarray, alpha: float, seed: int):
    """21 clients with the natural wave sizes; the share of POSITIVE rows per client ~ Dirichlet(alpha * 1).

    Positive-row targets = water-filled total positives by the Dirichlet proportions (capped at client size);
    negative targets = size - positive target. Whole leakage groups are then dealt by quota (as for 'iid').
    """
    nat = natural(meta)
    sizes = np.bincount(nat)
    rng = np.random.default_rng([seed, 1])
    p = rng.dirichlet([alpha] * len(sizes))
    t_pos = _waterfill(int(y.sum()), p, sizes.astype(float))
    t_neg = sizes - t_pos
    assign = _deal(_groups(meta, y), t_pos, t_neg, rng)
    client = pd.Series(assign).loc[meta.leakage_group_id.to_numpy()].to_numpy().astype(int)
    return client, {"dirichlet_proportions": p.tolist(), "target_pos": t_pos.tolist(), "target_neg": t_neg.tolist()}
