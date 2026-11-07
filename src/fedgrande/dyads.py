"""Deterministic reconstruction of SpeedDating dyads (the two partner-perspective rows of one speed date).

Each OpenML row describes one date from the perspective of person P with partner Q. The mirror row describes the
same date from Q's perspective. Columns describing Q in P's row ("partner side") must equal the columns describing
Q as the row owner in Q's row ("self side"):

    P's row: partner side            Q's row: self side
    age_o                       ==   age
    race_o                      ==   race
    pref_o_<attr>               ==   <attr>_important        (Q's stated preferences)
    <attr>_o                    ==   <attr>_partner          (Q's ratings of P)

plus same ``wave``, opposite ``gender``, and identical pair-level values ``samerace``, ``d_age`` and
``interests_correlate``. ``match``, ``decision`` and ``decision_o`` are NOT used to define pairs; they are only used
afterwards to validate the result. Missing values are compared as an explicit token (missing == missing).

Grouping rule (conservative, so that no candidate mirror is ever separated): build an undirected graph with an edge
i--j whenever j's self side equals i's partner side (in either direction); every connected component is one
``dyad_id``. Component types:

* ``pair_mutual``      2 rows, exact match in both directions (the normal case);
* ``pair_one_way``     2 rows, exact in one direction, the other direction differs in >= 1 column;
* ``ambiguous_group``  > 2 rows (some row has several exact candidates); kept together as one group;
* ``singleton``        1 row without any exact candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (self-side column, partner-side column)
MIRROR_PAIRS = [
    ("age", "age_o"),
    ("race", "race_o"),
    ("attractive_important", "pref_o_attractive"),
    ("sincere_important", "pref_o_sincere"),
    ("intellicence_important", "pref_o_intelligence"),
    ("funny_important", "pref_o_funny"),
    ("ambtition_important", "pref_o_ambitious"),
    ("shared_interests_important", "pref_o_shared_interests"),
    ("attractive_partner", "attractive_o"),
    ("sincere_partner", "sinsere_o"),
    ("intelligence_partner", "intelligence_o"),
    ("funny_partner", "funny_o"),
    ("ambition_partner", "ambitous_o"),
    ("shared_interests_partner", "shared_interests_o"),
]
SYMMETRIC = ["samerace", "d_age", "interests_correlate"]
HARD = ["wave", "gender"]  # same wave, opposite gender
FORBIDDEN = {"match", "decision", "decision_o"}
ALGORITHM_VERSION = "dyads-v1"


def _tokens(s: pd.Series) -> np.ndarray:
    return np.array(["<NA>" if pd.isna(v) else repr(float(v)) if isinstance(v, (int, float, np.floating)) else str(v)
                     for v in s.astype(object)], dtype=object)


def _codes(full: pd.DataFrame):
    """Integer codes: S (self side), P (partner side) share one code space per mirror pair; Y symmetric columns."""
    n = len(full)
    S = np.empty((n, len(MIRROR_PAIRS)), dtype=np.int64)
    P = np.empty_like(S)
    for k, (s, p) in enumerate(MIRROR_PAIRS):
        codes, _ = pd.factorize(np.concatenate([_tokens(full[s]), _tokens(full[p])]))
        S[:, k], P[:, k] = codes[:n], codes[n:]
    Y = np.column_stack([pd.factorize(_tokens(full[c]))[0] for c in SYMMETRIC])
    return S, P, Y


def mismatch_matrix(full: pd.DataFrame, rows_i: np.ndarray, rows_j: np.ndarray, S, P, Y) -> np.ndarray:
    """Number of differing mirror/symmetric columns when j is proposed as the mirror of i (directional)."""
    return (P[rows_i][:, None, :] != S[rows_j][None, :, :]).sum(-1) + (Y[rows_i][:, None, :] != Y[rows_j][None, :, :]).sum(-1)


def reconstruct(full: pd.DataFrame):
    """Return (per-row DataFrame, report dict). ``full`` must be in OpenML file order (row_index = position)."""
    used = {c for pair in MIRROR_PAIRS for c in pair} | set(SYMMETRIC) | set(HARD)
    assert not used & FORBIDDEN, "label columns must not define pairs"
    n = len(full)
    S, P, Y = _codes(full)
    wave = full["wave"].to_numpy()
    gender = full["gender"].astype(str).to_numpy()

    cand = [np.empty(0, dtype=np.int64)] * n  # exact candidates j for each i (directional)
    best_dist = np.full(n, -1)
    best_rows = [np.empty(0, dtype=np.int64)] * n
    for w in np.unique(wave):
        idx = np.flatnonzero(wave == w)
        D = mismatch_matrix(full, idx, idx, S, P, Y)
        same_gender = gender[idx][:, None] == gender[idx][None, :]
        D = np.where(same_gender, 10 ** 6, D)
        for a, i in enumerate(idx):
            cand[i] = idx[D[a] == 0]
            best_dist[i] = D[a].min()
            best_rows[i] = idx[D[a] == best_dist[i]]

    # union-find over undirected exact-candidate edges
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in cand[i]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    # deterministic ids: components ordered by their smallest row index
    order = {r: k for k, r in enumerate(sorted(set(roots.tolist()), key=lambda r: np.flatnonzero(roots == r)[0]))}
    dyad_id = np.array([order[r] for r in roots])

    status = np.empty(n, dtype=object)
    mirror = np.full(n, -1)
    size = np.bincount(dyad_id)[dyad_id]
    for d in np.unique(dyad_id):
        members = np.flatnonzero(dyad_id == d)
        if len(members) == 1:
            status[members] = "singleton"
        elif len(members) == 2:
            i, j = members
            mutual = j in cand[i] and i in cand[j]
            status[members] = "pair_mutual" if mutual else "pair_one_way"
            mirror[i], mirror[j] = j, i
        else:
            status[members] = "ambiguous_group"

    rows = pd.DataFrame({
        "row_index": np.arange(n),
        "wave": wave.astype(int),
        "gender": gender,
        "dyad_id": dyad_id,
        "dyad_status": status,
        "dyad_size": size,
        "mirror_row": mirror,
        "n_exact_candidates": [len(c) for c in cand],
        "best_mismatch": best_dist,
    })
    report = _report(full, rows, cand, best_rows, S, P, Y)
    return rows, report


def _report(full, rows, cand, best_rows, S, P, Y) -> dict:
    y = full["match"].astype(str).to_numpy()
    dec = full["decision"].astype(str).to_numpy()
    dec_o = full["decision_o"].astype(str).to_numpy()
    st = rows["dyad_status"].to_numpy()
    pairs = rows[rows["dyad_size"] == 2]
    i = pairs["row_index"].to_numpy()
    j = pairs["mirror_row"].to_numpy()
    groups = rows.groupby("dyad_id")

    one_way = []
    for a in np.flatnonzero(st == "pair_one_way"):
        b = rows["mirror_row"][a]
        if b in cand[a]:
            continue  # report each one-way pair from the failing direction
        diffs = [MIRROR_PAIRS[k][1] for k in range(len(MIRROR_PAIRS)) if P[a, k] != S[b, k]]
        diffs += [SYMMETRIC[k] for k in range(len(SYMMETRIC)) if Y[a, k] != Y[b, k]]
        one_way.append({"row": int(a), "proposed_mirror": int(b), "differing_partner_side_columns": diffs,
                        "values_row": {c: str(full[c].iloc[a]) for c in diffs},
                        "values_mirror_self_side": {dict((p, s) for s, p in MIRROR_PAIRS).get(c, c):
                                                    str(full[dict((p, s) for s, p in MIRROR_PAIRS).get(c, c)].iloc[b])
                                                    for c in diffs}})
    partner_cols = [p for _, p in MIRROR_PAIRS]
    singles = []
    for a in np.flatnonzero(st == "singleton"):
        singles.append({"row": int(a), "wave": int(rows["wave"][a]), "gender": rows["gender"][a],
                        "partner_side_missing": int(full[partner_cols].iloc[a].isna().sum()),
                        "partner_side_columns": len(partner_cols),
                        "best_mismatch": int(rows["best_mismatch"][a]),
                        "n_rows_at_best_mismatch": int(len(best_rows[a]))})
    amb = []
    for d, g in groups:
        if len(g) > 2:
            m = g["row_index"].to_numpy()
            amb.append({"dyad_id": int(d), "rows": m.tolist(), "wave": int(g["wave"].iloc[0]),
                        "genders": g["gender"].value_counts().to_dict(),
                        "exact_candidate_lists": {int(r): [int(c) for c in cand[r]] for r in m},
                        "match_values": y[m].tolist()})

    return {
        "algorithm_version": ALGORITHM_VERSION,
        "mirror_pairs_self_partner": MIRROR_PAIRS,
        "symmetric_columns": SYMMETRIC,
        "hard_constraints": "same wave, opposite gender",
        "label_columns_used_to_define_pairs": [],
        "n_rows": int(len(rows)),
        "n_groups": int(rows["dyad_id"].nunique()),
        "status_rows": rows["dyad_status"].value_counts().to_dict(),
        "status_groups": rows.drop_duplicates("dyad_id")["dyad_status"].value_counts().to_dict(),
        "rows_with_exact_candidate_count": rows["n_exact_candidates"].value_counts().sort_index().to_dict(),
        "paired_rows_total": int(len(pairs)),
        "dyads_size2": int(len(pairs) // 2),
        "validation_pairs": {
            "pairs_same_wave": bool((rows["wave"].to_numpy()[i] == rows["wave"].to_numpy()[j]).all()),
            "pairs_opposite_gender": bool((rows["gender"].to_numpy()[i] != rows["gender"].to_numpy()[j]).all()),
            "pairs_same_match": bool((y[i] == y[j]).all()),
            "pairs_with_different_match": int((y[i] != y[j]).sum() // 2),
            "decision_equals_mirror_decision_o (validation only)": bool((dec[i] == dec_o[j]).all()),
            "decision_mismatches_rows (validation only)": int((dec[i] != dec_o[j]).sum()),
        },
        "validation_groups": {
            "groups_single_wave": bool(groups["wave"].nunique().max() == 1),
            "ambiguous_groups_with_mixed_match": int(sum(len(set(a["match_values"])) > 1 for a in amb)),
        },
        "one_way_pairs": one_way,
        "ambiguous_groups": amb,
        "singletons": singles,
    }
