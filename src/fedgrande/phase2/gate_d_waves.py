"""Gate D: audit ``wave`` as a candidate natural client variable (no partitions are built here)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency, kruskal  # noqa: E402

from fedgrande.data import load_xy  # noqa: E402
from fedgrande.paths import DERIVED, RESULTS  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402

IMPORTANCE = ["attractive_important", "sincere_important", "intellicence_important", "funny_important",
              "ambtition_important", "shared_interests_important"]


def main():
    X, y, full = load_xy()
    d = pd.read_csv(DERIVED / "dyads_v1.csv")
    w = X["wave"].astype(int).to_numpy()
    df = pd.DataFrame({"wave": w, "y": y, "dyad_id": d["dyad_id"], "status": d["dyad_status"], "gender": X["gender"].astype(str)})

    # dyads crossing waves
    crossing = int((df.groupby("dyad_id")["wave"].nunique() > 1).sum())

    rows = []
    for wv, g in df.groupby("wave"):
        pairs = g[g["status"].isin(["pair_mutual", "pair_one_way"])].drop_duplicates("dyad_id")
        rows.append({
            "wave": int(wv), "rows": len(g), "female_rows": int((g.gender == "female").sum()), "male_rows": int((g.gender == "male").sum()),
            "pos_rows": int(g.y.sum()), "row_match_rate": round(float(g.y.mean()), 4),
            "dyad_groups": int(g.dyad_id.nunique()), "pair_dyads": len(pairs), "pos_pair_dyads": int(pairs.y.sum()),
            "dyad_match_rate": round(float(pairs.y.mean()), 4) if len(pairs) else None,
            "rows_in_ambiguous_groups": int((g.status == "ambiguous_group").sum()), "singleton_rows": int((g.status == "singleton").sum()),
        })
    table = pd.DataFrame(rows)

    # missingness by wave for the most-missing predictors
    miss = X.isna().mean().sort_values(ascending=False)
    top = [c for c in miss.index if miss[c] > 0.03]
    miss_by_wave = X[top].isna().groupby(w).mean().round(4)
    miss_by_wave.index.name = "wave"

    # protocol checks: preference allocations (sum of the six importance columns)
    pref_sum = X[IMPORTANCE].sum(axis=1, min_count=6)
    pref = pd.DataFrame({"wave": w, "s": pref_sum}).dropna().groupby("wave")["s"].agg(
        median="median", min="min", max="max", frac_exactly_100=lambda s: float(np.isclose(s, 100).mean()))

    # distribution shift across waves (descriptive tests; not used for any decision rule)
    num_cols = [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype) and c != "wave"]
    kw = {}
    for c in num_cols:
        groups = [X.loc[w == wv, c].dropna().to_numpy() for wv in np.unique(w)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1 and len(np.unique(np.concatenate(groups))) > 1:
            kw[c] = float(kruskal(*groups).pvalue)
    cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype) and not c.startswith("d_")]
    chi = {}
    for c in cat_cols:
        ct = pd.crosstab(w, X[c].astype(str))
        if ct.shape[1] > 1:
            chi[c] = float(chi2_contingency(ct)[1])
    label_shift_p = float(chi2_contingency(pd.crosstab(w, y))[1])

    out_csv = RESULTS / "gate_d" / "wave_table.csv"
    os.makedirs(out_csv.parent, exist_ok=True)
    table.to_csv(out_csv, index=False)
    miss_by_wave.to_csv(RESULTS / "gate_d" / "missingness_by_wave.csv")
    pref.to_csv(RESULTS / "gate_d" / "preference_allocation_by_wave.csv")

    ein = miss_by_wave["expected_num_interested_in_me"] if "expected_num_interested_in_me" in miss_by_wave else None
    report = {
        "n_waves": int(len(table)),
        "rows_per_wave": dict(zip(table.wave, table.rows)),
        "dyads_crossing_waves": crossing,
        "quantity_skew": {"min_rows": int(table.rows.min()), "max_rows": int(table.rows.max()),
                          "max_over_min": round(float(table.rows.max() / table.rows.min()), 2),
                          "min_pos_pair_dyads": int(table.pos_pair_dyads.min()),
                          "waves_with_lt_10_positive_dyads": table.loc[table.pos_pair_dyads < 10, "wave"].tolist()},
        "label_shift": {"row_match_rate_min": float(table.row_match_rate.min()), "row_match_rate_max": float(table.row_match_rate.max()),
                        "dyad_match_rate_min": float(table.dyad_match_rate.min()), "dyad_match_rate_max": float(table.dyad_match_rate.max()),
                        "chi2_wave_x_match_p_rowlevel_(rows_not_independent)": label_shift_p},
        "feature_shift": {"numeric_features_tested": len(kw), "numeric_kruskal_p_lt_1e-3": int(sum(p < 1e-3 for p in kw.values())),
                          "categorical_features_tested": len(chi), "categorical_chi2_p_lt_1e-3": int(sum(p < 1e-3 for p in chi.values())),
                          "note": "row-level tests; rows are not independent (participants repeat), so p-values are descriptive only"},
        "protocol_missingness_shift": {
            "expected_num_interested_in_me_missing_by_wave": ein.to_dict() if ein is not None else None,
            "waves_fully_missing": [int(k) for k, v in (ein.items() if ein is not None else []) if v == 1.0],
            "waves_mostly_present": [int(k) for k, v in (ein.items() if ein is not None else []) if v < 0.5],
        },
        "preference_allocation_frac_exactly_100_min_over_waves": float(pref.frac_exactly_100.min()),
        "outputs": {"wave_table": str(out_csv), "wave_table_sha256": sha256_file(out_csv),
                    "missingness_by_wave": str(RESULTS / "gate_d" / "missingness_by_wave.csv"),
                    "preference_allocation_by_wave": str(RESULTS / "gate_d" / "preference_allocation_by_wave.csv")},
        "meta": run_metadata(),
    }
    write_json(RESULTS / "gate_d" / "wave_report.json", report)
    print(table.to_string(index=False))
    print(miss_by_wave.to_string())
    print(pref.round(3).to_string())
    print({k: report[k] for k in ("n_waves", "dyads_crossing_waves", "quantity_skew", "label_shift", "feature_shift")})


if __name__ == "__main__":
    main()
