#!/usr/bin/env python3
"""Build the curated result tables from saved Phase 2B-2F summaries. Safe: reads files only, never trains.

Outputs:
- results/curated/*.csv
- results/curated/final_key_numbers.json
- results/curated/PROVENANCE.json
- the generated tables in docs/RESULTS_LEDGER.md (between the GENERATED markers)

Every number comes from a machine-generated result file of the frozen experiments (git tag `experiments-final`).
Each row carries its source file, the file's sha256, its phase, the commit that last wrote the source file, and the
frozen config sha256 where one exists.

Usage: python -I scripts/build_curated_results.py [--check]
--check exits non-zero if a regenerated file would differ from the committed one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "curated"
LEDGER = ROOT / "docs" / "RESULTS_LEDGER.md"
TAG = "experiments-final"
CONFIGS = {"2B": "configs/phase2b/core.json", "2B-A1": "configs/phase2b/amendment_A1_dirichlet.json",
           "2C": "configs/phase2c/personalization.json", "2D": "configs/phase2d/compression.json",
           "2E": "configs/phase2e/structure.json", "2F": "configs/phase2f/cgsf.json"}


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def commit_of(path: str) -> str:
    """Commit (at or before the experiments-final tag) that last changed this file."""
    out = subprocess.run(["git", "log", "-1", "--format=%h", TAG, "--", path], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "untracked"


def prov(df: pd.DataFrame, paths: list[str], phase: str, config: str | None) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = "; ".join(paths)
    df["source_sha256"] = "; ".join(sha(p)[:16] for p in paths)
    df["phase"] = phase
    df["results_commit"] = "; ".join(sorted({commit_of(p) for p in paths}))
    df["config_sha256"] = sha(CONFIGS[config])[:16] if config else ""
    return df


def pm(s: str) -> tuple[float, float]:
    """Parse a machine-written 'mean ± sd' cell (used only for Phase 2C's frozen string table)."""
    a, b = s.split("±")
    return float(a), float(b)


def main_performance():
    p_oof, p_fold = "results/phase2b/summary/oof_summary_all.csv", "results/phase2b/summary/per_fold_summary_all.csv"
    o, f = pd.read_csv(ROOT / p_oof), pd.read_csv(ROOT / p_fold)
    df = o.merge(f, on=["method", "partition"])
    def role(m, p):
        if m == "central_grande":
            return "primary reference (centralized)"
        if m == "fedavg_grande":
            return "primary" if p == "natural" else "primary, IID control partition"
        if m == "local_grande":
            return "control (no collaboration)"
        return "control (MLP model class)"
    df.insert(2, "role", [role(m, p) for m, p in zip(df.method, df.partition)])
    df = df[df.partition != "dirichlet0.1"].reset_index(drop=True)
    return prov(df, [p_oof, p_fold], "2B", "2B")


def heterogeneity():
    base = "results/phase2b/summary/"
    o = pd.read_csv(ROOT / base / "oof_summary_all.csv")
    pf = pd.read_csv(ROOT / base / "per_fold_summary_all.csv")
    cs = pd.read_csv(ROOT / base / "consensus_summary_all.csv")
    fx = pd.read_csv(ROOT / base / "consensus_fixed_rounds_summary_all.csv")
    gaps = json.load(open(ROOT / base / "gaps_all.json"))
    loc = pd.read_csv(ROOT / base / "local_model_consensus_all.csv").set_index("partition")
    svc = pd.read_csv(ROOT / base / "structure_vs_central_all.csv")
    svc = svc[svc.method == "fedavg_grande"].groupby("partition").frac_nodes_same_feature_as_central_mean.mean()
    rows = []
    for part in ("natural", "iid", "dirichlet0.1"):
        g = o[(o.method == "fedavg_grande") & (o.partition == part)].iloc[0]
        p = pf[(pf.method == "fedavg_grande") & (pf.partition == part)].iloc[0]
        c = cs[cs.partition == part].iloc[0]
        r20 = fx[(fx.partition == part) & (fx["round"] == 20)].iloc[0]
        gk = gaps[f"central_minus_fedavg_grande_{part}_oof_auc"]
        rows.append({"partition": part, "fedavg_grande_oof_auc": g.oof_auc_mean, "fedavg_grande_oof_auc_sd": g.oof_auc_sd,
                     "fedavg_grande_fold_auc": p.fold_mean_auc, "central_minus_fedavg_oof_auc": gk["mean"],
                     "central_minus_fedavg_oof_auc_sd": gk["sd"], "best_round_mean": c.best_round_mean,
                     "consensus_round20": r20.cons_consensus_mean_mean, "unanimous_frac_round20": r20.cons_frac_nodes_unanimous_mean,
                     "update_cosine_round20": r20.cos_to_aggregate_mean_mean,
                     "consensus_at_best": c.cons_consensus_mean_at_best_mean, "unanimous_frac_at_best": c.cons_frac_nodes_unanimous_at_best_mean,
                     "update_cosine_mean_to_best": c.cos_to_aggregate_mean_mean_to_best_mean,
                     "local_only_final_consensus": loc.loc[part, "mean"], "frac_nodes_same_feature_as_central": svc[part]})
    df = pd.DataFrame(rows)
    return prov(df, [base + n for n in ("oof_summary_all.csv", "per_fold_summary_all.csv", "consensus_summary_all.csv",
                                         "consensus_fixed_rounds_summary_all.csv", "local_model_consensus_all.csv",
                                         "structure_vs_central_all.csv", "gaps_all.json")], "2B (+A1, A3)", "2B")


def personalization():
    p_seed = "results/phase2c/summary/per_seed_fold_means.csv"
    p_cells = "results/phase2c/summary/within_client_per_fold_cells.csv"
    p_json = "results/phase2c/summary/phase2c_summary.json"
    s = pd.read_csv(ROOT / p_seed)
    cells = pd.read_csv(ROOT / p_cells)
    mt = {r["method"]: r for r in json.load(open(ROOT / p_json))["main_table"]}
    names = {"P0": "Global FedAvg GRANDE", "P1": "+ client intercept calibration", "P2": "+ personalized leaves",
             "P3": "+ personalized leaves and estimator weights"}
    rows = []
    for m, label in names.items():
        g = s[s.method == m]
        wc = cells[cells.method == m].groupby("seed").auc.mean()
        r = {"method": m, "label": label}
        for col, key in (("fold_auc", "roc_auc_mean"), ("f1_05", "macro_f1_05_mean"), ("f1_valthr", "macro_f1_valthr_mean"),
                         ("log_loss", "log_loss_mean"), ("brier", "brier_mean"), ("ece10", "ece10_mean")):
            r[col], r[col + "_sd"] = g[key].mean(), g[key].std()
        r["within_client_auc_per_fold_A1"], r["within_client_auc_per_fold_A1_sd"] = wc.mean(), wc.std()
        r["within_client_auc_pooled_frozen_contaminated"] = pm(mt[label]["within-client AUC (mixed, n=5)"])[0]
        r["pooled_oof_auc_secondary"] = pm(mt[label]["pooled OOF AUC (secondary)"])[0]
        # consistency with the frozen generated table (4-decimal rounding)
        assert abs(round(r["fold_auc"], 4) - pm(mt[label]["per-fold AUC"])[0]) < 1.5e-4
        assert abs(round(r["within_client_auc_per_fold_A1"], 4) - pm(mt[label]["A1 within-client AUC per fold (48 client-fold cells)"])[0]) < 1.5e-4
        rows.append(r)
    return prov(pd.DataFrame(rows), [p_seed, p_cells, p_json], "2C (+A1)", "2C")


def compression():
    p = "results/phase2d/summary/main_table.csv"
    d = pd.read_csv(ROOT / p)
    keep = ["config", "partition", "params", "auc", "auc_sd", "d_auc", "d_auc_sd", "retention", "f1_05", "f1_vt", "best_round",
            "gb_to_best", "gb_to_final", "total_round_mb", "total_compression_to_best_x", "uplink_compression_per_round_x",
            "seconds", "peak_rss_mb"]
    d = d[keep]
    d.insert(1, "description", d.config.map({"C0": "256 trees, FP32 (Phase 2B FedAvg, reused)", "C1": "64 trees, FP32",
                                             "C2": "256 trees, INT8 uplink", "C3": "64 trees, INT8 uplink"}))
    return prov(d, [p], "2D", "2D")


def structure_freezing():
    p = "results/phase2e/summary/main_table.csv"
    g0 = json.load(open(ROOT / "results/phase2e/gate0.json"))
    d = pd.read_csv(ROOT / p)
    keep = ["config", "partition", "auc", "auc_sd", "d_auc", "d_auc_sd", "retention", "f1_05", "f1_vt", "best_round", "mb_round_pre",
            "mb_round_post", "post_freeze_round_reduction_pct", "gb_to_best", "gb_to_final", "compression_to_best_x", "seconds", "peak_rss_mb"]
    d = d[keep]
    d.insert(1, "description", d.config.map({"F0": "no freeze (Phase 2B FedAvg, reused)", "F1": "freeze split_index_array after round 20",
                                             "F2": "freeze split_index_array + split_values after round 20",
                                             "F3": "F2 + INT8 uplink"}))
    share = {t["tensor"]: t["pct_fp32_payload"] for t in g0["tensors"]}
    d["structure_share_of_payload_pct"] = share["split_values"] + share["split_index_array"]
    return prov(d, [p, "results/phase2e/gate0.json"], "2E", "2E")


def adaptive_freezing():
    p = "results/phase2f/summary/main_table.csv"
    d = pd.read_csv(ROOT / p)
    keep = ["config", "partition", "auc", "auc_sd", "d_auc", "d_auc_sd", "retention", "fold_auc", "fold_auc_sd", "f1_05", "f1_vt",
            "best_round", "gb_to_best", "gb_to_final", "compression_to_best_x", "mb_round_last", "seconds", "peak_rss_mb", "trigger_round",
            "feat_freeze_median", "full_freeze_median", "threshold_lag_median", "never_feat_frozen", "never_full_frozen",
            "feat_frozen_frac_20", "full_frozen_frac_20", "feat_frozen_frac_final", "full_frozen_frac_final"]
    d = d[keep]
    d.insert(1, "description", d.config.map({"F0": "no freeze (Phase 2B, reused)", "F2": "fixed round-20 structure freeze (Phase 2E, reused)",
                                             "G1": "consensus-gated global freeze", "G2": "consensus-gated progressive node-wise freeze"}))
    return prov(d, [p], "2F", "2F")


def key_numbers(tabs):
    mp, het, per, com, st, ad = (tabs[k] for k in ("main_performance", "heterogeneity", "personalization", "compression",
                                                   "structure_freezing", "adaptive_freezing"))
    f2 = json.load(open(ROOT / "results/phase2f/summary/phase2f_summary.json"))
    def v(df, val, **kw):
        m = np.ones(len(df), bool)
        for k, x in kw.items():
            m &= (df[k] == x).to_numpy()
        assert m.sum() == 1, (kw, m.sum())
        return float(df.loc[m, val].iloc[0])
    K = {}
    def put(name, value, source, note=""):
        K[name] = {"value": value, "source": source, "note": note}
    put("central_grande_oof_auc", v(mp, "oof_auc_mean", method="central_grande"), "results/curated/main_performance.csv")
    for part in ("natural", "iid"):
        put(f"fedavg_grande_{part}_oof_auc", v(mp, "oof_auc_mean", method="fedavg_grande", partition=part), "results/curated/main_performance.csv")
        put(f"local_grande_{part}_oof_auc", v(mp, "oof_auc_mean", method="local_grande", partition=part), "results/curated/main_performance.csv")
        put(f"fedavg_mlp_{part}_oof_auc", v(mp, "oof_auc_mean", method="fedavg_mlp", partition=part), "results/curated/main_performance.csv")
    put("central_mlp_oof_auc", v(mp, "oof_auc_mean", method="central_mlp"), "results/curated/main_performance.csv")
    for part in ("natural", "iid", "dirichlet0.1"):
        put(f"central_minus_fedavg_grande_{part}_oof_auc", v(het, "central_minus_fedavg_oof_auc", partition=part), "results/curated/heterogeneity.csv")
        put(f"fedavg_grande_{part}_consensus_round20", v(het, "consensus_round20", partition=part), "results/curated/heterogeneity.csv")
    put("fedavg_grande_dirichlet_oof_auc", v(het, "fedavg_grande_oof_auc", partition="dirichlet0.1"), "results/curated/heterogeneity.csv",
        "pooled OOF; distorted by cross-fold calibration shifts (2B amendment A3)")
    put("fedavg_grande_dirichlet_fold_auc", v(het, "fedavg_grande_fold_auc", partition="dirichlet0.1"), "results/curated/heterogeneity.csv")
    for m in ("P0", "P1", "P2", "P3"):
        put(f"{m}_within_client_auc_per_fold", v(per, "within_client_auc_per_fold_A1", method=m), "results/curated/personalization.csv")
        put(f"{m}_log_loss", v(per, "log_loss", method=m), "results/curated/personalization.csv")
    put("structure_share_of_payload_pct", v(st, "structure_share_of_payload_pct", config="F0", partition="natural"), "results/phase2e/gate0.json")
    for c in ("C1", "C2", "C3"):
        for part in ("natural", "iid"):
            put(f"{c}_{part}_d_auc", v(com, "d_auc", config=c, partition=part), "results/curated/compression.csv")
            put(f"{c}_{part}_compression_to_best_x", v(com, "total_compression_to_best_x", config=c, partition=part), "results/curated/compression.csv")
    for c in ("F1", "F2", "F3"):
        for part in ("natural", "iid"):
            put(f"{c}_{part}_d_auc", v(st, "d_auc", config=c, partition=part), "results/curated/structure_freezing.csv")
            put(f"{c}_{part}_gb_to_best", v(st, "gb_to_best", config=c, partition=part), "results/curated/structure_freezing.csv")
    for c in ("F0", "F2", "G1", "G2"):
        for part in ("natural", "iid"):
            put(f"2F_{c}_{part}_d_auc", v(ad, "d_auc", config=c, partition=part), "results/curated/adaptive_freezing.csv")
            put(f"2F_{c}_{part}_gb_to_best", v(ad, "gb_to_best", config=c, partition=part), "results/curated/adaptive_freezing.csv")
    put("G1_trigger_round_mean", f2["G1_trigger"]["mean"], "results/phase2f/summary/phase2f_summary.json")
    put("G1_trigger_round_sd", f2["G1_trigger"]["sd"], "results/phase2f/summary/phase2f_summary.json")
    put("G1_never_triggered", f2["G1_trigger"]["never_triggered"], "results/phase2f/summary/phase2f_summary.json")
    put("G3_eligible", f2["G3_eligibility"]["eligible"], "results/phase2f/summary/phase2f_summary.json")
    return K


def fmt(df, cols, nd=4):
    d = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.{nd}f}")
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in d.itertuples(index=False))


def ledger_tables(t):
    L = ["<!-- GENERATED by scripts/build_curated_results.py; do not edit between these markers -->\n"]
    L += ["### Main performance (Phase 2B; pooled OOF AUC = mean ± SD over 3 seeds; per-fold AUC = mean of 5 folds)\n\n",
          fmt(t["main_performance"], ["method", "partition", "role", "oof_auc_mean", "oof_auc_sd", "fold_mean_auc", "oof_f1_valthr_mean", "oof_f1_05_mean"]),
          "\nSource: `results/curated/main_performance.csv`\n\n"]
    L += ["### Heterogeneity and structure (FedAvg GRANDE, Phase 2B)\n\n",
          fmt(t["heterogeneity"], ["partition", "fedavg_grande_oof_auc", "fedavg_grande_fold_auc", "central_minus_fedavg_oof_auc",
                                   "consensus_round20", "unanimous_frac_round20", "update_cosine_round20", "consensus_at_best",
                                   "best_round_mean", "local_only_final_consensus"]),
          "\nSource: `results/curated/heterogeneity.csv`. Dirichlet pooled OOF AUC is distorted by cross-fold score shifts; use the per-fold column.\n\n"]
    L += ["### Personalization under Dirichlet(0.1) label skew (Phase 2C; mean ± SD over 3 seeds of 5-fold means)\n\n",
          fmt(t["personalization"], ["method", "label", "within_client_auc_per_fold_A1", "within_client_auc_per_fold_A1_sd", "fold_auc",
                                     "f1_05", "f1_valthr", "log_loss", "brier", "ece10"]),
          "\nSource: `results/curated/personalization.csv`. The aggregate `fold_auc` is inflated by client identity under label skew; "
          "the within-client per-fold AUC (amendment A1) is the discrimination measure.\n\n"]
    L += ["### Static compression (Phase 2D; ΔAUC and retention vs C0, paired by partition and seed)\n\n",
          fmt(t["compression"], ["config", "description", "partition", "auc", "d_auc", "retention", "gb_to_best", "total_compression_to_best_x", "seconds"]),
          "\nSource: `results/curated/compression.csv`. GB = conceptual tensor bytes, 21 clients, downlink + uplink.\n\n"]
    L += ["### Fixed structural freezing after round 20 (Phase 2E)\n\n",
          fmt(t["structure_freezing"], ["config", "partition", "auc", "d_auc", "retention", "mb_round_post", "gb_to_best",
                                        "compression_to_best_x", "seconds"]),
          f"\nStructure (`split_values` + `split_index_array`) = {t['structure_freezing'].structure_share_of_payload_pct.iloc[0]:.2f}% of the trainable payload. "
          "Source: `results/curated/structure_freezing.csv`.\n\n"]
    L += ["### Consensus-gated adaptive freezing (Phase 2F)\n\n",
          fmt(t["adaptive_freezing"], ["config", "partition", "auc", "d_auc", "retention", "fold_auc", "gb_to_best", "gb_to_final",
                                       "trigger_round", "feat_freeze_median", "full_freeze_median", "never_full_frozen", "seconds"]),
          "\nSource: `results/curated/adaptive_freezing.csv`.\n"]
    L += ["<!-- END GENERATED -->\n"]
    return "".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify committed outputs are up to date; write nothing")
    a = ap.parse_args()
    tabs = {"main_performance": main_performance(), "heterogeneity": heterogeneity(), "personalization": personalization(),
            "compression": compression(), "structure_freezing": structure_freezing(), "adaptive_freezing": adaptive_freezing()}
    files = {f"{k}.csv": v.to_csv(index=False, float_format="%.10g") for k, v in tabs.items()}
    K = key_numbers(tabs)
    files["final_key_numbers.json"] = json.dumps({"generated_by": "scripts/build_curated_results.py", "experiments_tag": TAG,
                                                  "numbers": K}, indent=1, default=float) + "\n"
    files["PROVENANCE.json"] = json.dumps({"experiments_tag": TAG, "configs": {k: {"path": p, "sha256": sha(p), "committed_in": commit_of(p)}
                                                                               for k, p in CONFIGS.items()}}, indent=1) + "\n"
    led = LEDGER.read_text() if LEDGER.exists() else ""
    s, e = "<!-- GENERATED", "<!-- END GENERATED -->\n"
    new_led = None
    if s in led:
        new_led = led[:led.index(s)] + ledger_tables(tabs) + led[led.index(e) + len(e):]
    stale = [n for n, c in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != c]
    if new_led is not None and new_led != led:
        stale.append("docs/RESULTS_LEDGER.md")
    if a.check:
        print("curated results up to date" if not stale else f"STALE: {stale}")
        sys.exit(1 if stale else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for n, c in files.items():
        (OUT / n).write_text(c)
    if new_led is not None:
        LEDGER.write_text(new_led)
    print(f"wrote {len(files)} files to results/curated/" + (" and refreshed docs/RESULTS_LEDGER.md tables" if new_led else ""))


if __name__ == "__main__":
    main()
