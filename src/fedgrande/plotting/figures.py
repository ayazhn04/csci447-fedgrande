"""Final-style figures, built only from saved, git-tracked result files (no training, no model evaluation).

Every figure:
- uses `fedgrande.plotting.theme` (style matched to benchmark_figures);
- writes its exact plotted data to reports/figures/final_style/data/<name>.csv;
- checks anchor values against the original phase summaries with asserts, so a figure cannot silently drift from the
  frozen results.

Entry point: scripts/regenerate_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from . import theme as T

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
CUR = RES / "curated"
OUT = ROOT / "reports" / "figures" / "final_style"
DATA = OUT / "data"
PARTS = ("natural", "iid")
FOLDS, SEEDS = range(5), (42, 447, 2026)
N_CLIENTS = 21
GB, MB = 1e9, 1e6


def _save(fig, name, data: pd.DataFrame):
    DATA.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA / f"{name}.csv", index=False, float_format="%.10g")
    return T.save(fig, OUT / name)


def _close(a, b, tol=1e-9, what=""):
    assert abs(float(a) - float(b)) <= tol, f"anchor mismatch {what}: {a} vs {b}"


def _runs(phase, method, parts=PARTS):
    for p in parts:
        for f in FOLDS:
            for s in SEEDS:
                d = RES / phase / "runs" / f"{method}__{p}__f{f}__s{s}"
                yield p, f, s, d


def _active_mean(frames, col):
    """Mean/SD per round over runs that are still running (each run contributes only rounds it executed)."""
    x = pd.concat(frames)
    g = x.groupby("round")[col]
    return pd.DataFrame({"round": g.mean().index, "mean": g.mean().values, "sd": g.std().values, "n_runs": g.count().values})


# ---------------------------------------------------------------------------------------------------------------------
def fig01_pipeline():
    """Schematic of the study design (no data)."""
    fig, ax = T.figure("wide")
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 33)
    boxes = [
        (1, "SpeedDating\nOpenML 40536\n8,378 rows\n120 predictors", T.GRAY),
        (21, "Leakage groups\nmirror pairs →\ngrouped 5-fold CV\n(no group split)", T.SLATE),
        (41, "21 clients\nnatural waves\n+ IID control\n+ Dirichlet stress", T.LAVENDER),
        (61, "FedAvg GRANDE\n256 trees, depth 5\nlocal Adam\n≤100 rounds", T.PURPLE),
        (81, "Extensions\ncalibration · INT8\nfixed / consensus\nstructure freeze", T.ORANGE),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 9), 17.5, 22, boxstyle="round,pad=0.4,rounding_size=1.6", fc="white", ec=col, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, 28.5), 17.5, 2.5, boxstyle="round,pad=0.4,rounding_size=1.0", fc=col, ec=col, lw=0))
        ax.text(x + 8.75, 18.5, txt, ha="center", va="center", fontsize=6.6, color=T.TEXT, linespacing=1.35)
    for x in (19.5, 39.5, 59.5, 79.5):
        ax.annotate("", xy=(x + 1.4, 20), xytext=(x - 0.6, 20), arrowprops=dict(arrowstyle="-|>", color=T.SUBTEXT, lw=1.0))
    ax.text(50, 3.2, "All preprocessing is label-free and fitted on training rows only; every phase's protocol is frozen before its final runs.",
            ha="center", fontsize=6.4, color=T.SUBTEXT)
    T.title(ax, "Study design: leakage-aware federated GRANDE")
    return _save(fig, "fig01_study_design", pd.DataFrame({"stage": [b[1].replace("\n", " | ") for b in boxes]}))


def fig02_main_performance():
    mp = pd.read_csv(CUR / "main_performance.csv")
    order = [("central_grande", "none", "Centralized GRANDE"), ("fedavg_grande", "natural", "FedAvg GRANDE · natural"),
             ("fedavg_grande", "iid", "FedAvg GRANDE · IID"), ("local_grande", "natural", "Local-only GRANDE · natural"),
             ("local_grande", "iid", "Local-only GRANDE · IID"), ("central_mlp", "none", "Centralized MLP"),
             ("fedavg_mlp", "natural", "FedAvg MLP · natural"), ("fedavg_mlp", "iid", "FedAvg MLP · IID")]
    rows = []
    for m, p, lab in order:
        r = mp[(mp.method == m) & (mp.partition == p)].iloc[0]
        rows.append({"label": lab, "method": m, "partition": p, "oof_auc_mean": r.oof_auc_mean, "oof_auc_sd": r.oof_auc_sd})
    d = pd.DataFrame(rows)
    src = pd.read_csv(RES / "phase2b/summary/oof_summary_all.csv")
    for r in d.itertuples():
        _close(r.oof_auc_mean, src[(src.method == r.method) & (src.partition == r.partition)].oof_auc_mean.iloc[0], what=r.label)
    fig, ax = T.figure("medium")
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        c = T.METHOD[r.method]
        ax.errorbar(r.oof_auc_mean, yi, xerr=r.oof_auc_sd, fmt=T.MARKER.get(r.partition, "o") if r.partition != "none" else "s",
                    color=c, mfc=c, mec="black", mew=T.MEW, ms=T.MS, capsize=2.5, lw=1.2)
        ax.text(r.oof_auc_mean + r.oof_auc_sd + 0.004, yi + 0.18, f"{r.oof_auc_mean:.3f}", va="center", fontsize=6.4, color=T.SUBTEXT)
    ax.axvline(d.oof_auc_mean.iloc[0], color=T.GRAY, ls=(0, (1, 2)), lw=T.LW_THIN)
    ax.set_yticks(y, d.label)
    ax.set_xlim(0.70, 0.885)
    ax.set_xlabel("pooled out-of-fold ROC-AUC (mean ± SD, 3 seeds)")
    T.style_axis(ax, grid_axis="x")
    T.title(ax, "FedAvg keeps most of the centralized AUC")
    return _save(fig, "fig02_main_performance", d)


def fig03_consensus_rounds():
    rows, curves = [], {}
    for p in ("natural", "iid", "dirichlet0.1"):
        fr = [pd.read_csv(d / "rounds.csv")[["round", "cons_consensus_mean"]] for _, _, _, d in _runs("phase2b", "fedavg_grande", (p,))]
        c = _active_mean(fr, "cons_consensus_mean").assign(partition=p)
        curves[p] = c
        rows.append(c)
    d = pd.concat(rows)
    fx = pd.read_csv(RES / "phase2b/summary/consensus_fixed_rounds_summary_all.csv")
    for p, c in curves.items():   # anchor: round-20 mean equals the Phase 2B fixed-round summary (all runs reach round 20)
        _close(c[c["round"] == 20]["mean"].iloc[0], fx[(fx.partition == p) & (fx["round"] == 20)].cons_consensus_mean_mean.iloc[0], 1e-12, p)
    fig, ax = T.figure("single")
    for p, c in curves.items():
        ax.plot(c["round"], c["mean"], color=T.PARTITION[p], ls=T.PARTITION_LS[p], label=T.PARTITION_LABEL[p])
        ax.fill_between(c["round"], c["mean"] - c["sd"], c["mean"] + c["sd"], color=T.PARTITION[p], alpha=0.18, lw=0)
    ax.axvline(20, color=T.GRAY, ls=(0, (1, 2)), lw=T.LW_THIN)
    ax.set_xlabel("communication round")
    ax.set_ylabel("mean node feature consensus")
    T.style_axis(ax)
    T.legend(ax, loc="lower right")
    T.title(ax, "Client split-feature consensus rises fast")
    T.note(ax, "FedAvg GRANDE; mean ± SD over runs still active (5 folds × 3 seeds).", y=-0.24)
    return _save(fig, "fig03_consensus_by_round", d)


def fig04_consensus_vs_gap():
    x = pd.read_csv(RES / "phase2b/summary/consensus_fixed_rounds_by_run_all.csv")
    x = x[x["round"] == 20][["partition", "fold", "seed", "cons_consensus_mean", "gap_grande"]].reset_index(drop=True)
    g = pd.read_csv(RES / "phase2b/summary/fold_gaps_all.csv")
    m = x.merge(g, on=["partition", "fold", "seed"], suffixes=("", "_src"))
    assert len(m) == 45 and np.allclose(m.gap_grande, m.gap_grande_src, atol=0, rtol=0)
    fig, ax = T.figure("single")
    for p in ("natural", "iid", "dirichlet0.1"):
        s = x[x.partition == p]
        ax.scatter(s.cons_consensus_mean, s.gap_grande, s=20, color=T.PARTITION[p], edgecolor="black", lw=0.5,
                   marker=T.MARKER[p], label=T.PARTITION_LABEL[p], zorder=3)
    ax.axhline(0, color=T.GRAY, lw=T.LW_THIN)
    ax.set_xlabel("mean node feature consensus at round 20")
    ax.set_ylabel("centralized − FedAvg test AUC\n(same fold and seed)")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(4))
    T.style_axis(ax)
    T.legend(ax, loc="upper right")
    T.title(ax, "Label skew: lower consensus, larger gap")
    T.note(ax, "Descriptive association across 45 fold × seed runs; not causal.", y=-0.27)
    return _save(fig, "fig04_consensus_vs_gap", x)


def fig05_calibration():
    d = pd.read_csv(CUR / "personalization.csv")
    lab = {"P0": "P0\nglobal", "P1": "P1\n+intercept", "P2": "P2\n+leaves", "P3": "P3\n+leaves+EW"}
    fig, axes = T.figure("wide", ncols=2)
    x = np.arange(4)
    cols = [T.CONFIG[m] for m in d.method]
    for ax, (col, sd, ttl, yl) in zip(axes, (("within_client_auc_per_fold_A1", "within_client_auc_per_fold_A1_sd",
                                               "Discrimination within clients is unchanged", "within-client ROC-AUC (per fold)"),
                                              ("log_loss", "log_loss_sd", "Calibration is where the gain is", "test log loss (lower is better)"))):
        ax.bar(x, d[col], yerr=d[sd], color=cols, edgecolor="black", lw=0.6, width=0.62, capsize=2.5, error_kw={"lw": 0.9})
        for xi, v, e in zip(x, d[col], d[sd]):
            ax.text(xi, v + e + (0.01 if col == "log_loss" else 0.0015), f"{v:.3f}", ha="center", va="bottom", fontsize=6.4, color=T.SUBTEXT)
        ax.set_xticks(x, [lab[m] for m in d.method])
        ax.set_ylabel(yl)
        if col.startswith("within"):
            ax.set_ylim(0.70, 0.79)
        T.style_axis(ax, grid_axis="y")
        T.title(ax, ttl)
    T.note(axes[0], "Dirichlet(0.1) label skew; 48 client × fold cells with both classes; mean ± SD over 3 seeds.", y=-0.24)
    return _save(fig, "fig05_calibration_vs_personalization", d[["method", "within_client_auc_per_fold_A1", "within_client_auc_per_fold_A1_sd", "log_loss", "log_loss_sd"]])


def fig06_payload_share():
    g0 = json.load(open(RES / "phase2e/gate0.json"))
    d = pd.DataFrame([{"tensor": t["tensor"], "params": t["params"], "pct_payload": t["pct_fp32_payload"]} for t in g0["tensors"]])
    assert int(d.params.sum()) == g0["total_params"] == 809984
    fig, ax = T.figure("medium")
    name = {"split_values": "split thresholds\n(split_values)", "split_index_array": "split feature logits\n(split_index_array)",
            "estimator_weights": "estimator weights", "leaf_classes_array": "leaf values"}
    col = {"split_values": T.PURPLE, "split_index_array": T.DARK, "estimator_weights": T.AMBER, "leaf_classes_array": T.ORANGE}
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.pct_payload, color=[col[t] for t in d.tensor], edgecolor="black", lw=0.6, height=0.6)
    for yi, v in zip(y, d.pct_payload):
        ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=6.6, color=T.SUBTEXT)
    ax.set_yticks(y, [name[t] for t in d.tensor])
    ax.set_xlim(0, 60)
    ax.set_xlabel("share of FP32 trainable payload (%)")
    T.style_axis(ax, grid_axis="x")
    T.title(ax, "Tree structure ≈ 98% of the FedAvg payload")
    T.note(ax, "256 trees, depth 5, 502 inputs, V=50: 809,984 parameters (3.24 MB) per model per direction.", y=-0.24)
    return _save(fig, "fig06_payload_share", d)


def fig07_post_round20_drift():
    s = pd.read_csv(RES / "phase2e/stability_by_run.csv")
    summ = json.load(open(RES / "phase2e/stability_summary.json"))["summary"]
    rows = []
    for p in PARTS:
        g = s[s.partition == p]
        for k in ("40", "60", "best"):
            rows.append({"partition": p, "at": k, "feature_stability_mean": g[f"feature_stability_{k}"].mean(),
                         "feature_stability_sd": g[f"feature_stability_{k}"].std(),
                         "thr_drift_mean": g[f"thr_drift_numeric_sd_mean_{k}"].mean(), "thr_drift_sd": g[f"thr_drift_numeric_sd_mean_{k}"].std()})
    d = pd.DataFrame(rows)
    for e in summ:
        for k in ("40", "60", "best"):
            _close(d[(d.partition == e["partition"]) & (d["at"] == k)].feature_stability_mean.iloc[0], e[f"feature_stability_{k}_mean"], 1e-12, "stability")
    fig, axes = T.figure("wide", ncols=2)
    xs = np.arange(3)
    for p in PARTS:
        g = d[d.partition == p]
        off = -0.06 if p == "natural" else 0.06
        for ax, m, sd in ((axes[0], "feature_stability_mean", "feature_stability_sd"), (axes[1], "thr_drift_mean", "thr_drift_sd")):
            ax.errorbar(xs + off, g[m], yerr=g[sd], color=T.PARTITION[p], ls=T.PARTITION_LS[p], marker=T.MARKER[p], mec="black",
                        mew=T.MEW, ms=T.MS - 1, capsize=2.5, lw=1.4, label=T.PARTITION_LABEL[p])
    for ax, yl, ttl in ((axes[0], "nodes with the same feature as at round 20", "Feature identities mostly settle"),
                        (axes[1], "|threshold − round-20 threshold| (SD units)", "Thresholds keep drifting slowly")):
        ax.set_xticks(xs, ["round 40", "round 60", "best round"])
        ax.set_ylabel(yl)
        T.style_axis(ax)
        T.title(ax, ttl)
    T.legend(axes[0], loc="upper right")
    T.note(axes[0], "DIAGNOSTIC: ordinary FedAvg (F0) reruns, seed 42, 5 folds, bitwise-identical to Phase 2B.", y=-0.2)
    return _save(fig, "fig07_post_round20_drift", d)


def _frontier_data():
    st = pd.read_csv(CUR / "structure_freezing.csv")
    ad = pd.read_csv(CUR / "adaptive_freezing.csv")
    co = pd.read_csv(CUR / "compression.csv")
    a = st[["config", "partition", "auc", "auc_sd", "gb_to_best"]].assign(group="fixed freeze (2E)")
    b = ad[ad.config.isin(["G1", "G2"])][["config", "partition", "auc", "auc_sd", "gb_to_best"]].assign(group="consensus-gated (2F)")
    c = co[co.config.isin(["C1", "C2", "C3"])][["config", "partition", "auc", "auc_sd", "gb_to_best"]].assign(group="static compression (2D)")
    return pd.concat([a, b, c], ignore_index=True)


def fig08_frontier():
    d = _frontier_data()
    lab = {"F0": "F0 FedAvg (no freeze)", "F1": "F1 freeze features @20", "F2": "F2 freeze structure @20", "F3": "F3 = F2 + INT8",
           "G1": "G1 consensus global freeze", "G2": "G2 consensus node-wise", "C1": "C1 64 trees", "C2": "C2 INT8 uplink", "C3": "C3 64 trees + INT8"}
    fig, ax = T.figure("wide_tall")
    for r in d.itertuples():
        faded = r.group.startswith("static")
        c = T.CONFIG[r.config]
        mk = T.MARKER[r.config]
        ax.errorbar(r.gb_to_best, r.auc, yerr=r.auc_sd, fmt=mk, ms=T.MS + (2 if r.config in ("G1", "G2") else 0),
                    color=c, mfc=c if r.partition == "natural" else "white", mec=c if faded else "black", mew=T.MEW if not faded else 1.0,
                    capsize=2.5, lw=1.0, alpha=0.45 if faded else 1.0, zorder=2 if faded else 3)
    ax.set_xscale("log")
    ax.set_xlabel("cumulative communication to best validation round (GB, conceptual tensor bytes, log scale)")
    ax.set_ylabel("pooled OOF ROC-AUC (mean ± SD, 3 seeds)")
    handles = [Line2D([], [], color=T.CONFIG[k], marker=T.MARKER[k], ls="none", mec="black", mew=T.MEW, ms=T.MS, label=v,
                      alpha=0.45 if k.startswith("C") else 1) for k, v in lab.items()]
    handles += [Line2D([], [], color=T.SUBTEXT, marker="o", ls="none", mfc=T.SUBTEXT, label="filled = natural waves"),
                Line2D([], [], color=T.SUBTEXT, marker="o", ls="none", mfc="white", label="open = size-matched IID")]
    T.legend(ax, handles=handles, loc="lower right", ncol=2, fontsize=6.3)
    T.style_axis(ax)
    T.title(ax, "Freezing tree structure after it stabilizes cuts communication ~4× for ≈0.002 AUC")
    return _save(fig, "fig08_communication_frontier", d)


def _per_round_bytes():
    rows = []
    for p, f, s, dd in _runs("phase2b", "fedavg_grande"):
        r = pd.read_csv(dd / "rounds.csv")
        rows.append(pd.DataFrame({"config": "F0", "partition": p, "fold": f, "seed": s, "round": r["round"], "mb": 136077312 / MB}))
    for cfg, ph, me in (("F2", "phase2e", "fedavg_grande_frzstruct"), ("G1", "phase2f", "fedavg_grande_cgsf_global"),
                        ("G2", "phase2f", "fedavg_grande_cgsf_node")):
        for p, f, s, dd in _runs(ph, me):
            r = pd.read_csv(dd / "rounds.csv")
            rows.append(pd.DataFrame({"config": cfg, "partition": p, "fold": f, "seed": s, "round": r["round"], "mb": (r.bytes_down + r.bytes_up) / MB}))
    x = pd.concat(rows, ignore_index=True)
    init = 8 * 256 * 50 * N_CLIENTS   # features_by_estimator (int64, E×V) once per client at initialization
    v3 = json.load(open(RES / "phase2f/runs/fedavg_grande_cgsf_node__iid__f0__s42/result.json"))["communication"]["v3"]
    assert v3["bytes_once_at_init_buffers"] == init
    x["cum_gb"] = (init + x.groupby(["config", "partition", "fold", "seed"]).mb.cumsum() * MB) / GB
    rt = pd.read_csv(RES / "phase2f/summary/runs_table.csv")   # anchor: per-run cumulative GB at the best round
    for r in rt.itertuples():
        v = x[(x.config == r.config) & (x.partition == r.partition) & (x.fold == r.fold) & (x.seed == r.seed) & (x["round"] == r.best_round)].cum_gb
        _close(v.iloc[0], r.gb_to_best, 1e-9, f"{r.config} {r.partition} f{r.fold} s{r.seed}")
    return x


def fig09_10_communication_curves():
    x = _per_round_bytes()
    per = x.groupby(["config", "partition", "round"]).mb.mean().reset_index()
    cum = []
    for (c, p), g in x.groupby(["config", "partition"]):
        w = g.pivot_table(index="round", columns=["fold", "seed"], values="cum_gb").reindex(range(1, 101)).ffill()
        cum.append(pd.DataFrame({"config": c, "partition": p, "round": w.index, "cum_gb_mean": w.mean(1).values}))
    cum = pd.concat(cum, ignore_index=True)
    out = []
    lab = {"F0": "F0 FedAvg", "F2": "F2 fixed freeze @20", "G1": "G1 consensus global", "G2": "G2 consensus node-wise"}
    fig, ax = T.figure("single_tall")
    for c in ("F0", "F2", "G1", "G2"):
        for p in PARTS:
            g = per[(per.config == c) & (per.partition == p)]
            ax.plot(g["round"], g.mb, color=T.CONFIG[c], ls=T.PARTITION_LS[p], lw=1.5, label=f"{lab[c]} · {p}" if p == "natural" else None)
    ax.set_yscale("log")
    ax.set_ylim(1.5, 4000)
    ax.set_xlabel("communication round")
    ax.set_ylabel("MB per round (21 clients, down + up)")
    T.style_axis(ax)
    T.legend(ax, loc="upper right", fontsize=6.3)
    T.title(ax, "Per-round traffic")
    T.note(ax, "Mean over runs still active; solid = natural, dashed = IID.", y=-0.2)
    out += _save(fig, "fig09_traffic_per_round", per)
    fig, ax = T.figure("single_tall")
    for c in ("F0", "F2", "G1", "G2"):
        for p in PARTS:
            g = cum[(cum.config == c) & (cum.partition == p)]
            ax.plot(g["round"], g.cum_gb_mean, color=T.CONFIG[c], ls=T.PARTITION_LS[p], lw=1.5, label=f"{lab[c]}" if p == "natural" else None)
    ax.set_xlabel("communication round")
    ax.set_ylabel("cumulative GB (incl. initialization)")
    T.style_axis(ax)
    T.legend(ax, loc="upper left", fontsize=6.3)
    T.title(ax, "Cumulative communication")
    T.note(ax, "Mean over 15 runs; a stopped run's total is carried forward. Solid = natural, dashed = IID.", y=-0.2)
    out += _save(fig, "fig10_cumulative_communication", cum)
    return out


def fig11_g1_trigger():
    s = json.load(open(RES / "phase2f/summary/phase2f_summary.json"))["G1_trigger"]
    d = pd.DataFrame(s["per_run"])
    _close(d.trigger_round.mean(), s["mean"], 1e-12, "trigger mean")
    fig, ax = T.figure("single")
    bins = np.arange(16.5, 25.5, 1)
    ax.hist([d[d.partition == p].trigger_round for p in PARTS], bins=bins, stacked=True, color=[T.PARTITION[p] for p in PARTS],
            edgecolor="black", lw=0.6, label=[T.PARTITION_LABEL[p] for p in PARTS])
    ax.axvline(20, color=T.DARK, ls=(0, (5.5, 2.2)), lw=1.4, label="F2 fixed freeze (round 20)")
    ax.set_xlabel("G1 trigger round (freeze applies from the next round)")
    ax.set_ylabel("runs (of 30)")
    ax.set_xticks(range(17, 25))
    ax.set_ylim(0, 11.5)
    T.style_axis(ax, grid_axis="y")
    T.legend(ax, loc="upper right", fontsize=6.3)
    T.title(ax, f"G1 triggers at round {s['mean']:.1f} ± {s['sd']:.1f}")
    return _save(fig, "fig11_g1_trigger_rounds", d)


def fig12_g2_freezing():
    fr, tm = [], []
    for p, f, s, dd in _runs("phase2f", "fedavg_grande_cgsf_node"):
        r = pd.read_csv(dd / "rounds.csv")
        fr.append(r[["round", "frac_feat_frozen", "frac_full_frozen"]].assign(partition=p))
        tm.append(r[["round", "newly_feat_frozen", "newly_full_frozen"]].assign(partition=p, fold=f, seed=s))
    fr, tm = pd.concat(fr), pd.concat(tm)
    rt = pd.read_csv(RES / "phase2f/summary/runs_table.csv")
    rt = rt[rt.config == "G2"]
    n_nodes = 256 * 31
    for r in rt.itertuples():   # anchor: nodes ever frozen per run equals the run table's never-frozen fraction
        t = tm[(tm.partition == r.partition) & (tm.fold == r.fold) & (tm.seed == r.seed)]
        _close(t.newly_feat_frozen.sum(), round((1 - r.never_feat_frozen) * n_nodes), 0, "feat frozen count")
        _close(t.newly_full_frozen.sum(), round((1 - r.never_full_frozen) * n_nodes), 0, "full frozen count")
    curve = fr.groupby(["partition", "round"])[["frac_feat_frozen", "frac_full_frozen"]].mean().reset_index()
    out = []
    fig, ax = T.figure("single")
    for p in PARTS:
        g = curve[curve.partition == p]
        ax.plot(g["round"], g.frac_feat_frozen, color=T.PARTITION[p], ls="-", label=f"feature-frozen · {p}")
        ax.plot(g["round"], g.frac_full_frozen, color=T.PARTITION[p], ls=(0, (1.4, 1.6)), label=f"threshold-frozen · {p}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("communication round")
    ax.set_ylabel("fraction of internal nodes frozen")
    T.style_axis(ax)
    T.legend(ax, loc="lower right", fontsize=6.3)
    T.title(ax, "G2 progressive freezing")
    T.note(ax, "Mean over runs still active (fraction at the start of each round).", y=-0.24)
    out += _save(fig, "fig12_g2_progressive_freezing", curve)
    h = tm.groupby("round")[["newly_feat_frozen", "newly_full_frozen"]].sum().reset_index()
    fig, ax = T.figure("single")
    ax.bar(h["round"] - 0.2, h.newly_feat_frozen, width=0.4, color=T.DARK, label="feature logits")
    ax.bar(h["round"] + 0.2, h.newly_full_frozen, width=0.4, color=T.ORANGE, label="thresholds")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("freeze decision round")
    ax.set_ylabel("nodes frozen (all 30 G2 runs)")
    T.style_axis(ax, grid_axis="y")
    T.legend(ax, loc="upper right")
    T.title(ax, "G2 freezes at the earliest allowed rounds")
    T.note(ax, "Decision after round r applies from r+1; earliest: features 13, thresholds 16.", y=-0.24)
    out += _save(fig, "fig13_g2_freeze_timing", h)
    return out


ALL = [fig01_pipeline, fig02_main_performance, fig03_consensus_rounds, fig04_consensus_vs_gap, fig05_calibration,
       fig06_payload_share, fig07_post_round20_drift, fig08_frontier, fig09_10_communication_curves, fig11_g1_trigger,
       fig12_g2_freezing]


def build_all() -> list[Path]:
    out = []
    for f in ALL:
        out += f()
    return out
