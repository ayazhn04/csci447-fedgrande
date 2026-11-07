"""Figures generated only from saved Phase 2B machine-readable outputs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fedgrande.paths import ROOT  # noqa: E402

RUNS = ROOT / "results" / "phase2b" / "runs"
SUM = ROOT / "results" / "phase2b" / "summary"
FIG = ROOT / "reports" / "figures" / "phase2b"
COLORS = {"natural": "#2a6fdb", "iid": "#e07b39", "dirichlet0.1": "#7a3db8"}


def rounds_frame(method, parts):
    rows = []
    runs = pd.read_csv(SUM / "runs_table_core.csv") if (SUM / "runs_table_core.csv").exists() else None
    for d in sorted(RUNS.glob(f"{method}__*")):
        part = d.name.split("__")[1]
        if part in parts and (d / "rounds.csv").exists():
            r = pd.read_csv(d / "rounds.csv")
            r["partition"], r["job"] = part, d.name
            rows.append(r)
    return pd.concat(rows) if rows else pd.DataFrame()


def band(ax, df, col, part, label=None):
    g = df[df.partition == part].groupby("round")[col]
    m, s, n = g.mean(), g.std(), g.size()
    ax.plot(m.index, m.values, color=COLORS[part], label=label or part)
    ax.fill_between(m.index, (m - s).values, (m + s).values, color=COLORS[part], alpha=0.15, linewidth=0)


def main(parts=("natural", "iid")):
    os.makedirs(FIG, exist_ok=True)
    R = rounds_frame("fedavg_grande", parts)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for p in parts:
        band(axes[0], R, "cons_consensus_mean", p)
        band(axes[1], R, "cons_global_agreement", p)
        band(axes[2], R, "val_auc", p)
    for ax, t in zip(axes, ("Mean node feature consensus across clients", "Client selections matching global model", "Pooled validation ROC-AUC (global model)")):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("communication round")
        ax.legend(fontsize=8)
    fig.suptitle("FedAvg GRANDE (256 estimators, depth 5): mean ± SD over folds × seeds; runs shown until they stop", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fedavg_grande_consensus_rounds.png", dpi=150)
    plt.close(fig)

    M = rounds_frame("fedavg_mlp", parts)
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in parts:
        band(ax, R, "val_auc", p, f"GRANDE {p}")
        g = M[M.partition == p].groupby("round").val_auc.mean()
        ax.plot(g.index, g.values, color=COLORS[p], linestyle="--", label=f"MLP {p}")
    ax.set_xlabel("communication round")
    ax.set_ylabel("pooled validation ROC-AUC")
    ax.legend(fontsize=8)
    ax.set_title("FedAvg validation curves", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fedavg_val_auc_rounds.png", dpi=150)
    plt.close(fig)

    C = pd.read_csv(SUM / "consensus_by_run_core.csv") if not os.environ.get("CONS_FILE") else pd.read_csv(os.environ["CONS_FILE"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p, g in C.groupby("partition"):
        ax.scatter(g.cons_consensus_mean_mean_to_best, g.gap_grande, color=COLORS.get(p, "k"), label=p, s=22)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("mean node consensus (rounds 1..best)")
    ax.set_ylabel("central − FedAvg GRANDE test AUC (fold)")
    ax.legend(fontsize=8)
    ax.set_title("Structural consensus vs federation gap (fold × seed runs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "consensus_vs_gap.png", dpi=150)
    plt.close(fig)

    W = pd.read_csv(SUM / "per_wave_mean_over_seeds_core.csv")
    nat = W[W.partition.isin(["none", "natural"])]
    piv = nat.pivot_table(index="wave", columns="method", values="auc_mean")
    cols = [c for c in ("central_grande", "fedavg_grande", "local_grande", "central_mlp", "fedavg_mlp") if c in piv]
    fig, ax = plt.subplots(figsize=(11, 4))
    piv[cols].plot(kind="bar", ax=ax, width=0.8)
    sizes = nat.drop_duplicates("wave").set_index("wave")
    ax.set_xticklabels([f"{w}\n(n={int(sizes.loc[w, 'n'])}, +{int(sizes.loc[w, 'pos'])})" for w in piv.index], fontsize=7)
    ax.set_ylim(max(0.0, float(piv[cols].min().min()) - 0.05), 1.0)
    ax.set_ylabel("pooled-OOF ROC-AUC (mean over 3 seeds)")
    ax.set_title("Per-wave ROC-AUC, natural partition (small waves are unstable)", fontsize=10)
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(FIG / "per_wave_auc_natural.png", dpi=150)
    plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("natural", "iid"))
