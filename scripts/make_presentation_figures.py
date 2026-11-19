#!/usr/bin/env python3
"""Regenerate the two presentation figures (final deck, slides 19 and 21) from saved results. No training.

Writes reports/figures/presentation/<name>.png (200 dpi, the exact image used in the deck), <name>.pdf (vector) and
data/<name>.csv (the plotted values). The style matches the deck's other charts, not the final_style theme.

- slide19_training_curves: hold-out fold, seed 42. Centralized best checkpoint at epoch 51, early stop at epoch 61
  (patience 10); FedAvg runs the full 100 rounds and the selected checkpoint is round 93.
- slide21_consensus_by_round: mean node consensus per round over the runs still active (same data as fig03).

Usage: python -I scripts/make_presentation_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "figures" / "presentation"
(OUT / "data").mkdir(parents=True, exist_ok=True)


def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})
    data.to_csv(OUT / "data" / f"{name}.csv", index=False, float_format="%.10g")
    plt.close(fig)
BLUE, ORANGE, PURPLE, DARK = "#2a78d6", "#eb6834", "#6250d6", "#52514e"
GRID, AXIS, TICK, LABEL = "#e4e3df", "#8a8984", "#52514e", "#0b0b0b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12.5, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11.5, "axes.edgecolor": AXIS, "xtick.color": AXIS,
                     "ytick.color": AXIS, "xtick.labelcolor": TICK, "ytick.labelcolor": TICK, "axes.labelcolor": LABEL})


def style(ax):
    ax.grid(color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.9)


# ---------- slide 19: training curves (hold-out fold, seed 42) ----------
c = pd.read_csv(f"{REPO}/submission/artifacts/holdout_training_curves.csv")
info = pd.read_csv(f"{REPO}/submission/artifacts/core_run_info.csv")
cen, fed = c[c.model == "Centralized GRANDE"], c[c.model == "FedAvg GRANDE"]
rc = info[(info.method == "central_grande") & (info.fold == 0) & (info.seed == 42)].iloc[0]
rf = info[(info.method == "fedavg_grande") & (info.partition == "natural") & (info.fold == 0) & (info.seed == 42)].iloc[0]
best_c, end_c, best_f, end_f = int(rc.best_step), int(rc.steps_executed), int(rf.best_step), int(rf.steps_executed)
assert (best_c, end_c, best_f, end_f) == (51, 61, 93, 100)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 7.0), dpi=200, sharex=True)
lw = 2.5
a1.plot(cen.step, cen.train_loss, color=DARK, lw=lw, label="Centralized, train")
a1.plot(cen.step, cen.val_loss, color=DARK, lw=lw, ls=(0, (3.5, 2)), label="Centralized, validation")
a1.plot(fed.step, fed.train_loss, color=BLUE, lw=lw, label="FedAvg, train")
a1.plot(fed.step, fed.val_loss, color=BLUE, lw=lw, ls=(0, (3.5, 2)), label="FedAvg, validation")
a1.set_ylabel("weighted cross-entropy")
a1.set_ylim(0.34, 0.81)
a1.legend(ncol=2, frameon=False, loc="upper center", handlelength=2.2, columnspacing=1.2)

a2.plot(cen.step, cen.val_auc, color=DARK, lw=lw, label="Centralized GRANDE")
a2.plot(fed.step, fed.val_auc, color=BLUE, lw=lw, label="FedAvg GRANDE")
for a, top in ((a1, 0.74), (a2, 1.0)):
    a.axvline(best_c, ymax=top, color=DARK, ls=":", lw=1.2)
    a.axvline(end_c, ymax=top, color=AXIS, ls=(0, (1, 1.5)), lw=1.2)
    a.axvline(best_f, ymax=top, color=BLUE, ls=":", lw=1.2)
    style(a)
yc = float(cen.val_auc[cen.step == best_c].iloc[0])
yf = float(fed.val_auc[fed.step == best_f].iloc[0])
a2.scatter([best_c], [yc], s=60, color=DARK, zorder=4)
a2.scatter([end_c], [float(cen.val_auc[cen.step == end_c].iloc[0])], s=60, marker="X", color=DARK, zorder=4)
a2.scatter([best_f], [yf], s=60, color=BLUE, zorder=4)
a2.annotate(f"best checkpoint: {best_c}", (best_c, yc), xytext=(best_c - 3, 0.879), ha="right", va="center", fontsize=10.5, color=DARK)
a2.annotate(f"early stop: {end_c}\n(patience 10)", (end_c, 0.866), xytext=(end_c + 2, 0.8735), ha="left", va="center", fontsize=10.5, color=DARK)
a2.annotate(f"selected checkpoint: {best_f}", (best_f, yf), xytext=(best_f - 2, 0.8445), ha="right", va="center", fontsize=10.5, color=BLUE)
a2.annotate("runs to the\n100-round limit", (end_f, float(fed.val_auc.iloc[-1])), xytext=(best_f - 2, 0.810), ha="right", va="center",
            fontsize=10.5, color=BLUE)
a2.set_ylabel("validation ROC-AUC")
a2.set_ylim(0.738, 0.888)
a2.set_xlabel("epoch (centralized) or communication round (FedAvg)")
a2.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=2.2)
a2.set_xlim(-3, 104)
fig.tight_layout(h_pad=2.2)
save(fig, "slide19_training_curves", c)

# ---------- slide 21: consensus per round (mean over runs still active) ----------
d = pd.read_csv(f"{REPO}/reports/figures/final_style/data/fig03_consensus_by_round.csv")
fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=200)
spec = [("iid", ORANGE, (0, (3.5, 2)), "D", "IID control", 3),
        ("natural", BLUE, "-", "o", "natural waves", 4),
        ("dirichlet0.1", PURPLE, (0, (5, 1.5, 1.2, 1.5)), "^", "Dirichlet(0.1) label skew", 2)]
for p, col, ls, mk, lab, z in spec:
    g = d[d.partition == p]
    ax.plot(g["round"], g["mean"], color=col, lw=2.5, ls=ls, zorder=z, label=lab,
            marker=mk, markevery=list(range(9, len(g), 10)), ms=6.5, mec="white", mew=0.9)
ax.axvline(20, color=AXIS, ls=":", lw=1.4)
ax.axhline(0.98, color=AXIS, ls=":", lw=1.4)
ax.text(21, 0.9418, "round 20", color=TICK, fontsize=11)
ax.text(100, 0.9808, "0.98", color=TICK, fontsize=11, ha="right", va="bottom")
gd = d[d.partition == "dirichlet0.1"]
ax.annotate("Dirichlet runs stop early\n(1 run left at round 53)", (gd["round"].iloc[-1], gd["mean"].iloc[-1]),
            xytext=(57, 0.9715), fontsize=10.5, color=PURPLE, va="center")
ax.set_xlim(1, 100)
ax.set_ylim(0.9375, 0.9955)
ax.set_xticks([20, 40, 60, 80, 100])
ax.set_xlabel("communication round")
ax.set_ylabel("mean node consensus")
style(ax)
h, l = ax.get_legend_handles_labels()
ax.legend([h[1], h[0], h[2]], [l[1], l[0], l[2]], frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.03), handlelength=2.6)
fig.tight_layout()
save(fig, "slide21_consensus_by_round", d)
print(f"wrote 2 figures to {OUT.relative_to(REPO)} (centralized best {best_c}, stop {end_c}; FedAvg best {best_f} of {end_f})")
