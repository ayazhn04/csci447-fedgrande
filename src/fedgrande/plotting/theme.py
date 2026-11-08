"""Central plotting theme for final-facing figures.

The visual language (palette, typography, line/marker rules, axes, grid, legend, export) is matched to the private
reference repository `ayazhn04/benchmark_figures` (scripts/make_fig4_*_composite_magma_final.py). Only the style is
reused; figure composition is specific to this project. See docs/FIGURE_STYLE.md.

Use:
    from fedgrande.plotting import theme
    theme.apply()
    fig, ax = theme.figure("single")
    ...
    theme.style_axis(ax)
    theme.save(fig, out_dir / "fig_name")   # writes .png (450 dpi) and .pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ---- neutrals (exact values from benchmark_figures) -------------------------------------------------------------------
BG = "#FBFAFE"          # figure background
CARD_FACE = "#FFFFFF"   # axes background
CARD_EDGE = "#E6DCEF"
GRID = "#EBE4F2"
SPINE = "#DCD2E8"
TEXT = "#1A0F2B"        # titles, primary text
SUBTEXT = "#59496A"     # axis labels, ticks, secondary text

# ---- magma-family accents (exact values from benchmark_figures) -------------------------------------------------------
PURPLE = "#772A8E"      # reference: "GAN"/main method
AMBER = "#F2A93B"       # reference: "DIFFUSION"/comparator
GRAY = "#8C93A1"        # reference: "REFERENCE"
ORANGE = "#E0792A"      # reference: annotations / largest component
DARK = "#3C1A5B"        # reference: solid phase
SLATE = "#738196"       # reference: bicubic baseline
LAVENDER = "#9483B0"    # reference: connectors
LIGHT_GRAY = "#C0C4CC"  # reference: low-resolution baseline
YELLOW = "#F5E27A"      # reference: disconnected fragments (use sparingly; low contrast on white)

# ---- project semantic mappings -------------------------------------------------------------------------------------
PARTITION = {"natural": PURPLE, "iid": AMBER, "dirichlet0.1": SLATE, "none": GRAY}
PARTITION_LABEL = {"natural": "natural waves", "iid": "size-matched IID", "dirichlet0.1": "Dirichlet α=0.1", "none": "centralized"}
PARTITION_LS = {"natural": "-", "iid": (0, (5.5, 2.2)), "dirichlet0.1": (0, (1.4, 1.6)), "none": "-"}
METHOD = {"central_grande": GRAY, "fedavg_grande": PURPLE, "local_grande": ORANGE, "central_mlp": LIGHT_GRAY, "fedavg_mlp": LAVENDER}
CONFIG = {"F0": GRAY, "F1": LAVENDER, "F2": DARK, "F3": SLATE, "G1": AMBER, "G2": ORANGE,
          "C0": GRAY, "C1": LAVENDER, "C2": PURPLE, "C3": ORANGE, "P0": GRAY, "P1": PURPLE, "P2": AMBER, "P3": ORANGE}
MARKER = {"F0": "o", "F1": "s", "F2": "^", "F3": "v", "G1": "D", "G2": "*", "C0": "o", "C1": "s", "C2": "^", "C3": "D",
          "natural": "o", "iid": "D", "dirichlet0.1": "^"}

# ---- sizes ----------------------------------------------------------------------------------------------------------
SIZES = {"single": (3.5, 2.6), "single_tall": (3.5, 3.2), "medium": (4.6, 3.0), "wide": (7.2, 2.8), "wide_tall": (7.2, 3.6)}
LW = 1.85          # data lines
LW_THIN = 1.0      # reference / helper lines
MS = 5.5           # markers
MEW = 0.6          # marker edge width (black edge, as in the reference scatter style)
DPI = 450          # PNG export
RC = {
    "font.family": "DejaVu Sans",   # bundled with matplotlib, so always installed
    "font.size": 8.0,
    "axes.titlesize": 8.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": SUBTEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "text.color": TEXT,
    "axes.facecolor": CARD_FACE,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.55,
    "grid.alpha": 0.9,
    "axes.axisbelow": True,
    "lines.linewidth": LW,
    "lines.markersize": MS,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": SPINE,
    "legend.framealpha": 0.94,
    "legend.borderpad": 0.45,
    "legend.labelspacing": 0.34,
    "legend.handlelength": 2.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply() -> None:
    mpl.rcParams.update(RC)


def figure(size: str = "single", ncols: int = 1, **kw):
    apply()
    return plt.subplots(1, ncols, figsize=SIZES[size], **kw)


def style_axis(ax, grid: bool = True, grid_axis: str = "both") -> None:
    """Spines, ticks and grid exactly as `style_axis` in benchmark_figures."""
    for sp in ax.spines.values():
        sp.set_color(SPINE)
        sp.set_linewidth(0.7)
    ax.tick_params(colors=SUBTEXT, labelcolor=SUBTEXT)
    ax.grid(grid, axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.9)
    ax.set_axisbelow(True)


def title(ax, text: str, pad: float = 4.5) -> None:
    ax.set_title(text, pad=pad, color=TEXT, fontweight="bold")


def legend(ax, **kw):
    leg = ax.legend(**kw)
    if leg is not None:
        leg.get_frame().set_linewidth(0.6)
        leg.set_zorder(8)
    return leg


def note(ax, text: str, **_ignored) -> None:
    """Small source/caveat line placed under the whole figure (added in `save`, outside the layout engine)."""
    ax.figure._fedgrande_note = text


def save(fig, stem: Path, pdf: bool = True) -> list[Path]:
    """Write <stem>.png at 450 dpi (+ <stem>.pdf, vector, TrueType fonts embedded). Metadata dates are fixed so outputs
    are byte-stable across reruns."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    txt = getattr(fig, "_fedgrande_note", None)
    if txt:
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        fig.text(0.012, 0.012, txt, fontsize=6.2, color=SUBTEXT, ha="left", va="bottom")
    else:
        fig.tight_layout()
    kw = {"facecolor": BG, "bbox_inches": "tight", "pad_inches": 0.05}
    out = [stem.with_suffix(".png")]
    fig.savefig(out[0], dpi=DPI, metadata={"Software": None}, **kw)
    if pdf:
        out.append(stem.with_suffix(".pdf"))
        fig.savefig(out[1], metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}, **kw)
    plt.close(fig)
    return out
