# Figure style

**The style is matched to `benchmark_figures`. The plot composition remains specific to this project.**

We reuse the reference repository's visual language: palette, typography, lines and markers, axes, grid, legend and
export settings. We do **not** copy its composite multi-panel layouts.

## Reference inspected

- **Repository:** `https://github.com/ayazhn04/benchmark_figures` (private).
  - Read-only shallow clone on 2026-11-08 of branch `claude/figure-4-1-composite-tkphge`, the only branch.
  - Nothing in that repository was modified.
- **Files inspected:**
  - `scripts/make_fig4_1_composite_magma_final.py` (and its `_v4_before_final` variant)
  - `make_fig4_2_…`, `make_fig4_3_…`, `make_fig4_4_…`, `make_fig4_5_…`, `make_fig4_6_…`, `make_fig4_7_composite_magma_final.py`
  - `scripts/make_visual_abstract_magma_final.py`
- **Content:** the repository contains plotting scripts only; no exported images are committed. Style values were
  therefore extracted from the code: `mpl.rcParams.update`, the color constants, `style_axis()`, the legend calls and
  the `savefig` calls. The values were consistent across all 9 scripts.

**Implementation:** `src/fedgrande/plotting/theme.py` is the only place where style is defined. Figures are built in
`src/fedgrande/plotting/figures.py`.

## Palette

**Neutrals.** These are the exact reference values.

| Token | Hex | Use |
|---|---|---|
| `BG` | `#FBFAFE` | figure background |
| `CARD_FACE` | `#FFFFFF` | axes background |
| `TEXT` | `#1A0F2B` | titles, primary text |
| `SUBTEXT` | `#59496A` | axis labels, tick labels, notes |
| `GRID` | `#EBE4F2` | grid lines |
| `SPINE` | `#DCD2E8` | axis spines, legend frame |
| `CARD_EDGE` | `#E6DCEF` | card/box outlines |

**Magma-family accents.** These are the exact reference values; the role in the reference is in parentheses.

| Token | Hex | Reference role | Project mapping |
|---|---|---|---|
| `PURPLE` | `#772A8E` | main method ("GAN") | natural partition; FedAvg GRANDE; C2; P1 |
| `AMBER` | `#F2A93B` | comparator ("diffusion") | IID partition; G1; P2 |
| `GRAY` | `#8C93A1` | reference | centralized GRANDE; F0/C0/P0 baselines |
| `ORANGE` | `#E0792A` | annotation / highlight | local-only GRANDE; G2; C3; P3 |
| `DARK` | `#3C1A5B` | solid phase | F2 (fixed structure freeze) |
| `SLATE` | `#738196` | bicubic baseline | Dirichlet stress partition; F3 |
| `LAVENDER` | `#9483B0` | connectors | FedAvg MLP; F1; C1 |
| `LIGHT_GRAY` | `#C0C4CC` | low-resolution baseline | centralized MLP |
| `YELLOW` | `#F5E27A` | fragments | not used: too little contrast on white |

## Typography

- **Font:** DejaVu Sans, the reference's `font.family`. It ships with matplotlib, so it is always installed and needs no
  fallback. No font files are copied into this repository.
- **Hierarchy** (points):

  | Element | Size and weight |
  |---|---|
  | base text | 8.0 |
  | axes title | 8.6, **bold**, `TEXT`, pad 4.5 |
  | axis labels | 7.6, `SUBTEXT` |
  | tick labels | 6.8, `SUBTEXT` |
  | legend | 6.9 (6.3 in dense legends) |
  | value annotations | 6.4–6.6 |
  | figure notes | 6.2, `SUBTEXT` |

- **Text in exported files:**
  - PDF and PS embed TrueType fonts (`pdf.fonttype = 42`), so text stays editable and selectable.
  - SVG keeps text as text (`svg.fonttype = none`).

## Lines, markers, axes

| Element | Rule |
|---|---|
| Data lines | 1.85 pt (1.4–1.5 in dense multi-series plots) |
| Helper / reference lines | 1.0 pt, dotted `(0, (1, 2))` |
| Line styles | solid = natural / primary; dashed `(0, (5.5, 2.2))` = IID / secondary; dotted `(0, (1.4, 1.6))` = Dirichlet / thresholds |
| Markers | 5.5 pt with a **black edge of 0.6 pt** (the reference scatter style). Filled = natural, open = IID in the frontier figure. Shape encodes method (o F0/C0, s F1/C1, ^ F2/C2, v F3, D G1/C3, * G2). |
| Uncertainty | Error bars with 2.5 pt caps for mean ± SD over seeds; shaded bands at alpha 0.18 for SD over runs |
| Spines | all four kept, `SPINE` color, 0.7 pt |
| Ticks | major size 2.6, width 0.65, `SUBTEXT` |
| Grid | on, behind the data, `GRID` color, 0.55 pt, alpha 0.9. Bar charts use a single-axis grid. |
| Legend | framed, white face, `SPINE` edge, frame 0.6 pt, alpha 0.94, borderpad 0.45, labelspacing 0.34, handlelength 2.3 |

## Sizes and export

| Name | Inches | Use |
|---|---|---|
| `single` | 3.5 × 2.6 | single-column figure |
| `single_tall` | 3.5 × 3.2 | single column with a long legend |
| `medium` | 4.6 × 3.0 | long category labels |
| `wide` | 7.2 × 2.8 | full-width figure or two related panels |
| `wide_tall` | 7.2 × 3.6 | full-width frontier |

- **PNG:** 450 dpi, as in the reference.
- **PDF:** vector, the preferred format for the report.
- Both are saved with `bbox_inches="tight"`. Metadata dates are removed, so a rerun gives byte-identical files.

**Deviation from the reference.** The reference positions panels by hand and avoids `tight_layout`, because its figures
are large composites. Ours are individual figures, so we use `tight_layout` plus a tight bounding box.

## Accessibility

- Partition encodings never rely on color alone: natural, IID and Dirichlet also differ in line style and marker shape.
- Method encodings differ in marker shape.
- Purple, amber and slate differ clearly in lightness, so they remain distinguishable in grayscale and for the common
  color-vision deficiencies.
- Yellow is avoided.
- Text is at least 6.2 pt at print size. All text on light backgrounds uses `TEXT` or `SUBTEXT`.

## Adding a new figure

```python
from fedgrande.plotting import theme as T
fig, ax = T.figure("single")
ax.plot(x, y, color=T.PARTITION["natural"], ls=T.PARTITION_LS["natural"], label=T.PARTITION_LABEL["natural"])
T.style_axis(ax)
T.legend(ax, loc="lower right")
T.title(ax, "Short claim-style title")
T.note(ax, "Source / caveat line.")
T.save(fig, out_dir / "figXX_name")   # writes .png and .pdf
```

Read data only from saved result files. Add an anchor `assert` against the phase summary, and register the function in
`figures.ALL`.
