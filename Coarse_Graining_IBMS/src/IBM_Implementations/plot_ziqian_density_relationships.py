"""Figure: adult-larval density relationships from Ziqian's IBM (house style).

A faceted scatter of density-normalised adult output against larval input for
Ziqian's R larval IBM, reproducing the structure of that model's ``fig_4`` but
rendered in the thesis house style (see ``plot_style``).  Each panel is one
(food regime x temperature x larval-density) treatment; overlapping runs are
counted so a point's *area* is proportional to how many identical
(larvae, adults) outcomes it represents (ggplot's ``scale_size_area``).

The data live under ``data/ziqian_ibm_runs`` as ``S*/csv/<RUN_ID>/stage_df.csv``,
where ``<RUN_ID>`` matches ``S<d>_T<d>_<COMP>_F<d>_D<DENS>``.  Only the
stage-count file is needed; larval-trajectory logs are not.

Run via the thin wrapper ``src/conference_figures.py --figure 9``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import plot_style
from paths import ZIQIAN_RUNS_DIR

#: Directory names look like ``S1_T20_EX_F21_DL``.
RUN_ID_RE = re.compile(r"^S\d+_T\d+_[A-Z]+_F\d+_D[A-Z]+$")

# LaTeX-safe facet / legend labels (``text.usetex`` is on: escape ``&``, wrap
# maths in ``$...$``).
TEMP_LABELS = {
    "T20": r"Temperature $=20^{\circ}$C",
    "T30": r"Temperature $=30^{\circ}$C",
}
FOOD_LABELS = {
    "F21": r"Modest \& Early Food",
    "F22": r"Modest \& Late Food",
    "F31": r"Ample \& Early Food",
    "F32": r"Ample \& Late Food",
}
DENSITY_LABELS = {
    "DH": r"High density (7 larvae/cell)",
    "DL": r"Low density (1 larva/cell)",
}
COMPETITION_LABELS = {"EX": "Exploitation", "IN": "Interference"}

#: Two on-palette colours for the larval-density levels (IBM blue / magenta-pink).
DENSITY_COLOURS = {"DH": plot_style.IBM_PALETTE[0], "DL": plot_style.IBM_PALETTE[2]}

#: Density normalisation: high-density ponds seed 7 larvae/cell, low-density 1.
DENSITY_SCALE = {"DH": 7.0, "DL": 1.0}

DEFAULT_COMP_ORDER = ["Exploitation", "Interference"]
DEFAULT_TEMP_ORDER = ["T20", "T30"]
DEFAULT_DENSITY_ORDER = ["DH", "DL"]
DEFAULT_FOOD_ORDER = ["F21", "F22", "F31", "F32"]

STRIP_GREY = "#D9D9D9"      # theme_bw facet strips
REF_LINE_GREY = "#B3B3B3"   # 1:1 reference line (ggplot grey70)
GRID_GREY = "#CCCCCC"

#: ggplot ``scale_size_area(max_size=...)``: point diameter (mm) at the largest
#: overlap count.  10 mm matches the reference ``fig_4`` visual weight given this
#: dataset's global maximum overlap count.
DESIRED_MAX_SIZE = 10.0

#: Floor on the drawn point diameter (mm).  Strict area-proportional sizing makes
#: non-overlapping (n = 1) points nearly invisible against the huge global max, so
#: clamp small points to this minimum diameter to keep them legible.
MIN_POINT_MM = 1.6

#: Square-panel side length (inches) for the full 4x4 grid; the cropped figure
#: uses larger panels (fewer of them) for legibility.  The figure size is derived
#: from the panel size, not the other way round.
PANEL_IN_FULL = 1.44
PANEL_IN_CROPPED = 2.35


# ---------------------------------------------------------------------------
# Data import
# ---------------------------------------------------------------------------
def _discover_stage_files(root: Path) -> list[Path]:
    """Every ``stage_df.csv`` whose parent directory is a valid run id."""
    return sorted(
        f for f in root.rglob("stage_df.csv") if RUN_ID_RE.match(f.parent.name)
    )


def _extract_meta(stage_file: Path) -> dict[str, str]:
    """Parse Scenario/Temperature/Competition/Food/Density from the run id."""
    parts = stage_file.parent.name.split("_")
    if len(parts) != 5:
        raise ValueError(f"Cannot parse run id: {stage_file.parent.name}")
    keys = ["Scenario", "Temperature", "Competition", "Food", "Density"]
    return dict(zip(keys, parts))


def _read_stage_file(stage_file: Path) -> pd.DataFrame:
    df = pd.read_csv(stage_file)
    for key, value in _extract_meta(stage_file).items():
        df[key] = value
    return df


def _size_legend_breaks(max_count: int) -> list[int]:
    """Preferred round breaks filtered to ``[1, max_count]`` plus the max."""
    preferred = [1, 25, 75, 150, int(max_count)]
    return sorted({b for b in preferred if 1 <= b <= max_count})


def load_counts(root: Path = ZIQIAN_RUNS_DIR):
    """Load every run, density-normalise, and count coincident coordinates.

    Returns ``(all_df, global_max_count, size_legend_breaks)`` where ``all_df``
    carries the plotting columns ``Larvae_plot``/``Adults_plot`` and the overlap
    count ``n`` for each row.
    """
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Ziqian runs directory not found: {root}")
    stage_files = _discover_stage_files(root)
    if not stage_files:
        raise SystemExit(f"No 'stage_df.csv' files found under {root}.")

    all_df = pd.concat([_read_stage_file(f) for f in stage_files], ignore_index=True)
    all_df["Competition"] = all_df["Competition"].map(
        lambda c: COMPETITION_LABELS.get(c, c)
    )
    all_df["scale_factor"] = all_df["Density"].map(DENSITY_SCALE)
    if all_df["scale_factor"].isna().any():
        bad = sorted(all_df.loc[all_df["scale_factor"].isna(), "Density"].unique())
        raise SystemExit(f"Unexpected density code(s): {', '.join(bad)}")

    all_df["Larvae_plot"] = all_df["Larvae"] / all_df["scale_factor"]
    all_df["Adults_plot"] = all_df["Adults"] / all_df["scale_factor"]

    counts = (
        all_df.groupby(["Competition", "Larvae_plot", "Adults_plot"])
        .size()
        .rename("n")
        .reset_index()
    )
    global_max_count = max(int(counts["n"].max()), 1)
    all_df = all_df.merge(
        counts, on=["Competition", "Larvae_plot", "Adults_plot"], how="left"
    )
    all_df["n"] = all_df["n"].fillna(1).astype(int)
    return all_df, global_max_count, _size_legend_breaks(global_max_count)


def _present_order(values, default: list[str]) -> list[str]:
    """Default ordering restricted to present values, with any extras appended."""
    present = set(values)
    return [v for v in default if v in present] + sorted(present - set(default))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _marker_area(n, global_max_count: int):
    """Marker area (points^2) reproducing ggplot's ``scale_size_area``.

    Area is proportional to the count, with the global maximum mapped to a point
    of diameter ``DESIRED_MAX_SIZE`` mm (converted to matplotlib points).  Areas
    below the ``MIN_POINT_MM`` floor are clamped up so small counts stay legible.
    Accepts a scalar or an array-like ``n``.
    """
    max_diameter_pt = DESIRED_MAX_SIZE * 72.0 / 25.4  # mm -> points
    max_area_pt2 = math.pi / 4.0 * max_diameter_pt**2
    min_area_pt2 = math.pi / 4.0 * (MIN_POINT_MM * 72.0 / 25.4) ** 2
    return np.maximum(max_area_pt2 * (n / global_max_count), min_area_pt2)


def plot_competition(
    all_df: pd.DataFrame,
    competition: str,
    global_max_count: int,
    size_legend_breaks: list[int],
    *,
    temperatures: list[str] | None = None,
    foods: list[str] | None = None,
    densities: list[str] | None = None,
    panel_in: float = 1.44,
    name: str | None = None,
    save: bool = True,
    tiny_pad: float = 0.5,
) -> plt.Figure:
    """Build (and optionally save) the faceted figure for one competition type.

    Rows are ``interaction(Temperature, Density)`` (temperature slowest) and
    columns are the food regimes, matching the reference ``fig_4`` layout.

    *temperatures*, *foods* and *densities* optionally restrict which facets are
    drawn (e.g. ``temperatures=["T20"], foods=["F21", "F22"]`` for the simplified
    top-left quadrant).  *panel_in* is the square-panel side length in inches; the
    overall figure size is derived from it so any grid shape stays balanced.
    """
    df_sub = all_df[all_df["Competition"] == competition]
    if df_sub.empty:
        raise ValueError(f"No data for competition '{competition}'.")

    food_order = _present_order(all_df["Food"], DEFAULT_FOOD_ORDER)
    temp_order = _present_order(all_df["Temperature"], DEFAULT_TEMP_ORDER)
    density_order = _present_order(all_df["Density"], DEFAULT_DENSITY_ORDER)
    if foods is not None:
        food_order = [f for f in food_order if f in set(foods)]
    if temperatures is not None:
        temp_order = [t for t in temp_order if t in set(temperatures)]
    if densities is not None:
        density_order = [d for d in density_order if d in set(densities)]
    if not (food_order and temp_order and density_order):
        raise SystemExit("No panels remain after temperature/food/density filtering.")

    # Axis limits / ticks derived from the shown panels only (same rule as R).
    shown = df_sub[
        df_sub["Temperature"].isin(temp_order)
        & df_sub["Food"].isin(food_order)
        & df_sub["Density"].isin(density_order)
    ]
    data_max = max(shown["Larvae_plot"].max(), shown["Adults_plot"].max())
    axis_upper = min(650, max(600, math.ceil(data_max / 200) * 200))
    tick_upper = math.floor(axis_upper / 200) * 200
    brks = list(range(0, tick_upper + 1, 200)) or [0, 200, 400, 600]

    n_rows = len(temp_order) * len(density_order)
    n_cols = len(food_order)
    row_keys = [(t, d) for t in temp_order for d in density_order]

    conference_rc = {**plot_style.HOUSE_RC, "font.size": 13}
    with plt.rc_context(conference_rc):
        # Inch-based layout so panels come out genuinely square (coord_fixed).
        # The figure size is derived from the (fixed) panel size, so the grid
        # fills the canvas and stays balanced whatever its shape.
        left_pad, right_margin = 0.12, 0.15
        top_margin, bottom_margin = 0.95, 1.60
        gap = 0.07
        strip_col_in = 0.24        # food strips across the top
        ytitle_gutter_in = 0.30    # rotated y-axis title
        ylabel_gutter_in = 0.42    # y tick labels

        grid_w = n_cols * panel_in + (n_cols - 1) * gap
        grid_h = n_rows * panel_in + (n_rows - 1) * gap
        fig_w = left_pad + ytitle_gutter_in + ylabel_gutter_in + grid_w + right_margin
        fig_h = bottom_margin + grid_h + strip_col_in + top_margin

        fig = plt.figure(figsize=(fig_w, fig_h))

        plot_left_in = left_pad
        plot_right_in = fig_w - right_margin
        plot_top_in = fig_h - top_margin
        plot_bottom_in = bottom_margin

        grid_left_in = plot_left_in + ytitle_gutter_in + ylabel_gutter_in
        grid_top_in = plot_top_in - strip_col_in

        def to_frac(x_in, y_in, w_in, h_in):
            return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

        axes = [[None] * n_cols for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                x0 = grid_left_in + c * (panel_in + gap)
                y0 = grid_top_in - (r + 1) * panel_in - r * gap
                axes[r][c] = fig.add_axes(to_frac(x0, y0, panel_in, panel_in))

        # Figure-fraction helpers for the annotations below.
        right = plot_right_in / fig_w
        bottom = plot_bottom_in / fig_h
        top = plot_top_in / fig_h
        plot_left = grid_left_in / fig_w
        plot_top = grid_top_in / fig_h
        strip_col_h = strip_col_in / fig_h
        legend_left = plot_left_in / fig_w

        for r, (temp, dens) in enumerate(row_keys):
            colour = DENSITY_COLOURS[dens]
            for c, food in enumerate(food_order):
                ax = axes[r][c]
                cell = df_sub[
                    (df_sub["Temperature"] == temp)
                    & (df_sub["Density"] == dens)
                    & (df_sub["Food"] == food)
                ]
                ax.plot(
                    [0, axis_upper], [0, axis_upper],
                    color=REF_LINE_GREY, linewidth=0.6, zorder=1,
                )
                if not cell.empty:
                    ax.scatter(
                        cell["Larvae_plot"], cell["Adults_plot"],
                        s=_marker_area(cell["n"], global_max_count),
                        c=colour, alpha=0.5, linewidths=0, zorder=2,
                    )
                ax.set_xlim(-tiny_pad, axis_upper)
                ax.set_ylim(-tiny_pad, axis_upper)
                ax.set_aspect("equal")
                ax.set_xticks(brks)
                ax.set_yticks(brks)
                ax.tick_params(direction="in", length=3, width=0.4, labelsize=9)
                ax.grid(True, color=GRID_GREY, linewidth=0.25, zorder=0)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.3)
                # Open the axes: drop the top and right spines (keep left/bottom).
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if c != 0:
                    ax.set_yticklabels([])
                if r != n_rows - 1:
                    ax.set_xticklabels([])

        # --- Column strips (food) across the top ---
        for c, food in enumerate(food_order):
            pos = axes[0][c].get_position()
            fig.patches.append(
                Rectangle(
                    (pos.x0, pos.y1 + 0.002), pos.width, strip_col_h,
                    transform=fig.transFigure, facecolor=STRIP_GREY,
                    edgecolor="black", linewidth=0.3, zorder=3,
                )
            )
            fig.text(
                pos.x0 + pos.width / 2, pos.y1 + 0.002 + strip_col_h / 2,
                FOOD_LABELS.get(food, food),
                ha="center", va="center", fontsize=10, zorder=4,
            )

        # --- Titles / axis labels ---
        fig.text(
            (plot_left + right) / 2, top + 0.025,
            rf"Adult--Larval Relationships under {competition} Competition",
            ha="center", va="bottom", fontsize=16,
        )
        fig.text(
            (plot_left + right) / 2, 1.02 / fig_h,
            r"Larval input (density-normalised)",
            ha="center", va="center", fontsize=13,
        )
        fig.text(
            (plot_left_in + ytitle_gutter_in / 2) / fig_w, (bottom + plot_top) / 2,
            r"Adult output (density-normalised)",
            ha="center", va="center", rotation=90, fontsize=13,
        )

        # --- Legends along the bottom ---
        colour_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=8,
                markerfacecolor=DENSITY_COLOURS[d], markeredgecolor="none",
                alpha=0.7, label=DENSITY_LABELS[d],
            )
            for d in density_order
        ]
        # Single-column so the density legend stays in the left third and never
        # runs into the size legend on the right.
        leg1 = fig.legend(
            handles=colour_handles, title=r"Larval Density",
            loc="lower left", bbox_to_anchor=(legend_left, 0.012),
            frameon=False, fontsize=11,
            title_fontproperties={"weight": "bold", "size": 12},
            handletextpad=0.3, ncol=1, labelspacing=0.6, borderaxespad=0.0,
        )
        leg1._legend_box.align = "left"
        fig.add_artist(leg1)

        # n = 1 maps to a zero-area, invisible marker; drop it from the legend.
        display_breaks = [b for b in size_legend_breaks if b > 1]
        size_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markersize=math.sqrt(_marker_area(b, global_max_count)),
                markerfacecolor="#4d4d4d", markeredgecolor="none", alpha=0.6,
                label=str(b),
            )
            for b in display_breaks
        ]
        leg2 = fig.legend(
            handles=size_handles, title=r"Overlapping points ($n$)",
            loc="lower right", bbox_to_anchor=(right, 0.012),
            frameon=False, fontsize=11,
            title_fontproperties={"weight": "bold", "size": 12},
            handletextpad=0.3, ncol=len(size_handles),
            columnspacing=0.7, borderaxespad=0.0,
        )
        leg2._legend_box.align = "left"

        if save:
            plot_style.save_fig(fig, name or f"ziqian_avl_{competition.lower()}")

    return fig


def plot_ziqian_density_relationships(
    competition: str = "Interference",
    *,
    temperatures: list[str] | None = None,
    foods: list[str] | None = None,
    densities: list[str] | None = None,
    save: bool = True,
) -> plt.Figure:
    """Entry point: the house-style adult-vs-larval figure for *competition*.

    Defaults to the interference-competition treatment (the ``fig_4`` analogue).
    Pass ``competition="Exploitation"`` for the exploitation-competition set.

    *temperatures*, *foods* and *densities* optionally restrict the facets shown
    (e.g. ``temperatures=["T20"], foods=["F21", "F22"]`` for the simplified
    top-left quadrant).  When cropping, the point-size *scale* is kept global (so
    a given overlap count is drawn the same size as in the full figure and the
    dense low-density band does not balloon into a solid blob); only the legend
    breaks are trimmed to those at or below the largest overlap actually shown.
    Larger panels are used for legibility.
    """
    all_df, global_max_count, breaks = load_counts()
    present = _present_order(all_df["Competition"], DEFAULT_COMP_ORDER)
    if competition not in present:
        raise SystemExit(
            f"Competition '{competition}' not in data; available: {', '.join(present)}"
        )

    cropped = temperatures is not None or foods is not None or densities is not None
    if cropped:
        # Trim the legend to breaks that actually occur in the shown panels,
        # keeping the global size scale so points stay comparable to the full
        # figure.
        sub = all_df[all_df["Competition"] == competition]
        if temperatures is not None:
            sub = sub[sub["Temperature"].isin(temperatures)]
        if foods is not None:
            sub = sub[sub["Food"].isin(foods)]
        if densities is not None:
            sub = sub[sub["Density"].isin(densities)]
        if not sub.empty:
            shown_max = int(sub["n"].max())
            breaks = [b for b in breaks if b <= shown_max]

    return plot_competition(
        all_df, competition, global_max_count, breaks,
        temperatures=temperatures, foods=foods, densities=densities,
        panel_in=PANEL_IN_CROPPED if cropped else PANEL_IN_FULL,
        save=save,
    )


if __name__ == "__main__":
    plot_ziqian_density_relationships()
