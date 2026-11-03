"""Plotting utilities for the underdominance gene-drive optimisation results."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import plot_style
from matplotlib.transforms import blended_transform_factory
from paths import IBM_OUTPUT_DIR, IMAGES_DIR
from torch import Tensor

# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

#: Colourblind-friendly palette (IBM Design Language), for consistent point colours.
IBM_PALETTE: list[str] = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
    "#FE6100",
    "#FFB000",
]

#: Blue/gold pairing from the IBM palette used by every two-objective figure
#: (surrogate validation, Sobol indices, local sensitivity) so they stay
#: visually consistent with the other two-colour plots.
_TWO_OBJ_COLOURS: list[str] = [IBM_PALETTE[0], IBM_PALETTE[4]]

#: Marker shapes matching :data:`_TWO_OBJ_COLOURS`, so the individual points
#: overlaid on a two-cluster bar chart stay attributable in greyscale.
_CLUSTER_MARKERS: list[str] = ["o", "s"]

#: rcParams applied inside both plot functions to render all text in LaTeX font.
#:
#: The house spine convention (:data:`plot_style.OPEN_SPINES_RC`) is folded in
#: here rather than adopting :data:`plot_style.HOUSE_RC` wholesale: this module
#: sets its font sizes per call site, so inheriting the house sizes would reflow
#: every figure it produces.
_LATEX_RC: dict = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "savefig.dpi": plot_style.SAVE_DPI,
    # Thin strokes so the hatching on the second series of a two-series figure
    # reads as texture rather than as a block of solid ink.
    "hatch.linewidth": 0.5,
    **plot_style.OPEN_SPINES_RC,
}

#: Figure size for the objective-space Pareto-frontier figure.  The paper
#: embeds it at ``0.6\textwidth`` (``\textwidth`` is 6.0 in, so 3.6 in on the
#: page), and matplotlib's point sizes are only preserved through that
#: scaling if the figure is drawn at close to its printed size: at 4.2 in
#: wide the 13 pt axis labels land at ~11 pt on paper, which is where the
#: old 7.0 in figure put them when it was embedded at full ``\textwidth``.
#: The 7:5.5 aspect ratio of that figure is kept.
_FRONTIER_FIGSIZE: tuple[float, float] = (4.2, 3.3)

_OBJ_LATEX: list[str] = [
    r"$f_{\mathrm{extinction}}$",
    r"$f_{\mathrm{fixation}}$",
]

#: Compact objective labels for small panels / tick labels.
_OBJ_SHORT: list[str] = [
    r"$f_{\mathrm{ext}}$",
    r"$f_{\mathrm{fix}}$",
]

_PARAM_LATEX: list[str] = [
    "C. rate",
    r"$a_{WG}$",
    r"$\alpha_{WG}$",
    r"$\beta_{WG}$",
    r"$g_{WG}$",
    r"$a_{GG}$",
    r"$\alpha_{GG}$",
    r"$\beta_{GG}$",
    r"$g_{GG}$",
]

#: Colormap built from the IBM palette, used to interpolate n distinct colours.
_IBM_CMAP = mcolors.LinearSegmentedColormap.from_list("ibm", IBM_PALETTE)

#: Diverging house-style colormap (IBM blue → white → IBM yellow) for signed
#: quantities such as correlations, centred at zero.
_IBM_DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ibm_diverging", ["#648FFF", "#FFFFFF", "#FFB000"]
)

#: Short names + house-style colours for the three parameter categories
#: (conversion rate | WG heterozygote penalties | GG homozygote penalties).
_PARAM_GROUP_NAMES: list[str] = ["Genetic", "Heterozygote", "Homozygote"]
_PARAM_GROUP_COLOURS: list[str] = ["#648FFF", "#DC267F", "#FE6100"]


def _make_palette(n: int) -> list[str]:
    """Return *n* perceptually-spaced hex colours sampled from the IBM palette.

    For n ≤ 5 the original IBM colours are returned directly (no interpolation).
    For n > 5 colours are interpolated along the IBM gradient so that every
    point on the Pareto front gets a unique, visually distinct colour.
    """
    if n <= 0:
        return []
    if n == 1:
        return [IBM_PALETTE[0]]
    if n <= len(IBM_PALETTE):
        # Sample evenly across the palette so the full colour range is used
        indices = np.round(np.linspace(0, len(IBM_PALETTE) - 1, n)).astype(int)
        return [IBM_PALETTE[i] for i in indices]
    return [mcolors.to_hex(_IBM_CMAP(t)) for t in np.linspace(0, 1, n)]


def _ordered_palette(n: int, *, for_lines: bool = False) -> list[str]:
    """Return *n* colours for an *ordered* series (thresholds, BO replicates).

    Unlike :func:`_make_palette`, which spaces hues for maximum colour contrast,
    this ramp is monotone in luminance, so the ordering survives a
    black-and-white print.  Pass ``for_lines=True`` for thin lines and small
    markers, which need the palest end lifted to stay visible against white.
    """
    return plot_style.ordered_colors(n, lo=0.22 if for_lines else 0.0)


def _two_series_bar_style(k: int, *, hatch: str = "///") -> dict:
    """Return :meth:`~matplotlib.axes.Axes.bar` kwargs for series *k* of two.

    The second series is hatched as well as recoloured: IBM blue and IBM amber
    sit only ~14% apart in luminance, so colour alone collapses into a single
    grey in print.  Both bars get a thin black edge so the pair looks
    deliberate (and so the palest fills keep a boundary).
    """
    return {
        "color": _TWO_OBJ_COLOURS[k % len(_TWO_OBJ_COLOURS)],
        "hatch": (None, hatch)[k % 2],
        "edgecolor": "black",
        "linewidth": 0.6,
    }


def _open_axes(*axes) -> None:
    """Open the axes: drop the top and right spines (keep left/bottom)."""
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_outputs(
    verified_pareto_Y: Tensor | None = None,
    verified_pareto_Yvar: Tensor | None = None,
    pareto_X: Tensor | None = None,
    history: list[tuple[Tensor, Tensor]] | None = None,
    validation_results: dict | None = None,
    sobol_results: dict | None = None,
    local_sensitivity_results: dict | None = None,
    filename_suffix: str = "",
) -> None:
    """Plot all gene-drive optimisation outputs and save to ``data/images/``.

    Every argument is optional; only the plots for which the relevant data is
    supplied will be produced.  Pass any combination to suit the calling
    context (e.g. sensitivity-analysis scripts may only supply
    *sobol_results*).

    Parameters
    ----------
    verified_pareto_Y:
        Objective values on the verified Pareto front, shape ``(n, 2)``.
    verified_pareto_Yvar:
        Posterior variance for each verified Pareto objective, shape ``(n, 2)``.
    pareto_X:
        Decision variables for each verified Pareto point, shape ``(n, d)``.
    history:
        Per-step Pareto snapshots from the BO loop.
    validation_results:
        Dict returned by :func:`optimise_gene_drives.validate_surrogate`.
    sobol_results:
        Dict returned by :func:`sensitivity_analysis.compute_sobol_indices`.
        When *S2* indices are present (``calc_second_order=True``), the
        second-order interaction grid is also saved.
    local_sensitivity_results:
        Dict returned by :func:`sensitivity_analysis.compute_local_sensitivity`.
    filename_suffix:
        Optional string appended to every saved figure's filename stem (before
        the ``.png`` extension).  Defaults to ``""``, which preserves the
        original filenames.  Useful for tagging outputs with, e.g., a model id
        so that separate runs do not overwrite one another.
    """
    if verified_pareto_Y is not None and pareto_X is not None:
        sort_idx = verified_pareto_Y[:, 0].argsort()
        sorted_X = pareto_X[sort_idx]
        sorted_Y = verified_pareto_Y[sort_idx]
        sorted_Yvar = (
            verified_pareto_Yvar[sort_idx] if verified_pareto_Yvar is not None else None
        )
        colours = _make_palette(sorted_X.shape[0])

        fig_params = plot_parameters(sorted_X, sorted_Y, colours=colours)
        plt.figure(fig_params.number)
        plt.savefig(IMAGES_DIR / f"gene_drive_optimisation_params{filename_suffix}.png")

        fig_obj = plot_objectives(
            sorted_Y, sorted_Yvar, colours=colours, history=history
        )
        plt.figure(fig_obj.number)
        plt.savefig(
            IMAGES_DIR / f"gene_drive_optimisation_objectives{filename_suffix}.png"
        )

    if validation_results is not None:
        fig_val = plot_model_vs_ground_truth(validation_results)
        plt.figure(fig_val.number)
        plt.savefig(
            IMAGES_DIR / f"gene_drive_surrogate_validation{filename_suffix}.png"
        )

    if sobol_results is not None:
        fig_sobol = plot_sobol_indices(sobol_results)
        plt.figure(fig_sobol.number)
        plt.savefig(
            IMAGES_DIR / f"gene_drive_sobol_total_indices{filename_suffix}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )

        first_obj = sobol_results[sobol_results["objective_names"][0]]
        if first_obj.get("S2") is not None:
            fig_s2 = plot_sobol_second_order(sobol_results)
            plt.figure(fig_s2.number)
            plt.savefig(
                IMAGES_DIR / f"gene_drive_sobol_second_order{filename_suffix}.png",
                dpi=plot_style.SAVE_DPI,
                bbox_inches="tight",
            )

    if local_sensitivity_results is not None:
        fig_local = plot_local_sensitivity(local_sensitivity_results)
        plt.figure(fig_local.number)
        plt.savefig(
            IMAGES_DIR / f"gene_drive_local_sensitivity{filename_suffix}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )


def plot_objectives(
    pareto_Y: Tensor,
    pareto_Yvar: Tensor | None = None,
    *,
    colours: list[str] | None = None,
    obj_names: list[str] | None = None,
    history: list[tuple[Tensor, Tensor]] | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
    title: str | None = None,
):
    """Scatter plot of f_fixation vs f_extinction for the Pareto-optimal points.

    Parameters
    ----------
    pareto_Y : Tensor  shape (n_pareto, 2)
        Objective values [f_extinction, f_fixation] of the final Pareto front.
    pareto_Yvar : Tensor  shape (n_pareto, 2), optional
        Variance of each objective estimate.  When provided, ±2 sd
        (±2 × sqrt(var)) error bars are drawn.
    colours : list[str], optional
        One colour per Pareto point.  Defaults to ``IBM_PALETTE``.
    obj_names : list[str], optional
        Axis labels for the two objectives.  Defaults to ``_OBJ_LATEX``.
    history : list of (pareto_X, pareto_Y) tuples, optional
        Per-step Pareto-front snapshots.  Earlier steps are shown as small
        semi-transparent blue dots (light → dark) with a colorbar.
    figsize : tuple[float, float]
        Figure size in inches (width, height).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Y_np = pareto_Y.detach().cpu().numpy()
    n_pareto = Y_np.shape[0]
    names = obj_names if obj_names is not None else _OBJ_LATEX
    _colours = colours if colours is not None else _make_palette(n_pareto)

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if history:
        n_steps = len(history)
        blues = matplotlib.colormaps["Blues"](np.linspace(0.25, 0.80, n_steps))
        for step_idx, (_, hY) in enumerate(history):
            hY_np = hY.detach().cpu().numpy()
            ax.scatter(
                hY_np[:, 0],
                hY_np[:, 1],
                color=blues[step_idx],
                alpha=0.25 + 0.55 * (step_idx / max(n_steps - 1, 1)),
                s=12,
                linewidths=0,
                zorder=1,
            )
        norm = mcolors.Normalize(vmin=1, vmax=n_steps)
        sm = cm.ScalarMappable(cmap="Blues", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("BO step", fontsize=12)

    Yvar_np = pareto_Yvar.detach().cpu().numpy() if pareto_Yvar is not None else None

    for pt_idx in range(n_pareto):
        c = _colours[pt_idx % len(_colours)]
        if Yvar_np is not None:
            ax.errorbar(
                Y_np[pt_idx, 0],
                Y_np[pt_idx, 1],
                xerr=2.0 * np.sqrt(Yvar_np[pt_idx, 0]),
                yerr=2.0 * np.sqrt(Yvar_np[pt_idx, 1]),
                fmt="none",
                color=c,
                alpha=0.6,
                linewidth=1.2,
                zorder=2,
            )
        ax.scatter(
            Y_np[pt_idx, 0],
            Y_np[pt_idx, 1],
            color=c,
            s=60,
            alpha=0.9,
            linewidths=0.4,
            edgecolors="white",
            zorder=3,
        )

    ax.set_xlabel(names[0], fontsize=13)
    ax.set_ylabel(names[1], fontsize=13)
    ax.tick_params(labelsize=11)

    return fig


def plot_parameters(
    pareto_X: Tensor,
    pareto_Y: Tensor,
    *,
    colours: list[str] | None = None,
    param_names: list[str] | None = None,
    obj_names: list[str] | None = None,
    category_breaks: list[int] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Grouped bar chart of parameter values across Pareto-optimal points.

    Each parameter is one group of bars; each bar in the group corresponds to
    one Pareto-optimal point, coloured distinctly.  The legend lists objective
    values for each point so the colours can be interpreted.

    Parameters
    ----------
    pareto_X : Tensor  shape (n_pareto, d)
        Decision-variable coordinates of the Pareto front.
    pareto_Y : Tensor  shape (n_pareto, 2)
        Objective values [f_extinction, f_fixation] used to annotate the legend.
    colours : list[str], optional
        One colour per Pareto point.  Defaults to ``IBM_PALETTE``.
    param_names : list[str], optional
        Labels for the decision variables.  Defaults to ``_PARAM_LATEX``.
    obj_names : list[str], optional
        Labels for the objectives used in the legend.  Defaults to ``_OBJ_LATEX``.
    category_breaks : list[int], optional
        Parameter indices where a new category begins (e.g. ``[1, 5]`` puts
        separators between index 0 and 1, and between 4 and 5).  Defaults to
        ``[1, 5]`` matching the Conversion-rate / WG / GG structure.
    figsize : tuple[float, float], optional
        Figure size in inches.  Defaults to ``(5.5, 5.0)``.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    X_np = pareto_X.detach().cpu().numpy()
    Y_np = pareto_Y.detach().cpu().numpy()

    sort_idx = np.argsort(Y_np[:, 0])
    X_np = X_np[sort_idx]
    Y_np = Y_np[sort_idx]

    n_pareto, d = X_np.shape
    names = param_names if param_names is not None else _PARAM_LATEX[:d]
    obj_labels = obj_names if obj_names is not None else _OBJ_LATEX[: Y_np.shape[1]]
    _colours = colours if colours is not None else _make_palette(n_pareto)
    breaks = category_breaks if category_breaks is not None else [1, 5]

    bar_width = 0.8 / n_pareto

    category_gap = 0.7
    break_set = set(breaks)
    extra = 0.0
    x_pos = np.empty(d)
    for i in range(d):
        if i in break_set:
            extra += category_gap
        x_pos[i] = i + extra

    sep_positions = [
        (x_pos[brk - 1] + x_pos[brk]) / 2.0 for brk in breaks if 0 < brk < d
    ]

    if figsize is None:
        figsize = (5.5, 5.0)

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for pt_idx in range(n_pareto):
        offset = (pt_idx - (n_pareto - 1) / 2.0) * bar_width
        label_parts = [
            f"{lbl.split()[0]}={Y_np[pt_idx, k]:.3f}"
            for k, lbl in enumerate(obj_labels)
        ]
        label = ",  ".join(label_parts)
        ax.bar(
            x_pos + offset,
            X_np[pt_idx],
            width=bar_width * 0.9,
            color=_colours[pt_idx % len(_colours)],
            label=label,
            alpha=0.85,
        )

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for sx in sep_positions:
        ax.plot(
            [sx, sx],
            [-0.19, 1.0],
            transform=trans,
            clip_on=False,
            color="black",
            linewidth=0.8,
            linestyle=":",
        )

    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=13)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.legend(
        fontsize=11,
        title="Pareto points",
        title_fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        borderaxespad=0,
        ncols=2,
    )

    return fig


def plot_sobol_indices(
    results: dict,
    *,
    param_names: list[str] | None = None,
    objective_names: list[str] | None = None,
    category_breaks: list[int] | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """Grouped bar chart of total-effect Sobol indices per input parameter.

    Each parameter gets one group of bars — one bar per objective — showing its
    total-effect index S_Ti (fraction of output variance attributable to that
    parameter and all its interactions).  Error bars show ±2 bootstrap standard
    errors (2 × SALib's ``ST_se``).

    Saves the figure to ``data/images/gene_drive_parameter_sensitivity.png``.

    Parameters
    ----------
    results:
        Dict returned by :func:`sensitivity_analysis.compute_sobol_indices`.
    param_names:
        Override the x-tick labels.  Defaults to the LaTeX names in
        ``_PARAM_LATEX`` matched by position to the names stored in *results*.
    objective_names:
        Subset or reorder the objectives to plot.  Defaults to all objectives.
    category_breaks:
        Parameter indices at which to draw a vertical category separator.
        Defaults to ``[1, 5]`` (conversion rate / WG penalties / GG penalties).
    figsize:
        Figure size in inches ``(width, height)``.
    title:
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    obj_names = (
        objective_names if objective_names is not None else results["objective_names"]
    )
    p_names = results["param_names"]
    d = len(p_names)

    _name_to_latex = dict(
        zip(
            ["conversion_rate"]
            + [f"wg_{p}_fac" for p in ["a", "alpha", "beta", "g"]]
            + [f"gg_{p}_fac" for p in ["a", "alpha", "beta", "g"]],
            _PARAM_LATEX,
        )
    )
    _obj_to_latex = dict(
        zip(["f_1 (low g frequency)", "f_2 (high g frequency)"], _OBJ_LATEX)
    )

    x_labels = (
        list(param_names)
        if param_names is not None
        else [_name_to_latex.get(n, n) for n in p_names]
    )

    # x positions with category gaps (same logic as plot_parameters)
    breaks = category_breaks if category_breaks is not None else [1, 5]
    x_pos, sep_positions = _param_x_positions(d, breaks)

    n_obj = len(obj_names)
    bar_width = 0.7 / n_obj

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for k, obj in enumerate(obj_names):
        Si = results[obj]
        ST = np.array(Si["ST"])
        ST_err = 2.0 * np.array(Si["ST_se"])
        offset = (k - (n_obj - 1) / 2.0) * bar_width
        ax.bar(
            x_pos + offset,
            np.clip(ST, 0, None),
            width=bar_width * 0.9,
            **_two_series_bar_style(k),
            label=_obj_to_latex.get(obj, obj),
            yerr=ST_err,
            capsize=3,
            error_kw={"linewidth": 1.0, "alpha": 0.7},
        )

    # Category separators + group labels beneath the bars (house style).
    _draw_category_separators(ax, sep_positions, y0=-0.30)
    _annotate_param_groups(ax, x_pos, breaks)

    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel(r"Total-effect Sobol' index $S_{Ti}$", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)
    ax.legend(
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        borderaxespad=0,
        ncols=n_obj,
        frameon=False,
    )

    return fig


def plot_local_sensitivity(
    results: dict,
    *,
    param_names: list[str] | None = None,
    objective_names: list[str] | None = None,
    category_breaks: list[int] | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """Grouped bar chart of the objective Jacobian at the best Pareto point.

    Each parameter gets one group of two bars — one per objective — showing
    the signed partial derivative ``∂f_k/∂x_j`` (estimated by central finite
    differences of the underlying objective).  A positive value means
    increasing that parameter increases the objective at the optimum; negative
    means the opposite.  The zero baseline is marked with a dashed horizontal
    line.

    Parameters
    ----------
    results:
        Dict returned by :func:`sensitivity_analysis.compute_local_sensitivity`.
    param_names:
        Override the x-tick labels.  Defaults to the LaTeX names in
        ``_PARAM_LATEX`` matched by position.
    objective_names:
        Subset or reorder the objectives to plot.  Defaults to all objectives.
    category_breaks:
        Parameter indices at which to draw a vertical separator.
        Defaults to ``[1, 5]`` (conversion rate / WG / GG structure).
    figsize:
        Figure size in inches ``(width, height)``.
    title:
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    obj_names = (
        objective_names if objective_names is not None else results["objective_names"]
    )
    p_names = results["param_names"]
    jacobian = results["jacobian"]  # shape (m, d)
    d = len(p_names)

    _name_to_latex = dict(
        zip(
            ["conversion_rate"]
            + [f"wg_{p}_fac" for p in ["a", "alpha", "beta", "g"]]
            + [f"gg_{p}_fac" for p in ["a", "alpha", "beta", "g"]],
            _PARAM_LATEX,
        )
    )
    _obj_to_latex = dict(
        zip(["f_1 (low g frequency)", "f_2 (high g frequency)"], _OBJ_LATEX)
    )

    x_labels = (
        list(param_names)
        if param_names is not None
        else [_name_to_latex.get(n, n) for n in p_names]
    )

    breaks = category_breaks if category_breaks is not None else [1, 5]
    x_pos, sep_positions = _param_x_positions(d, breaks)

    n_obj = len(obj_names)
    bar_width = 0.7 / n_obj

    all_obj = results["objective_names"]
    obj_indices = [all_obj.index(o) for o in obj_names]

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for k, (obj, obj_idx) in enumerate(zip(obj_names, obj_indices)):
        offset = (k - (n_obj - 1) / 2.0) * bar_width
        ax.bar(
            x_pos + offset,
            jacobian[obj_idx],
            width=bar_width * 0.9,
            **_two_series_bar_style(k),
            label=_obj_to_latex.get(obj, obj),
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)

    # Category separators + group labels beneath the bars (house style).
    _draw_category_separators(ax, sep_positions, y0=-0.30)
    _annotate_param_groups(ax, x_pos, breaks)

    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel(r"$\partial f \,/\, \partial x_j$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        borderaxespad=0,
        ncols=n_obj,
        frameon=False,
    )

    return fig


def plot_sobol_second_order(
    results: dict,
    *,
    param_names: list[str] | None = None,
    objective_names: list[str] | None = None,
    figsize: tuple[float, float] = (15.0, 14.0),
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """9×9 grid of bar charts showing second-order Sobol interaction indices.

    Each off-diagonal cell (i, j) shows a grouped bar chart — one bar per
    objective — of S2[i, j] with ±2 bootstrap standard-error error bars
    (2 × ``S2_se``).  Diagonal cells display the parameter name as a label.  Requires
    *results* computed with ``calc_second_order=True``.

    Saves the figure to ``data/images/gene_drive_sobol_second_order.png``.

    Parameters
    ----------
    results:
        Dict returned by :func:`sensitivity_analysis.compute_sobol_indices`
        called with ``calc_second_order=True``.
    param_names:
        Override axis labels.  Defaults to LaTeX names in ``_PARAM_LATEX``.
    objective_names:
        Subset or reorder objectives to plot.  Defaults to all objectives.
    figsize:
        Figure size in inches ``(width, height)``.
    title:
        Optional super-title for the whole figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    obj_names = (
        objective_names if objective_names is not None else results["objective_names"]
    )
    p_names = results["param_names"]
    d = len(p_names)

    first_obj = results[obj_names[0]]
    if "S2" not in first_obj or first_obj["S2"] is None:
        raise ValueError(
            "S2 indices not found. Re-run compute_sobol_indices with calc_second_order=True."
        )

    _name_to_latex = dict(
        zip(
            ["conversion_rate"]
            + [f"wg_{p}_fac" for p in ["a", "alpha", "beta", "g"]]
            + [f"gg_{p}_fac" for p in ["a", "alpha", "beta", "g"]],
            _PARAM_LATEX,
        )
    )
    _obj_to_latex = dict(
        zip(["f_1 (low g frequency)", "f_2 (high g frequency)"], _OBJ_LATEX)
    )

    x_labels = (
        list(param_names)
        if param_names is not None
        else [_name_to_latex.get(n, n) for n in p_names]
    )

    n_obj = len(obj_names)
    bar_width = 0.65

    # Compute global y-max so all cells share the same scale
    global_max = 0.0
    for obj in obj_names:
        s2 = np.array(results[obj]["S2"], dtype=float)
        s2_err = 2.0 * np.array(results[obj]["S2_se"], dtype=float)
        valid = ~np.isnan(s2)
        if valid.any():
            global_max = max(global_max, float(np.nanmax(s2[valid] + s2_err[valid])))
    y_max = max(global_max * 1.15, 1e-6)

    matplotlib.rcParams.update(_LATEX_RC)
    fig, axes = plt.subplots(d, d, figsize=figsize)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.text(
                    0.5,
                    0.5,
                    x_labels[i],
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
                ax.set_facecolor("#f0f0f0")
                ax.set_axis_off()
            else:
                for k, obj in enumerate(obj_names):
                    s2_val = float(results[obj]["S2"][i, j])
                    s2_err = 2.0 * float(results[obj]["S2_se"][i, j])
                    if np.isnan(s2_val):
                        s2_val, s2_err = 0.0, 0.0
                    ax.bar(
                        k,
                        max(s2_val, 0.0),
                        width=bar_width,
                        # Sparser hatch than the full-size bar charts: the cells
                        # of this 9x9 grid are only a few millimetres wide.
                        **_two_series_bar_style(k, hatch="//"),
                        yerr=max(s2_err, 0.0),
                        capsize=2,
                        error_kw={"linewidth": 0.7, "alpha": 0.6},
                    )

                ax.set_xlim(-0.5, n_obj - 0.5)
                ax.set_ylim(0, y_max)
                ax.set_xticks([])
                ax.tick_params(axis="y", labelsize=5, pad=1, length=2)
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(nbins=3, prune="both")
                )

                # Only show y-tick labels on the leftmost column
                if j != 0:
                    ax.set_yticklabels([])

    # Column headers (top row) and row labels (left column)
    for j in range(d):
        axes[0, j].set_title(x_labels[j], fontsize=8, pad=4)
    for i in range(d):
        axes[i, 0].set_ylabel(
            x_labels[i], fontsize=8, rotation=0, labelpad=32, va="center"
        )

    # Legend below the grid.  Built from the same style dict as the bars so the
    # swatches carry the hatching too.
    legend_handles = []
    for k, obj in enumerate(obj_names):
        style = _two_series_bar_style(k, hatch="//")
        legend_handles.append(
            matplotlib.patches.Patch(
                facecolor=style["color"],
                hatch=style["hatch"],
                edgecolor=style["edgecolor"],
                linewidth=style["linewidth"],
                label=_obj_to_latex.get(obj, obj),
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncols=n_obj,
        fontsize=11,
        frameon=False,
    )

    fig.tight_layout()
    return fig


def plot_model_vs_ground_truth(
    validation_results: dict,
    *,
    obj_names: list[str] | None = None,
    figsize: tuple[float, float] = (4.0, 4.0),
    title: str | None = None,
) -> "matplotlib.figure.Figure":
    """Bar chart of GP surrogate relative error versus the IBM ground truth.

    Displays one bar per objective, where bar height = mean absolute relative
    error and error bars span ±2 standard deviations across the validation
    points.  Intended to be called with the dict returned by
    :func:`optimise_gene_drives.validate_surrogate`.

    Parameters
    ----------
    validation_results:
        Dict returned by :func:`validate_surrogate`, containing at minimum
        ``'f1_rel_err_mean'``, ``'f1_rel_err_std'``,
        ``'f2_rel_err_mean'``, ``'f2_rel_err_std'``.
    obj_names:
        x-tick labels for the two objectives.  Defaults to ``_OBJ_LATEX``.
    figsize:
        Figure size in inches ``(width, height)``.
    title:
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = obj_names if obj_names is not None else _OBJ_LATEX

    means = [
        validation_results["f1_rel_err_mean"],
        validation_results["f2_rel_err_mean"],
    ]
    stds = [
        validation_results["f1_rel_err_std"],
        validation_results["f2_rel_err_std"],
    ]

    x = np.arange(len(names))
    styles = [_two_series_bar_style(k) for k in range(len(names))]

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    bars = ax.bar(
        x,
        means,
        yerr=[2.0 * s for s in stds],
        width=0.45,
        color=[s["color"] for s in styles],
        hatch=[s["hatch"] for s in styles],
        edgecolor="black",
        linewidth=0.6,
        error_kw={"linewidth": 1.2, "capsize": 5, "alpha": 0.7},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("Relative error", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)

    return fig


def plot_best_parameters_across_thresholds(
    pareto_dir: Path | str | None = None,
    *,
    top_n: int = 1,
    use_verified: bool = True,
    show_objectives: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Bar chart of the top-point parameters across threshold pairs.

    For each release-threshold window in *pareto_dir* (a ``low<lo>_high<hi>/``
    subdirectory, or legacy flat files tagged ``_low<lo>_high<hi>``), pools every
    Pareto point across that window's replicates, ranks them by the mean of the
    two objectives, and takes the highest-scoring *top_n*.  Each parameter bar is
    then the **mean** of those top points' value, with the points themselves
    overlaid (filled in that threshold's colour, ringed in black), so how the
    optimal architecture shifts as the threshold window moves is visible
    alongside how tightly the top solutions agree.  With ``top_n=1`` this
    reduces to the single best point per threshold (bars only, no overlay).

    Layout:

    - **Parameter panel** – grouped bar chart of parameter values (y ∈ [0, 1]).
      Each parameter is one group of bars; each bar is one threshold window,
      coloured distinctly, with that window's top points overlaid as markers in
      the bar's own colour.  Category separators mark the conversion-rate / WG /
      GG boundaries.
    - **Objective panel** *(only when* ``show_objectives=True``*)* – grouped
      bar chart of the mean f1 (extinction) and f2 (fixation) objectives over the
      top points, with error bars (±1 sd over the top points when ``top_n>1``,
      else the single point's ±2 sd).

    Saved to ``IMAGES_DIR / "best_parameters_across_thresholds.png"`` (or
    ``top<top_n>_parameters_across_thresholds.png`` when ``top_n>1``) at 400 dpi.

    Parameters
    ----------
    pareto_dir:
        Directory containing the threshold subdirectories / Pareto CSV files.
        Defaults to ``IBM_OUTPUT_DIR`` from ``paths.py``.
    top_n:
        Number of highest-scoring points (pooled across each threshold's
        replicates) to average per threshold.  Default 1 (single best point).
    use_verified:
        Rank/score using IBM-verified objective means (default) or surrogate.
    show_objectives:
        When ``True``, add a second panel showing the achieved objectives at
        each threshold.  Default ``False``.
    figsize:
        Figure size in inches ``(width, height)``.  Defaults to
        ``(8.0, 6.5)`` without the objective panel, ``(8.0, 9.0)`` with it.
    title:
        Optional super-title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    thresholds = _collect_top_points_per_threshold(
        pareto_dir, top_n=top_n, use_verified=use_verified
    )

    entries: list[dict] = []
    for t in thresholds:
        P = t["params"]  # (k, 9)
        Yt = t["Y"]  # (k, 2)
        k = P.shape[0]
        f1_vals, f2_vals = Yt[:, 0], Yt[:, 1]
        if k > 1:
            f1_err, f2_err = float(f1_vals.std()), float(f2_vals.std())
        elif t["Yvar"] is not None:
            f1_err = float(2.0 * np.sqrt(t["Yvar"][0, 0]))
            f2_err = float(2.0 * np.sqrt(t["Yvar"][0, 1]))
        else:
            f1_err = f2_err = 0.0
        entries.append(
            {
                "low": t["low"],
                "high": t["high"],
                "k": k,
                "params": P,
                "param_mean": P.mean(axis=0),
                "param_sd": P.std(axis=0),
                "f1": float(f1_vals.mean()),
                "f2": float(f2_vals.mean()),
                "f1_err": f1_err,
                "f2_err": f2_err,
            }
        )

    n = len(entries)
    x = np.arange(n)
    x_labels = [rf"${0.5 * (e['low'] + e['high']):.2g}$" for e in entries]
    f1_arr = np.array([e["f1"] for e in entries])
    f2_arr = np.array([e["f2"] for e in entries])
    f1_ci = np.array([e["f1_err"] for e in entries])
    f2_ci = np.array([e["f2_err"] for e in entries])

    d = entries[0]["param_mean"].shape[0]
    breaks = [1, 5]
    x_pos, sep_positions = _param_x_positions(d, breaks)

    # Ordered series (increasing release threshold): use the monotone
    # luminance ramp so the ordering survives a black-and-white print.
    colours = _ordered_palette(n)
    bar_width = 0.8 / n
    _short_obj = _OBJ_SHORT

    if figsize is None:
        figsize = (8.0, 9.0) if show_objectives else (8.0, 6.5)

    matplotlib.rcParams.update(_LATEX_RC)
    if show_objectives:
        fig, (ax_params, ax_obj) = plt.subplots(
            2,
            1,
            figsize=figsize,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.5, 1]},
        )
    else:
        fig, ax_params = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    # ── Parameter panel ──────────────────────────────────────────────────────
    for thresh_idx, (entry, colour) in enumerate(zip(entries, colours)):
        offset = (thresh_idx - (n - 1) / 2.0) * bar_width
        # Report the target threshold the window brackets, rather than the
        # window's end points: one number reads more clearly in the legend.
        tau = 0.5 * (entry["low"] + entry["high"])
        label = rf"$\tau={tau:.2g}$,  " + ",  ".join(
            f"{lbl}={val:.3f}"
            for lbl, val in zip(_short_obj, [entry["f1"], entry["f2"]])
        )
        ax_params.bar(
            x_pos + offset,
            entry["param_mean"],
            width=bar_width * 0.9,
            color=colour,
            # The palest end of the ordered ramp is nearly white, so every bar
            # gets a thin black edge to keep its extent readable.
            edgecolor="black",
            linewidth=0.4,
            label=label,
            alpha=1.0,
            zorder=3,
        )
        # Show the threshold's top points themselves rather than a mean +/-1 sd
        # error bar: with only a handful of points per threshold the raw spread
        # is the honest summary, and filling each marker with its bar's colour
        # keeps it attributable in a chart this densely grouped.
        if entry["k"] > 1:
            _scatter_bar_points(
                ax_params,
                x_pos + offset,
                entry["params"],
                bar_width=bar_width * 0.9,
                colour=colour,
                marker_size=10.0,
            )

    # Category separators + group labels beneath the bars (house style).  This
    # is a taller figure, so pull the dotted rule + labels up closer to the
    # tick labels to avoid a large gap beneath the parameter names.
    _draw_category_separators(ax_params, sep_positions, y0=-0.22)
    _annotate_param_groups(ax_params, x_pos, breaks, y_sep=-0.165, y_text=-0.195)

    ax_params.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax_params.set_xticks(x_pos)
    ax_params.set_xticklabels(_PARAM_LATEX, rotation=45, ha="right", fontsize=12)
    ax_params.set_ylabel("Parameter value", fontsize=13)
    ax_params.set_ylim(0, 1)
    ax_params.tick_params(labelsize=11)
    legend_title = (
        r"Target threshold $\tau$,  best verified objectives"
        if top_n == 1
        else rf"Target threshold $\tau$,  top-{top_n} mean objectives"
    )
    ax_params.legend(
        fontsize=9,
        title=legend_title,
        title_fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        borderaxespad=0,
        ncols=2,
        frameon=False,
    )

    # ── Objective panel ──────────────────────────────────────────────────────
    if show_objectives:
        obj_bar_width = 0.35
        ax_obj.bar(
            x - obj_bar_width / 2,
            f1_arr,
            width=obj_bar_width,
            color=IBM_PALETTE[0],
            alpha=0.85,
            yerr=f1_ci,
            capsize=4,
            error_kw={"linewidth": 1.2, "alpha": 0.6},
            label=_OBJ_LATEX[0],
        )
        ax_obj.bar(
            x + obj_bar_width / 2,
            f2_arr,
            width=obj_bar_width,
            color=IBM_PALETTE[2],
            alpha=0.85,
            yerr=f2_ci,
            capsize=4,
            error_kw={"linewidth": 1.2, "alpha": 0.6},
            label=_OBJ_LATEX[1],
        )
        ax_obj.set_xticks(x)
        ax_obj.set_xticklabels(x_labels, fontsize=11)
        ax_obj.set_xlabel(r"Target threshold $\tau$", fontsize=13)
        ax_obj.set_ylabel("Verified objective", fontsize=13)
        ax_obj.set_ylim(0, 1.15)
        ax_obj.tick_params(labelsize=11)
        ax_obj.legend(
            fontsize=11,
            loc="upper center",
            # Clear of the x-axis label, which sits directly above it.
            bbox_to_anchor=(0.5, -0.30),
            borderaxespad=0,
            ncols=2,
            frameon=False,
        )

    if save:
        fname = (
            "best_parameters_across_thresholds.png"
            if top_n == 1
            else f"top{top_n}_parameters_across_thresholds.png"
        )
        fig.savefig(IMAGES_DIR / fname, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    return fig


#: Column names of the 9 decision variables, in CSV order.
_PARAM_COLS: list[str] = [
    "conversion_rate",
    "wg_a_fac",
    "wg_alpha_fac",
    "wg_beta_fac",
    "wg_g_fac",
    "gg_a_fac",
    "gg_alpha_fac",
    "gg_beta_fac",
    "gg_g_fac",
]


def _load_replicate_pareto_fronts(
    pareto_dir: Path | str,
    *,
    use_verified: bool = True,
) -> list[dict]:
    """Load every Pareto-frontier CSV in *pareto_dir* as one replicate each.

    Each ``gene_drive_pareto_frontier_points_*.csv`` file is treated as one
    optimisation replicate.  Rows are the Pareto-optimal points of that run.

    Parameters
    ----------
    pareto_dir:
        Directory containing the per-replicate Pareto CSV files.
    use_verified:
        When ``True`` (default) read the IBM-verified objective means/variances;
        otherwise read the raw surrogate objective values (no variances).

    Returns
    -------
    list[dict]
        One dict per replicate (sorted by filename), each with keys
        ``label`` (str), ``path`` (Path), ``params`` ``(m, 9)`` array,
        ``Y`` ``(m, 2)`` objective array, and ``Yvar`` ``(m, 2)`` variance
        array (``None`` when *use_verified* is ``False``).  ``m`` is that
        replicate's number of Pareto points.
    """
    pareto_dir = Path(pareto_dir)
    csv_paths = sorted(pareto_dir.glob("gene_drive_pareto_frontier_points_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No gene_drive_pareto_frontier_points_*.csv files found in {pareto_dir}"
        )

    if use_verified:
        f1_col = "f_1 (low g frequency) [verified mean]"
        f2_col = "f_2 (high g frequency) [verified mean]"
        f1_var_col = "f_1 (low g frequency) [verified var]"
        f2_var_col = "f_2 (high g frequency) [verified var]"
    else:
        f1_col = "f_1 (low g frequency)"
        f2_col = "f_2 (high g frequency)"

    replicates: list[dict] = []
    for idx, csv_path in enumerate(csv_paths, start=1):
        df = pd.read_csv(csv_path)
        Y = df[[f1_col, f2_col]].to_numpy(dtype=float)
        Yvar = (
            df[[f1_var_col, f2_var_col]].to_numpy(dtype=float) if use_verified else None
        )
        # Derive a short label from the trailing timestamp if present.
        m = re.search(r"_(\d{8})_(\d{6})", csv_path.stem)
        tag = m.group(2) if m else csv_path.stem
        replicates.append(
            {
                "label": f"Rep {idx} ({tag}), n={len(df)}",
                "path": csv_path,
                "params": df[_PARAM_COLS].to_numpy(dtype=float),
                "Y": Y,
                "Yvar": Yvar,
            }
        )
    return replicates


def _cluster_top_by_outcome(
    top_Y: np.ndarray,
    top_scores: np.ndarray,
    max_cluster_distance: float,
) -> tuple[np.ndarray, int]:
    """Cluster the top points by their objective *outcomes* and relabel by score.

    Groups the rows of *top_Y* ``(n, 2)`` — the ``(f_extinction, f_fixation)``
    values of the top points — with agglomerative, complete-linkage clustering,
    so no two points in a cluster sit farther apart in objective space than
    *max_cluster_distance*.  Clusters are then renumbered so cluster ``0`` has
    the highest mean score, cluster ``1`` the next, and so on.

    Returns ``(labels, n_clusters)`` where ``labels`` is an integer array of
    length ``n``.
    """
    n = top_Y.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int), max(n, 1)

    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=max_cluster_distance,
        linkage="complete",
        metric="euclidean",
    ).fit_predict(top_Y)
    order = sorted(
        np.unique(labels),
        key=lambda c: top_scores[labels == c].mean(),
        reverse=True,
    )
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[c] for c in labels]), len(order)


def _param_x_positions(d: int, breaks: list[int]) -> tuple[np.ndarray, list[float]]:
    """Return grouped x-positions and category-separator x-positions.

    Inserts a 0.7-wide gap before every index in *breaks* (same logic shared by
    ``plot_parameters`` / ``plot_sobol_indices``) so parameters cluster into
    visual categories.
    """
    break_set = set(breaks)
    extra = 0.0
    x_pos = np.empty(d)
    for i in range(d):
        if i in break_set:
            extra += 0.7
        x_pos[i] = i + extra
    sep_positions = [
        (x_pos[brk - 1] + x_pos[brk]) / 2.0 for brk in breaks if 0 < brk < d
    ]
    return x_pos, sep_positions


def _scatter_bar_points(
    ax,
    x_centres: np.ndarray,
    values: np.ndarray,
    *,
    bar_width: float,
    colour: str | None = None,
    marker: str = "o",
    marker_size: float = 26.0,
) -> None:
    """Overlay the individual observations behind each bar as ringed markers.

    *values* is ``(m, d)``: one row per observation, one column per parameter
    (i.e. per bar).  With the handful of Pareto points these figures pool, the
    raw points are more informative than a mean +/-1 sd error bar, which implies
    a spread the sample size cannot support.  Points are fanned evenly across
    the bar so coincident values stay countable; the horizontal offset carries
    no meaning.  Markers are drawn unclipped so that values sitting on the
    [0, 1] parameter bounds are not sliced in half by the axes limits.

    Markers that sit *above* their bar float in white space, where it is easy to
    misread which bar (and so which band/cluster) they belong to.  Each marker
    is therefore filled with its own bar's *colour* and ringed in black: the
    fill names the column it belongs to, while the black ring keeps it legible
    against both the white background and the bar it sits on.  Without a
    *colour* the markers fall back to plain black x's.

    *marker* varies the shape between series (e.g. per cluster), which keeps the
    points attributable in a black-and-white print where their fill colours
    collapse to similar greys.

    *marker_size* is the marker area in points²; shrink it for charts with many
    narrow bars per group (the cross-threshold figure), where the default would
    have the markers of neighbouring bars run into one another.
    """
    m = values.shape[0]
    if m == 0:
        return
    spread = np.zeros(1) if m == 1 else np.linspace(-0.3, 0.3, m) * bar_width
    for k, x_centre in enumerate(x_centres):
        column = values[:, k]
        if colour is None:
            ax.scatter(
                x_centre + spread,
                column,
                marker="x",
                color="black",
                s=14,
                linewidths=0.8,
                alpha=0.9,
                clip_on=False,
                zorder=5,
            )
            continue
        ax.scatter(
            x_centre + spread,
            column,
            marker=marker,
            facecolors=colour,
            edgecolors="black",
            s=marker_size,
            # Thin the ring in step with the marker so small markers do not
            # turn into solid black dots.
            linewidths=max(0.4, 0.8 * np.sqrt(marker_size / 26.0)),
            alpha=1.0,
            clip_on=False,
            zorder=5,
        )


def _draw_category_separators(
    ax, sep_positions: list[float], *, y0: float = -0.19
) -> None:
    """Draw dotted vertical category separators spanning the full axes height."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for sx in sep_positions:
        ax.plot(
            [sx, sx],
            [y0, 1.0],
            transform=trans,
            clip_on=False,
            color="black",
            linewidth=0.8,
            linestyle=":",
        )


def _annotate_param_groups(
    ax,
    x_pos: np.ndarray,
    breaks: list[int],
    *,
    names: list[str] | None = None,
    colours: list[str] | None = None,
    position: str = "bottom",
    y_sep: float = -0.22,
    y_text: float = -0.25,
) -> None:
    """Label the parameter categories beneath (or above) the bars.

    Groups are the index ranges delimited by *breaks* (e.g. ``[1, 5]`` →
    ``[0:1]``, ``[1:5]``, ``[5:9]``), naming what each block of parameters
    controls.

    ``position="bottom"`` (default) writes black centred titles below the
    parameter tick labels, separated from them by a dotted horizontal rule.
    ``position="top"`` instead draws house-style coloured brackets above the
    bars with matching coloured titles.

    *y_sep* / *y_text* (axes-fraction y of the dotted rule and the titles) let
    callers tighten the spacing on taller figures, where the default offsets
    leave too large a gap between the rule and the labels.
    """
    d = len(x_pos)
    bounds = [0, *breaks, d]
    names = names if names is not None else _PARAM_GROUP_NAMES
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    centres = [(x_pos[s] + x_pos[e - 1]) / 2.0 for s, e in zip(bounds[:-1], bounds[1:])]

    if position == "bottom":
        ax.plot(
            [x_pos[0] - 0.6, x_pos[-1] + 0.6],
            [y_sep, y_sep],
            transform=trans,
            clip_on=False,
            color="black",
            linewidth=0.8,
            linestyle=":",
        )
        for centre, name in zip(centres, names):
            ax.text(
                centre,
                y_text,
                name,
                transform=trans,
                clip_on=False,
                ha="center",
                va="top",
                color="black",
                fontsize=12,
            )
        return

    colours = colours if colours is not None else _PARAM_GROUP_COLOURS
    for g, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        colour = colours[g % len(colours)]
        x0, x1 = x_pos[s], x_pos[e - 1]
        ax.plot(
            [x0 - 0.4, x1 + 0.4],
            [1.015, 1.015],
            transform=trans,
            clip_on=False,
            color=colour,
            linewidth=2.5,
            solid_capstyle="round",
        )
        ax.text(
            (x0 + x1) / 2.0,
            1.05,
            names[g % len(names)],
            transform=trans,
            clip_on=False,
            ha="center",
            va="bottom",
            color=colour,
            fontsize=12,
        )


def plot_pareto_replicates(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    show_error_bars: bool = True,
    show_replicate_labels: bool = False,
    figsize: tuple[float, float] = _FRONTIER_FIGSIZE,
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Overlay the objective-space Pareto frontiers of every replicate in a dir.

    Treats each ``gene_drive_pareto_frontier_points_*.csv`` in *pareto_dir* as
    one optimisation replicate and plots its Pareto front in objective space
    (f_extinction vs f_fixation), sorted by f_extinction and joined by a line,
    in a distinct colour.  With ``show_error_bars`` and verified objectives,
    ±2 sd (±2 √var) error bars are drawn per point.

    The companion :func:`plot_parameter_presentations` shows how the decision
    variables vary along these frontiers.

    Saved to ``IMAGES_DIR / "pareto_replicates_<dirname>.png"`` at 400 dpi when
    ``save`` is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of per-replicate Pareto CSVs.  Defaults to ``IBM_OUTPUT_DIR``.
    use_verified:
        Plot IBM-verified objective means (default) or raw surrogate values.
    show_error_bars:
        Draw ±2 sd error bars (verified data only).
    show_replicate_labels:
        Add a per-replicate legend keyed by source CSV.  Off by default — the
        tags clutter a paper figure but are useful for replicability and
        validation when comparing runs.
    figsize:
        Figure size in inches.  Defaults to :data:`_FRONTIER_FIGSIZE`, sized so
        the figure prints at ``0.6\\textwidth`` with the house font sizes
        landing just above the 10 pt body text (see that constant).
    title:
        Optional super-title.
    save:
        Whether to write the figure to ``IMAGES_DIR``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    replicates = _load_replicate_pareto_fronts(pareto_dir, use_verified=use_verified)
    colours = _ordered_palette(len(replicates), for_lines=True)

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax_obj = plt.subplots(figsize=figsize, constrained_layout=True)
    _open_axes(ax_obj)

    for rep, colour in zip(replicates, colours):
        Y = rep["Y"]
        order = np.argsort(Y[:, 0])
        Ys = Y[order]
        # The palest end of the ordered ramp needs a little more ink than the
        # old equal-luminance palette did to stay legible against white: hence
        # the heavier line, the less transparent whiskers, and black (rather
        # than white) marker rings.  The sizes below are then trimmed back a
        # little from what a full-width figure carried: at _FRONTIER_FIGSIZE
        # the axes hold the same points in 0.6 of the width, so markers and
        # strokes at their old point sizes would crowd the dense right-hand
        # end of the frontiers.  Fonts are deliberately *not* trimmed.
        ax_obj.plot(
            Ys[:, 0],
            Ys[:, 1],
            color=colour,
            linewidth=1.4,
            alpha=0.7,
            zorder=2,
        )
        if show_error_bars and rep["Yvar"] is not None:
            Yvar = rep["Yvar"][order]
            ax_obj.errorbar(
                Ys[:, 0],
                Ys[:, 1],
                xerr=2.0 * np.sqrt(Yvar[:, 0]),
                yerr=2.0 * np.sqrt(Yvar[:, 1]),
                fmt="none",
                color=colour,
                alpha=0.55,
                linewidth=0.8,
                zorder=1,
            )
        ax_obj.scatter(
            Ys[:, 0],
            Ys[:, 1],
            color=colour,
            s=30,
            alpha=0.9,
            linewidths=0.45,
            edgecolors="black",
            zorder=3,
            label=rep["label"] if show_replicate_labels else None,
        )

    ax_obj.set_xlabel(_OBJ_LATEX[0], fontsize=13)
    ax_obj.set_ylabel(_OBJ_LATEX[1], fontsize=13)
    ax_obj.tick_params(labelsize=11)
    if show_replicate_labels:
        ax_obj.legend(
            fontsize=9,
            title="Replicate",
            title_fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            borderaxespad=0,
            ncols=2,
            frameon=False,
        )

    if save:
        fig.savefig(
            IMAGES_DIR / f"pareto_replicates_{pareto_dir.name}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


# ── Parameter-presentation panels ───────────────────────────────────────────
# Each helper draws one candidate visualisation of "how do the 9 decision
# variables change as you move along the Pareto frontier" into a SubFigure, so
# the four can be compared side by side in plot_parameter_presentations.


def _panel_param_lines(subfig, replicates: list[dict], colours: list[str]) -> None:
    """(A) 3×3 small multiples: one parameter per cell, value vs f_extinction."""
    subfig.suptitle(
        r"(A) Line small-multiples: value vs $f_{\mathrm{ext}}$", fontsize=12
    )
    axes = subfig.subplots(3, 3, sharex=True, sharey=True)
    flat = axes.flat
    for i, ax in enumerate(flat):
        for rep, colour in zip(replicates, colours):
            f = rep["Y"][:, 0]
            order = np.argsort(f)
            ax.plot(
                f[order],
                rep["params"][order, i],
                color=colour,
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                alpha=0.8,
                label=rep["label"] if i == 0 else None,
            )
        ax.set_title(_PARAM_LATEX[i], fontsize=9, pad=2)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(_OBJ_SHORT[0], fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("value", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    subfig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(len(replicates), 3),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )


def _panel_param_scatter(
    subfig, params_pooled, f_ext_pooled, x_pos, sep_positions, norm, cmap
) -> None:
    """(B) Grouped points on a single axes, coloured by f_extinction."""
    subfig.suptitle(r"(B) Grouped points, colour $= f_{\mathrm{ext}}$", fontsize=12)
    ax = subfig.subplots()
    rng = np.random.default_rng(0)
    M = params_pooled.shape[0]
    for i in range(len(x_pos)):
        jitter = (rng.random(M) - 0.5) * 0.4
        ax.scatter(
            np.full(M, x_pos[i]) + jitter,
            params_pooled[:, i],
            c=f_ext_pooled,
            cmap=cmap,
            norm=norm,
            s=16,
            alpha=0.85,
            linewidths=0.2,
            edgecolors="white",
            zorder=3,
        )
    _draw_category_separators(ax, sep_positions)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_PARAM_LATEX, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=9)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    subfig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label(
        _OBJ_LATEX[0], fontsize=10
    )


def _panel_param_parallel(subfig, params_pooled, f_ext_pooled, norm, cmap) -> None:
    """(C) Parallel coordinates: one polyline per Pareto point, colour=f_ext."""
    subfig.suptitle(
        r"(C) Parallel coordinates, colour $= f_{\mathrm{ext}}$", fontsize=12
    )
    ax = subfig.subplots()
    d = params_pooled.shape[1]
    xs = np.arange(d)
    cmap_obj = matplotlib.colormaps[cmap]
    for x in xs:
        ax.axvline(x, color="0.85", linewidth=0.6, zorder=0)
    # Draw low f_ext first so high-extinction lines sit on top.
    for idx in np.argsort(f_ext_pooled):
        ax.plot(
            xs,
            params_pooled[idx],
            color=cmap_obj(norm(f_ext_pooled[idx])),
            linewidth=0.8,
            alpha=0.5,
            zorder=2,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(_PARAM_LATEX, rotation=45, ha="right", fontsize=10)
    ax.set_xlim(-0.4, d - 0.6)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=9)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    subfig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02).set_label(
        _OBJ_LATEX[0], fontsize=10
    )


def _panel_param_binned_bars(
    subfig, params_pooled, f_ext_pooled, x_pos, sep_positions, norm, cmap, n_bins
) -> None:
    """(D) Grouped bars: mean ±1 sd per f_extinction band, per parameter."""
    subfig.suptitle(
        r"(D) Binned bars by $f_{\mathrm{ext}}$ (mean $\pm$ 1 sd)", fontsize=12
    )
    ax = subfig.subplots()
    d = params_pooled.shape[1]
    edges = np.linspace(f_ext_pooled.min(), f_ext_pooled.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(f_ext_pooled, edges[1:-1]), 0, n_bins - 1)
    cmap_obj = matplotlib.colormaps[cmap]
    bar_width = 0.8 / n_bins
    for b in range(n_bins):
        mask = bin_idx == b
        means = np.full(d, np.nan)
        sds = np.full(d, np.nan)
        if mask.any():
            means = params_pooled[mask].mean(axis=0)
            sds = params_pooled[mask].std(axis=0)
        centre = (edges[b] + edges[b + 1]) / 2.0
        offset = (b - (n_bins - 1) / 2.0) * bar_width
        ax.bar(
            x_pos + offset,
            means,
            width=bar_width * 0.9,
            color=cmap_obj(norm(centre)),
            alpha=0.85,
            yerr=sds,
            capsize=2,
            error_kw={"linewidth": 0.8, "alpha": 0.6},
            label=f"{edges[b]:.2f}–{edges[b + 1]:.2f}",
            zorder=3,
        )
    _draw_category_separators(ax, sep_positions)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_PARAM_LATEX, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=9)
    ax.legend(
        fontsize=8,
        title=_OBJ_LATEX[0] + " band",
        title_fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=n_bins,
        frameon=False,
    )


def plot_parameter_presentations(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    category_breaks: list[int] | None = None,
    cmap: str = "viridis",
    n_bins: int = 3,
    figsize: tuple[float, float] = (15.0, 12.0),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Compare four ways of showing parameter variation along the Pareto front.

    Loads every ``gene_drive_pareto_frontier_points_*.csv`` in *pareto_dir* and
    lays out four candidate visualisations as a 2×2 grid of sub-figures so the
    trade-offs of each can be assessed:

    - **(A)** 3×3 line small-multiples — one parameter per cell, value vs
      ``f_extinction``, one line per replicate (colours match
      :func:`plot_pareto_replicates`).
    - **(B)** grouped points on a single axes, coloured by ``f_extinction``.
    - **(C)** parallel coordinates — one polyline per Pareto point, coloured by
      ``f_extinction``.
    - **(D)** grouped bars, binned by ``f_extinction`` band (mean ±1 sd).

    Panels B–D pool all Pareto points across replicates.  Saved to
    ``IMAGES_DIR / "pareto_param_presentations_<dirname>.png"`` at 400 dpi when
    ``save`` is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of per-replicate Pareto CSVs.  Defaults to ``IBM_OUTPUT_DIR``.
    use_verified:
        Use IBM-verified objective means (default) or raw surrogate values.
    category_breaks:
        Parameter indices where a new category begins.  Defaults to ``[1, 5]``.
    cmap:
        Continuous colormap encoding ``f_extinction``.  Defaults to
        ``"viridis"`` (perceptually uniform).
    n_bins:
        Number of ``f_extinction`` bands for panel (D).  Defaults to ``3``.
    figsize:
        Figure size in inches.  Defaults to ``(15.0, 12.0)``.
    title:
        Optional super-title.
    save:
        Whether to write the figure to ``IMAGES_DIR``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    replicates = _load_replicate_pareto_fronts(pareto_dir, use_verified=use_verified)
    colours = _ordered_palette(len(replicates), for_lines=True)

    d = replicates[0]["params"].shape[1]
    breaks = category_breaks if category_breaks is not None else [1, 5]
    x_pos, sep_positions = _param_x_positions(d, breaks)

    params_pooled = np.vstack([rep["params"] for rep in replicates])  # (M, d)
    f_ext_pooled = np.concatenate([rep["Y"][:, 0] for rep in replicates])  # (M,)
    norm = mcolors.Normalize(vmin=f_ext_pooled.min(), vmax=f_ext_pooled.max())

    matplotlib.rcParams.update(_LATEX_RC)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(2, 2, wspace=0.04, hspace=0.04)

    _panel_param_lines(subfigs[0, 0], replicates, colours)
    _panel_param_scatter(
        subfigs[0, 1], params_pooled, f_ext_pooled, x_pos, sep_positions, norm, cmap
    )
    _panel_param_parallel(subfigs[1, 0], params_pooled, f_ext_pooled, norm, cmap)
    _panel_param_binned_bars(
        subfigs[1, 1],
        params_pooled,
        f_ext_pooled,
        x_pos,
        sep_positions,
        norm,
        cmap,
        n_bins,
    )

    if save:
        fig.savefig(
            IMAGES_DIR / f"pareto_param_presentations_{pareto_dir.name}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


def plot_parameters_binned(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    category_breaks: list[int] | None = None,
    n_bins: int = 3,
    figsize: tuple[float, float] = (8.0, 5.5),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Grouped bars of each parameter binned by position along the frontier.

    Pools every Pareto point across all ``gene_drive_pareto_frontier_points_*``
    replicates in *pareto_dir*, splits them into *n_bins* equal-width
    ``f_extinction`` bands, and draws one bar per band per parameter showing the
    band mean, with every pooled point in that band overlaid as a marker filled
    in the band's own colour and ringed in black.  This reveals how each
    decision variable shifts as you move along the Pareto frontier (low → high
    extinction), while the points keep the (small) sample behind each bar
    visible rather than hiding it in an error bar.

    Saved to ``IMAGES_DIR / "pareto_params_binned_<dirname>.png"`` at 400 dpi
    when ``save`` is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of per-replicate Pareto CSVs.  Defaults to ``IBM_OUTPUT_DIR``.
    use_verified:
        Use IBM-verified objective means (default) or raw surrogate values.
    category_breaks:
        Parameter indices where a new category begins.  Defaults to ``[1, 5]``.
    n_bins:
        Number of ``f_extinction`` bands.  Defaults to ``3``.  Each band gets a
        distinct house-style (IBM palette) colour.
    figsize:
        Figure size in inches.  Defaults to ``(8.0, 5.5)``.
    title:
        Optional super-title.
    save:
        Whether to write the figure to ``IMAGES_DIR``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    replicates = _load_replicate_pareto_fronts(pareto_dir, use_verified=use_verified)
    params_pooled = np.vstack([rep["params"] for rep in replicates])  # (M, d)
    f_ext_pooled = np.concatenate([rep["Y"][:, 0] for rep in replicates])  # (M,)

    d = params_pooled.shape[1]
    breaks = category_breaks if category_breaks is not None else [1, 5]
    x_pos, sep_positions = _param_x_positions(d, breaks)

    edges = np.linspace(f_ext_pooled.min(), f_ext_pooled.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(f_ext_pooled, edges[1:-1]), 0, n_bins - 1)
    bar_colours = _make_palette(n_bins)
    bar_width = 0.8 / n_bins

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _open_axes(ax)

    for b in range(n_bins):
        mask = bin_idx == b
        means = np.full(d, np.nan)
        if mask.any():
            means = params_pooled[mask].mean(axis=0)
        offset = (b - (n_bins - 1) / 2.0) * bar_width
        ax.bar(
            x_pos + offset,
            means,
            width=bar_width * 0.9,
            color=bar_colours[b],
            alpha=0.85,
            label=f"{edges[b]:.2f}–{edges[b + 1]:.2f}",
            zorder=3,
        )
        # Show the band's points themselves instead of a mean ±1 sd error bar.
        if mask.any():
            _scatter_bar_points(
                ax,
                x_pos + offset,
                params_pooled[mask],
                bar_width=bar_width * 0.9,
                colour=bar_colours[b],
            )

    # Extend the vertical group separators down to meet the group labels.
    _draw_category_separators(ax, sep_positions, y0=-0.30)
    _annotate_param_groups(ax, x_pos, breaks)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_PARAM_LATEX, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=13)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.legend(
        fontsize=11,
        title=_OBJ_LATEX[0] + " band",
        title_fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        borderaxespad=0,
        ncols=n_bins,
        frameon=False,
    )

    if save:
        fig.savefig(
            IMAGES_DIR / f"pareto_params_binned_{pareto_dir.name}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


def plot_best_parameters(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    top_n: int = 5,
    aggregate: bool = False,
    cluster: bool = False,
    max_cluster_distance: float = 0.2,
    category_breaks: list[int] | None = None,
    figsize: tuple[float, float] = (8.0, 8.5),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Best-scoring points: where they sit on the front and their parameters.

    Pools every Pareto point across all ``gene_drive_pareto_frontier_points_*``
    replicates in *pareto_dir*, ranks them by their average objective score
    (the mean of ``f_extinction`` and ``f_fixation``), and shows the *top_n*
    points in two stacked panels:

    - **(top)** objective space — every replicate frontier drawn faintly in
      grey, with the top points overlaid in colour and numbered by rank, so you
      can see where the best points sit along the frontier.
    - **(bottom)** the 9 decision variables of those same points.  This is the
      best-parameters counterpart to :func:`plot_parameters_binned`: instead of
      binning the whole frontier it focuses on the handful of points that score
      best across all runs.

    Three display modes for the bottom panel (checked in priority order
    ``cluster`` → ``aggregate`` → per-point):

    - ``cluster=True`` – group the top points by their objective *outcomes*
      ``(f_extinction, f_fixation)`` with agglomerative clustering (complete
      linkage, so no two points in a cluster are farther apart in objective
      space than *max_cluster_distance*) and show one mean bar per parameter
      *per cluster*, with the cluster's individual points overlaid as markers
      filled in that cluster's colour and ringed in black, so a point floating
      above its bar is still readable as that cluster's.  Clusters are numbered
      by descending mean score and coloured to match their markers above.  This
      groups solutions that sit in the same region of the frontier, so the bars
      reveal how the underlying
      parameter values vary among points that achieve a similar trade-off —
      rather than washing every top point into one average.
    - ``aggregate=True`` – a single bar per parameter showing the mean over the
      top *top_n* points with those points overlaid, matching the bar style of
      :func:`plot_parameters_binned`.
    - neither (default) – one bar per top point per parameter, each top point
      coloured distinctly (matching its marker above), legend labelled by rank
      and average score.  Lets you read off how consistent the best parameter
      sets are.

    Saved to ``IMAGES_DIR / "pareto_best_params_<dirname>.png"`` at 400 dpi when
    ``save`` is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of per-replicate Pareto CSVs.  Defaults to ``IBM_OUTPUT_DIR``.
    use_verified:
        Score and select using IBM-verified objective means (default) or raw
        surrogate values.
    top_n:
        Number of highest-scoring points to display.  Defaults to ``5``.  Capped
        at the number of pooled Pareto points available.
    aggregate:
        When ``True`` show one mean bar per parameter (with the individual
        points overlaid) instead of one bar per top point.  Default ``False``.
        Ignored when ``cluster`` is ``True``.
    cluster:
        When ``True`` cluster the top points by their parameter vectors and show
        one mean bar per parameter per cluster, with the cluster's points
        overlaid.  Takes precedence over ``aggregate``.  Default ``False``.
    max_cluster_distance:
        Maximum within-cluster Euclidean distance (complete-linkage threshold)
        used when ``cluster`` is ``True``.  Clustering is on the objective
        outcomes ``(f_extinction, f_fixation)``, which lie in ``[0, 1]``, so this
        is in the same units; larger values merge more points into fewer, looser
        clusters.  Defaults to ``0.2``.
    category_breaks:
        Parameter indices where a new category begins.  Defaults to ``[1, 5]``.
    figsize:
        Figure size in inches.  Defaults to ``(8.0, 8.5)``.
    title:
        Optional super-title.
    save:
        Whether to write the figure to ``IMAGES_DIR``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    replicates = _load_replicate_pareto_fronts(pareto_dir, use_verified=use_verified)
    params_pooled = np.vstack([rep["params"] for rep in replicates])  # (M, d)
    Y_pooled = np.vstack([rep["Y"] for rep in replicates])  # (M, 2)
    scores = Y_pooled.mean(axis=1)  # (M,)

    n = min(top_n, params_pooled.shape[0])
    # Highest scores first.
    top_idx = np.argsort(scores)[::-1][:n]
    top_params = params_pooled[top_idx]  # (n, d)
    top_scores = scores[top_idx]  # (n,)
    top_Y = Y_pooled[top_idx]  # (n, 2)

    # Decide bottom-panel mode and the colour of each top point's marker.
    #   cluster   → group the top points by similar objective outcomes (one
    #               colour each), then show the parameter spread within each
    #   aggregate → a single shared bar colour
    #   per-point → one distinct colour per point
    cluster_labels: np.ndarray | None = None
    cluster_colours: list[str] = []
    if cluster:
        cluster_labels, n_clusters = _cluster_top_by_outcome(
            top_Y, top_scores, max_cluster_distance
        )
        cluster_colours = _make_palette(n_clusters)
        point_colours = [cluster_colours[c] for c in cluster_labels]
        # Shape as well as colour, so the two clusters stay apart in greyscale.
        point_markers = [
            _CLUSTER_MARKERS[c % len(_CLUSTER_MARKERS)] for c in cluster_labels
        ]
    elif aggregate:
        point_colours = [IBM_PALETTE[0]] * n
        point_markers = ["o"] * n
    else:
        point_colours = _make_palette(n)
        point_markers = ["o"] * n

    d = params_pooled.shape[1]
    breaks = category_breaks if category_breaks is not None else [1, 5]
    x_pos, sep_positions = _param_x_positions(d, breaks)

    matplotlib.rcParams.update(_LATEX_RC)
    fig, (ax_obj, ax) = plt.subplots(
        2,
        1,
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    _open_axes(ax_obj, ax)

    # ── (top) objective space: faint frontiers + highlighted top points ──────
    for rep in replicates:
        Y = rep["Y"]
        order = np.argsort(Y[:, 0])
        ax_obj.plot(
            Y[order, 0],
            Y[order, 1],
            color="0.6",
            linewidth=1.0,
            alpha=0.5,
            zorder=1,
        )
        ax_obj.scatter(
            Y[:, 0],
            Y[:, 1],
            color="0.6",
            s=18,
            alpha=0.5,
            zorder=1,
        )
    for j in range(n):
        ax_obj.scatter(
            top_Y[j, 0],
            top_Y[j, 1],
            color=point_colours[j],
            marker=point_markers[j],
            s=70,
            alpha=0.95,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
        )
        ax_obj.annotate(
            f"{j + 1}",
            (top_Y[j, 0], top_Y[j, 1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            color=point_colours[j],
            zorder=4,
        )
    ax_obj.set_xlabel(_OBJ_LATEX[0], fontsize=13)
    ax_obj.set_ylabel(_OBJ_LATEX[1], fontsize=13)
    ax_obj.tick_params(labelsize=11)

    # ── (bottom) decision variables of the top points ───────────────────────
    if cluster:
        bar_width = 0.8 / n_clusters
        for c in range(n_clusters):
            members = cluster_labels == c
            m = int(members.sum())
            means = top_params[members].mean(axis=0)
            offset = (c - (n_clusters - 1) / 2.0) * bar_width
            # Hatch the second cluster: the blue/amber pair is ~14% apart in
            # luminance, so colour alone does not separate them in greyscale.
            ax.bar(
                x_pos + offset,
                means,
                width=bar_width * 0.9,
                **_two_series_bar_style(c),
                label=f"Cluster {c + 1} ({m} pt{'' if m == 1 else 's'}, "
                f"score {top_scores[members].mean():.3f})",
                zorder=3,
            )
            # Show the cluster's points themselves rather than an error bar.
            # The marker shape carries the cluster too, for the same reason.
            _scatter_bar_points(
                ax,
                x_pos + offset,
                top_params[members],
                bar_width=bar_width * 0.9,
                colour=cluster_colours[c],
                marker=_CLUSTER_MARKERS[c % len(_CLUSTER_MARKERS)],
            )
        legend_kw = dict(ncols=min(n_clusters, 3))
        legend_title = "Strategy clusters (avg.\\ score)"
    elif aggregate:
        means = top_params.mean(axis=0)
        ax.bar(
            x_pos,
            means,
            width=0.8,
            color=IBM_PALETTE[0],
            alpha=0.85,
            label=f"Top {n} mean",
            zorder=3,
        )
        # Show the top points themselves rather than an error bar.
        _scatter_bar_points(
            ax,
            x_pos,
            top_params,
            bar_width=0.8,
            colour=IBM_PALETTE[0],
        )
        legend_kw = dict(ncols=1)
        legend_title = "Best points (avg.\\ score)"
    else:
        bar_width = 0.8 / n
        for j in range(n):
            offset = (j - (n - 1) / 2.0) * bar_width
            ax.bar(
                x_pos + offset,
                top_params[j],
                width=bar_width * 0.9,
                color=point_colours[j],
                alpha=0.85,
                label=f"\\#{j + 1} (score {top_scores[j]:.3f})",
                zorder=3,
            )
        legend_kw = dict(ncols=min(n, 5))
        legend_title = "Best points (avg.\\ score)"

    # Extend the vertical group separators down to meet the group labels.
    _draw_category_separators(ax, sep_positions, y0=-0.30)
    _annotate_param_groups(ax, x_pos, breaks)
    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_PARAM_LATEX, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=13)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.legend(
        fontsize=11,
        title=legend_title,
        title_fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        borderaxespad=0,
        frameon=False,
        **legend_kw,
    )

    if save:
        fig.savefig(
            IMAGES_DIR / f"pareto_best_params_{pareto_dir.name}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


#: Genotype line styles for the mass-growth figure (WG/GG carry the
#: cluster-specific penalties; WW is the unpenalised wild-type reference).
_GENOTYPE_LATEX: dict[str, str] = {
    "WW": "WW",
    "WG": "WG",
    "GG": "GG",
}

#: Fixed reserve/mass constants for the infinite-food inner growth ODE, matching
#: ``initialize_simulation`` in agent_based_larval_model.py.
_GROWTH_RTH, _GROWTH_SIGMA = 15.0, 23.0
_GROWTH_W0, _GROWTH_RPUP = 4.1, 87.9
_GROWTH_R0 = 0.4 * _GROWTH_W0


def _base_larval_params(fixed_temperature: int) -> dict[str, float]:
    """Base (unpenalised) temperature-*fixed_temperature* larval parameters.

    Returns the ``a``, ``c``, ``alpha``, ``beta`` and ``g`` values the IBM uses
    at that temperature (see ``build_constant_functions``), keyed ``a0`` … ``g0``.
    """
    from IBM_Implementations.helpers.model_config import build_constant_functions

    funs, _ = build_constant_functions(fixed_temperature)
    return {
        "a0": funs["a_fun"](fixed_temperature),
        "c0": funs["c_fun"](fixed_temperature),
        "alpha0": funs["alpha_fun"](fixed_temperature),
        "beta0": funs["beta_fun"](fixed_temperature),
        "g0": funs["g_fun"](fixed_temperature),
    }


def _infinite_food_trajectory(
    factors: np.ndarray,
    base: dict[str, float],
    *,
    max_steps: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the IBM's inner growth ODE under infinite food.

    *factors* are the genotype penalty multipliers ``[a, alpha, beta, g]`` that
    scale the base larval parameters in *base* (from :func:`_base_larval_params`).
    Each hourly step uses the demanded food allocation
    ``FA = g·W·exp(−(R−Rth)²/2σ²)`` recomputed from the current state, matching
    the IBM's per-hour fixed ``FA_actual`` under unlimited food.  Integration
    starts from ``(W0, R0)`` and stops at pupation (``R ≥ Rpup``), starvation
    (``R ≤ 0``) or *max_steps* hours.

    Returns ``(t, W, R, dWdt)`` arrays of equal length, where ``t`` is elapsed
    hours, ``W`` the larval mass, ``R`` the reserves and ``dWdt`` the
    instantaneous mass growth rate at each recorded state.
    """
    from IBM_Implementations.helpers.feeding_and_movement import (
        rdg_model,
        state_update,
    )

    a_fac, alpha_fac, beta_fac, g_fac = factors
    a0, c0 = base["a0"], base["c0"]
    alpha0, beta0, g0 = base["alpha0"], base["beta0"], base["g0"]
    pars = np.array(
        [
            [
                a0 * a_fac,
                c0,
                alpha0 * alpha_fac,
                beta0 * beta_fac,
                g0 * g_fac,
                _GROWTH_RTH,
                _GROWTH_SIGMA,
            ]
        ]
    )
    state = np.array([[_GROWTH_W0, _GROWTH_R0]], dtype=float)
    ts, Ws, Rs, dWs = [], [], [], []
    for step in range(max_steps):
        W_now, R_now = state[0]
        FA = np.array(
            [
                g0
                * g_fac
                * W_now
                * np.exp(-((R_now - _GROWTH_RTH) ** 2) / (2 * _GROWTH_SIGMA**2))
            ]
        )
        ts.append(float(step))
        Ws.append(W_now)
        Rs.append(R_now)
        dWs.append(rdg_model(state, pars, FA=FA)[0, 0])
        if R_now >= _GROWTH_RPUP:
            break
        state = state_update(state, pars, rdg_model, FA=FA)
        if state[0, 1] <= 0.0:  # reserves exhausted: starvation
            ts.append(float(step + 1))
            Ws.append(state[0, 0])
            Rs.append(0.0)
            dWs.append(0.0)
            break
    return np.array(ts), np.array(Ws), np.array(Rs), np.array(dWs)


#: Penalty factors [a, alpha, beta, g] per genotype: WW is unpenalised, WG uses
#: param columns 1:5 and GG columns 5:9 of a 9-vector gene-drive design.
def _genotype_factors(param_vec: np.ndarray, genotype: str) -> np.ndarray:
    if genotype == "WW":
        return np.ones(4)
    if genotype == "WG":
        return param_vec[1:5]
    if genotype == "GG":
        return param_vec[5:9]
    raise ValueError(f"unknown genotype {genotype!r}")


#: Regexes matching threshold-pair subdirectories / legacy flat filenames.
_THRESH_DIR_RE = re.compile(r"^low(\d+p\d+)_high(\d+p\d+)$")
_THRESH_FILE_RE = re.compile(r"_low(\d+p\d+)_high(\d+p\d+)(?:_\d{8}_\d{6})?\.csv$")


def _collect_top_points_per_threshold(
    pareto_dir: Path | str,
    *,
    top_n: int,
    use_verified: bool = True,
) -> list[dict]:
    """Pool each threshold's replicates and return its top-*top_n* Pareto points.

    Scans *pareto_dir* for threshold-pair subdirectories (``low<lo>_high<hi>/``,
    new style) and legacy flat files
    (``gene_drive_pareto_frontier_points_low<lo>_high<hi>*.csv``).  For each
    threshold it pools every Pareto point across that threshold's replicate CSVs,
    scores them by the mean of the two objectives and keeps the highest-scoring
    *top_n* (fewer if the threshold has fewer points).

    Returns one dict per threshold, sorted by ``(low, high)``, with keys
    ``low``/``high`` (floats), ``params`` ``(k, 9)``, ``Y`` ``(k, 2)``,
    ``Yvar`` ``(k, 2)`` or ``None`` (surrogate), and ``scores`` ``(k,)``.
    """
    from collections import defaultdict

    pareto_dir = Path(pareto_dir)
    if use_verified:
        f1_col = "f_1 (low g frequency) [verified mean]"
        f2_col = "f_2 (high g frequency) [verified mean]"
        var_cols = [
            "f_1 (low g frequency) [verified var]",
            "f_2 (high g frequency) [verified var]",
        ]
    else:
        f1_col = "f_1 (low g frequency)"
        f2_col = "f_2 (high g frequency)"
        var_cols = None

    pooled: dict[tuple[float, float], dict[str, list]] = defaultdict(
        lambda: {"params": [], "Y": [], "Yvar": []}
    )

    def _add(df: pd.DataFrame, key: tuple[float, float]) -> None:
        pooled[key]["params"].append(df[_PARAM_COLS].to_numpy(dtype=float))
        pooled[key]["Y"].append(df[[f1_col, f2_col]].to_numpy(dtype=float))
        if var_cols is not None:
            pooled[key]["Yvar"].append(df[var_cols].to_numpy(dtype=float))

    # Pass 1: new-style — one subdirectory per threshold pair.
    for subdir in sorted(pareto_dir.iterdir()):
        if not subdir.is_dir():
            continue
        m = _THRESH_DIR_RE.match(subdir.name)
        if m is None:
            continue
        key = (
            float(m.group(1).replace("p", ".")),
            float(m.group(2).replace("p", ".")),
        )
        for csv_path in sorted(subdir.glob("gene_drive_pareto_frontier_points_*.csv")):
            _add(pd.read_csv(csv_path), key)

    # Pass 2: legacy — flat files with threshold tags in the filename.
    for csv_path in sorted(
        pareto_dir.glob("gene_drive_pareto_frontier_points_low*_high*.csv")
    ):
        m = _THRESH_FILE_RE.search(csv_path.name)
        if m is None:
            continue
        key = (
            float(m.group(1).replace("p", ".")),
            float(m.group(2).replace("p", ".")),
        )
        _add(pd.read_csv(csv_path), key)

    if not pooled:
        raise FileNotFoundError(
            f"No threshold-labelled Pareto CSV files found in {pareto_dir}"
        )

    entries: list[dict] = []
    for (low, high), d in sorted(pooled.items()):
        params = np.vstack(d["params"])
        Y = np.vstack(d["Y"])
        Yvar = np.vstack(d["Yvar"]) if var_cols is not None else None
        scores = Y.mean(axis=1)
        k = min(top_n, params.shape[0])
        idx = np.argsort(scores)[::-1][:k]
        entries.append(
            {
                "low": low,
                "high": high,
                "params": params[idx],
                "Y": Y[idx],
                "Yvar": Yvar[idx] if Yvar is not None else None,
                "scores": scores[idx],
            }
        )
    return entries


def plot_mass_growth_rate_by_cluster(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    top_n: int = 5,
    max_cluster_distance: float = 0.2,
    fixed_temperature: int = 20,
    genotypes: tuple[str, ...] = ("WG", "GG"),
    max_steps: int = 2000,
    loglog: bool = True,
    figsize: tuple[float, float] = (9.0, 4.2),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Larval mass growth rate ``dW/dt`` vs mass along the infinite-food trajectory.

    Clusters the top Pareto points by their objective outcomes (the same grouping
    as :func:`plot_best_parameters`), takes each cluster's **mean** 9-parameter
    gene-drive vector, and integrates the IBM's inner growth ODE
    (:func:`IBM_Implementations.helpers.feeding_and_movement.rdg_model`, advanced
    with :func:`~IBM_Implementations.helpers.feeding_and_movement.state_update`)
    forward in time from the initial state ``(W0, R0)``.  The instantaneous
    growth rate ``dW/dt`` is then plotted against mass ``W`` *along that
    trajectory*, rather than at a fixed reserve level.

    This matters because reserves ``R`` co-evolve with mass: the food the larva
    demands, ``FA = g·W·exp(−(R−Rth)²/2σ²)``, is homeostatic — it peaks when
    reserves sit at the target ``Rth`` and is suppressed once reserves climb
    above it.  So as reserves accumulate, intake (and hence ``dW/dt``) saturates
    and falls, giving the hump-shaped growth curve the model actually produces —
    not the straight line obtained by pinning ``R = Rth``.

    The genotype penalty factors scale the temperature-20 base larval parameters
    (``a``, ``alpha``, ``beta``, ``g``; ``c`` is unpenalised), so each point
    implies a different trajectory for the heterozygote (WG) and drive homozygote
    (GG).  One panel is drawn per genotype in *genotypes*.  Every individual top
    point is drawn as its own trajectory, coloured by cluster (colours matching
    :func:`plot_best_parameters`), with the unpenalised wild-type (WW) trajectory
    as a black dashed reference.  A dot marks each trajectory's end (pupation
    when ``R`` reaches ``Rpup``, or starvation when ``R`` falls to zero).  No
    cluster-mean line is drawn: clusters that are heterogeneous in a genotype
    (e.g. WG, whose top points range from starvation to runaway growth) have no
    meaningful mean trajectory.

    Modelling assumptions (matching the gene-drive optimisation set-up):

    - **Infinite food**: the larva always receives the allocation it demands —
      each hourly step is integrated with the demanded ``FA`` (recomputed from the
      current state), i.e. no competition or food limitation.
    - Integration starts from ``(W0, R0)`` and runs until pupation
      (``R ≥ Rpup``), starvation (``R ≤ 0``) or *max_steps* hours.

    Saved to ``IMAGES_DIR / "mass_growth_rate_<dirname>.png"`` at 400 dpi when
    *save* is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of per-replicate Pareto CSVs.  Defaults to ``IBM_OUTPUT_DIR``.
    use_verified:
        Score/select using IBM-verified objective means (default) or surrogate.
    top_n:
        Number of highest-scoring points pooled before clustering.  Default 5.
    max_cluster_distance:
        Maximum within-cluster objective-space distance.  Default 0.2.
    fixed_temperature:
        Temperature (°C) whose base larval parameters are used.  Default 20.
    genotypes:
        Which genotypes to draw a panel for.  Default ``("WG", "GG")``.
    max_steps:
        Maximum number of hourly integration steps.  Default 2000.
    loglog:
        Use logarithmic mass (x) and growth-rate (y) axes.  Default ``True`` —
        trajectories span several decades in both under infinite food.
    figsize, title, save:
        Figure size, optional super-title, and whether to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    replicates = _load_replicate_pareto_fronts(pareto_dir, use_verified=use_verified)
    params_pooled = np.vstack([rep["params"] for rep in replicates])
    Y_pooled = np.vstack([rep["Y"] for rep in replicates])
    scores = Y_pooled.mean(axis=1)

    n = min(top_n, params_pooled.shape[0])
    top_idx = np.argsort(scores)[::-1][:n]
    top_params = params_pooled[top_idx]
    top_scores = scores[top_idx]
    top_Y = Y_pooled[top_idx]

    cluster_labels, n_clusters = _cluster_top_by_outcome(
        top_Y, top_scores, max_cluster_distance
    )
    cluster_colours = _make_palette(n_clusters)
    cluster_sizes = [int((cluster_labels == c).sum()) for c in range(n_clusters)]
    cluster_mean_scores = [
        top_scores[cluster_labels == c].mean() for c in range(n_clusters)
    ]

    base = _base_larval_params(fixed_temperature)
    W0 = _GROWTH_W0

    def _trajectory(factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(W, dW/dt)`` along the infinite-food trajectory for *factors*."""
        _, W, _, dW = _infinite_food_trajectory(factors, base, max_steps=max_steps)
        return W, dW

    ww_W, ww_dW = _trajectory(np.ones(4))

    matplotlib.rcParams.update(_LATEX_RC)
    fig, axes = plt.subplots(
        1,
        len(genotypes),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    _open_axes(*axes)

    # On log axes only strictly-positive growth rates can be drawn; mask the
    # zero-growth (starvation / pupation) tail before plotting.  Track the extent
    # of every drawn trajectory so the view can be bounded to the sane region
    # (infinite food lets low-allocation genotypes run away to absurd masses).
    extents: list[tuple[float, float]] = []

    def _draw(
        ax, W, dW, *, colour, lw, alpha=1.0, label=None, ls="-", end_marker=False
    ):
        m = dW > 1e-9 if loglog else np.ones_like(dW, dtype=bool)
        if m.sum() == 0:
            return
        extents.append((W[m].max(), dW[m].max()))
        ax.plot(
            W[m],
            dW[m],
            color=colour,
            linewidth=lw,
            alpha=alpha,
            linestyle=ls,
            label=label,
            zorder=3,
        )
        if end_marker:
            ax.scatter(
                W[m][-1],
                dW[m][-1],
                color=colour,
                s=22,
                zorder=4,
                edgecolors="white",
                linewidths=0.5,
            )

    for ax, genotype in zip(axes, genotypes):
        _draw(
            ax,
            ww_W,
            ww_dW,
            colour="black",
            lw=1.6,
            ls=":",
            label="WW (wild type)",
            end_marker=True,
        )
        # One trajectory per individual top point, coloured by cluster; no mean.
        # The dash pattern repeats the cluster distinction for greyscale prints.
        for c in range(n_clusters):
            cluster_ls = plot_style.TWO_SERIES_LINESTYLE[
                c % len(plot_style.TWO_SERIES_LINESTYLE)
            ]
            for member in top_params[cluster_labels == c]:
                Wm, dWm = _trajectory(_genotype_factors(member, genotype))
                _draw(
                    ax,
                    Wm,
                    dWm,
                    colour=cluster_colours[c],
                    lw=1.4,
                    alpha=0.75,
                    ls=cluster_ls,
                    end_marker=True,
                )
        ax.set_xlabel(r"Larval mass $W$", fontsize=13)
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.tick_params(labelsize=11)
        ax.set_title(f"{_GENOTYPE_LATEX.get(genotype, genotype)} genotype", fontsize=13)

    # Bound the (shared) axes to the sane region: a trajectory is "sane" if it
    # stays below sane_cap; pathological runaways then visibly exit the panels.
    if loglog and extents:
        sane_cap = 1e5
        sane = [(w, d) for w, d in extents if w < sane_cap] or extents
        x_hi = max(w for w, _ in sane) * 3.0
        y_hi = max(d for _, d in sane) * 3.0
        for ax in axes:
            ax.set_xlim(W0 * 0.5, x_hi)
            ax.set_ylim(y_hi * 1e-4, y_hi)

    axes[0].set_ylabel(r"Growth rate $\mathrm{d}W/\mathrm{d}t$", fontsize=13)
    # Build the legend explicitly so every cluster is listed even when all of its
    # trajectories for a genotype are non-viable (starve instantly, never drawn).
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color="black", linewidth=1.6, linestyle=":")]
    labels = ["WW (wild type)"]
    for c in range(n_clusters):
        handles.append(
            Line2D(
                [],
                [],
                color=cluster_colours[c],
                linewidth=1.4,
                alpha=0.75,
                linestyle=plot_style.TWO_SERIES_LINESTYLE[
                    c % len(plot_style.TWO_SERIES_LINESTYLE)
                ],
            )
        )
        labels.append(
            f"Cluster {c + 1} top points "
            f"({cluster_sizes[c]}, score {cluster_mean_scores[c]:.3f})"
        )
    fig.legend(
        handles,
        labels,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=min(len(labels), 4),
        frameon=False,
        borderaxespad=0,
    )

    if save:
        fig.savefig(
            IMAGES_DIR / f"mass_growth_rate_{pareto_dir.name}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


def plot_mass_growth_rate_across_thresholds(
    pareto_dir: Path | str | None = None,
    *,
    top_n: int = 3,
    use_verified: bool = True,
    fixed_temperature: int = 20,
    genotypes: tuple[str, ...] = ("WG", "GG"),
    max_steps: int = 2000,
    loglog: bool = True,
    figsize: tuple[float, float] = (9.0, 4.6),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Infinite-food growth curves of the top points, coloured by threshold.

    The across-thresholds counterpart of :func:`plot_mass_growth_rate_by_cluster`
    with **no clustering**: for each release-threshold window in *pareto_dir* it
    pools the window's replicates, takes the top-*top_n* Pareto points (ranked by
    mean objective) and integrates each point's infinite-food larval-growth
    trajectory (:func:`_infinite_food_trajectory`).  The instantaneous growth
    rate ``dW/dt`` is plotted against mass ``W`` along that trajectory — the
    hump-shaped curve the model actually produces once homeostatic feeding
    saturates reserves (see :func:`plot_mass_growth_rate_by_cluster` for the
    mechanism).

    One panel per genotype in *genotypes*.  Every top point is drawn as its own
    trajectory, **coloured by its threshold window** (so the architectures of
    different thresholds can be compared directly), with the unpenalised
    wild-type (WW) trajectory as a black dashed reference.  A dot marks each
    trajectory's end (pupation or starvation).

    Saved to ``IMAGES_DIR / "mass_growth_rate_across_thresholds.png"`` at 400 dpi
    when *save* is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of threshold subdirectories / Pareto CSVs.  Defaults to
        ``IBM_OUTPUT_DIR``.
    top_n:
        Number of highest-scoring points per threshold to draw.  Default 3.
    use_verified:
        Score/select using IBM-verified objective means (default) or surrogate.
    fixed_temperature:
        Temperature (°C) whose base larval parameters are used.  Default 20.
    genotypes:
        Which genotypes to draw a panel for.  Default ``("WG", "GG")``.
    max_steps:
        Maximum number of hourly integration steps.  Default 2000.
    loglog:
        Use logarithmic mass (x) and growth-rate (y) axes.  Default ``True``.
    figsize, title, save:
        Figure size, optional super-title, and whether to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    thresholds = _collect_top_points_per_threshold(
        pareto_dir, top_n=top_n, use_verified=use_verified
    )
    n_thresh = len(thresholds)
    thresh_colours = _ordered_palette(n_thresh, for_lines=True)
    thresh_linestyles = plot_style.ordered_linestyles(n_thresh)

    base = _base_larval_params(fixed_temperature)

    def _trajectory(factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, W, _, dW = _infinite_food_trajectory(factors, base, max_steps=max_steps)
        return W, dW

    ww_W, ww_dW = _trajectory(np.ones(4))

    matplotlib.rcParams.update(_LATEX_RC)
    fig, axes = plt.subplots(
        1,
        len(genotypes),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    # On log axes only strictly-positive growth rates can be drawn; mask the
    # zero-growth (starvation / pupation) tail before plotting, and track every
    # drawn trajectory's extent so the view can be bounded to the sane region.
    extents: list[tuple[float, float]] = []

    def _draw(
        ax, W, dW, *, colour, lw, alpha=1.0, label=None, ls="-", end_marker=False
    ):
        m = dW > 1e-9 if loglog else np.ones_like(dW, dtype=bool)
        if m.sum() == 0:
            return
        extents.append((W[m].max(), dW[m].max()))
        ax.plot(
            W[m],
            dW[m],
            color=colour,
            linewidth=lw,
            alpha=alpha,
            linestyle=ls,
            label=label,
            zorder=3,
        )
        if end_marker:
            ax.scatter(
                W[m][-1],
                dW[m][-1],
                color=colour,
                s=22,
                zorder=4,
                edgecolors="black",
                linewidths=0.5,
            )

    for ax, genotype in zip(axes, genotypes):
        _draw(
            ax,
            ww_W,
            ww_dW,
            colour="black",
            lw=1.6,
            ls=":",
            label="WW (wild type)",
            end_marker=True,
        )
        # One trajectory per individual top point, coloured by its threshold.
        for t_idx, t in enumerate(thresholds):
            for member in t["params"]:
                Wm, dWm = _trajectory(_genotype_factors(member, genotype))
                _draw(
                    ax,
                    Wm,
                    dWm,
                    colour=thresh_colours[t_idx],
                    lw=1.4,
                    alpha=0.75,
                    ls=thresh_linestyles[t_idx],
                    end_marker=True,
                )
        ax.set_xlabel(r"Larval mass $W$", fontsize=13)
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.tick_params(labelsize=11)
        ax.set_title(f"{_GENOTYPE_LATEX.get(genotype, genotype)} genotype", fontsize=13)

    # Bound the (shared) axes to the sane region: a trajectory is "sane" if it
    # stays below sane_cap; pathological runaways then visibly exit the panels.
    if loglog and extents:
        sane_cap = 1e5
        sane = [(w, d) for w, d in extents if w < sane_cap] or extents
        x_hi = max(w for w, _ in sane) * 3.0
        y_hi = max(d for _, d in sane) * 3.0
        for ax in axes:
            ax.set_xlim(_GROWTH_W0 * 0.5, x_hi)
            ax.set_ylim(y_hi * 1e-4, y_hi)

    axes[0].set_ylabel(r"Growth rate $\mathrm{d}W/\mathrm{d}t$", fontsize=13)
    # Explicit legend so every threshold is listed even when all of its
    # trajectories for a genotype are non-viable (starve instantly, never drawn).
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color="black", linewidth=1.6, linestyle=":")]
    labels = ["WW (wild type)"]
    for t_idx, t in enumerate(thresholds):
        handles.append(
            Line2D(
            [],
            [],
            color=thresh_colours[t_idx],
            linewidth=1.4,
            alpha=0.75,
            linestyle=thresh_linestyles[t_idx],
        )
        )
        labels.append(f"({t['low']:.2g}, {t['high']:.2g}) (top {t['params'].shape[0]})")
    fig.legend(
        handles,
        labels,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=min(len(labels), 4),
        frameon=False,
        borderaxespad=0,
    )

    if save:
        fig.savefig(
            IMAGES_DIR / "mass_growth_rate_across_thresholds.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


def plot_reserve_trajectory_by_cluster(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    top_n: int = 5,
    max_cluster_distance: float = 0.2,
    fixed_temperature: int = 20,
    genotypes: tuple[str, ...] = ("WG", "GG"),
    max_steps: int = 2000,
    t_max_hours: float = 300.0,
    figsize: tuple[float, float] = (9.0, 7.4),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Reserves ``R`` and mass ``W`` vs time ``t`` along the infinite-food trajectory.

    Companion to :func:`plot_mass_growth_rate_by_cluster`: it integrates the
    **same** infinite-food trajectory (the IBM's inner growth ODE
    :func:`~IBM_Implementations.helpers.feeding_and_movement.rdg_model`, advanced
    hourly with
    :func:`~IBM_Implementations.helpers.feeding_and_movement.state_update` from
    the initial state ``(W0, R0)`` with the demanded food allocation at every
    step) but plots the state variables — **reserve** ``R`` and **mass** ``W`` —
    against elapsed time ``t`` (hours) rather than the growth rate against mass.

    This makes the reserve dynamics directly visible: a viable larva fills its
    reserves until they reach the pupation threshold ``Rpup`` (dotted line);
    homeostatic feedback around the target ``Rth`` shapes how fast
    intake — and so reserve accumulation — proceeds.  Genotypes that route little
    surplus into reserves (low ``alpha``) never fill, so their reserve curve
    plateaus below ``Rpup`` and they run to *max_steps* without pupating; a
    starving larva's reserves fall to zero.

    The genotype penalty factors scale the temperature-20 base larval parameters
    (``a``, ``alpha``, ``beta``, ``g``; ``c`` is unpenalised).  One panel is drawn
    per genotype in *genotypes*, with two rows: reserves ``R`` (top) and larval
    mass ``W`` (bottom, on a log axis) against the same time axis.  Every
    individual top point is drawn as its own trajectory, coloured by cluster
    (colours matching :func:`plot_best_parameters`), with the unpenalised
    wild-type (WW) trajectory as a black dashed reference.  A dot marks each
    trajectory's end (pupation at ``R = Rpup``, starvation at ``R = 0``, or
    *max_steps* hours).  No cluster-mean line is drawn, matching
    :func:`plot_mass_growth_rate_by_cluster`.  The mass axis is bounded to the
    trajectories that pupate within the window, so low-``alpha`` genotypes that
    run away to absurd masses under infinite food visibly exit the top.

    Saved to ``IMAGES_DIR / "reserve_trajectory_<dirname>.png"`` at 400 dpi when
    *save* is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of per-replicate Pareto CSVs.  Defaults to ``IBM_OUTPUT_DIR``.
    use_verified:
        Score/select using IBM-verified objective means (default) or surrogate.
    top_n:
        Number of highest-scoring points pooled before clustering.  Default 5.
    max_cluster_distance:
        Maximum within-cluster objective-space distance.  Default 0.2.
    fixed_temperature:
        Temperature (°C) whose base larval parameters are used.  Default 20.
    genotypes:
        Which genotypes to draw a panel for.  Default ``("WG", "GG")``.
    max_steps:
        Maximum number of hourly integration steps.  Default 2000.
    t_max_hours:
        Upper limit of the (shared) time axis, in hours.  Default 300 — the
        length of one IBM larval cycle (``time_steps`` in
        ``agent_based_larval_model.py``): a larva that has not pupated by then is
        culled in the model, so trajectories beyond it are not biologically
        realised and only clutter the early dynamics.
    figsize, title, save:
        Figure size, optional super-title, and whether to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    replicates = _load_replicate_pareto_fronts(pareto_dir, use_verified=use_verified)
    params_pooled = np.vstack([rep["params"] for rep in replicates])
    Y_pooled = np.vstack([rep["Y"] for rep in replicates])
    scores = Y_pooled.mean(axis=1)

    n = min(top_n, params_pooled.shape[0])
    top_idx = np.argsort(scores)[::-1][:n]
    top_params = params_pooled[top_idx]
    top_scores = scores[top_idx]
    top_Y = Y_pooled[top_idx]

    cluster_labels, n_clusters = _cluster_top_by_outcome(
        top_Y, top_scores, max_cluster_distance
    )
    cluster_colours = _make_palette(n_clusters)
    cluster_sizes = [int((cluster_labels == c).sum()) for c in range(n_clusters)]
    cluster_mean_scores = [
        top_scores[cluster_labels == c].mean() for c in range(n_clusters)
    ]

    base = _base_larval_params(fixed_temperature)
    Rpup, W0 = _GROWTH_RPUP, _GROWTH_W0

    def _trajectory(
        factors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(t, R, W)`` along the infinite-food trajectory for *factors*."""
        t, W, R, _ = _infinite_food_trajectory(factors, base, max_steps=max_steps)
        return t, R, W

    ww_t, ww_R, ww_W = _trajectory(np.ones(4))

    matplotlib.rcParams.update(_LATEX_RC)
    # Two rows (reserves R on top, mass W below) x one column per genotype; the
    # time axis is shared throughout, the y axes are shared within each row.
    fig, axes = plt.subplots(
        2,
        len(genotypes),
        figsize=figsize,
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    r_axes, w_axes = axes[0], axes[1]
    _open_axes(*axes.ravel())

    # Trajectories from a zero-conversion-rate design (pure underdominance, no
    # gene-drive conversion) are flagged with open-diamond markers along the line
    # so they can be told apart from the converting designs of the same cluster.
    conv_tol = 1e-6

    def _draw(
        ax,
        t,
        y,
        *,
        colour,
        lw,
        alpha=1.0,
        label=None,
        ls="-",
        end_marker=False,
        zero_conv=False,
    ):
        ax.plot(
            t,
            y,
            color=colour,
            linewidth=lw,
            alpha=alpha,
            linestyle=ls,
            label=label,
            zorder=3,
            marker="D" if zero_conv else None,
            markevery=20,
            markersize=4.5,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=0.9,
        )
        if end_marker:
            ax.scatter(
                t[-1],
                y[-1],
                color=colour,
                s=30 if zero_conv else 22,
                marker="D" if zero_conv else "o",
                zorder=4,
                edgecolors="white",
                linewidths=0.5,
            )

    # Mass spans many decades under infinite food (low-alpha genotypes route all
    # surplus to growth and run away), so the W row uses a log axis bounded to the
    # "sane" region — trajectories that actually pupate within the window.  Track
    # the peak mass of those so pathological runaways visibly exit the top.
    pupation_W: list[float] = []

    for col, genotype in enumerate(genotypes):
        ax_R, ax_W = r_axes[col], w_axes[col]
        # Reference reserve level: the pupation threshold.
        ax_R.axhline(Rpup, color="grey", lw=0.9, ls=":", zorder=1)
        _draw(
            ax_R,
            ww_t,
            ww_R,
            colour="black",
            lw=1.6,
            ls=":",
            label="WW (wild type)",
            end_marker=True,
        )
        _draw(
            ax_W,
            ww_t,
            ww_W,
            colour="black",
            lw=1.6,
            ls=":",
            label="WW (wild type)",
            end_marker=True,
        )
        in_window = ww_t <= t_max_hours
        if ww_R[in_window][-1] >= Rpup:
            pupation_W.append(ww_W[in_window].max())
        # One trajectory per individual top point, coloured by cluster; no mean.
        # The dash pattern repeats the cluster distinction, which the blue/amber
        # pair alone loses in a black-and-white print.
        for c in range(n_clusters):
            cluster_ls = plot_style.TWO_SERIES_LINESTYLE[
                c % len(plot_style.TWO_SERIES_LINESTYLE)
            ]
            for member in top_params[cluster_labels == c]:
                zero_conv = member[0] <= conv_tol
                tm, Rm, Wm = _trajectory(_genotype_factors(member, genotype))
                _draw(
                    ax_R,
                    tm,
                    Rm,
                    colour=cluster_colours[c],
                    lw=1.4,
                    alpha=0.75,
                    ls=cluster_ls,
                    end_marker=True,
                    zero_conv=zero_conv,
                )
                _draw(
                    ax_W,
                    tm,
                    Wm,
                    colour=cluster_colours[c],
                    lw=1.4,
                    alpha=0.75,
                    ls=cluster_ls,
                    end_marker=True,
                    zero_conv=zero_conv,
                )
                win = tm <= t_max_hours
                if win.any() and Rm[win][-1] >= Rpup:
                    pupation_W.append(Wm[win].max())
        ax_W.set_xlabel(r"Time $t$ (hours)", fontsize=13)
        ax_R.set_xlim(0, t_max_hours)
        ax_R.set_title(
            f"{_GENOTYPE_LATEX.get(genotype, genotype)} genotype", fontsize=13
        )
        for ax in (ax_R, ax_W):
            ax.tick_params(labelsize=11)
        ax_W.set_yscale("log")

    # Bound the (shared) mass axis to the pupating trajectories; runaways exit.
    w_hi = (max(pupation_W) if pupation_W else float(ww_W.max())) * 3.0
    for ax in w_axes:
        ax.set_ylim(W0 * 0.7, w_hi)

    r_axes[0].set_ylabel(r"Reserves $R$", fontsize=13)
    w_axes[0].set_ylabel(r"Larval mass $W$", fontsize=13)
    # Annotate the pupation reserve threshold on the right-hand reserve panel.
    r_axes[-1].text(
        0.99,
        Rpup,
        r"$R_{\mathrm{pup}}$",
        color="grey",
        fontsize=10,
        ha="right",
        va="bottom",
        transform=r_axes[-1].get_yaxis_transform(),
    )
    # Build the legend explicitly, split over two rows so it stays within the
    # figure width: the per-cluster entries on top, the genotype/design "type"
    # entries (WW reference, zero-conversion marker) underneath.
    from matplotlib.lines import Line2D

    cluster_handles = [
        Line2D(
            [],
            [],
            color=cluster_colours[c],
            linewidth=1.4,
            alpha=0.75,
            linestyle=plot_style.TWO_SERIES_LINESTYLE[
                c % len(plot_style.TWO_SERIES_LINESTYLE)
            ],
        )
        for c in range(n_clusters)
    ]
    cluster_labels = [
        f"Cluster {c + 1} top points "
        f"({cluster_sizes[c]}, score {cluster_mean_scores[c]:.3f})"
        for c in range(n_clusters)
    ]
    type_handles = [Line2D([], [], color="black", linewidth=1.6, linestyle=":")]
    type_labels = ["WW (wild type)"]
    if (top_params[:, 0] <= conv_tol).any():
        type_handles.append(
            Line2D(
                [],
                [],
                color="0.4",
                linewidth=1.4,
                linestyle="-",
                marker="D",
                markerfacecolor="none",
                markeredgecolor="0.4",
                markersize=4.5,
            )
        )
        type_labels.append("Zero conversion rate (pure underdominance)")
    fig.legend(
        cluster_handles,
        cluster_labels,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=len(cluster_handles),
        frameon=False,
        borderaxespad=0,
    )
    fig.legend(
        type_handles,
        type_labels,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncols=len(type_handles),
        frameon=False,
        borderaxespad=0,
    )

    if save:
        fig.savefig(
            IMAGES_DIR / f"reserve_trajectory_{pareto_dir.name}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


def plot_reserve_trajectory_across_thresholds(
    pareto_dir: Path | str | None = None,
    *,
    top_n: int = 3,
    use_verified: bool = True,
    fixed_temperature: int = 20,
    genotypes: tuple[str, ...] = ("WG", "GG"),
    max_steps: int = 2000,
    t_max_hours: float = 300.0,
    figsize: tuple[float, float] = (9.0, 7.4),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Reserves ``R`` and mass ``W`` vs time ``t``, coloured by threshold.

    The across-thresholds counterpart of
    :func:`plot_reserve_trajectory_by_cluster` with **no clustering**: for each
    release-threshold window in *pareto_dir* it pools the window's replicates,
    takes the top-*top_n* Pareto points (ranked by mean objective) and integrates
    each point's infinite-food larval trajectory
    (:func:`_infinite_food_trajectory`), then plots the state variables —
    **reserve** ``R`` (top row) and **mass** ``W`` (bottom row, log axis) —
    against elapsed time ``t`` (hours).

    One column per genotype in *genotypes*.  Every top point is drawn as its own
    trajectory, **coloured by its threshold window**, with the unpenalised
    wild-type (WW) trajectory as a black dashed reference.  Zero-conversion-rate
    (pure-underdominance) designs are flagged with open-diamond markers.  A dot
    marks each trajectory's end (pupation at ``R = Rpup``, starvation at
    ``R = 0``, or *max_steps* hours).  The mass axis is bounded to the
    trajectories that pupate within the window, so low-``alpha`` genotypes that
    run away to absurd masses under infinite food visibly exit the top.

    Saved to ``IMAGES_DIR / "reserve_trajectory_across_thresholds.png"`` at 400
    dpi when *save* is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of threshold subdirectories / Pareto CSVs.  Defaults to
        ``IBM_OUTPUT_DIR``.
    top_n:
        Number of highest-scoring points per threshold to draw.  Default 3.
    use_verified:
        Score/select using IBM-verified objective means (default) or surrogate.
    fixed_temperature:
        Temperature (°C) whose base larval parameters are used.  Default 20.
    genotypes:
        Which genotypes to draw a column for.  Default ``("WG", "GG")``.
    max_steps:
        Maximum number of hourly integration steps.  Default 2000.
    t_max_hours:
        Upper limit of the (shared) time axis, in hours.  Default 300 — one IBM
        larval cycle, beyond which an un-pupated larva is culled.
    figsize, title, save:
        Figure size, optional super-title, and whether to write the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    thresholds = _collect_top_points_per_threshold(
        pareto_dir, top_n=top_n, use_verified=use_verified
    )
    n_thresh = len(thresholds)
    thresh_colours = _ordered_palette(n_thresh, for_lines=True)
    thresh_linestyles = plot_style.ordered_linestyles(n_thresh)

    base = _base_larval_params(fixed_temperature)
    Rpup, W0 = _GROWTH_RPUP, _GROWTH_W0

    def _trajectory(
        factors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(t, R, W)`` along the infinite-food trajectory for *factors*."""
        t, W, R, _ = _infinite_food_trajectory(factors, base, max_steps=max_steps)
        return t, R, W

    ww_t, ww_R, ww_W = _trajectory(np.ones(4))

    matplotlib.rcParams.update(_LATEX_RC)
    # Two rows (reserves R on top, mass W below) x one column per genotype; the
    # time axis is shared throughout, the y axes are shared within each row.
    fig, axes = plt.subplots(
        2,
        len(genotypes),
        figsize=figsize,
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    r_axes, w_axes = axes[0], axes[1]

    # Trajectories from a zero-conversion-rate design (pure underdominance) are
    # flagged with open-diamond markers so they stand out from converting designs.
    conv_tol = 1e-6

    def _draw(
        ax,
        t,
        y,
        *,
        colour,
        lw,
        alpha=1.0,
        label=None,
        ls="-",
        end_marker=False,
        zero_conv=False,
    ):
        ax.plot(
            t,
            y,
            color=colour,
            linewidth=lw,
            alpha=alpha,
            linestyle=ls,
            label=label,
            zorder=3,
            marker="D" if zero_conv else None,
            markevery=20,
            markersize=4.5,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=0.9,
        )
        if end_marker:
            ax.scatter(
                t[-1],
                y[-1],
                color=colour,
                s=30 if zero_conv else 22,
                marker="D" if zero_conv else "o",
                zorder=4,
                edgecolors="black",
                linewidths=0.5,
            )

    # Mass spans many decades under infinite food, so the W row uses a log axis
    # bounded to the "sane" region — trajectories that pupate within the window.
    pupation_W: list[float] = []

    for col, genotype in enumerate(genotypes):
        ax_R, ax_W = r_axes[col], w_axes[col]
        # Reference reserve level: the pupation threshold.
        ax_R.axhline(Rpup, color="grey", lw=0.9, ls=":", zorder=1)
        _draw(
            ax_R,
            ww_t,
            ww_R,
            colour="black",
            lw=1.6,
            ls=":",
            label="WW (wild type)",
            end_marker=True,
        )
        _draw(
            ax_W,
            ww_t,
            ww_W,
            colour="black",
            lw=1.6,
            ls=":",
            label="WW (wild type)",
            end_marker=True,
        )
        in_window = ww_t <= t_max_hours
        if ww_R[in_window][-1] >= Rpup:
            pupation_W.append(ww_W[in_window].max())
        # One trajectory per individual top point, coloured by its threshold.
        # Seven thin overlapping lines crowd at the dark end of the ramp, so the
        # dash pattern breaks the tie; darkness still carries the rank.
        for t_idx, t in enumerate(thresholds):
            thresh_ls = thresh_linestyles[t_idx]
            for member in t["params"]:
                zero_conv = member[0] <= conv_tol
                tm, Rm, Wm = _trajectory(_genotype_factors(member, genotype))
                _draw(
                    ax_R,
                    tm,
                    Rm,
                    colour=thresh_colours[t_idx],
                    lw=1.4,
                    alpha=0.75,
                    ls=thresh_ls,
                    end_marker=True,
                    zero_conv=zero_conv,
                )
                _draw(
                    ax_W,
                    tm,
                    Wm,
                    colour=thresh_colours[t_idx],
                    lw=1.4,
                    alpha=0.75,
                    ls=thresh_ls,
                    end_marker=True,
                    zero_conv=zero_conv,
                )
                win = tm <= t_max_hours
                if win.any() and Rm[win][-1] >= Rpup:
                    pupation_W.append(Wm[win].max())
        ax_W.set_xlabel(r"Time $t$ (hours)", fontsize=13)
        ax_R.set_xlim(0, t_max_hours)
        ax_R.set_title(
            f"{_GENOTYPE_LATEX.get(genotype, genotype)} genotype", fontsize=13
        )
        for ax in (ax_R, ax_W):
            ax.tick_params(labelsize=11)
        ax_W.set_yscale("log")

    # Bound the (shared) mass axis to the pupating trajectories; runaways exit.
    w_hi = (max(pupation_W) if pupation_W else float(ww_W.max())) * 3.0
    for ax in w_axes:
        ax.set_ylim(W0 * 0.7, w_hi)

    r_axes[0].set_ylabel(r"Reserves $R$", fontsize=13)
    w_axes[0].set_ylabel(r"Larval mass $W$", fontsize=13)
    # Annotate the pupation reserve threshold on the right-hand reserve panel.
    r_axes[-1].text(
        0.99,
        Rpup,
        r"$R_{\mathrm{pup}}$",
        color="grey",
        fontsize=10,
        ha="right",
        va="bottom",
        transform=r_axes[-1].get_yaxis_transform(),
    )
    # Build the legend explicitly, split over two rows: the per-threshold entries
    # on top, the design "type" entries (WW reference, zero-conversion) beneath.
    from matplotlib.lines import Line2D

    thresh_handles = [
        Line2D(
            [],
            [],
            color=thresh_colours[t_idx],
            linewidth=1.4,
            alpha=0.75,
            linestyle=thresh_linestyles[t_idx],
        )
        for t_idx in range(n_thresh)
    ]
    thresh_labels = [f"({t['low']:.2g}, {t['high']:.2g})" for t in thresholds]
    type_handles = [Line2D([], [], color="black", linewidth=1.6, linestyle=":")]
    type_labels = ["WW (wild type)"]
    if any((t["params"][:, 0] <= conv_tol).any() for t in thresholds):
        type_handles.append(
            Line2D(
                [],
                [],
                color="0.4",
                linewidth=1.4,
                linestyle="-",
                marker="D",
                markerfacecolor="none",
                markeredgecolor="0.4",
                markersize=4.5,
            )
        )
        type_labels.append("Zero conversion rate (pure underdominance)")
    fig.legend(
        thresh_handles,
        thresh_labels,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=min(len(thresh_handles), 4),
        frameon=False,
        borderaxespad=0,
    )
    fig.legend(
        type_handles,
        type_labels,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncols=len(type_handles),
        frameon=False,
        borderaxespad=0,
    )

    if save:
        fig.savefig(
            IMAGES_DIR / "reserve_trajectory_across_thresholds.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


def plot_parameter_correlations(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    method: str = "pearson",
    category_breaks: list[int] | None = None,
    cmap: "str | matplotlib.colors.Colormap | None" = None,
    figsize: tuple[float, float] = (7.5, 6.5),
    title: str | None = None,
    save: bool = True,
) -> "matplotlib.figure.Figure":
    """Heatmap of pairwise correlations between the 9 decision variables.

    Pools every Pareto point across all replicates in *pareto_dir* and shows the
    parameter-by-parameter correlation matrix as an annotated heatmap.  This
    surfaces joint structure on the Pareto front — e.g. parameters that trade
    off against one another — that the per-parameter views cannot show.  Only
    the lower triangle is drawn; the diagonal and the (duplicate) upper triangle
    are blanked.  Parameters that are constant across the whole front have
    undefined correlation; those cells are also left blank.

    Saved to ``IMAGES_DIR / "pareto_param_correlations_<dirname>.png"`` at 400
    dpi when ``save`` is ``True``.

    Parameters
    ----------
    pareto_dir:
        Directory of per-replicate Pareto CSVs.  Defaults to ``IBM_OUTPUT_DIR``.
    use_verified:
        Use IBM-verified objective means (default) or raw surrogate values.
        (Only affects which rows are loaded; correlations use the 9 parameters.)
    method:
        Correlation method passed to :meth:`pandas.DataFrame.corr` —
        ``"pearson"`` (linear, default) or ``"spearman"`` (rank/monotonic,
        better for non-linear trends).
    category_breaks:
        Parameter indices where a category boundary is drawn.  Defaults to
        ``[1, 5]`` (conversion rate / WG penalties / GG penalties).
    cmap:
        Diverging colormap (name or ``Colormap``), centred at zero over
        ``[-1, 1]``.  Defaults to the house-style IBM diverging map
        (blue → white → yellow).
    figsize:
        Figure size in inches.  Defaults to ``(7.5, 6.5)``.
    title:
        Optional figure title.
    save:
        Whether to write the figure to ``IMAGES_DIR``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    replicates = _load_replicate_pareto_fronts(pareto_dir, use_verified=use_verified)
    params_pooled = np.vstack([rep["params"] for rep in replicates])  # (M, d)
    d = params_pooled.shape[1]

    corr = (
        pd.DataFrame(params_pooled, columns=_PARAM_COLS).corr(method=method).to_numpy()
    )
    # Blank the diagonal and the upper triangle (it duplicates the lower one).
    corr[np.triu_indices(d, k=0)] = np.nan
    masked = np.ma.masked_invalid(corr)

    breaks = category_breaks if category_breaks is not None else [1, 5]

    if cmap is None:
        cmap_obj = _IBM_DIVERGING_CMAP.copy()
    elif isinstance(cmap, str):
        cmap_obj = matplotlib.colormaps[cmap].copy()
    else:
        cmap_obj = cmap.copy()
    cmap_obj.set_bad("white")

    matplotlib.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Open the axes fully: the matrix is a lower triangle, so a bounding box
    # (or even a left/bottom pair) leaves stubs running past the blank cells.
    # The staircase and category rules drawn below outline the data instead.
    for spine in ax.spines.values():
        spine.set_visible(False)

    im = ax.imshow(masked, cmap=cmap_obj, vmin=-1.0, vmax=1.0, aspect="equal")

    for i in range(d):
        for j in range(d):
            v = corr[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    # Outline the lower triangle: a staircase along the blanked diagonal edge.
    stair_x: list[float] = []
    stair_y: list[float] = []
    for i in range(1, d):
        stair_x += [i - 1.5, i - 0.5, i - 0.5]
        stair_y += [i - 0.5, i - 0.5, i + 0.5]
    ax.plot(stair_x, stair_y, color="black", linewidth=1.0, zorder=5)

    # Category boundaries, clipped to the lower triangle (they meet the
    # staircase at the diagonal corner (b-0.5, b-0.5)).
    for b in breaks:
        if not 0 < b < d:
            continue
        e = b - 0.5
        ax.plot([e, e], [e, d - 0.5], color="black", linewidth=1.0, zorder=5)
        ax.plot([-0.5, e], [e, e], color="black", linewidth=1.0, zorder=5)

    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))
    ax.set_xticklabels(_PARAM_LATEX, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(_PARAM_LATEX, fontsize=11)
    ax.set_xticks(np.arange(-0.5, d, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, d, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(f"{method.capitalize()} correlation", fontsize=12)

    if save:
        fig.savefig(
            IMAGES_DIR / f"pareto_param_correlations_{pareto_dir.name}.png",
            dpi=plot_style.SAVE_DPI,
            bbox_inches="tight",
        )
    return fig


def _population_g_frequency_series(
    penalties: dict,
    conversion_rate: float,
    prop_gg: float,
    seed: int,
    cycles: int,
    fixed_temperature: int = 20,
) -> np.ndarray:
    """Per-cycle G-allele frequency for one IBM population run.

    Runs the IBM forward for *cycles* life-history cycles starting from a GG-egg
    fraction of *prop_gg*, and returns the adult G-allele frequency
    ``(2·GG + WG) / (2·(WW + WG + GG))`` at each cycle (mirrors
    :func:`~Gene_Drive_Optimisation.gene_drive_objective._g_gene_frequency` but
    for every cycle, not just the last).
    """
    from IBM_Implementations.agent_based_larval_model import run_simulation

    state = run_simulation(
        seed=seed,
        cycles=cycles,
        fixed_temperature=fixed_temperature,
        random_mating=True,
        genotype_penalties=penalties,
        prop_eggs_genetically_modified=prop_gg,
        conversion_rate=conversion_rate,
    )
    sp = state["stage_pop_including_genotypes"]
    freqs = []
    for _, row in sp.iterrows():
        ww = float(row.get("Adults_WW", 0) or 0)
        wg = float(row.get("Adults_WG", 0) or 0)
        gg = float(row.get("Adults_GG", 0) or 0)
        total = ww + wg + gg
        freqs.append((2 * gg + wg) / (2 * total) if total > 0 else np.nan)
    return np.array(freqs)


def plot_emergent_underdominance(
    pareto_dir: Path | str | None = None,
    *,
    use_verified: bool = True,
    fixed_temperature: int = 20,
    threshold_low: float = 0.3,
    threshold_high: float = 0.5,
    prop_low: float = 0.3,
    prop_high: float = 0.5,
    cycles: int = 8,
    n_seeds: int = 20,
    base_seed: int = 101,
    max_steps: int = 2000,
    t_max_hours: float = 300.0,
    save: bool = True,
):
    """Figure: emergent underdominance of the best (0.3, 0.5) gene-drive design.

    Left column (two stacked axes): the reserve ``R`` and mass ``W`` of a single
    larva of each genotype (WW, WG, GG) over time, along the IBM's infinite-food
    inner-growth trajectory, for the top-scoring Pareto design at the
    ``(threshold_low, threshold_high)`` optimisation window (``low0p30_high0p50``).
    Underdominance is visible as the heterozygote (WG) failing to reach the
    pupation threshold ``R_pup`` while both homozygotes do.

    Right column: the population consequence — the adult G-allele frequency over
    generations for two starting GG proportions.  *prop_low* is below the
    invasion threshold -> the drive goes **extinct**; *prop_high* is above it ->
    the drive **fixes**.  Note these are the demonstration release fractions and
    are distinct from ``threshold_low``/``threshold_high``, which only select
    which optimised design is used.  ``prop_high`` defaults to 0.7 rather than the
    0.5 optimisation target because 0.5 sits *on* the (unstable) invasion
    threshold, where the outcome is stochastic and does not cleanly fix.  Each
    start fraction is run for *n_seeds* IBM replicates; the mean is drawn solid
    over faint per-replicate lines.
    """
    import plot_style

    if pareto_dir is None:
        pareto_dir = IBM_OUTPUT_DIR
    pareto_dir = Path(pareto_dir)

    entries = _collect_top_points_per_threshold(
        pareto_dir, top_n=1, use_verified=use_verified
    )
    entry = next(
        (
            e
            for e in entries
            if abs(e["low"] - threshold_low) < 1e-6
            and abs(e["high"] - threshold_high) < 1e-6
        ),
        None,
    )
    if entry is None:
        raise ValueError(
            f"No Pareto results for optimisation thresholds "
            f"(low={threshold_low}, high={threshold_high}) under {pareto_dir}"
        )
    param_vec = entry["params"][0]
    conversion_rate = float(param_vec[0])
    penalties = {
        "WW": {"a_fac": 1.0, "alpha_fac": 1.0, "beta_fac": 1.0, "g_fac": 1.0},
        "WG": {
            "a_fac": float(param_vec[1]),
            "alpha_fac": float(param_vec[2]),
            "beta_fac": float(param_vec[3]),
            "g_fac": float(param_vec[4]),
        },
        "GG": {
            "a_fac": float(param_vec[5]),
            "alpha_fac": float(param_vec[6]),
            "beta_fac": float(param_vec[7]),
            "g_fac": float(param_vec[8]),
        },
    }

    # ---- Left column data: per-genotype infinite-food trajectories -----------
    base = _base_larval_params(fixed_temperature)
    traj = {}
    for genotype in ("WW", "WG", "GG"):
        t, W, R, _ = _infinite_food_trajectory(
            _genotype_factors(param_vec, genotype), base, max_steps=max_steps
        )
        traj[genotype] = (t, R, W)

    # ---- Right column data: population G-frequency trajectories ---------------
    # Each IBM run (especially the fixation case, which grows to carrying
    # capacity) is expensive, so run all (prop, seed) combinations in parallel.
    import multiprocess
    from cpu_resources import available_cpu_count

    jobs = [
        (penalties, conversion_rate, prop, base_seed + i, cycles, fixed_temperature)
        for prop in (prop_low, prop_high)
        for i in range(n_seeds)
    ]
    with multiprocess.Pool(min(len(jobs), available_cpu_count())) as pool:
        results = pool.starmap(_population_g_frequency_series, jobs)

    pop_series = {
        prop: np.array(results[k * n_seeds : (k + 1) * n_seeds])
        for k, prop in enumerate((prop_low, prop_high))
    }

    with plt.rc_context(plot_style.HOUSE_RC):
        fig, axd = plt.subplot_mosaic(
            [["R", "pop"], ["W", "pop"]],
            figsize=(13, 6.5),
            gridspec_kw={"width_ratios": [1.0, 1.1]},
        )
        ax_R, ax_W, ax_pop = axd["R"], axd["W"], axd["pop"]
        # Open the axes: drop the top and right spines on the line panels.
        for _ax in (ax_R, ax_W, ax_pop):
            _ax.spines["top"].set_visible(False)
            _ax.spines["right"].set_visible(False)

        # Left: reserves (top) and mass (bottom, log) for each genotype.
        for genotype in ("WW", "WG", "GG"):
            t, R, W = traj[genotype]
            colour = plot_style.GENOTYPE_COLORS[genotype]
            ax_R.plot(t, R, color=colour, linewidth=2, label=genotype)
            ax_W.plot(t, W, color=colour, linewidth=2, label=genotype)
            # Mark each trajectory's endpoint (pupation / plateau / starvation).
            if t[-1] <= t_max_hours:
                ax_R.plot(t[-1], R[-1], "o", color=colour, markersize=5)

        ax_R.axhline(
            _GROWTH_RPUP,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$R_{\mathrm{pup}}$",
        )
        ax_R.set_xlim(0, t_max_hours)
        ax_R.set_ylim(0, _GROWTH_RPUP * 1.1)
        ax_R.set_ylabel(r"Reserves ($\mu$g) $R$")
        ax_R.set_title(r"Individual growth (best design)")
        ax_R.legend(loc="lower right", ncol=2, framealpha=0.9)
        ax_R.tick_params(labelbottom=False)

        ax_W.set_xlim(0, t_max_hours)
        ax_W.set_yscale("log")
        ax_W.set_xlabel(r"Time (hours)")
        ax_W.set_ylabel(r"Mass ($\mu$g) $W$")

        # Right: G-allele frequency over generations.
        prop_colours = {
            prop_low: plot_style.IBM_PALETTE[3],
            prop_high: plot_style.GENOTYPE_COLORS["WW"],
        }
        prop_labels = {
            prop_low: rf"Start {int(round(prop_low * 100))}\% GG (extinction)",
            prop_high: rf"Start {int(round(prop_high * 100))}\% GG (fixation)",
        }
        generations = np.arange(1, cycles + 1)
        for prop in (prop_low, prop_high):
            runs = pop_series[prop]
            colour = prop_colours[prop]
            for r in runs:
                ax_pop.plot(generations, r, color=colour, alpha=0.25, linewidth=1)
            ax_pop.plot(
                generations,
                np.nanmean(runs, axis=0),
                color=colour,
                linewidth=2.5,
                label=prop_labels[prop],
            )
        ax_pop.axhline(0.5, color="0.5", linestyle=":", linewidth=1.2)
        ax_pop.set_ylim(-0.02, 1.02)
        ax_pop.set_xlim(1, cycles)
        ax_pop.set_xlabel(r"Generation")
        ax_pop.set_ylabel(r"G-allele frequency")
        ax_pop.set_title(r"Population outcome")
        ax_pop.legend(loc="center right", framealpha=0.9)

        fig.tight_layout()
        if save:
            plot_style.save_fig(fig, "emergent_underdominance")

    return fig
