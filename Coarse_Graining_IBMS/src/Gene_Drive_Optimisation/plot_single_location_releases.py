"""Figures for the single-location gene-drive release experiments.

Every figure here is a view of the one tidy table
:func:`Gene_Drive_Optimisation.single_location_releases.simulate_single_location_releases`
produces, so they can all be rebuilt from a saved CSV without touching the IBM.

Two are drawn by default:

- :func:`plot_release_trajectories` -- a row per release frequency showing what
  happened (abundance, genotype frequency) and why (per-genotype egg-to-adult
  survival).
- :func:`plot_release_diagnostics` -- where the drive's advantage comes from
  and how the replicates actually ended.

A third, :func:`plot_threshold_diagnostics`, is drawn on request and is the one
to reach for when the question is *where the invasion threshold sits* rather
than what happened at a particular release frequency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_style
from Gene_Drive_Optimisation.single_location_releases import (
    PHASE_POST_RELEASE,
    POND_CARRYING_CAPACITY,
    add_derived_columns,
    classify_outcomes,
)
from IBM_Implementations.input_output_pairs import GENOTYPES

#: Spatial and thesis multi-panel figures are saved at this resolution.
_DPI = 400

#: Colours for the four end states in the outcome summary.  Loss and fixation
#: take the wild-type and drive genotype colours, so the bar reads in the same
#: palette as the trajectories above it; extinction is near-black because it is
#: an outcome for the population rather than for an allele.
_OUTCOME_COLORS: dict[str, str] = {
    "lost": plot_style.GENOTYPE_COLORS["WW"],
    "polymorphic": plot_style.GENOTYPE_COLORS["WG"],
    "fixed": plot_style.GENOTYPE_COLORS["GG"],
    "extinct": "0.25",
}

#: Order the outcome bars stack in: from "the drive did nothing" to "the
#: population is gone", so a bar can be read left to right as severity.
_OUTCOME_ORDER: tuple[str, ...] = ("lost", "polymorphic", "fixed", "extinct")

#: Larval fates, in the order they stack: the count column recording the fate
#: and its key in the house fate palette.  Shares are recomputed from the counts
#: rather than averaged, so a generation with three larvae cannot outvote one
#: with three hundred.
_FATE_COUNTS: tuple[tuple[str, str], ...] = (
    ("pupae", "pupated"),
    ("starved", "starved"),
    ("timed_out", "timed_out"),
)


def _frequency_label(frequency: float) -> str:
    """TeX label for a release frequency."""
    return rf"$f = {frequency:g}$"


def _mark_release(ax, generations) -> None:
    """Mark the release on a generation axis.

    Generation 0 is the moment of release, and everything left of it is the
    wild-type burn-in.  Shading that region is what stops a reader mistaking
    the pre-release equilibrium for part of the drive's dynamics.

    The extent comes from *generations* rather than from the axis limits:
    ``axvspan`` widens the limits to contain itself, and on a shared axis the
    next panel would read those widened limits back and shade wider still.
    """
    first, last = int(np.min(generations)), int(np.max(generations))
    if first < 0:
        ax.axvspan(first, 0, color="0.92", zorder=0, linewidth=0)
    ax.axvline(0, color="0.4", linestyle=(0, (4, 2)), linewidth=1, zorder=1)
    ax.set_xlim(first, last)


def _plot_by_genotype(
    ax,
    dataframe: pd.DataFrame,
    column: str,
    *,
    show_replicates: bool = True,
) -> None:
    """Draw *column* against generation, one line per genotype.

    Replicate means are drawn solid over semi-transparent individual replicates,
    so the spread that a mean hides -- which is most of the story for a drive
    near its threshold -- stays visible.
    """
    for genotype in GENOTYPES:
        genotype_rows = dataframe[dataframe["genotype"] == genotype]
        if genotype_rows.empty:
            continue
        color = plot_style.GENOTYPE_COLORS[genotype]

        if show_replicates:
            for _, replicate_rows in genotype_rows.groupby("replicate"):
                replicate_rows = replicate_rows.sort_values("generation")
                ax.plot(
                    replicate_rows["generation"],
                    replicate_rows[column],
                    color=color,
                    alpha=0.25,
                    linewidth=0.8,
                    zorder=2,
                )

        mean_by_generation = (
            genotype_rows.groupby("generation")[column].mean().sort_index()
        )
        ax.plot(
            mean_by_generation.index,
            mean_by_generation.values,
            color=color,
            linewidth=2,
            zorder=3,
            label=genotype,
        )


def plot_release_trajectories(
    dataframe: pd.DataFrame,
    save_path: Path,
    *,
    dpi: int = _DPI,
) -> Path:
    """Abundance, genotype frequency and survival, a row per release frequency.

    The three columns answer the three questions in order: how many of each
    genotype there were, what share of the population they made up, and what
    fraction of the eggs each genotype laid reached adulthood.

    The survival column is the diagnostic one.  Egg-to-larva and pupa-to-adult
    survival are deterministic and identical across genotypes in this model, so
    *every* genotype difference in egg-to-adult survival is imposed during
    larval development -- which is exactly where the drive's fitness penalties
    act.

    **Arguments**
    1. dataframe: pd.DataFrame
        - tidy output of ``simulate_single_location_releases``
    2. save_path: Path
    3. dpi: int = 400

    **Returns**
    - the Path the figure was written to
    """
    derived = add_derived_columns(dataframe)
    frequencies = sorted(derived["release_frequency"].unique())

    columns = [
        ("adults", "Adult abundance"),
        ("genotype_frequency", "Genotype frequency"),
        ("egg_to_adult_survival", "Egg-to-adult survival"),
    ]

    with plt.rc_context(plot_style.HOUSE_RC):
        fig, axes = plt.subplots(
            len(frequencies),
            len(columns),
            figsize=(4.2 * len(columns), 2.9 * len(frequencies)),
            squeeze=False,
            sharex=True,
        )

        for row, frequency in enumerate(frequencies):
            frequency_rows = derived[derived["release_frequency"] == frequency]
            for column_index, (column, label) in enumerate(columns):
                ax = axes[row][column_index]
                # Survival is undefined for the assembled states (no eggs were
                # laid to reach them), so those rows would break the line.
                plotted = (
                    frequency_rows.dropna(subset=[column])
                    if column == "egg_to_adult_survival"
                    else frequency_rows
                )
                _plot_by_genotype(ax, plotted, column)
                _mark_release(ax, derived["generation"])

                if row == 0:
                    ax.set_title(label)
                if row == len(frequencies) - 1:
                    ax.set_xlabel("Generation")
                if column_index == 0:
                    ax.set_ylabel(_frequency_label(frequency))
                if column == "genotype_frequency":
                    ax.set_ylim(-0.02, 1.02)
                else:
                    # Survival and abundance are left to autoscale: pinning
                    # survival to [0, 1] would flatten the range it actually
                    # occupies, which is where the genotypes differ.
                    ax.set_ylim(bottom=0)

        handles = [
            Line2D([], [], color=plot_style.GENOTYPE_COLORS[genotype], linewidth=2, label=genotype)
            for genotype in GENOTYPES
        ]
        handles.append(
            Patch(facecolor="0.92", edgecolor="none", label="Wild-type burn-in")
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return save_path


def plot_release_diagnostics(
    dataframe: pd.DataFrame,
    save_path: Path,
    *,
    dpi: int = _DPI,
) -> Path:
    """Four panels on where the drive's effect came from and how runs ended.

    1. **G allele frequency**, one line per replicate, all release frequencies
       overlaid.  This is the quantity the optimisation objectives score, and
       drawing replicates individually is what makes a threshold visible: near
       it, runs launched from the same frequency separate instead of tracking
       a common mean.
    2. **Larval fate composition** by genotype.  Splits the larval bottleneck
       into starvation and running out of development time -- two different
       ways a fitness penalty can act, indistinguishable in an adult count.
    3. **Total abundance** relative to the pre-release equilibrium: the
       suppression the release bought, and whether the population recovered.
    4. **Outcomes** across replicates, which is what a mean allele frequency
       cannot tell you when runs end up at opposite extremes.

    **Arguments**
    1. dataframe: pd.DataFrame
    2. save_path: Path
    3. dpi: int = 400

    **Returns**
    - the Path the figure was written to
    """
    derived = add_derived_columns(dataframe)
    frequencies = sorted(derived["release_frequency"].unique())
    colors = plot_style.ordered_colors(len(frequencies), lo=0.22)
    linestyles = plot_style.ordered_linestyles(len(frequencies))

    with plt.rc_context(plot_style.HOUSE_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

        # ── 1. G allele frequency, per replicate ─────────────────────────────
        ax = axes[0][0]
        for index, frequency in enumerate(frequencies):
            frequency_rows = derived[
                (derived["release_frequency"] == frequency)
                & (derived["genotype"] == "WW")  # one row per generation
                & (derived["generation"] >= 0)
            ]
            for _, replicate_rows in frequency_rows.groupby("replicate"):
                replicate_rows = replicate_rows.sort_values("generation")
                ax.plot(
                    replicate_rows["generation"],
                    replicate_rows["allele_frequency"],
                    color=colors[index],
                    linestyle=linestyles[index],
                    linewidth=1.2,
                    alpha=0.85,
                )
        ax.set_xlabel("Generation since release")
        ax.set_ylabel("G allele frequency")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Allele frequency by replicate")
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=colors[index],
                    linestyle=linestyles[index],
                    linewidth=1.5,
                    label=_frequency_label(frequency),
                )
                for index, frequency in enumerate(frequencies)
            ],
            frameon=False,
            fontsize=10,
        )

        # ── 2. Larval fate composition by genotype ───────────────────────────
        ax = axes[0][1]
        post_release = derived[derived["phase"] == PHASE_POST_RELEASE]
        bar_width = 0.8 / max(len(GENOTYPES), 1)
        positions = np.arange(len(frequencies), dtype=float)
        for genotype_index, genotype in enumerate(GENOTYPES):
            offset = (genotype_index - (len(GENOTYPES) - 1) / 2) * bar_width
            bottoms = np.zeros(len(frequencies))
            for column, fate in _FATE_COUNTS:
                shares = []
                for frequency in frequencies:
                    rows = post_release[
                        (post_release["release_frequency"] == frequency)
                        & (post_release["genotype"] == genotype)
                    ]
                    larvae = rows["larvae"].sum()
                    shares.append(rows[column].sum() / larvae if larvae > 0 else 0.0)
                shares = np.array(shares)
                ax.bar(
                    positions + offset,
                    shares,
                    bar_width * 0.9,
                    bottom=bottoms,
                    color=plot_style.FATE_COLORS[fate],
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms += shares
        # Genotypes are named under their own bars rather than in the title:
        # the bars are filled with the fate palette, so nothing else in the
        # panel says which of the three a bar belongs to.  The release
        # frequencies go below them as free text, because a minor tick label
        # sharing a location with a major tick is silently dropped.
        genotype_positions = [
            position + (index - (len(GENOTYPES) - 1) / 2) * bar_width
            for position in positions
            for index in range(len(GENOTYPES))
        ]
        ax.set_xticks(genotype_positions)
        ax.set_xticklabels(list(GENOTYPES) * len(frequencies), fontsize=9)
        ax.tick_params(axis="x", length=0)
        for position, frequency in zip(positions, frequencies):
            ax.text(
                position,
                -0.13,
                _frequency_label(frequency),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
            )
        ax.set_ylabel("Share of larvae")
        ax.set_title("Larval fate by genotype")
        ax.set_ylim(0, 1)
        ax.legend(
            handles=[
                Patch(
                    facecolor=plot_style.FATE_COLORS[fate],
                    label=plot_style.FATE_LABELS[fate],
                )
                for _, fate in _FATE_COUNTS
            ],
            frameon=False,
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )

        # ── 3. Total abundance relative to the pre-release equilibrium ───────
        ax = axes[1][0]
        # Generation -1 is the last wild-type generation: the equilibrium the
        # release landed in, and the only baseline the suppression is against.
        baselines = (
            derived[derived["generation"] == -1]
            .groupby("replicate")["total_adults"]
            .first()
        )
        for index, frequency in enumerate(frequencies):
            frequency_rows = derived[
                (derived["release_frequency"] == frequency)
                & (derived["genotype"] == "WW")
                & (derived["generation"] >= 0)
            ]
            relative = []
            for _, replicate_rows in frequency_rows.groupby("replicate"):
                replicate_rows = replicate_rows.sort_values("generation")
                baseline = baselines.get(replicate_rows["replicate"].iloc[0], np.nan)
                if not np.isfinite(baseline) or baseline <= 0:
                    continue
                relative.append(
                    pd.Series(
                        (replicate_rows["total_adults"] / baseline).values,
                        index=replicate_rows["generation"].values,
                    )
                )
            if not relative:
                continue
            stacked = pd.concat(relative, axis=1)
            for column in stacked.columns:
                ax.plot(
                    stacked.index,
                    stacked[column],
                    color=colors[index],
                    alpha=0.25,
                    linewidth=0.8,
                )
            ax.plot(
                stacked.index,
                stacked.mean(axis=1),
                color=colors[index],
                linestyle=linestyles[index],
                linewidth=2,
                label=_frequency_label(frequency),
            )
        ax.axhline(1.0, color="0.4", linestyle=(0, (4, 2)), linewidth=1)
        ax.set_xlabel("Generation since release")
        ax.set_ylabel("Abundance / pre-release")
        ax.set_title("Population suppression")
        ax.legend(frameon=False, fontsize=10)

        # ── 4. Outcome classification ────────────────────────────────────────
        ax = axes[1][1]
        outcomes = classify_outcomes(dataframe)
        bottoms = np.zeros(len(frequencies))
        for outcome in _OUTCOME_ORDER:
            shares = []
            for frequency in frequencies:
                rows = outcomes[outcomes["release_frequency"] == frequency]
                shares.append(
                    (rows["outcome"] == outcome).mean() if len(rows) else 0.0
                )
            shares = np.array(shares)
            ax.bar(
                positions,
                shares,
                0.6,
                bottom=bottoms,
                color=_OUTCOME_COLORS[outcome],
                edgecolor="white",
                linewidth=0.5,
                label=outcome.capitalize(),
            )
            bottoms += shares
        ax.set_xticks(positions)
        ax.set_xticklabels([_frequency_label(f) for f in frequencies])
        ax.set_ylabel("Share of replicates")
        ax.set_ylim(0, 1)
        ax.set_title("Outcome at the final generation")
        ax.legend(
            frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5)
        )

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return save_path


def plot_threshold_diagnostics(
    dataframe: pd.DataFrame,
    save_path: Path,
    *,
    pond_carrying_capacity: int = POND_CARRYING_CAPACITY,
    dpi: int = _DPI,
) -> Path:
    """Three panels aimed squarely at the invasion threshold.

    1. **Next generation's allele frequency against this one's**, pooled over
       every replicate and release frequency, with the identity line.  Where the
       cloud crosses that line the allele is neutral: below it the drive is
       being lost, above it it is invading, and for an underdominant drive the
       crossing *is* the threshold frequency.  Reading it off this panel needs
       no assumption about which release frequencies happened to be simulated.
    2. **Realised growth rate by genotype**, the factor each genotype's adult
       count changed by from one generation to the next.
    3. **Eggs laid against the pond's egg carrying capacity**.  Density
       dependence in this model acts through larval competition *and* through
       that cap; when the egg pool is at the cap, the cap is what is regulating
       the population, and a comparison of drives at that point is partly a
       comparison of how they interact with the ceiling.

    **Arguments**
    1. dataframe: pd.DataFrame
    2. save_path: Path
    3. pond_carrying_capacity: int = POND_CARRYING_CAPACITY
    4. dpi: int = 400

    **Returns**
    - the Path the figure was written to
    """
    derived = add_derived_columns(dataframe)
    frequencies = sorted(derived["release_frequency"].unique())
    colors = plot_style.ordered_colors(len(frequencies), lo=0.22)

    with plt.rc_context(plot_style.HOUSE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

        # ── 1. p_{t+1} against p_t ───────────────────────────────────────────
        ax = axes[0]
        for index, frequency in enumerate(frequencies):
            current, following = [], []
            frequency_rows = derived[
                (derived["release_frequency"] == frequency)
                & (derived["genotype"] == "WW")
                & (derived["generation"] >= 0)
            ]
            for _, replicate_rows in frequency_rows.groupby("replicate"):
                series = (
                    replicate_rows.sort_values("generation")["allele_frequency"]
                    .to_numpy(dtype=float)
                )
                current.extend(series[:-1])
                following.extend(series[1:])
            ax.scatter(
                current,
                following,
                s=22,
                color=colors[index],
                alpha=0.8,
                edgecolor="none",
                label=_frequency_label(frequency),
            )
        ax.plot([0, 1], [0, 1], color="0.4", linestyle=(0, (4, 2)), linewidth=1)
        ax.set_xlabel(r"$p_t$")
        ax.set_ylabel(r"$p_{t+1}$")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("G allele frequency map")
        ax.legend(frameon=False, fontsize=10, loc="upper left")

        # ── 2. Realised growth rate by genotype ──────────────────────────────
        ax = axes[1]
        for genotype in GENOTYPES:
            growth_by_generation: dict[int, list[float]] = {}
            genotype_rows = derived[
                (derived["genotype"] == genotype) & (derived["generation"] >= 0)
            ]
            for _, replicate_rows in genotype_rows.groupby(
                ["replicate", "release_frequency"]
            ):
                replicate_rows = replicate_rows.sort_values("generation")
                counts = replicate_rows["adults"].to_numpy(dtype=float)
                generations = replicate_rows["generation"].to_numpy()
                for step in range(len(counts) - 1):
                    if counts[step] > 0:
                        growth_by_generation.setdefault(
                            int(generations[step + 1]), []
                        ).append(counts[step + 1] / counts[step])
            if not growth_by_generation:
                continue
            generations = sorted(growth_by_generation)
            means = [float(np.mean(growth_by_generation[g])) for g in generations]
            ax.plot(
                generations,
                means,
                color=plot_style.GENOTYPE_COLORS[genotype],
                linewidth=2,
                marker="o",
                markersize=4,
                label=genotype,
            )
        ax.axhline(1.0, color="0.4", linestyle=(0, (4, 2)), linewidth=1)
        ax.set_xlabel("Generation since release")
        ax.set_ylabel(r"Adults$_{t+1}$ / Adults$_t$")
        ax.set_title("Realised growth rate")
        ax.legend(frameon=False, fontsize=10)

        # ── 3. Egg pool against the pond cap ─────────────────────────────────
        ax = axes[2]
        eggs = (
            derived[derived["generation"] >= 1]
            .groupby(["release_frequency", "replicate", "generation"])["eggs"]
            .sum()
            .reset_index()
        )
        for index, frequency in enumerate(frequencies):
            frequency_eggs = eggs[eggs["release_frequency"] == frequency]
            means = frequency_eggs.groupby("generation")["eggs"].mean().sort_index()
            ax.plot(
                means.index,
                means.values,
                color=colors[index],
                linewidth=2,
                marker="o",
                markersize=4,
                label=_frequency_label(frequency),
            )
        ax.axhline(
            pond_carrying_capacity,
            color="0.4",
            linestyle=(0, (4, 2)),
            linewidth=1,
        )
        ax.annotate(
            "Pond egg capacity",
            xy=(0.02, pond_carrying_capacity),
            xycoords=("axes fraction", "data"),
            va="bottom",
            fontsize=10,
            color="0.4",
        )
        ax.set_xlabel("Generation since release")
        ax.set_ylabel("Eggs laid")
        ax.set_title("Density dependence")
        ax.legend(frameon=False, fontsize=10)

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return save_path
