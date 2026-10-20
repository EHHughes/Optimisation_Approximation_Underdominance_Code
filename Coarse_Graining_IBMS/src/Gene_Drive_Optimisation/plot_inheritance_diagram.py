"""Figure: one-locus underdominance inheritance diagram.

A schematic figure of the three *single-outcome* crosses between homozygous
parents, illustrating the genetics of a single-locus underdominance / CI-style
gene drive:

* ``WW x WW`` -> all ``WW`` (wild-type, viable);
* ``WW x GG`` -> all ``WG`` (heterozygotes, inviable);
* ``GG x GG`` -> all ``GG`` (drive homozygote, viable).

Each cross has a single possible offspring genotype.  Offspring are coloured by
genotype (the house genotype palette) and carry a viability badge.  Because
heterozygotes arise only from mixing the two homozygous types and are inviable,
the two homozygous types are mutually incompatible — the defining feature of
underdominance — creating a bistable release-frequency threshold above which the
drive fixes and below which it is lost.

This is a pure schematic (no simulation data).  Run via the thin wrapper
``src/conference_figures.py --figure 6``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import plot_style

#: Per-allele colours: wild-type allele blue, drive allele magenta (reusing the
#: genotype-palette hues so WW reads blue, GG magenta, WG a mix).
ALLELE_COLORS: dict = {
    "W": plot_style.GENOTYPE_COLORS["WW"],
    "G": plot_style.GENOTYPE_COLORS["GG"],
}

#: Only the homozygotes are viable; the heterozygote is inviable (underdominance).
GENOTYPE_VIABLE: dict = {"WW": True, "WG": False, "GG": True}

_VIABLE_COLOR = "#2ca02c"  # green
_INVIABLE_COLOR = "#d62728"  # red

#: The three single-outcome crosses: (parent 1, parent 2, offspring).
CROSSES: list[tuple[str, str, str]] = [
    ("WW", "WW", "WW"),
    ("WW", "GG", "WG"),
    ("GG", "GG", "GG"),
]


def _allele_chip(ax, x: float, y: float, allele: str, r: float = 0.28) -> None:
    """Draw a filled circular chip carrying a single allele letter."""
    ax.add_patch(
        Circle(
            (x, y),
            r,
            facecolor=ALLELE_COLORS[allele],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
    )
    ax.text(
        x,
        y,
        rf"$\mathrm{{{allele}}}$",
        ha="center",
        va="center_baseline",
        color="white",
        fontsize=19,
        zorder=4,
    )


def _viability_badge(ax, cx: float, cy: float, viable: bool) -> None:
    """Draw a coloured 'viable' / 'non-viable' pill badge."""
    color = _VIABLE_COLOR if viable else _INVIABLE_COLOR
    label = r"viable" if viable else r"non-viable"
    width = 1.1 if viable else 1.55
    ax.add_patch(
        FancyBboxPatch(
            (cx - width / 2, cy - 0.19),
            width,
            0.38,
            boxstyle="round,pad=0.02,rounding_size=0.19",
            facecolor=color,
            edgecolor="none",
            zorder=5,
        )
    )
    ax.text(
        cx, cy, label, ha="center", va="center", color="white", fontsize=14, zorder=6
    )


def _parent_card(ax, cx: float, cy: float, genotype: str) -> None:
    """Draw a homozygous parent card: genotype-tinted box, two allele chips and a
    genotype label."""
    color = plot_style.GENOTYPE_COLORS[genotype]
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.85, cy - 0.45),
            1.7,
            0.9,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            alpha=0.16,
            edgecolor=color,
            linewidth=1.4,
            zorder=1,
        )
    )
    _allele_chip(ax, cx - 0.4, cy, genotype[0])
    _allele_chip(ax, cx + 0.4, cy, genotype[1])
    ax.text(cx, cy + 0.74, rf"{genotype}", ha="center", va="center", fontsize=16)


def _offspring_card(ax, cx: float, cy: float, genotype: str) -> None:
    """Draw the offspring card: genotype-tinted box, allele chips, genotype label
    and a viability badge.  Inviable offspring are struck through in red."""
    color = plot_style.GENOTYPE_COLORS[genotype]
    viable = GENOTYPE_VIABLE[genotype]
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.95, cy - 0.85),
            1.9,
            1.7,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color,
            alpha=0.16,
            edgecolor=color,
            linewidth=1.6,
            zorder=1,
        )
    )
    _allele_chip(ax, cx - 0.4, cy + 0.45, genotype[0], r=0.26)
    _allele_chip(ax, cx + 0.4, cy + 0.45, genotype[1], r=0.26)
    ax.text(cx, cy - 0.05, rf"{genotype}", ha="center", va="center", fontsize=19)

    # Strike inviable offspring through with a translucent red cross (behind the
    # chips, over the tint) so the crossed-out genotype stays legible.
    if not viable:
        for (xa, ya), (xb, yb) in (
            ((cx - 0.8, cy + 0.72), (cx + 0.8, cy - 0.72)),
            ((cx - 0.8, cy - 0.72), (cx + 0.8, cy + 0.72)),
        ):
            ax.plot(
                [xa, xb],
                [ya, yb],
                color=_INVIABLE_COLOR,
                linewidth=3.0,
                alpha=0.45,
                solid_capstyle="round",
                zorder=2,
            )

    _viability_badge(ax, cx, cy - 0.58, viable)


def plot_inheritance_diagram(save: bool = True):
    """Build and (optionally) save the single-outcome underdominance crosses."""
    p1_x, p2_x, off_x = 2.0, 4.8, 8.5
    cross_x = 3.4
    row_y = [6.6, 4.2, 1.8]

    with plt.rc_context(plot_style.HOUSE_RC):
        fig, ax = plt.subplots(figsize=(11.5, 9.0))
        ax.set_xlim(0, 12)
        ax.set_ylim(0.6, 9.0)
        ax.axis("off")

        # ---- Allele legend -------------------------------------------------
        _allele_chip(ax, 3.6, 8.5, "W", r=0.26)
        ax.text(3.95, 8.5, r"wild-type allele", ha="left", va="center", fontsize=14)
        _allele_chip(ax, 6.9, 8.5, "G", r=0.26)
        ax.text(7.25, 8.5, r"drive allele", ha="left", va="center", fontsize=14)

        # ---- Column headers ------------------------------------------------
        ax.text(cross_x, 7.8, r"Parents", ha="center", va="center", fontsize=18)
        ax.text(off_x, 7.8, r"Offspring", ha="center", va="center", fontsize=18)

        # ---- Crosses (one per row) ----------------------------------------
        for (p1, p2, offspring), y in zip(CROSSES, row_y):
            _parent_card(ax, p1_x, y, p1)
            ax.text(cross_x, y, r"$\times$", ha="center", va="center", fontsize=26)
            _parent_card(ax, p2_x, y, p2)
            ax.add_patch(
                FancyArrowPatch(
                    (5.85, y),
                    (7.45, y),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    color="0.35",
                    linewidth=1.6,
                    zorder=2,
                )
            )
            _offspring_card(ax, off_x, y, offspring)

        if save:
            # Pure line art (no rasterised data), so it is worth carrying at
            # twice the project-wide raster resolution: the thin rules and
            # arrowheads alias badly at 600.
            plot_style.save_fig(fig, "underdominance_inheritance", dpi=1200)

    return fig
