"""House style for all publication / conference figures.

This module is the single importable source of truth for the plotting
conventions that are otherwise copy-pasted inline across the ``plot_*.py``
modules (usetex Computer Modern Roman, the IBM colour palettes, and the
dpi / bbox saving rules).  New figure code should ``import plot_style`` and use
``HOUSE_RC`` (via ``plt.rc_context``), the palettes, and :func:`save_fig`.

See ``.claude/skills/plot/SKILL.md`` for the documented conventions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure

from paths import IMAGES_DIR

#: Resolution every saved figure is written at.  A single value across the whole
#: project so that figures dropped side by side into the paper or a talk match.
SAVE_DPI: int = 600

#: Spine rules for the house style: "open" axes with no top/right bounding box.
#: Kept separate from :data:`HOUSE_RC` so that modules which cannot adopt the
#: full house style (because it would reflow their font sizes) can still opt
#: into the spine convention on its own.
OPEN_SPINES_RC: dict = {
    "axes.spines.top": False,
    "axes.spines.right": False,
}

#: Canonical rcParams block (matches ``plot_spatial_simulations.py``).  Apply
#: with ``with plt.rc_context(HOUSE_RC): ...`` around figure construction.
HOUSE_RC: dict = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "savefig.dpi": SAVE_DPI,
    **OPEN_SPINES_RC,
}

#: Genotype palette (WW blue, WG amber, GG magenta) shared with the spatial plots.
GENOTYPE_COLORS: dict = {
    "WW": "#0C7BDC",
    "WG": "#FFC20A",
    "GG": "#DC267F",
}

#: Per-genotype colormaps: white (zero abundance) -> genotype colour.
GENOTYPE_CMAPS: dict = {
    key: LinearSegmentedColormap.from_list(key, ["white", color])
    for key, color in GENOTYPE_COLORS.items()
}

#: IBM Design Language qualitative palette (from ``plot_gene_drives.py``).
IBM_PALETTE: list[str] = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

#: House sequential colormap (white -> deck blue) for neutral density /
#: probability heatmaps that are not tied to a single genotype.  Use in place of
#: ``viridis`` so density figures stay on-palette.  Shares the WW/deck blue so
#: the whole conference set reads as one colour scheme.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "ibm_sequential", ["white", GENOTYPE_COLORS["WW"]]
)

#: Anchors of the ordered ramp: pale -> near-black house blue.  Seven anchors so
#: that the seven release thresholds land on them exactly.
_ORDERED_ANCHORS: list[str] = [
    "#D6E4FA",
    "#A8C6F2",
    "#7AA6E6",
    "#4F84D6",
    "#2F62B4",
    "#1B4184",
    "#0C2350",
]

#: Monotone-luminance ramp for *ordered* categorical series (release thresholds,
#: optimisation replicates).  Unlike :data:`IBM_PALETTE`, whose hues are chosen
#: for colour separation and are near-identical in luminance, this ramp is
#: greyscale-safe: rank maps directly onto darkness, so the ordering survives a
#: black-and-white print.
ORDERED_CMAP = LinearSegmentedColormap.from_list("ibm_ordered", _ORDERED_ANCHORS)

#: Redundant (non-colour) encodings for the two-series figures.  IBM blue and
#: IBM amber sit only ~14% apart in luminance, so on their own they collapse
#: into one grey; the hatch / dash pattern carries the distinction in print.
TWO_SERIES_HATCH: tuple = (None, "///")
TWO_SERIES_LINESTYLE: tuple = ("-", (0, (5, 2)))

#: Dash cycle to lay over :func:`ordered_colors` when an ordered series has many
#: members *and* its lines overlap.  The ramp alone separates a handful of
#: series, but by seven the dark end crowds together on thin overlapping lines;
#: the dash pattern breaks the tie while darkness still carries the rank.
ORDERED_LINESTYLES: tuple = (
    "-",
    (0, (5, 1.5)),
    (0, (1, 1.5)),
    (0, (6, 1.5, 1, 1.5)),
    (0, (3, 1.5)),
    (0, (6, 1.5, 1, 1.5, 1, 1.5)),
    (0, (1, 1, 4, 1)),
)


def ordered_linestyles(n: int) -> list:
    """Return *n* dash patterns cycled from :data:`ORDERED_LINESTYLES`."""
    return [ORDERED_LINESTYLES[k % len(ORDERED_LINESTYLES)] for k in range(n)]


def ordered_colors(n: int, *, lo: float = 0.0, hi: float = 1.0) -> list[str]:
    """Return *n* hex colours sampled from :data:`ORDERED_CMAP`.

    The samples are monotone in luminance, so the *k*-th colour is always darker
    than the (*k*-1)-th and an ordered series stays readable in greyscale.
    ``ordered_colors(7)`` reproduces :data:`_ORDERED_ANCHORS` exactly.

    *lo* raises the pale end of the range.  Bars are large filled shapes and can
    use the palest anchor (``lo=0.0``); thin lines and small markers cannot, and
    should pass ``lo=0.22`` or so to keep the first series visible against white.
    """
    if n <= 0:
        return []
    if n == 1:
        return [to_hex(ORDERED_CMAP(hi))]
    return [to_hex(ORDERED_CMAP(t)) for t in np.linspace(lo, hi, n)]


#: Larval fate palette for the larvae-behaviour figure.
FATE_COLORS: dict = {
    "pupated": GENOTYPE_COLORS["WW"],  # blue (shared with the WW genotype blue)
    "timed_out": IBM_PALETTE[4],  # gold
    "starved": IBM_PALETTE[2],  # magenta
}

#: Human-readable fate labels (LaTeX-safe) for legends.
FATE_LABELS: dict = {
    "pupated": "Pupated",
    "timed_out": "Timed out",
    "starved": "Starved",
}

#: Subdirectory that all conference-slide figures are written to.
CONFERENCE_DIR = IMAGES_DIR / "conference_slides"


def latex_escape(text: str) -> str:
    """Make *text* safe to render under ``text.usetex``.

    Labels built from filenames, Pareto tags and parameter names routinely
    contain underscores and percent signs that TeX would treat as markup (an
    unescaped ``_`` is a hard TeX error, not a cosmetic one).  Since every
    house-style figure renders through real LaTeX, this belongs with the rest
    of the house style rather than in whichever figure module needed it first.
    """
    for character in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(character, "\\" + character)
    return text


def save_fig(fig: Figure, name: str, *, dpi: int = SAVE_DPI) -> None:
    """Save *fig* into the conference-slides image directory as a tight PNG.

    Creates ``CONFERENCE_DIR`` if needed and follows the house convention of
    ``bbox_inches="tight"``.  *name* may be given with or without a ``.png``
    suffix.
    """
    CONFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".png"):
        name = f"{name}.png"
    fig.savefig(CONFERENCE_DIR / name, dpi=dpi, bbox_inches="tight")
    print(f"Saved {CONFERENCE_DIR / name}")
