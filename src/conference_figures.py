"""Generate the conference-slide figures.

Thin entry-point wrapper: the figure logic lives in the ``plot_*`` modules of
each sub-package; this script just selects and runs them.  All figures are
written to ``data/images/conference_slides/``.

Run from ``src/``::

    python conference_figures.py --figure all      # every figure
    python conference_figures.py --figure 1        # just the larvae-behaviour figure

Figures
-------
1. Larvae behaviour   — reserve dynamics by fate + spatial density heatmap (IBM).
2. Density dependence — emergent adults-in vs adults-out relation.
3. Underdominance     — per-genotype growth + population fixation/extinction.
   (Figures 4 and 5 — the 2D MAF release and the dispersal kernel — were
   spatial slides and are not part of this repository; the numbering of the
   remaining figures is left unchanged.)
6. Inheritance        — WG x WG cross Punnett square (one-locus underdominance).
7. Multigen NN        — IBM vs feedforward-NN adult trajectories over generations.
8. Multigen MAF       — IBM vs MAF adult trajectories over generations.
9. Adult-larval AvL   — density-normalised adult-vs-larval scatter (Ziqian's IBM,
                        20 C and the two modest-food regimes).
"""

import argparse

import numpy as np

#: Seed identifying the trained surrogates on disk (shared by figures 7 and 8).
TRAINED_MODEL_SEED = 1180468651


def _figure_1():
    from IBM_Implementations.plot_larvae_behaviour import plot_larvae_behaviour

    plot_larvae_behaviour()


def _figure_2():
    from IBM_Emulation.plot_surrogate_model_comparisons import (
        plot_emergent_density_dependence,
    )

    plot_emergent_density_dependence()


def _figure_3():
    from Gene_Drive_Optimisation.plot_gene_drives import plot_emergent_underdominance

    plot_emergent_underdominance()


def _figure_6():
    from Gene_Drive_Optimisation.plot_inheritance_diagram import (
        plot_inheritance_diagram,
    )

    plot_inheritance_diagram()


def _figure_7(n_replicates: int = 50):
    from IBM_Emulation.plot_surrogate_model_comparisons import (
        compare_multigeneration_ibm_to_nn,
    )

    seed = np.random.randint(0, 2**32 - 1)
    compare_multigeneration_ibm_to_nn(
        model_type="nn",
        n_replicates=n_replicates,
        model_seed=TRAINED_MODEL_SEED,
        for_conference=True,
        seed=seed,
    )


def _figure_8(n_replicates: int = 50):
    from IBM_Emulation.plot_surrogate_model_comparisons import (
        compare_multigeneration_ibm_to_nn,
    )

    seed = np.random.randint(0, 2**32 - 1)
    compare_multigeneration_ibm_to_nn(
        model_type="maf",
        n_replicates=n_replicates,
        model_seed=TRAINED_MODEL_SEED,
        for_conference=True,
        seed=seed,
    )


def _figure_9(competition: str = "Interference"):
    from IBM_Implementations.plot_ziqian_density_relationships import (
        plot_ziqian_density_relationships,
    )

    # Simplified to the top-left quadrant: 20 C only, and the two modest-food
    # regimes, keeping both larval densities.
    plot_ziqian_density_relationships(
        competition=competition,
        temperatures=["T20"],
        foods=["F21", "F22"],
    )


_FIGURES = {
    "1": _figure_1,
    "2": _figure_2,
    "3": _figure_3,
    "6": _figure_6,
    "7": _figure_7,
    "8": _figure_8,
    "9": _figure_9,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=[*_FIGURES.keys(), "all"],
        default="all",
        help="which conference figure to generate (default: all)",
    )
    args = parser.parse_args()

    selected = _FIGURES.keys() if args.figure == "all" else [args.figure]
    for key in selected:
        print(f"=== Generating figure {key} ===")
        _FIGURES[key]()
