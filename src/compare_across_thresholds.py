"""Compare the optimal gene-drive designs across release-threshold windows.

Run from ``src/``::

    python compare_across_thresholds.py

Scans the parent ``ibm_output`` directory (which holds one
``low<lo>_high<hi>/`` subdirectory of Pareto-frontier replicates per release
threshold) and builds two cross-threshold figures from each threshold's top
points (pooled across that threshold's replicates and ranked by mean verified
objective):

1. **Parameters** – a grouped bar chart of the 9 parameters we are optimising, with
   one bar per threshold, showing the mean of the top points with those points
   overlaid as markers in the bar's own colour
   (``plot_best_parameters_across_thresholds``).
2. **Reserve/mass trajectories** – the infinite-food reserve ``R`` and larval
   mass ``W`` against time ``t`` for every top point (WG and GG genotypes, WW
   wild type as reference), coloured by threshold and *not* clustered
   (``plot_reserve_trajectory_across_thresholds``).

``--top-n N`` sets how many top points per threshold to use (default 3).  Pass
``--surrogate`` to rank/plot the raw GP objectives instead of the
verified IBM means, and ``--show-objectives`` to add the achieved-objective
panel to the parameters figure.  With no positional argument the directory
falls back to ``IBM_OUTPUT_DIR`` from ``paths.py``.

Each figure is saved at 600 dpi into ``IMAGES_DIR`` and then shown unless
``--no-show`` is given.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import plot_style
from Gene_Drive_Optimisation.plot_gene_drives import (
    plot_best_parameters_across_thresholds,
    plot_reserve_trajectory_across_thresholds,
)
from paths import IBM_OUTPUT_DIR, IMAGES_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pareto_dir",
        nargs="?",
        default=None,
        help="Parent directory of low*_high* threshold subdirectories.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top points per threshold to compare (default 3).",
    )
    parser.add_argument(
        "--surrogate",
        action="store_true",
        help="Rank/plot raw surrogate objectives instead of IBM-verified means.",
    )
    parser.add_argument(
        "--show-objectives",
        action="store_true",
        help="Add the achieved-objective panel to the parameters figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figures without opening an interactive window.",
    )
    args = parser.parse_args()

    verified = not args.surrogate
    pareto_dir = (
        Path(args.pareto_dir) if args.pareto_dir is not None else IBM_OUTPUT_DIR
    )

    figures = {
        f"top{args.top_n}_parameters_across_thresholds": (
            plot_best_parameters_across_thresholds(
                pareto_dir,
                top_n=args.top_n,
                use_verified=verified,
                show_objectives=args.show_objectives,
                save=False,
            )
        ),
        "reserve_trajectory_across_thresholds": (
            plot_reserve_trajectory_across_thresholds(
                pareto_dir,
                top_n=args.top_n,
                use_verified=verified,
                save=False,
            )
        ),
    }

    for name, fig in figures.items():
        path = IMAGES_DIR / f"{name}.png"
        fig.savefig(path, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
        print(f"Saved {path}")

    if not args.no_show:
        plt.show()
