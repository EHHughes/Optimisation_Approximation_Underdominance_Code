"""Compare the Pareto frontiers + parameters across optimisation replicates.

Run from ``src/``::

    python compare_pareto_replicates.py ../data/ibm_output/low0p30_high0p50

Builds three figures: the objective-space Pareto frontiers of each replicate,
the parameters binned by position along the frontier (band mean, with the
band's individual points overlaid as black-ringed markers in the band's own
colour), and the pairwise correlations between parameters.  With no argument it
falls back to ``IBM_OUTPUT_DIR`` from ``paths.py``.  Pass
``--surrogate`` to plot the raw GP objectives instead of the verified
IBM means (generally this is unwise, unless you are sure that the GP is fitting well), and ``--spearman`` for rank correlations.  Pass ``--replicate-labels``
to add the per-replicate legend to the frontiers figure (off by default).

Pass ``--best`` to additionally build a parameters figure for just the
top-scoring points pooled across all runs (ranked by the mean of f_extinction
and f_fixation).  ``--top-n N`` sets how many points to show (default 5).  By
default the top points are clustered by their objective outcomes (f_extinction,
f_fixation) into frontier regions (one mean bar per parameter per cluster, with
that cluster's points overlaid) so the parameter spread among similar-outcome
solutions stays visible; ``--max-cluster-distance D`` tunes how tightly points
are grouped.  ``--aggregate-best`` instead collapses them to a single mean bar
per parameter, and ``--per-point-best`` shows one bar per point.

Pass ``--growth`` to additionally plot the larval mass growth rate dW/dt against
mass for each cluster's mean parameter set (WG and GG genotypes, with the WW
wild type as a reference), evaluated through the IBM's inner growth ODE.  Pass
``--reserve`` for the companion plot of reserves R against time t along the same
infinite-food trajectory.

Each figure is saved at 600 dpi into
``<IMAGES_DIR>/pareto_replicate_comparisons/<dirname>/`` (filenames carry the
correlation method and a ``_surrogate`` tag when relevant) and then shown unless
``--no-show`` is given.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import plot_style
from Gene_Drive_Optimisation.plot_gene_drives import (
    plot_best_parameters,
    plot_mass_growth_rate_by_cluster,
    plot_parameter_correlations,
    plot_parameters_binned,
    plot_pareto_replicates,
    plot_reserve_trajectory_by_cluster,
)
from paths import IBM_OUTPUT_DIR, IMAGES_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pareto_dir",
        nargs="?",
        default=None,
        help="Directory of gene_drive_pareto_frontier_points_*.csv replicates.",
    )
    parser.add_argument(
        "--surrogate",
        action="store_true",
        help="Plot raw surrogate objectives instead of IBM-verified means.",
    )
    parser.add_argument(
        "--spearman",
        action="store_true",
        help="Use Spearman rank correlations in the correlation figure.",
    )
    parser.add_argument(
        "--replicate-labels",
        action="store_true",
        help="Add a per-replicate legend to the frontiers figure (useful for "
        "replicability/validation; off by default as it clutters a paper figure).",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Also plot the parameters of the top-scoring points across all runs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of highest-scoring points to show with --best (default 5).",
    )
    best_mode = parser.add_mutually_exclusive_group()
    best_mode.add_argument(
        "--aggregate-best",
        action="store_true",
        help="With --best, show one mean bar per parameter over all top "
        "points instead of clustering them.",
    )
    best_mode.add_argument(
        "--per-point-best",
        action="store_true",
        help="With --best, show one bar per top point per parameter instead of "
        "clustering them.",
    )
    parser.add_argument(
        "--max-cluster-distance",
        type=float,
        default=0.2,
        help="With --best (default clustered mode), maximum within-cluster "
        "objective-space distance used to group the top points (default 0.2).",
    )
    parser.add_argument(
        "--growth",
        action="store_true",
        help="Also plot dW/dt vs larval mass for each cluster's mean parameter "
        "set (WG and GG genotypes, with the WW wild type as reference).",
    )
    parser.add_argument(
        "--reserve",
        action="store_true",
        help="Also plot reserves R vs time t along the infinite-food trajectory "
        "for each top point (WG and GG genotypes, WW wild type as reference).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figures without opening an interactive window.",
    )
    args = parser.parse_args()

    verified = not args.surrogate
    method = "spearman" if args.spearman else "pearson"

    pareto_dir = (
        Path(args.pareto_dir) if args.pareto_dir is not None else IBM_OUTPUT_DIR
    )
    out_dir = IMAGES_DIR / "pareto_replicate_comparisons" / pareto_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_surrogate" if args.surrogate else ""

    figures = {
        f"frontiers{tag}": plot_pareto_replicates(
            pareto_dir,
            use_verified=verified,
            show_replicate_labels=args.replicate_labels,
            save=False,
        ),
        f"parameters_binned{tag}": plot_parameters_binned(
            pareto_dir, use_verified=verified, save=False
        ),
        f"correlations_{method}{tag}": plot_parameter_correlations(
            pareto_dir, use_verified=verified, method=method, save=False
        ),
    }

    if args.best:
        # Default to the clustered view; the two flags opt back into the older
        # single-aggregate or one-bar-per-point displays.
        cluster_best = not (args.aggregate_best or args.per_point_best)
        figures[f"best_parameters{tag}"] = plot_best_parameters(
            pareto_dir,
            use_verified=verified,
            top_n=args.top_n,
            aggregate=args.aggregate_best,
            cluster=cluster_best,
            max_cluster_distance=args.max_cluster_distance,
            save=False,
        )

    if args.growth:
        figures[f"mass_growth_rate{tag}"] = plot_mass_growth_rate_by_cluster(
            pareto_dir,
            use_verified=verified,
            top_n=args.top_n,
            max_cluster_distance=args.max_cluster_distance,
            save=False,
        )

    if args.reserve:
        figures[f"reserve_trajectory{tag}"] = plot_reserve_trajectory_by_cluster(
            pareto_dir,
            use_verified=verified,
            top_n=args.top_n,
            max_cluster_distance=args.max_cluster_distance,
            save=False,
        )

    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=plot_style.SAVE_DPI, bbox_inches="tight")
        print(f"Saved {path}")

    if not args.no_show:
        plt.show()
