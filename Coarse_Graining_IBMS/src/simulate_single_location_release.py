"""Simulate gene-drive releases into one equilibrated wild-type population.

Run from ``Coarse_Graining_IBMS/src/``::

    python simulate_single_location_release.py --default
    python simulate_single_location_release.py \\
        --pareto-file ../data/ibm_output/low0p30_high0p50/gene_drive_pareto_frontier_points_20260601_101508.csv \\
        --pareto-row 0
    python simulate_single_location_release.py --best ../data/ibm_output/low0p30_high0p50
    python simulate_single_location_release.py --from-csv ../data/ibm_output/single_location_releases/<file>.csv

The gene-drive optimisation scores a drive from a founding cohort of eggs whose
GG fraction is set by hand.  This asks the more realistic question: released
into a wild-type population already sitting at its equilibrium, at a range of
release frequencies, what does the drive actually do?  A wild-type burn-in runs
first, GG adults are added to that equilibrium, and every stage count the IBM
produces is recorded (see
:mod:`Gene_Drive_Optimisation.single_location_releases`).

Two figures are written to ``SINGLE_LOCATION_RELEASE_IMAGES_DIR``: the
trajectories (abundance, genotype frequency and per-genotype egg-to-adult
survival, a row per release frequency) and the diagnostics (allele frequency by
replicate, larval fate composition, suppression, and how the replicates ended).
``--diagnostics`` adds a third aimed at the invasion threshold.

**This is slow.**  One generation is one IBM cycle -- roughly 15 s at
equilibrium abundance -- and a run costs ``n_replicates * (burn-in +
n_frequencies * n_generations)`` of them, spread over the available cores.  The
defaults are sized for prototyping (a few minutes); a publication run means
raising ``--n-replicates`` and walking away.  Every run therefore writes its raw
counts to CSV, and ``--from-csv`` rebuilds the figures from that file without
simulating anything.
"""

import argparse
from pathlib import Path

import matplotlib

# Batch helper that writes figures to disk; use a non-interactive backend so it
# runs headless and nothing tries to open a window.
matplotlib.use("Agg")

import pandas as pd
from paths import (
    SINGLE_LOCATION_RELEASE_IMAGES_DIR,
    single_location_release_path,
)

from Gene_Drive_Optimisation.gene_drive_objective import (
    load_gene_drive_parameters,
    rank_pareto_points,
)
from Gene_Drive_Optimisation.plot_single_location_releases import (
    plot_release_diagnostics,
    plot_release_trajectories,
    plot_threshold_diagnostics,
)
from Gene_Drive_Optimisation.single_location_releases import (
    POND_CARRYING_CAPACITY,
    simulate_single_location_releases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Release a gene drive into a single equilibrated wild-type "
            "population at a range of frequencies, and plot what happens."
        )
    )
    parser.add_argument(
        "--pareto-file",
        type=Path,
        default=None,
        dest="pareto_file",
        help=(
            "Path to a gene_drive_pareto_frontier_points_*.csv produced by "
            "optimise_gene_drive.py. Must be used together with --pareto-row."
        ),
    )
    parser.add_argument(
        "--pareto-row",
        type=int,
        default=None,
        dest="pareto_row",
        help=(
            "Zero-indexed row of --pareto-file to take the gene-drive "
            "parameters from (row 0 is the first Pareto point)."
        ),
    )
    parser.add_argument(
        "--best",
        type=Path,
        default=None,
        dest="best_dir",
        metavar="PARETO_DIR",
        help=(
            "Directory of per-replicate gene_drive_pareto_frontier_points_*.csv "
            "files (e.g. ../data/ibm_output/low0p30_high0p50). Pools every point "
            "across those runs, ranks them by the mean of their two IBM-verified "
            "objective scores, and simulates the best one. Mutually exclusive "
            "with --pareto-file/--pareto-row."
        ),
    )
    parser.add_argument(
        "--default",
        action="store_true",
        dest="use_default",
        help=(
            "Simulate the default gene-drive parameters hard-coded in "
            "initialize_simulation() instead of a Pareto point."
        ),
    )
    parser.add_argument(
        "--release-frequencies",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        dest="release_frequencies",
        help=(
            "GG proportions of the population immediately after release, one "
            "row of the trajectory figure each. Each must lie in [0, 1). "
            "Default: 0.1 0.3 0.5."
        ),
    )
    parser.add_argument(
        "--release-mode",
        choices=("additive", "replacement"),
        default="additive",
        dest="release_mode",
        help=(
            "'additive' (default) adds GG adults on top of the residents, as a "
            "release programme does, which lifts abundance above equilibrium. "
            "'replacement' holds the total fixed, isolating the genetic effect "
            "from the response to adding mosquitoes."
        ),
    )
    parser.add_argument(
        "--n-replicates",
        type=int,
        default=3,
        dest="n_replicates",
        help=(
            "Independent burn-in/release replicates. Small by default because "
            "each one costs minutes; raise it for anything conclusive. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--n-burn-in-generations",
        type=int,
        default=5,
        dest="n_burn_in_generations",
        help=(
            "Wild-type generations before the release. The population settles "
            "within about three from the default --initial-ww. Default: 5."
        ),
    )
    parser.add_argument(
        "--n-generations",
        type=int,
        default=6,
        dest="n_generations",
        help="Generations simulated after the release. Default: 6.",
    )
    parser.add_argument(
        "--initial-ww",
        type=float,
        default=70.0,
        dest="initial_ww",
        help=(
            "Wild-type adults the burn-in starts from. The default is already "
            "close to this model's equilibrium, so the burn-in has little to "
            "do. Default: 70."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=20,
        choices=(20, 22, 24, 26, 28, 30),
        help="Fixed simulation temperature in degrees C. Default: 20.",
    )
    parser.add_argument(
        "--eggs-per-female",
        type=int,
        default=30,
        dest="eggs_per_female",
        help="Eggs produced per mating female. Default: 30.",
    )
    parser.add_argument(
        "--pond-carrying-capacity",
        type=int,
        default=POND_CARRYING_CAPACITY,
        dest="pond_carrying_capacity",
        help=(
            "Maximum eggs the pond supports; excess are discarded at random. "
            f"Default: {POND_CARRYING_CAPACITY}, matching the spatial model."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Also draw the threshold diagnostics: the allele-frequency map "
            "p_(t+1) against p_t, realised growth rates by genotype, and the "
            "egg pool against the pond's capacity."
        ),
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        dest="from_csv",
        help=(
            "Rebuild the figures from a previously saved run instead of "
            "simulating. Every other simulation argument is ignored."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        dest="n_workers",
        help=(
            "Worker processes for the replicate/frequency branches. Defaults to "
            "the SLURM-allocated core count under SLURM, otherwise the machine's."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Master seed for the run's stochastic components. If omitted, a "
            "fresh one is generated, printed, and recorded in the output filenames."
        ),
    )
    args = parser.parse_args()

    if args.from_csv is not None:
        return args

    selectors = [
        args.best_dir is not None,
        args.pareto_file is not None or args.pareto_row is not None,
        args.use_default,
    ]
    if sum(selectors) != 1:
        parser.error(
            "Choose exactly one architecture to simulate: --default, "
            "--pareto-file with --pareto-row, or --best."
        )

    if (args.pareto_file is None) != (args.pareto_row is None):
        parser.error(
            "--pareto-file and --pareto-row must be given together: a Pareto "
            "file is meaningless without a row to select from it."
        )

    return args


def resolve_architecture(args: argparse.Namespace) -> tuple[dict | None, float, str | None]:
    """Return the ``(genotype_penalties, conversion_rate, param_tag)`` to simulate.

    The parameters are echoed in full because they are the only record of what
    was simulated that does not depend on reading the Pareto CSV back later.
    """
    if args.use_default:
        print(
            "Simulating the default gene-drive parameters from "
            "initialize_simulation() (conversion rate 0)."
        )
        return None, 0.0, None

    if args.best_dir is not None:
        ranked = rank_pareto_points(args.best_dir, top_n=1)
        pareto_file, pareto_row, score = ranked[0]
        print(
            f"Best Pareto point in {args.best_dir} by mean IBM-verified "
            f"objective score ({score:.2f}): row {pareto_row} of {pareto_file}"
        )
    else:
        pareto_file, pareto_row = args.pareto_file, args.pareto_row

    genotype_penalties, conversion_rate, param_tag = load_gene_drive_parameters(
        pareto_file, pareto_row
    )
    print(f"Using gene-drive parameters from row {pareto_row} of {pareto_file}:")
    print(f"  conversion_rate = {conversion_rate}")
    for genotype, penalties in genotype_penalties.items():
        formatted = ", ".join(f"{key}={value}" for key, value in penalties.items())
        print(f"  {genotype}: {formatted}")

    return genotype_penalties, conversion_rate, param_tag


def figure_paths(param_tag: str | None, seed) -> dict[str, Path]:
    """Output paths for the run's figures, keyed by figure name."""
    tag = param_tag if param_tag else "default"
    stem = f"single_location_release_params={tag}_seed={seed}"
    return {
        name: SINGLE_LOCATION_RELEASE_IMAGES_DIR / f"{stem}_{name}.png"
        for name in ("trajectories", "diagnostics", "threshold")
    }


if __name__ == "__main__":
    args = parse_args()

    if args.from_csv is not None:
        dataframe = pd.read_csv(args.from_csv)
        param_tag = str(dataframe["param_tag"].iloc[0])
        param_tag = None if param_tag == "default" else param_tag
        seed = dataframe["seed"].iloc[0]
        print(f"Rebuilding figures from {args.from_csv} ({len(dataframe)} rows).")
        csv_path = args.from_csv
    else:
        genotype_penalties, conversion_rate, param_tag = resolve_architecture(args)
        dataframe = simulate_single_location_releases(
            release_frequencies=args.release_frequencies,
            conversion_rate=conversion_rate,
            genotype_penalties=genotype_penalties,
            param_tag=param_tag,
            n_replicates=args.n_replicates,
            n_burn_in_generations=args.n_burn_in_generations,
            n_generations=args.n_generations,
            initial_ww=args.initial_ww,
            release_mode=args.release_mode,
            eggs_per_female=args.eggs_per_female,
            pond_carrying_capacity=args.pond_carrying_capacity,
            temperature=args.temperature,
            seed=args.seed,
            n_workers=args.n_workers,
        )
        seed = dataframe["seed"].iloc[0]
        csv_path = single_location_release_path(seed, param_tag)
        dataframe.to_csv(csv_path, index=False)
        print(f"Saved raw stage counts to {csv_path}")

    paths = figure_paths(param_tag, seed)
    print(f"Saved trajectory figure to {plot_release_trajectories(dataframe, paths['trajectories'])}")
    print(f"Saved diagnostics figure to {plot_release_diagnostics(dataframe, paths['diagnostics'])}")
    if args.diagnostics:
        print(
            "Saved threshold diagnostics figure to "
            f"{plot_threshold_diagnostics(dataframe, paths['threshold'])}"
        )
    if args.from_csv is None:
        print(f"Replot without re-simulating with:  --from-csv {csv_path}")
