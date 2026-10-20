"""This is the endpoint script to simulate non-spatial IBM and generate input-output pairs for coarse-graining.

By default the IBM runs with the gene-drive parameters hard-coded in
``initialize_simulation()``.  Passing ``--pareto-file`` and ``--pareto-row``
instead takes the full nine-parameter gene-drive design from one point on a
Bayesian-optimisation Pareto frontier, so a surrogate can be trained for that
optimised drive.  Data generated that way carries a ``_params=`` marker in its
filename and can never be confused with the default-parameter data.

``--homozygotes-only`` samples the input adults on the WW/GG face only, with no
heterozygotes.  Some underdominance architectures make the heterozygote non-viable,
so the dynamics never visit a state with WG adults and the default
three-genotype sampling spends most of its trials describing a region the
surrogate will never be asked about.  Such data carries a
``_homozygotes=True`` marker, and the same flag is passed to
``train_approximations.py`` to fit the matching two-genotype surrogates.
"""

import argparse
from pathlib import Path

import numpy as np
from Gene_Drive_Optimisation.gene_drive_objective import load_gene_drive_parameters
from IBM_Emulation.dataloaders import report_heterozygote_residual
from IBM_Implementations.input_output_pairs import generate_input_output_pairs
from paths import input_output_pairs_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set then saves the file as a test set which can be used for held out validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for regenerating a specific dataset. If none them samples the seed randomly",
    )
    parser.add_argument(
        "--homozygotes-only",
        action="store_true",
        dest="homozygotes_only",
        help=(
            "Sample the input adults on the WW/GG face only, with no "
            "heterozygotes, for underdominance architectures whose "
            "heterozygote is non-viable. The output filenames carry a "
            "_homozygotes=True marker; pass the same flag to "
            "train_approximations.py to fit the matching two-genotype "
            "surrogates."
        ),
    )
    parser.add_argument(
        "--pareto-file",
        type=Path,
        default=None,
        dest="pareto_file",
        help=(
            "Optional path to a gene_drive_pareto_frontier_points_*.csv produced by "
            "optimise_gene_drive.py. If given, the gene-drive parameters are taken "
            "from the row selected by --pareto-row rather than from the defaults. "
            "Must be used together with --pareto-row."
        ),
    )
    parser.add_argument(
        "--pareto-row",
        type=int,
        default=None,
        dest="pareto_row",
        help=(
            "Zero-indexed row of --pareto-file to take the gene-drive parameters "
            "from (row 0 is the first Pareto point). Must be used together with "
            "--pareto-file."
        ),
    )
    args = parser.parse_args()

    if (args.pareto_file is None) != (args.pareto_row is None):
        parser.error(
            "--pareto-file and --pareto-row must be given together: "
            "a Pareto file is meaningless without a row to select from it."
        )

    test = args.test
    if test:
        n_trials = 1000
    else:
        n_trials = 40000

    seed = args.seed
    if seed is None:
        seed = np.random.randint(400)

    genotype_penalties = None
    conversion_rate = 0.0
    param_tag = None

    if args.pareto_file is not None:
        genotype_penalties, conversion_rate, param_tag = load_gene_drive_parameters(
            args.pareto_file, args.pareto_row
        )
        print(
            f"Using gene-drive parameters from row {args.pareto_row} of "
            f"{args.pareto_file}:"
        )
        print(f"  conversion_rate = {conversion_rate}")
        for genotype, penalties in genotype_penalties.items():
            formatted = ", ".join(f"{k}={v}" for k, v in penalties.items())
            print(f"  {genotype}: {formatted}")
    else:
        print("Using the default gene-drive parameters from initialize_simulation().")

    if args.homozygotes_only:
        print(
            "Sampling input adults on the WW/GG face only (no heterozygotes). "
            "The recorded input_eggs_WG and output_adults_WG are still "
            "whatever the IBM produced."
        )
    else:
        print("Sampling input adults across all three genotypes.")

    # These runs are long, so show up front exactly what is being written.
    for with_genetics in (False, True):
        print(
            "Writing "
            f"{input_output_pairs_path(seed, with_genetics=with_genetics, test=test, param_tag=param_tag, homozygotes_only=args.homozygotes_only)}"
        )

    pairs = generate_input_output_pairs(
        n_trials=n_trials,
        append_to_existing=False,
        seed=seed,
        test=test,
        genotype_penalties=genotype_penalties,
        conversion_rate=conversion_rate,
        param_tag=param_tag,
        homozygotes_only=args.homozygotes_only,
    )

    if args.homozygotes_only:
        # A surrogate trained on this data will report WG = 0; this is how much
        # output mass that assertion discards, and so whether this
        # architecture's wg_* penalties are lethal enough for it to be sound.
        report_heterozygote_residual(pairs, label=f"generated data (seed={seed})")
