"""Train the IBM surrogate approximations reproducibly.

Runs both surrogate approaches over a single, reproducible seed:
  1. the feedforward neural net (``construct_surrogate_function``), and
  2. the masked-autoregressive-flow / NSF variational approximation
     (``train_maf`` followed by ``compare_pretrained_maf_to_data``).

Seeds control the numpy and pytorch RNG (and the train/test split) so a run
can be reproduced exactly. Each saved model encodes its seed in its filename,
so passing that seed back here reproduces the model. A seed is generated
randomly when one is not supplied on the command line.

By default the training data is the default-parameter dataset written by
``generate_training_data.py``. When that data was instead generated for one
point on a Bayesian-optimisation Pareto frontier (``generate_training_data.py
--pareto-file ... --pareto-row ...``), its filename carries a ``_params=``
marker and the same two flags must be passed here so the dataset can be found.
The marker is carried into the saved model filenames too, so a surrogate for an
optimised drive is never confused with the default-parameter one.

``--homozygotes-only`` trains two-genotype ``(WW, GG) -> (WW, GG)``
surrogates on the heterozygote-free data written by ``generate_training_data.py
--homozygotes-only``, for underdominance architectures whose heterozygote is
non-viable.  The saved models carry a ``_homozygotes=True`` marker, and the
loaders automatically wrap them with a function that reports WG=0 for all outputs, so they can be used as a drop-in replacement for the full three-genotype surrogates.
"""

import argparse
from pathlib import Path

import numpy as np
from Gene_Drive_Optimisation.gene_drive_objective import load_gene_drive_parameters
from IBM_Emulation.surrogates import construct_surrogate_function
from IBM_Emulation.variational_approximations import (
    compare_pretrained_maf_to_data,
    train_maf,
)


def generate_seed() -> int:
    """Generate a fresh, non-deterministic seed from the OS entropy pool."""
    return int(np.random.SeedSequence().generate_state(1)[0])


def resolve_param_tag(
    pareto_file: Path | None,
    pareto_row: int | None,
) -> str | None:
    """Return the ``param_tag`` identifying the dataset to train on.

    The tag is rebuilt from the Pareto CSV with the same function the data
    generator used (``load_gene_drive_parameters``), so it matches the marker in
    the data filename exactly rather than being retyped by hand. ``None`` --
    no Pareto point selected -- means the default-parameter dataset.
    """
    if pareto_file is None:
        print("Training on the default-parameter dataset.")
        return None

    genotype_penalties, conversion_rate, param_tag = load_gene_drive_parameters(
        pareto_file, pareto_row
    )
    print(
        f"Training on the dataset for row {pareto_row} of {pareto_file} "
        f"(param_tag={param_tag}):"
    )
    print(f"  conversion_rate = {conversion_rate}")
    for genotype, penalties in genotype_penalties.items():
        formatted = ", ".join(f"{k}={v}" for k, v in penalties.items())
        print(f"  {genotype}: {formatted}")
    return param_tag


def main(
    seed: int | None = None,
    data_seed: int = 27,
    test_data_seed: int | None = None,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
) -> None:
    """Train the feedforward NN and the MAF surrogate under a shared seed.

    ``data_seed`` selects the training dataset (the input-output pair file
    generated with that seed). ``test_data_seed``, when given, selects a
    held-out test set (generated with ``--test`` and carrying a ``test=True``
    marker in its filename) to evaluate against instead of splitting the
    training data. ``param_tag`` selects the datasets generated for a
    non-default gene-drive parameter set, and is carried into the saved models'
    filenames. ``homozygotes_only`` trains two-genotype ``(WW, GG)`` surrogates
    on the heterozygote-free data, and is carried into the filenames the same
    way.
    """
    if seed is None:
        seed = generate_seed()
    print(
        f"Training approximations with seed={seed}, data_seed={data_seed}, "
        f"test_data_seed={test_data_seed}, param_tag={param_tag}, "
        f"homozygotes_only={homozygotes_only}"
    )

    # 1. Feedforward neural-net surrogate (genetics-aware).
    construct_surrogate_function(
        with_genetics=True,
        seed=seed,
        data_seed=data_seed,
        test_data_seed=test_data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )

    # 2. Masked-autoregressive-flow / NSF variational approximation.
    train_maf(
        seed=seed,
        data_seed=data_seed,
        test_data_seed=test_data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )
    compare_pretrained_maf_to_data(
        seed=seed,
        data_seed=data_seed,
        test_data_seed=test_data_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility; randomly generated if omitted.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=27,
        help="Seed identifying the input-output pair file to train on.",
    )
    parser.add_argument(
        "--test-data-seed",
        type=int,
        default=None,
        help=(
            "Seed identifying a held-out test set (generated with --test) to "
            "evaluate against; if omitted the training data is split instead."
        ),
    )
    parser.add_argument(
        "--homozygotes-only",
        action="store_true",
        dest="homozygotes_only",
        help=(
            "Train two-genotype (WW, GG) surrogates on the heterozygote-free "
            "data written by generate_training_data.py --homozygotes-only. The "
            "models report WG = 0 by construction; the training run prints how "
            "much output mass that discards."
        ),
    )
    parser.add_argument(
        "--pareto-file",
        type=Path,
        default=None,
        dest="pareto_file",
        help=(
            "Path to the gene_drive_pareto_frontier_points_*.csv the training "
            "data was generated from. Required when the target dataset was "
            "generated with generate_training_data.py --pareto-file, since the "
            "gene-drive parameters are part of its filename. Must be used "
            "together with --pareto-row."
        ),
    )
    parser.add_argument(
        "--pareto-row",
        type=int,
        default=None,
        dest="pareto_row",
        help=(
            "Zero-indexed row of --pareto-file the training data was generated "
            "from (row 0 is the first Pareto point). Must be used together "
            "with --pareto-file."
        ),
    )
    args = parser.parse_args()

    if (args.pareto_file is None) != (args.pareto_row is None):
        parser.error(
            "--pareto-file and --pareto-row must be given together: "
            "a Pareto file is meaningless without a row to select from it."
        )

    main(
        seed=args.seed,
        data_seed=args.data_seed,
        test_data_seed=args.test_data_seed,
        param_tag=resolve_param_tag(args.pareto_file, args.pareto_row),
        homozygotes_only=args.homozygotes_only,
    )
