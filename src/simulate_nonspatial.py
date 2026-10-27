"""This is the endpoint script to simulate non-spatial IBM and generate input-output pairs for coarse-graining. If they are already generated (see input_output_pairs.py), then we can directly construct the surrogate function."""

import argparse

# Plotting / comparison functions live in IBM_Emulation/plot_surrogate_model_comparisons.py
import numpy as np
from IBM_Emulation.plot_surrogate_model_comparisons import (
    compare_multigeneration_ibm_to_nn,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the IBM simulations; a fresh one is generated if omitted.",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=None,
        help="Seed identifying which trained surrogate to load (matches the seed "
        "in the saved model's filename). Defaults to the unseeded model.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="nn",
        help="Which model to use for the approximation (out of NN, MAF). Defaults to NN",
    )
    parser.add_argument(
        "--homozygotes-only",
        action="store_true",
        dest="homozygotes_only",
        help=(
            "Load the two-genotype (WW, GG) surrogate trained with "
            "train_approximations.py --homozygotes-only. Required when it was "
            "produced that way, since the marker is part of its filename."
        ),
    )
    parser.add_argument(
        "--keep-heterozygotes",
        action="store_true",
        dest="keep_heterozygotes",
        help=(
            "Keep the WG panel in the multi-generation figure, restoring the "
            "legacy 2x2 (WW, WG, GG, Total) layout. By default the panel is "
            "dropped, since heterozygotes are non-viable under most "
            "architectures, giving a single row of three panels."
        ),
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(int(1e6)))
    model_seed = args.model_seed
    model_type: str = args.model_type
    print(f"Running with seed={seed}, model_seed={model_seed}")

    compare_multigeneration_ibm_to_nn(
        n_generations=10,
        init_WW=50,
        init_WG=0,
        init_GG=50,
        n_replicates=50,
        fixed_temperature=20,
        seed=seed,
        model_seed=model_seed,
        model_type=model_type,
        homozygotes_only=args.homozygotes_only,
        drop_heterozygotes=not args.keep_heterozygotes,
    )
