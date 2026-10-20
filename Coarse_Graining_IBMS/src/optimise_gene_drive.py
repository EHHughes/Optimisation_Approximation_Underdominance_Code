import argparse
import datetime
import random

from Gene_Drive_Optimisation.optimise_gene_drives import (
    PROP_GG_HIGH,
    PROP_GG_LOW,
    optimise_gene_drive,
    save_model,
    save_pareto_frontier_points,
)
from paths import IBM_OUTPUT_DIR


def _unit_float(value: str) -> float:
    x = float(value)
    if not (0.0 < x < 1.0):
        raise argparse.ArgumentTypeError(f"{value!r} must be strictly between 0 and 1")
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-objective Bayesian optimisation for underdominance gene drives."
    )
    parser.add_argument(
        "--prop-gg-low",
        type=_unit_float,
        default=PROP_GG_LOW,
        dest="prop_gg_low",
        help=(
            "GG release fraction for the extinction objective (f1). "
            f"Must be in (0, 1). Default: {PROP_GG_LOW}"
        ),
    )
    parser.add_argument(
        "--prop-gg-high",
        type=_unit_float,
        default=PROP_GG_HIGH,
        dest="prop_gg_high",
        help=(
            "GG release fraction for the fixation objective (f2). "
            f"Must be in (0, 1). Default: {PROP_GG_HIGH}"
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        dest="base_seed",
        help=(
            "Base random seed for the BoTorch Sobol initialisation and verification "
            "RNG. If omitted, a random seed is drawn at startup and printed. Note: "
            "IBM evaluation stochasticity is not controlled by this seed."
        ),
    )
    args = parser.parse_args()
    if args.prop_gg_low >= args.prop_gg_high:
        parser.error(
            f"--prop-gg-low ({args.prop_gg_low}) must be less than "
            f"--prop-gg-high ({args.prop_gg_high})"
        )
    return args


if __name__ == "__main__":
    args = parse_args()

    seed = (
        args.base_seed if args.base_seed is not None else random.randint(1, 2**31 - 1)
    )
    print(f"Run seed: {seed}")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    n_init = 256
    n_bo_steps = 150
    n_replicates = 5

    pareto_X, pareto_Y, verified_pareto_Y, verified_pareto_Yvar, model, history = (
        optimise_gene_drive(
            n_init=n_init,
            n_bo_steps=n_bo_steps,
            cycles=3,
            n_replicates=n_replicates,
            return_history=True,
            base_seed=seed,
            prop_gg_low=args.prop_gg_low,
            prop_gg_high=args.prop_gg_high,
        )
    )

    save_model(
        model,
        n_init=n_init,
        n_bo_steps=n_bo_steps,
        n_replicates=n_replicates,
        prop_gg_low=args.prop_gg_low,
        prop_gg_high=args.prop_gg_high,
        run_timestamp=run_timestamp,
    )

    save_pareto_frontier_points(
        pareto_X,
        pareto_Y,
        verified_pareto_Y,
        verified_pareto_Yvar,
        str(IBM_OUTPUT_DIR / "gene_drive_pareto_frontier_points.csv"),
        prop_gg_low=args.prop_gg_low,
        prop_gg_high=args.prop_gg_high,
        run_timestamp=run_timestamp,
    )
