"""Helper script to validate the trained surrogate approximations against a
held-out test set.

Run from ``Coarse_Graining_IBMS/src/``::

    python validate_approximations.py --model_seed 1180468651 --data_seed 27 \
        --scaling_seed 27

It loads the feedforward NN and the MAF normalising-flow surrogates trained
under ``--model_seed`` and the held-out test set generated under
``--data_seed``, then writes two diagnostic figures to ``IMAGES_DIR``:

* the NN's actual-vs-predicted test-set predictions
  (:func:`plot_test_set_predictions`), and
* the MAF's test outputs mapped into latent space, checked against the
  standard-normal base distribution (:func:`check_normality_of_test_set`).

``--data_seed`` identifies the held-out test set, which is free to vary so we
validate on data the models have never seen.

``--scaling_seed`` identifies the *training* CSV the input/output scalings are
read off. Those scalings must match the ones the models were trained under, so
this is the ``--data-seed`` that was passed to ``train_approximations.py``. The
train CSV is used solely for scalings -- it is *not* evaluated -- and the test
seed tracks ``--data_seed`` freely, i.e. ``test_train_seeds = (data_seed,
scaling_seed)``.

By default the models and data are the default-parameter ones. When they were
instead produced for one point on a Bayesian-optimisation Pareto frontier
(``generate_training_data.py``/``train_approximations.py --pareto-file ...
--pareto-row ...``), their filenames carry a ``_params=`` marker and the same
two flags must be passed here so the models, the data and the saved figures all
belong to that one drive design.
"""

import argparse
from pathlib import Path

import matplotlib

# Batch helper that writes figures to disk; use a non-interactive backend so it
# runs headless and the plotting functions' plt.show() calls don't block.
matplotlib.use("Agg")

import torch
from Gene_Drive_Optimisation.gene_drive_objective import load_gene_drive_parameters
from IBM_Emulation.dataloaders import (
    embed_homozygotes,
    get_test_train_datasets,
    report_heterozygote_residual_for_dataset,
)
from IBM_Emulation.surrogates import (
    load_pretrained_neural_net,
    plot_test_set_predictions,
)
from IBM_Emulation.variational_approximations import check_normality_of_test_set
from paths import IMAGES_DIR


def resolve_param_tag(
    pareto_file: Path | None,
    pareto_row: int | None,
) -> str | None:
    """Return the ``param_tag`` identifying the models and data to validate.

    The tag is rebuilt from the Pareto CSV with the same function the data
    generator and the trainer used (``load_gene_drive_parameters``), so it
    matches the marker in the filenames exactly rather than being retyped by
    hand. ``None`` -- no Pareto point selected -- means the default-parameter
    models and data.
    """
    if pareto_file is None:
        print("Validating the default-parameter approximations.")
        return None

    genotype_penalties, conversion_rate, param_tag = load_gene_drive_parameters(
        pareto_file, pareto_row
    )
    print(
        f"Validating the approximations for row {pareto_row} of {pareto_file} "
        f"(param_tag={param_tag}):"
    )
    print(f"  conversion_rate = {conversion_rate}")
    for genotype, penalties in genotype_penalties.items():
        formatted = ", ".join(f"{k}={v}" for k, v in penalties.items())
        print(f"  {genotype}: {formatted}")
    return param_tag


def validate_approximations(
    model_seed: int,
    data_seed: int,
    scaling_seed: int,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
) -> None:
    """Run the NN and MAF test-set diagnostics and save both figures.

    ``scaling_seed`` selects the training CSV the input/output scalings are
    taken from; it must be the data seed the models were trained under (see the
    module docstring). ``param_tag`` selects the models and data belonging to a
    non-default gene-drive parameter set, and is carried into the figure
    filenames. ``homozygotes_only`` selects the two-genotype ``(WW, GG)``
    models and the heterozygote-free data they were fit to.
    """
    device = torch.device("cpu")
    # Test set varies with --data_seed; the train seed is the models' training
    # data so the scalings match (see module docstring).
    test_train_seeds = (data_seed, scaling_seed)
    # Mirrors the ``_params=`` marker on the model and data filenames, so a
    # figure is never mistaken for the default-parameter one.
    params_marker = f"_params={param_tag}" if param_tag else ""
    homozygotes_marker = "_homozygotes=True" if homozygotes_only else ""

    if homozygotes_only:
        # The models assert WG = 0; report how much output mass the held-out
        # data says that discards, so the figures are read in that light.
        report_heterozygote_residual_for_dataset(
            data_seed,
            param_tag=param_tag,
            label=f"held-out test set (seed={data_seed})",
        )

    # --- NN surrogate: actual vs predicted on the held-out test set ---
    # Returned already wrapped as a three-genotype map when homozygotes_only,
    # so the plotting below needs no special case beyond widening the dataset.
    net, _ = load_pretrained_neural_net(
        with_genetics=True,
        seed=model_seed,
        data_seed=scaling_seed,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )

    # Build a loader over the held-out test set; plot_test_set_predictions reads
    # the underlying dataset's inputs/outputs, so shuffling is unnecessary. The
    # scalings come from this test set's (shared) train CSV.
    _train, test_dataset, scalings = get_test_train_datasets(
        device, test_train_seeds, param_tag=param_tag, homozygotes_only=homozygotes_only
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    # Test RMSE in real (un-normalised) adult counts: the net maps normalised
    # inputs -> normalised outputs, so rescale both sides by the output scaling.
    with torch.no_grad():
        predicted = net(test_dataset.inputs)
    actual = test_dataset.outputs
    rmse = (
        ((predicted - actual) * float(scalings[1])).pow(2).mean().sqrt().item()
    )
    print(f"NN test RMSE (adult counts): {rmse:.4f}")

    nn_save_path = (
        IMAGES_DIR
        / f"nn_test_set_predictions_model_seed={model_seed}"
        f"_data_seed={data_seed}_scaling_seed={scaling_seed}"
        f"{params_marker}{homozygotes_marker}.png"
    )
    plot_test_set_predictions(
        net,
        test_loader,
        scalings,
        title="NN test-set predictions",
        save_path=str(nn_save_path),
        subsample=5,
        embed=embed_homozygotes if homozygotes_only else None,
    )
    print(f"Saved NN test-set predictions to {nn_save_path}")

    # --- MAF surrogate: normality of test outputs in latent space ---
    maf_save_path = (
        IMAGES_DIR
        / f"maf_latent_normality_model_seed={model_seed}"
        f"_data_seed={data_seed}_scaling_seed={scaling_seed}"
        f"{params_marker}{homozygotes_marker}.png"
    )
    check_normality_of_test_set(
        test_train_seeds,
        model_seed=model_seed,
        save_path=maf_save_path,
        param_tag=param_tag,
        homozygotes_only=homozygotes_only,
    )
    print(f"Saved MAF latent-space normality check to {maf_save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the NN and MAF surrogate approximations against a "
            "held-out test set."
        )
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        required=True,
        help="Seed identifying the pretrained NN and MAF model weights to load.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        required=True,
        help=(
            "Seed identifying the held-out test set to validate against "
            "(may differ from the models' training data; scalings come from "
            "--scaling_seed)."
        ),
    )
    parser.add_argument(
        "--scaling_seed",
        type=int,
        required=True,
        help=(
            "Seed identifying the training input-output pair file the "
            "input/output scalings are read off. Must be the --data-seed the "
            "models were trained under, so the scalings match training time; "
            "the file is used for scalings only and is never evaluated."
        ),
    )
    parser.add_argument(
        "--homozygotes-only",
        action="store_true",
        dest="homozygotes_only",
        help=(
            "Validate the two-genotype (WW, GG) models trained with "
            "train_approximations.py --homozygotes-only, against the "
            "heterozygote-free data. Required when they were produced that "
            "way, since the marker is part of their filenames."
        ),
    )
    parser.add_argument(
        "--pareto-file",
        type=Path,
        default=None,
        dest="pareto_file",
        help=(
            "Path to the gene_drive_pareto_frontier_points_*.csv the models "
            "and data were generated from. Required when they were produced "
            "with --pareto-file, since the gene-drive parameters are part of "
            "their filenames. Must be used together with --pareto-row."
        ),
    )
    parser.add_argument(
        "--pareto-row",
        type=int,
        default=None,
        dest="pareto_row",
        help=(
            "Zero-indexed row of --pareto-file the models and data were "
            "generated from (row 0 is the first Pareto point). Must be used "
            "together with --pareto-file."
        ),
    )
    args = parser.parse_args()

    if (args.pareto_file is None) != (args.pareto_row is None):
        parser.error(
            "--pareto-file and --pareto-row must be given together: "
            "a Pareto file is meaningless without a row to select from it."
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    validate_approximations(
        model_seed=args.model_seed,
        data_seed=args.data_seed,
        scaling_seed=args.scaling_seed,
        param_tag=resolve_param_tag(args.pareto_file, args.pareto_row),
        homozygotes_only=args.homozygotes_only,
    )
