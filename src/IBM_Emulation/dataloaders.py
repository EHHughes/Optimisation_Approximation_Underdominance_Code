"""Module for loading input-output pairs from the CSV file and turning them into a Torch Dataset."""

import warnings

import numpy as np
import pandas as pd
import torch
from paths import input_output_pairs_path

#: Column offsets of the two homozygotes within every ``(..., 3)`` genotype
#: array in the pipeline, matching the ``(WW, WG, GG)`` order of
#: ``IBM_Implementations.input_output_pairs.GENOTYPES``.  This module is the
#: single owner of that mapping, so no call site re-derives ``[0, 2]``.
HOMOZYGOTE_INDICES: tuple[int, int] = (0, 2)

#: Position of the heterozygote in the same ordering.
HETEROZYGOTE_INDEX: int = 1

#: WG share of the total output adults above which forcing WG to zero is
#: flagged.  One percent is well below the point at which a homozygote-only
#: surrogate would misstate the total population, and well above the rounding
#: noise of a cohort of a few hundred adults.
HETEROZYGOTE_RESIDUAL_WARNING_THRESHOLD: float = 0.01


def select_homozygotes(array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Drop the heterozygote column: ``(..., 3)`` -> ``(..., 2)``.

    Works on numpy arrays and torch tensors alike, so the same helper serves
    the dataloaders and the surrogate wrapper.
    """
    return array[..., list(HOMOZYGOTE_INDICES)]


def embed_homozygotes(array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Re-insert a zero heterozygote column: ``(..., 2)`` -> ``(..., 3)``.

    The inverse of :func:`select_homozygotes`, and the step that lets a genuine
    two-genotype surrogate meet the pipeline's three-genotype contract.

    The result is built by concatenation rather than by assigning into a fresh
    array, because ``SpatialSimulation._update_population`` applies
    ``result[result < 1] = 0`` and then ``.detach().numpy()`` to whatever the
    simulator returns; a tensor assembled in-place from ``torch.zeros`` would
    not survive that.
    """
    if isinstance(array, torch.Tensor):
        heterozygotes = torch.zeros_like(array[..., :1])
        return torch.cat([array[..., :1], heterozygotes, array[..., 1:2]], dim=-1)
    array = np.asarray(array)
    heterozygotes = np.zeros_like(array[..., :1])
    return np.concatenate([array[..., :1], heterozygotes, array[..., 1:2]], axis=-1)


def report_heterozygote_residual(
    dataframe: pd.DataFrame,
    *,
    threshold: float = HETEROZYGOTE_RESIDUAL_WARNING_THRESHOLD,
    label: str = "dataset",
) -> dict[str, float]:
    """Measure what a homozygote-only surrogate throws away.

    A homozygote-only surrogate asserts ``WG = 0``.  That is an assertion about
    the architecture, not a measurement: with no WG adults in the input the IBM
    still breeds WG eggs from every WW x GG cross, and they still reach
    adulthood unless the architecture's ``wg_*`` penalties starve them.  The
    penalties are multiplicative scalings on larval life-history parameters,
    not a viability probability, so nothing guarantees it.  The WG share of the
    output adults is therefore exactly the mass that assertion discards, and
    this function is how you find out whether it is negligible.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A genetics-aware input-output pair frame, with the nine
        ``input_adults_*`` / ``input_eggs_*`` / ``output_adults_*`` columns.
    threshold : float
        Output share above which a warning is raised.
    label : str
        Name for this dataset in the printed report and the warning.

    Returns
    -------
    dict[str, float]
        ``input_share`` and ``output_share`` -- the WG fraction of the total
        input and output adults -- and the raw ``input_total`` /
        ``output_total`` / ``input_heterozygotes`` / ``output_heterozygotes``
        counts behind them.
    """
    input_columns = [f"input_adults_{genotype}" for genotype in ("WW", "WG", "GG")]
    output_columns = [f"output_adults_{genotype}" for genotype in ("WW", "WG", "GG")]
    missing = [
        column
        for column in input_columns + output_columns
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(
            f"{label} is not a genetics-aware input-output pair frame; missing "
            f"{missing}. report_heterozygote_residual needs the named "
            "per-genotype adult columns, not the aggregated counts."
        )

    input_total = float(dataframe[input_columns].to_numpy().sum())
    output_total = float(dataframe[output_columns].to_numpy().sum())
    input_heterozygotes = float(dataframe["input_adults_WG"].sum())
    output_heterozygotes = float(dataframe["output_adults_WG"].sum())

    if input_total == 0:
        raise ValueError(f"{label} has no input adults, so the WG share is undefined.")
    else:
        input_share = input_heterozygotes / input_total
    if output_total == 0:
        raise ValueError(f"{label} has no output adults, so the WG share is undefined.")
    else:
        output_share = output_heterozygotes / output_total

    print(
        f"Heterozygote residual in {label}: "
        f"WG is {input_share:.3%} of input adults, "
        f"{output_share:.3%} of output adults."
    )
    if output_share > threshold:
        warnings.warn(
            f"Forcing WG to zero discards {output_share:.2%} of the output "
            f"adults in {label}, above the {threshold:.2%} threshold. A "
            "homozygote-only surrogate is only a sound approximation when the "
            "heterozygote is near-non-viable; check this architecture's wg_* "
            "penalties.",
            stacklevel=2,
        )
    return {
        "input_share": input_share,
        "output_share": output_share,
        "input_total": input_total,
        "output_total": output_total,
        "input_heterozygotes": input_heterozygotes,
        "output_heterozygotes": output_heterozygotes,
    }


def report_heterozygote_residual_for_dataset(
    data_seed: int,
    *,
    param_tag: str | None = None,
    label: str | None = None,
    threshold: float = HETEROZYGOTE_RESIDUAL_WARNING_THRESHOLD,
) -> dict[str, float]:
    """Run :func:`report_heterozygote_residual` on a heterozygote-free dataset.

    The trainers want the residual for the file they are about to train on,
    named by its seed and tags rather than loaded by hand. Reading the CSV
    belongs here rather than in the trainers, which is why this wrapper exists.
    """
    path = input_output_pairs_path(
        data_seed, with_genetics=True, param_tag=param_tag, homozygotes_only=True
    )
    return report_heterozygote_residual(
        pd.read_csv(path), threshold=threshold, label=label or path.name
    )


def warn_if_heterozygote_inputs_present(
    dataframe: pd.DataFrame, label: str = "dataset"
) -> bool:
    """Warn when homozygote-only training is asked for on data that varies WG.

    Distinct from :func:`report_heterozygote_residual`: that one measures the
    *output* mass the ``WG = 0`` assertion discards, this one catches the
    separate mistake of dropping a WG *input* column the data actually varies,
    which hides a feature the surrogate would need.
    """
    if "input_adults_WG" not in dataframe.columns:
        return False
    heterozygote_inputs = dataframe["input_adults_WG"].to_numpy()
    if not (heterozygote_inputs > 0).any():
        return False
    fraction = float((heterozygote_inputs > 0).mean())
    warnings.warn(
        f"{label} has WG input adults in {fraction:.1%} of its rows, but "
        "homozygotes_only drops that column. The surrogate will not see a "
        "feature the data varies. Regenerate the data with "
        "generate_training_data.py --homozygotes-only.",
        stacklevel=2,
    )
    return True


def _load_inputs_outputs(
    dataframe: pd.DataFrame,
    with_genetics: bool,
    homozygotes_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Split an input-output pair dataframe into input/output arrays.

    The genetics-aware files store the three input genotype counts in the
    first columns and the three output counts from column 6 onwards; the
    aggregated files store a single input column and a single output column.

    ``homozygotes_only`` keeps just the WW and GG columns of both blocks, so
    the dataset is a genuine two-genotype one and the net trained on it is
    2-in / 2-out.  It has no effect on the aggregated files, which carry no
    genotype breakdown to subset.
    """
    if with_genetics:
        inputs = dataframe.iloc[:, :3].to_numpy()
        outputs = dataframe.iloc[:, 6:].to_numpy()
        dataset_shape = (1, 3)
        if homozygotes_only:
            inputs = select_homozygotes(inputs)
            outputs = select_homozygotes(outputs)
            dataset_shape = (1, len(HOMOZYGOTE_INDICES))
    else:
        inputs = dataframe.iloc[:, 0].to_numpy()
        outputs = dataframe.iloc[:, -1].to_numpy()
        dataset_shape = (1, 1)
    return inputs, outputs, dataset_shape


def get_test_train_datasets(
    device: torch.device,
    test_train_seeds: tuple[int, int],
    param_tag: str | None = None,
    homozygotes_only: bool = False,
):
    """Load a separate (genetics-aware) train file and held-out test file.

    ``test_train_seeds`` is ``(test_data_seed, train_data_seed)``: the training
    data is loaded from the file generated with the train seed, and the test
    data from the held-out test set generated with the test seed (which carries
    a ``test=True`` marker in its filename).

    ``param_tag`` selects the data generated for a non-default gene-drive
    parameter set (the ``_params=`` marker written by
    ``generate_training_data.py --pareto-file/--pareto-row``); ``None`` selects
    the default-parameter data. Both the train and test files must carry the
    same tag, since a surrogate is only meaningful for one drive design.

    ``homozygotes_only`` does two jobs at once: it selects the heterozygote-free
    data (the ``_homozygotes=True`` marker written by
    ``generate_training_data.py --homozygotes-only``) *and* it drops the WG
    columns, so the datasets are two-genotype ones. The two always travel
    together, which is why one flag covers both."""

    train_dataframe = pd.read_csv(
        input_output_pairs_path(
            test_train_seeds[1],
            with_genetics=True,
            param_tag=param_tag,
            homozygotes_only=homozygotes_only,
        )
    )
    if homozygotes_only:
        warn_if_heterozygote_inputs_present(
            train_dataframe, label=f"training data (seed={test_train_seeds[1]})"
        )
    train_inputs, train_outputs, dataset_shape = _load_inputs_outputs(
        train_dataframe, with_genetics=True, homozygotes_only=homozygotes_only
    )

    if device is None:
        device = torch.device("cpu")

    scalings = np.array([train_inputs.max(), train_outputs.max()])

    train_dataset = InputOutputDataset(
        train_inputs,
        train_outputs,
        dataset_shape,
        scalings=scalings,
    )

    test_dataframe = pd.read_csv(
        input_output_pairs_path(
            test_train_seeds[0],
            with_genetics=True,
            test=True,
            param_tag=param_tag,
            homozygotes_only=homozygotes_only,
        )
    )
    test_inputs, test_outputs, _ = _load_inputs_outputs(
        test_dataframe, with_genetics=True, homozygotes_only=homozygotes_only
    )

    test_dataset = InputOutputDataset(
        test_inputs,
        test_outputs,
        dataset_shape,
        scalings=scalings,
    )

    return train_dataset, test_dataset, scalings


class InputOutputDataset(torch.utils.data.Dataset):
    """Custom Torch Dataset for loading input-output pairs from the CSV file. Note that the input-output pairs are normalised, so predictions from the NN will need to be denormalised before they can be used."""

    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        dataset_shape: tuple[int, int],
        scalings: np.ndarray | None = None,
        device: torch.device | None = None,
    ):
        self.dataset_shape = dataset_shape

        if scalings is None:
            self.input_max = inputs.max()
            self.output_max = outputs.max()
        else:
            self.input_max = scalings[0]
            self.output_max = scalings[1]

        self.inputs = torch.tensor(
            inputs / self.input_max, dtype=torch.float32
        ).reshape((-1, self.dataset_shape[1]))
        self.outputs = torch.tensor(
            outputs / self.output_max, dtype=torch.float32
        ).reshape((-1, self.dataset_shape[1]))
        if device is not None:
            self.inputs = self.inputs.to(device)
            self.outputs = self.outputs.to(device)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        output_tensor = self.outputs[idx]
        return input_tensor, output_tensor


def get_test_train_dataloaders(
    batch_size: int = 32,
    with_genetics: bool = False,
    train_fraction: float = 0.95,
    seed: int | None = None,
    data_seed: int = 27,
    test_data_seed: int | None = None,
    param_tag: str | None = None,
    homozygotes_only: bool = False,
    device: torch.device | None = None,
):
    """Build the train and test dataloaders for our model.

    The dataset generated with ``data_seed`` is loaded as the training data.
    When ``test_data_seed`` is ``None`` it is randomly shuffled and split into
    disjoint train and test sets. When ``test_data_seed`` is given, the whole
    dataset is used for training and the held-out test set generated with that
    seed (carrying a ``test=True`` marker in its filename) is loaded for the
    test set instead of splitting.

    Parameters
    ----------
    batch_size : int
        Batch size for both dataloaders.
    with_genetics : bool
        Whether to use the genetics-aware dataset.
    train_fraction : float
        Fraction of data to use for training (remainder is test). Only used
        when ``test_data_seed`` is ``None``.
    seed : int | None
        Random seed for reproducible train/test splits.
    data_seed : int
        Seed identifying the input-output pair file to load as training data.
    test_data_seed : int | None
        If given, load the held-out test set generated with this seed for the
        test dataloader rather than splitting the training data.
    param_tag : str | None
        Marker identifying a non-default gene-drive parameter set, as written
        into the filename by ``generate_training_data.py --pareto-file
        --pareto-row`` (e.g. ``low0p30_high0p50_20260601_101511_row0``). Must
        be given whenever the target dataset was generated for such a parameter
        set, otherwise the default-parameter filename is looked for and no file
        is found. ``None`` selects the default-parameter data.
    homozygotes_only : bool
        Select the heterozygote-free data (the ``_homozygotes=True`` marker
        written by ``generate_training_data.py --homozygotes-only``) *and* drop
        the WG columns, giving two-genotype datasets for a 2-in / 2-out
        surrogate. The two go together, so one flag covers both; training a
        three-genotype net on heterozygote-free data is deliberately not
        reachable this way. To ask instead how much WG mass such a surrogate
        discards, use :func:`report_heterozygote_residual` on the raw frame.
    """
    rng = np.random.default_rng(seed)

    dataset = pd.read_csv(
        input_output_pairs_path(
            data_seed,
            with_genetics=with_genetics,
            param_tag=param_tag,
            homozygotes_only=homozygotes_only,
        )
    )
    if homozygotes_only:
        warn_if_heterozygote_inputs_present(
            dataset, label=f"training data (seed={data_seed})"
        )
    inputs, outputs, dataset_shape = _load_inputs_outputs(
        dataset, with_genetics, homozygotes_only=homozygotes_only
    )

    if device is None:
        device = torch.device("cpu")
    # Compute scalings from the full training dataset so they are independent of
    # the split / held-out test set.
    scalings = np.array([inputs.max(), outputs.max()])

    if test_data_seed is None:
        n = len(inputs)
        indices = rng.permutation(n)
        split = int(n * train_fraction)
        train_idx, test_idx = indices[:split], indices[split:]
        train_inputs, train_outputs = inputs[train_idx], outputs[train_idx]
        test_inputs, test_outputs = inputs[test_idx], outputs[test_idx]
    else:
        train_inputs, train_outputs = inputs, outputs
        test_dataset = pd.read_csv(
            input_output_pairs_path(
                test_data_seed,
                with_genetics=with_genetics,
                test=True,
                param_tag=param_tag,
                homozygotes_only=homozygotes_only,
            )
        )
        test_inputs, test_outputs, _ = _load_inputs_outputs(
            test_dataset, with_genetics, homozygotes_only=homozygotes_only
        )

    train_dataset = InputOutputDataset(
        train_inputs,
        train_outputs,
        dataset_shape,
        scalings=scalings,
        device=device,
    )

    test_dataset = InputOutputDataset(
        test_inputs,
        test_outputs,
        dataset_shape,
        scalings=scalings,
        device=device,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_dataloader, test_dataloader, scalings
