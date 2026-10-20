import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from IBM_Implementations.agent_based_larval_model import run_simulation
from cpu_resources import available_cpu_count
import multiprocess

#: Names of the nine decision variables, in the column order used by ``X`` and
#: by the Pareto-frontier CSVs.  Defined here — the lowest-level module that
#: needs them — and re-exported by ``optimise_gene_drives`` so the optimiser and
#: anything reading its output can never disagree on the ordering.
PARAM_NAMES = (
    ["conversion_rate"]
    + [f"wg_{param}_fac" for param in ["a", "alpha", "beta", "g"]]
    + [f"gg_{param}_fac" for param in ["a", "alpha", "beta", "g"]]
)

#: Names of the two objectives, in the column order used by ``Y`` and by the
#: Pareto-frontier CSVs.  Defined here for the same reason as ``PARAM_NAMES``,
#: and re-exported by ``optimise_gene_drives``.
OBJECTIVE_NAMES = ["f_1 (low g frequency)", "f_2 (high g frequency)"]

#: Matches the ``low<x>_high<y>`` subdirectory the BO runs write their Pareto
#: CSVs into, and the ``YYYYMMDD_HHMMSS`` run timestamp in the filename.
_THRESHOLD_DIR_RE = re.compile(r"^low[0-9p]+_high[0-9p]+$")
_RUN_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
_PARETO_STEM_PREFIX = "gene_drive_pareto_frontier_points_"


def gene_drive_objectives(
    X: Tensor,
    *,
    prop_gg_low: float,
    prop_gg_high: float,
    n_replicates: int = 5,
    cycles: int = 10,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    rng: torch.Generator | None = None,
    n_workers: int | None = None,
    common_seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate both gene-drive objectives for a batch of candidate points.

    Parameters
    ----------
    X:
        Tensor of shape ``(batch_size, 9)`` where column 0 is
        ``conversion_rate`` and columns 1–8 are the eight penalty factors
        (``wg_a_fac``, ``wg_alpha_fac``, ``wg_beta_fac``, ``wg_g_fac``,
        ``gg_a_fac``, ``gg_alpha_fac``, ``gg_beta_fac``, ``gg_g_fac``).
        Values should lie within ``PARAM_BOUNDS``.
    prop_gg_low:
        Release fraction used for the extinction objective (f1).  Must be < 0.5.
    prop_gg_high:
        Release fraction used for the fixation objective (f2).  Must be > 0.5.
    n_replicates:
        IBM replicates averaged per candidate evaluation.
    cycles:
        IBM life-history cycles per replicate.
    fixed_temperature:
        Simulation temperature in °C.
    feeding_type:
        "interference" or "exploitative".
    rng:
        Optional ``torch.Generator`` used to draw per-row base seeds.
        Ignored when *common_seed* is set.
    n_workers:
        Number of worker processes for ``multiprocess.Pool``.
        ``None`` uses :func:`cpu_resources.available_cpu_count` (the SLURM-allocated
        core count under SLURM, otherwise the full machine).
    common_seed:
        When set, **every** row of *X* is evaluated with the same pair of base
        seeds (derived deterministically from *common_seed*), i.e. common random
        numbers across the whole batch.  This makes each objective a
        deterministic function of the parameters for that seed, so the shared
        stochastic noise cancels when differencing rows — required for the
        finite-difference local sensitivity analysis.  When ``None`` (default),
        each row gets an independent seed drawn from *rng*.

    Returns
    -------
    means : Tensor  shape ``(batch_size, 2)``
        Column 0 (f1) = ``1 − mean_GG_freq`` at *prop_gg_low*;
        column 1 (f2) = ``mean_GG_freq`` at *prop_gg_high*.
    variances : Tensor  shape ``(batch_size, 2)``
        Estimated variance of each objective (variance of the mean across
        replicates).
    """
    X_np = X.detach().cpu().numpy()

    if common_seed is not None:
        # Common random numbers: one shared pair of base seeds for every row.
        base_pair = np.random.SeedSequence(common_seed).generate_state(2)
        seed_array = np.tile(base_pair.astype(np.int64), (X_np.shape[0], 1))
    else:
        seed_array = torch.randint(
            0, int(1e6), (X_np.shape[0], 2), generator=rng if rng is not None else None
        ).numpy()

    def _eval_single(args):
        row, seed = args
        conversion_rate = row[0]
        penalties_array = row[1:].reshape([2, -1]).T

        mean_low, std_low = evaluate_gene_drive(
            prop_eggs_genetically_modified=prop_gg_low,
            conversion_rate=conversion_rate,
            penalties_array=penalties_array,
            n_replicates=n_replicates,
            cycles=cycles,
            fixed_temperature=fixed_temperature,
            feeding_type=feeding_type,
            base_seed=int(seed[0]),
        )
        mean_high, std_high = evaluate_gene_drive(
            prop_eggs_genetically_modified=prop_gg_high,
            conversion_rate=conversion_rate,
            penalties_array=penalties_array,
            n_replicates=n_replicates,
            cycles=cycles,
            fixed_temperature=fixed_temperature,
            feeding_type=feeding_type,
            base_seed=int(seed[1]),
        )
        _MIN_NOISE = 1e-6
        return (
            [1.0 - mean_low, mean_high],
            [
                max(std_low**2 / n_replicates, _MIN_NOISE),
                max(std_high**2 / n_replicates, _MIN_NOISE),
            ],
        )

    if n_workers is None:
        n_workers = available_cpu_count()

    with multiprocess.Pool(n_workers) as pool:
        results = pool.map(_eval_single, zip(X_np, seed_array))

    means, variances = zip(*results)
    return (
        torch.tensor(list(means), dtype=X.dtype),
        torch.tensor(list(variances), dtype=X.dtype),
    )


def evaluate_gene_drive(
    prop_eggs_genetically_modified: float,
    conversion_rate: float,
    penalties_array: np.ndarray,
    *,
    n_replicates: int = 5,
    cycles: int = 10,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    base_seed: int = 0,
) -> tuple[float, float]:
    """Run ``n_replicates`` IBM simulations and return (mean, std) G gene frequency.

    Parameters
    ----------
    prop_eggs_genetically_modified:
        Fraction of initial eggs that carry two G alleles (GG).
    conversion_rate:
        Probability that a WG larva converts to GG each generation (gene drive).
    penalties_array:
        Array of shape ``(4, 2)`` containing multiplicative fitness penalties for
        WG (column 0) and GG (column 1) genotypes.  Rows correspond to
        ``a_fac``, ``alpha_fac``, ``beta_fac``, and ``g_fac`` respectively.
        WW genotype is always kept at 1.0 (wild-type).
    n_replicates:
        Number of independent stochastic IBM runs to average over.
    cycles:
        Number of life-history cycles to simulate per replicate.
    fixed_temperature:
        Temperature (°C); must be one of [20, 22, 24, 26, 28, 30].
    feeding_type:
        "interference" or "exploitative".
    base_seed:
        Base RNG seed; replicate *i* uses ``base_seed + i``.

    Returns
    -------
    mean_g_freq, std_g_freq
    """
    genotype_penalties = _build_genotype_penalties(penalties_array)

    def _run(seed: int) -> float:
        state = run_simulation(
            seed=seed,
            cycles=cycles,
            fixed_temperature=fixed_temperature,
            feeding_type=feeding_type,
            random_mating=True,
            genotype_penalties=genotype_penalties,
            prop_eggs_genetically_modified=float(prop_eggs_genetically_modified),
            conversion_rate=float(conversion_rate),
        )
        return _g_gene_frequency(state)

    seeds = [
        int(s.generate_state(1)[0])
        for s in np.random.SeedSequence(base_seed).spawn(n_replicates)
    ]

    freqs = [_run(s) for s in seeds]

    arr = np.array(freqs)
    return float(arr.mean()), float(arr.std())


def _build_genotype_penalties(penalties_array) -> dict:
    """Return the full genotype_penalties dict given an array of penalties for the WG (column 0) and GG (column 1) genotypes."""
    return {
        "WW": {
            "a_fac": 1,
            "alpha_fac": 1,
            "beta_fac": 1.0,
            "g_fac": 1.0,
        },
        "WG": {
            "a_fac": penalties_array[0, 0],
            "alpha_fac": penalties_array[1, 0],
            "beta_fac": penalties_array[2, 0],
            "g_fac": penalties_array[3, 0],
        },
        "GG": {
            "a_fac": penalties_array[0, 1],
            "alpha_fac": penalties_array[1, 1],
            "beta_fac": penalties_array[2, 1],
            "g_fac": penalties_array[3, 1],
        },
    }


def load_gene_drive_parameters(
    pareto_csv: str | Path, row: int
) -> tuple[dict, float, str]:
    """Read one Pareto-frontier point and return it as IBM simulation inputs.

    The BO pipeline (``optimise_gene_drive.py``) writes its Pareto frontier to
    ``data/ibm_output/low<x>_high<y>/gene_drive_pareto_frontier_points_<ts>.csv``,
    each row of which is a complete nine-parameter gene-drive design.  This
    turns one such row back into the arguments the IBM takes, using the same
    column ordering and the same reshape as
    :func:`gene_drive_objectives`, so a surrogate trained on the resulting data
    emulates exactly the drive the optimiser proposed.

    Parameters
    ----------
    pareto_csv:
        Path to a ``gene_drive_pareto_frontier_points_*.csv`` file.
    row:
        Zero-indexed row within that file (row 0 is the first Pareto point).

    Returns
    -------
    genotype_penalties : dict
        Per-genotype, per-parameter penalties, ready to pass to the IBM.
    conversion_rate : float
        The WG -> GG gene-drive conversion probability from the same row.
    param_tag : str
        Filename-safe marker identifying this parameter set, e.g.
        ``low0p50_high0p70_20260612_084001_row0``.  Pass to
        :func:`paths.input_output_pairs_path` so data generated with these
        parameters is never confused with the default-parameter data.
    """
    pareto_csv = Path(pareto_csv)
    dataframe = pd.read_csv(pareto_csv)

    missing = [name for name in PARAM_NAMES if name not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{pareto_csv} is missing the gene-drive parameter column(s) "
            f"{missing}; expected all of {PARAM_NAMES}."
        )

    if not -len(dataframe) <= row < len(dataframe):
        raise IndexError(
            f"Row {row} is out of range for {pareto_csv}, which has "
            f"{len(dataframe)} Pareto point(s) (valid rows: 0 to "
            f"{len(dataframe) - 1})."
        )

    # Select by name, not position: the CSV carries objective and verification
    # columns after the parameters.
    row_values = dataframe.iloc[row][PARAM_NAMES].to_numpy(dtype=float)

    conversion_rate = float(row_values[0])
    penalties_array = row_values[1:].reshape([2, -1]).T
    genotype_penalties = _build_genotype_penalties(penalties_array)

    return genotype_penalties, conversion_rate, _build_param_tag(pareto_csv, row)


def _build_param_tag(pareto_csv: Path, row: int) -> str:
    """Build a filename-safe tag identifying a Pareto point by its provenance."""
    parts = []

    if _THRESHOLD_DIR_RE.match(pareto_csv.parent.name):
        parts.append(pareto_csv.parent.name)

    stem = pareto_csv.stem
    timestamp_match = _RUN_TIMESTAMP_RE.search(stem)
    if timestamp_match:
        parts.append(timestamp_match.group(1))

    if not parts:
        # Legacy flat layout, or a CSV named some other way entirely: fall back
        # to the filename itself so the tag still identifies its source.
        fallback = stem.removeprefix(_PARETO_STEM_PREFIX) or stem
        parts.append(re.sub(r"[^0-9A-Za-z]+", "_", fallback).strip("_"))

    return "_".join(parts) + f"_row{row}"


def rank_pareto_points(
    pareto_dir: str | Path,
    *,
    top_n: int = 5,
    use_verified: bool = True,
) -> list[tuple[Path, int, float]]:
    """Rank every Pareto point in a directory of optimisation replicates.

    A single BO run writes one ``gene_drive_pareto_frontier_points_<ts>.csv``
    into ``data/ibm_output/low<x>_high<y>/``, so that directory holds one file
    per replicate of the same optimisation problem.  This pools the points of
    every replicate and orders them by their average objective score -- the mean
    of the extinction objective (f_1) and the fixation objective (f_2) -- which
    is the same scalarisation :func:`plot_best_parameters` ranks by, so the
    drives selected here are the ones that figure calls best.

    Parameters
    ----------
    pareto_dir:
        Directory of ``gene_drive_pareto_frontier_points_*.csv`` files.
    top_n:
        How many of the highest-scoring points to return.  Capped at the number
        of points available across the whole directory.
    use_verified:
        Score by the IBM-verified objective means (default), which are
        re-evaluated with many replicates once the BO loop has finished, rather
        than by the raw surrogate objective values the optimiser worked with.

    Returns
    -------
    list[tuple[Path, int, float]]
        ``(csv_path, row, score)`` for each selected point, best first, where
        ``row`` indexes into ``csv_path`` exactly as
        :func:`load_gene_drive_parameters` expects.  Ties are broken by filename
        and then by row, so the ordering is deterministic.
    """
    pareto_dir = Path(pareto_dir)
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1; got {top_n}.")

    csv_paths = sorted(pareto_dir.glob(f"{_PARETO_STEM_PREFIX}*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No {_PARETO_STEM_PREFIX}*.csv files found in {pareto_dir}; expected "
            "a directory of per-replicate Pareto frontiers written by "
            "optimise_gene_drive.py."
        )

    if use_verified:
        objective_columns = [f"{name} [verified mean]" for name in OBJECTIVE_NAMES]
    else:
        objective_columns = list(OBJECTIVE_NAMES)

    scored: list[tuple[float, Path, int]] = []
    for csv_path in csv_paths:
        dataframe = pd.read_csv(csv_path)
        missing = [name for name in objective_columns if name not in dataframe.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing the objective column(s) {missing}; "
                f"expected all of {objective_columns}."
            )
        scores = dataframe[objective_columns].to_numpy(dtype=float).mean(axis=1)
        scored.extend((float(score), csv_path, row) for row, score in enumerate(scores))

    # Highest score first; the remaining keys only make ties reproducible.
    scored.sort(key=lambda point: (-point[0], str(point[1]), point[2]))

    return [(csv_path, row, score) for score, csv_path, row in scored[:top_n]]


def _g_gene_frequency(state: dict) -> float:
    """Compute the frequency of the G gene among adults in the last completed cycle."""
    sp = state["stage_pop_including_genotypes"]
    last = sp.iloc[-1]
    ww = float(last.get("Adults_WW", 0) or 0)
    wg = float(last.get("Adults_WG", 0) or 0)
    gg = float(last.get("Adults_GG", 0) or 0)
    total_alleles = ww + wg + gg
    if total_alleles == 0:
        return 0.0
    return (2 * gg + wg) / (2 * total_alleles)
