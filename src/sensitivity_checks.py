"""Conducts sobol sensitivity analysis (e.g. global sensitivity analysis) and local sensitivity analysis around the drive architecture with the highest score. As before, it can be pointed at the correct dir with --prop-gg-low and --prop-gg-high. Other flags control the evaluation budget and number of workers used, so make use of these if running on a machine with constrained resources."""

import argparse
import glob
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from Gene_Drive_Optimisation.gene_drive_objective import (
    OBJECTIVE_NAMES,
    PARAM_NAMES,
    gene_drive_objectives,
    rank_pareto_points,
)
from Gene_Drive_Optimisation.optimise_gene_drives import (
    PROP_GG_HIGH,
    PROP_GG_LOW,
)
from Gene_Drive_Optimisation.plot_gene_drives import plot_outputs
from Gene_Drive_Optimisation.sensitivity_analysis import (
    LOCAL_ANALYSIS_TAG,
    SOBOL_ANALYSIS_TAG,
    compute_local_sensitivity,
    compute_sobol_indices,
    local_sensitivity_from_dataframe,
    local_sensitivity_to_dataframe,
    sobol_indices_from_dataframe,
    sobol_indices_to_dataframe,
)
from paths import (
    IBM_OUTPUT_DIR,
    IMAGES_DIR,
    OPTIMISATION_MODELS_DIR,
    sensitivity_results_path,
)

#: Fixed seed for the Sobol design's IBM evaluations.  A single seed (rather
#: than fresh randomness) keeps the sampled response a reproducible, quasi-
#: deterministic function of the parameters, as the Sobol estimator assumes.
_SOBOL_IBM_SEED = 20240607

#: Objective columns scored when choosing the point to compute the local
#: sensitivity around.  These are the IBM-verified means -- re-evaluated with
#: many replicates once each BO run has finished -- rather than the raw
#: surrogate values the optimiser worked with, so that points coming from runs
#: whose surrogates differ in accuracy are ranked on a common footing.
_VERIFIED_OBJECTIVE_COLS = [f"{name} [verified mean]" for name in OBJECTIVE_NAMES]

_DIR_RE = re.compile(r"^low(\d+p\d+)_high(\d+p\d+)$")
_FILE_RE = re.compile(r"_low(\d+p\d+)_high(\d+p\d+)_(\d{8}_\d{6})\.pt$")
_TS_RE = re.compile(r"_(\d{8}_\d{6})\.pt$")


def _unit_float(value: str) -> float:
    x = float(value)
    if not (0.0 < x < 1.0):
        raise argparse.ArgumentTypeError(f"{value!r} must be strictly between 0 and 1")
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sensitivity analyses on the underlying (IBM) gene-drive objective "
            "for a chosen underdominance gene-drive threshold pair.  The BO GP "
            "surrogate is used only for the separate surrogate-validation figure; "
            "the Sobol and local-sensitivity analyses evaluate the IBM directly."
        )
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
        "--sobol-samples",
        type=int,
        default=2048,
        dest="sobol_samples",
        help=(
            "SALib base sample size N for the Sobol analysis (use a power of "
            "two). The design has N*(2D+2) points with second-order indices "
            "(D=9), so this sets how finely the input space is explored. For a "
            "stochastic model the Monte-Carlo integration error over distinct "
            "points is the dominant error, so prefer spending budget here (more "
            "points) over --sobol-replicates. Default: 256"
        ),
    )
    parser.add_argument(
        "--sobol-replicates",
        type=int,
        default=1,
        dest="sobol_replicates",
        help=(
            "IBM replicates averaged per Sobol design point. By default this is set to one, as in increasing the number above one will make the total variance seem smaller than it actually is."
            "Default: 1"
        ),
    )
    parser.add_argument(
        "--no-second-order",
        action="store_true",
        dest="no_second_order",
        help=(
            "Skip second-order Sobol interaction indices. Halves the Sobol "
            "evaluation budget (N*(D+2) instead of N*(2D+2)) and omits the "
            "second-order figure."
        ),
    )
    parser.add_argument(
        "--fd-step",
        type=float,
        default=0.1,
        dest="fd_step",
        help=(
            "Central finite-difference half-step (in unit parameter space) for "
            "the local sensitivity Jacobian. Coordinates whose step is clamped "
            "by a bound on one side fall back to a one-sided first-order "
            "difference. Default: 0.1"
        ),
    )
    parser.add_argument(
        "--fd-replicates",
        type=int,
        default=100,
        dest="fd_replicates",
        help=("IBM replicates per finite-difference stencil point. Default: 100"),
    )
    parser.add_argument(
        "--local-only",
        dest="local_only",
        action="store_true",
        help=(
            "If True, only run the local sensitivity analysis and skip the Sobol "
            "analysis. Default: False"
        ),
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        nargs="+",
        default=None,
        dest="from_csv",
        metavar="CSV",
        help=(
            "Redraw the figures from results saved by an earlier run instead "
            "of evaluating the IBM at all. Pass the Sobol CSV, the local-"
            "sensitivity CSV, or both (each file says which analysis it "
            "holds); only the figures whose data is supplied are redrawn. "
            "Every other argument is ignored."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        dest="n_workers",
        help=(
            "Number of worker processes for the IBM evaluation pool. Each IBM "
            "call fans out to this many parallel processes; the default (None) "
            "uses every core the OS scheduler allows, which can swamp a laptop. "
            "Set e.g. --n-workers 2 to cap parallelism. Default: all available "
            "cores."
        ),
    )
    args = parser.parse_args()
    if args.n_workers is not None and args.n_workers < 1:
        parser.error(f"--n-workers ({args.n_workers}) must be a positive integer")
    if args.prop_gg_low >= args.prop_gg_high:
        parser.error(
            f"--prop-gg-low ({args.prop_gg_low}) must be less than "
            f"--prop-gg-high ({args.prop_gg_high})"
        )
    return args


def find_matching_models(low_tag: str, high_tag: str) -> list[str]:
    """Return BO model files for a threshold pair, newest-layout first.

    Searches the new-style ``low*_high*`` subdirectory layout, then falls back
    to the legacy flat layout whose threshold tags live in the filename.
    """
    model_files = glob.glob(
        str(
            OPTIMISATION_MODELS_DIR
            / f"low{low_tag}_high{high_tag}"
            / "gene_drive_bo_model_*.pt"
        )
    )
    if not model_files:
        model_files = glob.glob(
            str(
                OPTIMISATION_MODELS_DIR
                / f"gene_drive_bo_model_*_low{low_tag}_high{high_tag}_*.pt"
            )
        )
    return model_files


def derive_pareto_csv(model_path: Path) -> pd.DataFrame:
    """Load the Pareto-frontier CSV matching a given BO model file.

    Handles both the new-style subdirectory layout (thresholds in the parent
    directory name) and the legacy flat layout (thresholds in the filename).
    """
    dm = _DIR_RE.match(model_path.parent.name)
    fm = _FILE_RE.search(model_path.name)

    if dm:
        low_tag, high_tag = dm.group(1), dm.group(2)
        ts_m = _TS_RE.search(model_path.name)
        run_ts = ts_m.group(1) if ts_m else None
        csv_candidates = []
        if run_ts:
            csv_candidates.append(
                IBM_OUTPUT_DIR
                / f"low{low_tag}_high{high_tag}"
                / f"gene_drive_pareto_frontier_points_{run_ts}.csv"
            )
        csv_candidates.append(
            IBM_OUTPUT_DIR
            / f"gene_drive_pareto_frontier_points_low{low_tag}_high{high_tag}.csv"
        )
    elif fm:
        low_tag, high_tag, run_ts = fm.group(1), fm.group(2), fm.group(3)
        csv_candidates = [
            IBM_OUTPUT_DIR
            / f"gene_drive_pareto_frontier_points_low{low_tag}_high{high_tag}_{run_ts}.csv",
            IBM_OUTPUT_DIR
            / f"gene_drive_pareto_frontier_points_low{low_tag}_high{high_tag}.csv",
        ]
    else:
        raise ValueError(f"Cannot parse threshold tags from model path: {model_path}")

    for _csv_path in csv_candidates:
        if _csv_path.exists():
            return pd.read_csv(_csv_path)
    raise FileNotFoundError(
        f"No matching Pareto CSV found for thresholds {low_tag}/{high_tag}"
    )


def find_best_pareto_point(
    low_tag: str, high_tag: str
) -> tuple[torch.Tensor, torch.Tensor, Path, int, float]:
    """Find the best-scoring Pareto point across *every* run for a threshold pair.

    Each BO run writes its own ``gene_drive_pareto_frontier_points_<ts>.csv``
    into ``data/ibm_output/low<x>_high<y>/``, so that directory holds one
    frontier per replicate of the same optimisation problem.  This pools the
    points of every one of those runs and returns the single highest-scoring
    point, scored by the mean of the two IBM-verified (e.g. average over 20 replicates, rather than the 5 replicates used when conducting the search) objectives.

    Falls back to the legacy flat layout (a single pooled CSV per threshold
    pair, thresholds in the filename) when no per-run subdirectory exists.

    Returns
    -------
    tuple
        ``(X, Y, csv_path, row, score)`` where *X* is the ``(1, 9)`` parameter
        vector and *Y* the ``(1, 2)`` verified objective values of the selected
        point, both ready to pass to :func:`compute_local_sensitivity`, and
        ``csv_path``/``row``/``score`` record where it came from.
    """
    threshold_dir = IBM_OUTPUT_DIR / f"low{low_tag}_high{high_tag}"
    if any(threshold_dir.glob("gene_drive_pareto_frontier_points_*.csv")):
        ((csv_path, row, score),) = rank_pareto_points(threshold_dir, top_n=1)
        frontier = pd.read_csv(csv_path)
    else:
        csv_path = (
            IBM_OUTPUT_DIR
            / f"gene_drive_pareto_frontier_points_low{low_tag}_high{high_tag}.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No Pareto-frontier CSVs found for thresholds "
                f"low{low_tag}/high{high_tag}: neither per-run files in "
                f"{threshold_dir} nor a legacy pooled file at {csv_path}."
            )
        frontier = pd.read_csv(csv_path)
        missing = [c for c in _VERIFIED_OBJECTIVE_COLS if c not in frontier.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing the objective column(s) {missing}; "
                f"expected all of {_VERIFIED_OBJECTIVE_COLS}."
            )
        scores = frontier[_VERIFIED_OBJECTIVE_COLS].to_numpy(dtype=float).mean(axis=1)
        row = int(scores.argmax())
        score = float(scores[row])

    point = frontier.iloc[[row]]
    X = torch.tensor(point[PARAM_NAMES].to_numpy(dtype=float), dtype=torch.double)
    Y = torch.tensor(
        point[_VERIFIED_OBJECTIVE_COLS].to_numpy(dtype=float), dtype=torch.double
    )
    return X, Y, Path(csv_path), row, score


def save_sobol_indices(results: dict, args: argparse.Namespace, timestamp: str) -> Path:
    """Write the computed Sobol indices beside the figures they produced.

    The design behind them cost one IBM evaluation per point, so the numbers
    are saved with the run settings that produced them and the figures can be
    redrawn later with ``--from-csv``.
    """
    csv_path = sensitivity_results_path(
        SOBOL_ANALYSIS_TAG, args.prop_gg_low, args.prop_gg_high, timestamp
    )
    sobol_indices_to_dataframe(
        results,
        prop_gg_low=args.prop_gg_low,
        prop_gg_high=args.prop_gg_high,
        n_samples=args.sobol_samples,
        n_replicates=args.sobol_replicates,
        second_order=not args.no_second_order,
        ibm_seed=_SOBOL_IBM_SEED,
    ).to_csv(csv_path, index=False)
    return csv_path


def save_local_sensitivity(
    results: dict, args: argparse.Namespace, timestamp: str
) -> Path:
    """Write the finite-difference Jacobian beside the figure it produced."""
    csv_path = sensitivity_results_path(
        LOCAL_ANALYSIS_TAG, args.prop_gg_low, args.prop_gg_high, timestamp
    )
    local_sensitivity_to_dataframe(
        results,
        prop_gg_low=args.prop_gg_low,
        prop_gg_high=args.prop_gg_high,
        step=args.fd_step,
        n_replicates=args.fd_replicates,
    ).to_csv(csv_path, index=False)
    return csv_path


def threshold_tags_from_results(csv_path: Path, frame: pd.DataFrame) -> tuple[str, str]:
    """Recover the ``low``/``high`` filename tags a saved result belongs to.

    Taken from the metadata columns the analysis wrote, falling back to the
    threshold directory the file sits in.  The tags decide the figure
    filenames, so a replot has to land on the same pair the run did.
    """
    if {"prop_gg_low", "prop_gg_high"} <= set(frame.columns):
        low = float(frame["prop_gg_low"].iloc[0])
        high = float(frame["prop_gg_high"].iloc[0])
        return f"{low:.2f}".replace(".", "p"), f"{high:.2f}".replace(".", "p")

    match = _DIR_RE.match(csv_path.parent.name)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(
        f"Cannot tell which release thresholds {csv_path} belongs to: it has no "
        "prop_gg_low/prop_gg_high columns and does not sit in a low*_high* "
        "directory."
    )


def replot_from_csv(csv_paths: list[Path]) -> None:
    """Redraw the sensitivity figures from saved results, simulating nothing.

    Each file names its own analysis in its ``analysis`` column, so the Sobol
    and local-sensitivity CSVs can be passed in either order, or one alone --
    only the figures whose data is supplied are redrawn.  Every file must
    describe the same threshold pair, since all of them share one set of
    figure filenames.
    """
    results: dict[str, dict] = {}
    tags: set[tuple[str, str]] = set()

    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if "analysis" not in frame.columns:
            raise ValueError(
                f"{csv_path} has no 'analysis' column; it was not written by a "
                "sensitivity run."
            )
        tag = str(frame["analysis"].iloc[0])
        if tag not in (SOBOL_ANALYSIS_TAG, LOCAL_ANALYSIS_TAG):
            raise ValueError(
                f"{csv_path} holds unknown results {tag!r}; expected "
                f"{SOBOL_ANALYSIS_TAG!r} or {LOCAL_ANALYSIS_TAG!r}."
            )
        if (
            tag in results
        ):  # triggers if, as you loop over the csv's passed in, one of them is a duplicate of a previous one
            raise ValueError(f"Two {tag!r} files passed; pass at most one of each.")

        results[tag] = (
            sobol_indices_from_dataframe(frame)
            if tag == SOBOL_ANALYSIS_TAG
            else local_sensitivity_from_dataframe(frame)
        )
        tags.add(threshold_tags_from_results(csv_path, frame))
        print(f"Loaded {tag} from {csv_path} ({len(frame)} rows).")

    if len(tags) > 1:
        raise ValueError(
            "The files passed describe different threshold pairs "
            f"({sorted(tags)}); they would overwrite one another's figures."
        )

    (low_tag, high_tag) = tags.pop()
    plot_outputs(
        sobol_results=results.get(SOBOL_ANALYSIS_TAG),
        local_sensitivity_results=results.get(LOCAL_ANALYSIS_TAG),
        filename_suffix=f"_low{low_tag}_high{high_tag}",
    )
    print(f"Redrew figures for low{low_tag}/high{high_tag} in {IMAGES_DIR}.")


if __name__ == "__main__":
    args = parse_args()

    if args.from_csv is not None:
        replot_from_csv(args.from_csv)
        raise SystemExit(0)

    low_tag = f"{args.prop_gg_low:.2f}".replace(".", "p")
    high_tag = f"{args.prop_gg_high:.2f}".replace(".", "p")

    # One timestamp for the whole run, so the analyses it saves stay a matched
    # set rather than two files a second apart.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.local_only:
        print(
            f"Running only local sensitivity analysis for thresholds "
            f"low={args.prop_gg_low} / high={args.prop_gg_high}."
        )
    else:
        model_files = find_matching_models(low_tag, high_tag)
        if not model_files:
            raise FileNotFoundError(
                f"No BO model files found for thresholds low{low_tag}/high{high_tag} "
                f"in {OPTIMISATION_MODELS_DIR}."
            )

    (
        best_point_X,
        best_point_Y,
        best_point_csv,
        best_point_row,
        best_point_score,
    ) = find_best_pareto_point(low_tag, high_tag)
    print(
        f"\nBest Pareto point across all runs: {best_point_csv.name} row "
        f"{best_point_row} (mean verified objective = {best_point_score:.4f}). "
        "The local sensitivity is computed around this point."
    )

    # 2. Global and local sensitivity analyses on the underlying IBM objective.
    #    Both analyses call the IBM (via gene_drive_objectives) rather than the
    #    GP posterior, for greater confidence that the reported sensitivities
    #    reflect the true model and not surrogate artefacts.
    calc_second_order = not args.no_second_order

    def sobol_objective_fn(X: torch.Tensor) -> torch.Tensor:
        """IBM objective for the Sobol design.

        Uses independent per-point RNG seeds (a fixed master seed only for
        reproducibility), so the estimator targets the sensitivities of the
        mean response by integrating over the IBM's intrinsic randomness.
        """
        Y, _ = gene_drive_objectives(
            X,
            prop_gg_low=args.prop_gg_low,
            prop_gg_high=args.prop_gg_high,
            n_replicates=args.sobol_replicates,
            rng=torch.Generator().manual_seed(_SOBOL_IBM_SEED),
            n_workers=args.n_workers,
            cycles=3,
        )
        return Y

    def local_objective_fn(X: torch.Tensor) -> torch.Tensor:
        """IBM objective for the finite-difference stencil (common random numbers)."""
        Y, _ = gene_drive_objectives(
            X,
            prop_gg_low=args.prop_gg_low,
            prop_gg_high=args.prop_gg_high,
            n_replicates=args.fd_replicates,
            common_seed=None,
            cycles=3,
            n_workers=args.n_workers,
        )
        return Y

    if args.local_only:
        local_sensitivity_results = compute_local_sensitivity(
            local_objective_fn,
            pareto_X=best_point_X,
            pareto_Y=best_point_Y,
            step=args.fd_step,
        )
        local_csv = save_local_sensitivity(
            local_sensitivity_results, args, run_timestamp
        )
        print(f"Saved local sensitivity results to {local_csv}")

        plot_outputs(
            local_sensitivity_results=local_sensitivity_results,
        )
        print(f"Redraw this figure without re-running with:  --from-csv {local_csv}")
    else:
        n_dims = best_point_X.shape[1]
        n_design = args.sobol_samples * (
            2 * n_dims + 2 if calc_second_order else n_dims + 2
        )
        print(
            f"\nRunning Sobol analysis on the IBM: N={args.sobol_samples}, "
            f"second_order={calc_second_order}, {args.sobol_replicates} replicates/point "
            f"→ {n_design} design points × 2 objectives."
        )
        sobol_results = compute_sobol_indices(
            sobol_objective_fn,
            n_samples=args.sobol_samples,
            calc_second_order=calc_second_order,
            num_resamples=1000,
            conf_level=0.95,
            seed=42,
        )

        print(
            f"\nRunning local sensitivity on the IBM: step={args.fd_step}, "
            f"{args.fd_replicates} replicates/point (common random numbers)."
        )
        local_sensitivity_results = compute_local_sensitivity(
            local_objective_fn,
            pareto_X=best_point_X,
            pareto_Y=best_point_Y,
            step=args.fd_step,
        )

        # 3. Plot, embedding the model id in the output filenames.
        # ts_m = _TS_RE.search(best_model_path.name)
        # run_ts = ts_m.group(1) if ts_m else "unknown"
        # model_id = f"low{low_tag}_high{high_tag}_{run_ts}"
        model_id = f"low{low_tag}_high{high_tag}"

        sobol_csv = save_sobol_indices(sobol_results, args, run_timestamp)
        print(f"Saved Sobol indices to {sobol_csv}")
        local_csv = save_local_sensitivity(
            local_sensitivity_results, args, run_timestamp
        )
        print(f"Saved local sensitivity results to {local_csv}")

        plot_outputs(
            sobol_results=sobol_results,
            local_sensitivity_results=local_sensitivity_results,
            filename_suffix=f"_{model_id}",
        )
        print(
            "Redraw these figures without re-running with:  "
            f"--from-csv {sobol_csv} {local_csv}"
        )
