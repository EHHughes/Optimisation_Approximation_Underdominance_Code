"""BoTorch wrapper for optimising underdominance gene-drive release strategies.

Decision variables (9D, see PARAM_BOUNDS)
-----------------------------------------
X[:, 0]  conversion_rate  – WG → GG gene-drive conversion probability  [0, 1]
X[:, 1]  wg_a_fac         – a-parameter penalty for WG larvae           [0, 1]
X[:, 2]  wg_alpha_fac     – alpha-parameter penalty for WG larvae       [0, 1]
X[:, 3]  wg_beta_fac      – beta-parameter penalty for WG larvae        [0, 1]
X[:, 4]  wg_g_fac         – g-parameter penalty for WG larvae           [0, 1]
X[:, 5]  gg_a_fac         – a-parameter penalty for GG larvae           [0, 1]
X[:, 6]  gg_alpha_fac     – alpha-parameter penalty for GG larvae       [0, 1]
X[:, 7]  gg_beta_fac      – beta-parameter penalty for GG larvae        [0, 1]
X[:, 8]  gg_g_fac         – g-parameter penalty for GG larvae           [0, 1]

Objectives (both maximised simultaneously)
------------------------------------------
f1  extinction objective  = 1 − mean_GG_freq(prop_GG = PROP_GG_LOW)
    The drive should go to extinction when the initial release is below the
    threshold (GG allele frequency → 0).

f2  fixation objective    = mean_GG_freq(prop_GG = PROP_GG_HIGH)
    The drive should reach fixation when the initial release is above the
    threshold (GG allele frequency → 1).

Both objectives are evaluated at the **final** simulated cycle, averaged over
``n_replicates`` stochastic IBM runs.  Multi-objective Bayesian optimisation
(qLogNoisyExpectedHypervolumeImprovement) is used to explore the Pareto front.

Typical usage
-------------
    from optimise_gene_drives import optimise_gene_drive
    pareto_X, pareto_Y, verified_Y, verified_Yvar, model, _ = optimise_gene_drive(n_init=10, n_bo_steps=20)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from paths import OPTIMISATION_MODELS_DIR
from torch import Tensor

from .gene_drive_objective import OBJECTIVE_NAMES as _OBJ_NAMES
from .gene_drive_objective import PARAM_NAMES as _PARAM_NAMES
from .gene_drive_objective import gene_drive_objectives

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

#: Lower and upper bounds for each decision variable (shape 2 × d).
PARAM_BOUNDS: Tensor = torch.tensor(
    [
        9 * [0.0],
        9 * [1.0],
    ],
    dtype=torch.double,
)

#: Fixed release fraction used to evaluate the *extinction* objective (f1).
#: Must be < 0.5.
PROP_GG_LOW: float = 0.3

#: Fixed release fraction used to evaluate the *fixation* objective (f2).
#: Must be > 0.5.
PROP_GG_HIGH: float = 0.5

#: Reference point in objective space (must be strictly dominated by every
#: feasible point for the hypervolume calculation to be meaningful).
_REF_POINT: Tensor = torch.tensor([-0.01, -0.01], dtype=torch.double)


def optimise_gene_drive(
    n_init: int = 10,
    n_bo_steps: int = 20,
    n_replicates: int = 5,
    n_verify_replicates: int = 20,
    cycles: int = 10,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    base_seed: int = 0,
    bounds: Tensor | None = None,
    prop_gg_low: float = PROP_GG_LOW,
    prop_gg_high: float = PROP_GG_HIGH,
    ref_point: Tensor | None = None,
    verbose: bool = True,
    return_history: bool = False,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, ModelListGP, list[tuple[Tensor, Tensor]] | None
]:
    """Run a multi-objective Bayesian optimisation loop for underdominance gene drives.

    Starts with a Sobol quasi-random initial design, then iterates with
    qLogNoisyExpectedHypervolumeImprovement to jointly maximise:

    * f1 – extinction objective (drive dies out below the threshold)
    * f2 – fixation objective   (drive fixes above the threshold)

    After the BO loop converges, each Pareto-optimal point is re-evaluated with
    ``n_verify_replicates`` IBM runs to produce a high-fidelity estimate of the
    true objective values.

    Parameters
    ----------
    n_init:
        Number of initial quasi-random evaluations.
    n_bo_steps:
        Number of sequential BO iterations after the initial design.
    n_replicates:
        IBM replicates per candidate evaluation during the BO loop.
    n_verify_replicates:
        IBM replicates used to verify each final Pareto point.
    cycles:
        IBM life-history cycles per replicate.
    fixed_temperature:
        Simulation temperature in °C.
    feeding_type:
        "interference" or "exploitative".
    base_seed:
        Base random seed.
    bounds:
        Optional override for ``PARAM_BOUNDS`` (shape ``2 × d``).
    prop_gg_low:
        Release fraction for the extinction objective.  Must be < 0.5.
    prop_gg_high:
        Release fraction for the fixation objective.  Must be > 0.5.
    ref_point:
        Reference point in objective space for hypervolume calculation.
        Defaults to ``_REF_POINT = [-0.01, -0.01]``.
    verbose:
        If True, print progress each step.
    return_history:
        If True, the sixth return value is a list of ``(pareto_X, pareto_Y)``
        snapshots — one per completed BO step — suitable for passing directly
        to :func:`plot_pareto_grid` as the ``history`` argument.

    Returns
    -------
    pareto_X : Tensor  shape (n_pareto, d)
        Input points on the Pareto front.
    pareto_Y : Tensor  shape (n_pareto, 2)
        Objective values from the BO loop ``[f1_extinction, f2_fixation]``.
    verified_pareto_Y : Tensor  shape (n_pareto, 2)
        High-fidelity mean objectives re-evaluated with ``n_verify_replicates``.
    verified_pareto_Yvar : Tensor  shape (n_pareto, 2)
        Variance of each verified objective mean.
    model : ModelListGP
        Fitted multi-output GP model at the end of the run.
    history : list[tuple[Tensor, Tensor]] | None
        Per-step Pareto-front snapshots when ``return_history=True``,
        otherwise ``None``.
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    if ref_point is None:
        ref_point = _REF_POINT

    bounds_double = bounds.to(dtype=torch.double)
    ref_point_double = ref_point.to(dtype=torch.double)

    rng = torch.Generator().manual_seed(base_seed)

    def _eval(X: Tensor) -> tuple[Tensor, Tensor]:
        Y, Yvar = gene_drive_objectives(
            X,
            prop_gg_low=prop_gg_low,
            prop_gg_high=prop_gg_high,
            n_replicates=n_replicates,
            cycles=cycles,
            fixed_temperature=fixed_temperature,
            feeding_type=feeding_type,
            rng=rng,
        )
        return Y.to(dtype=torch.double), Yvar.to(dtype=torch.double)

    history: list[tuple[Tensor, Tensor]] | None = [] if return_history else None

    def _record_history(step: int, px: Tensor, py: Tensor) -> None:
        if history is not None:
            history.append((px, py))

    pareto_X, pareto_Y, model = optimise_objectives(
        objective_fn=_eval,
        bounds=bounds_double,
        ref_point=ref_point_double,
        n_init=n_init,
        n_bo_steps=n_bo_steps,
        base_seed=base_seed,
        verbose=verbose,
        step_callback=_record_history if return_history else None,
    )

    verified_pareto_Y, verified_pareto_Yvar = verify_pareto_front(
        pareto_X,
        n_verify_replicates=n_verify_replicates,
        cycles=cycles,
        fixed_temperature=fixed_temperature,
        feeding_type=feeding_type,
        prop_gg_low=prop_gg_low,
        prop_gg_high=prop_gg_high,
        base_seed=base_seed + 1,
        verbose=verbose,
    )

    return pareto_X, pareto_Y, verified_pareto_Y, verified_pareto_Yvar, model, history


def verify_pareto_front(
    pareto_X: Tensor,
    n_verify_replicates: int = 20,
    cycles: int = 10,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    prop_gg_low: float = PROP_GG_LOW,
    prop_gg_high: float = PROP_GG_HIGH,
    base_seed: int = 0,
    verbose: bool = True,
) -> tuple[Tensor, Tensor]:
    """Re-evaluate each Pareto-optimal point with a high-fidelity replicate budget.

    Parameters
    ----------
    pareto_X:
        Decision variables for each Pareto point, shape ``(n_pareto, d)``.
    n_verify_replicates:
        Number of IBM replicates per point.  Should be larger than the value
        used during the BO loop to reduce stochastic noise on the estimates.
    cycles, fixed_temperature, feeding_type, prop_gg_low, prop_gg_high:
        Passed directly to :func:`gene_drive_objectives`.
    base_seed:
        Seed for the verification RNG.  Should differ from the BO seed to
        avoid correlated estimates.
    verbose:
        If True, print the verified objective values for each point.

    Returns
    -------
    verified_Y : Tensor  shape (n_pareto, 2)
        Mean objective values ``[f1_extinction, f2_fixation]`` over
        ``n_verify_replicates`` runs.
    verified_Yvar : Tensor  shape (n_pareto, 2)
        Variance of each objective mean (variance of the mean across replicates).
    """
    verified_Y, verified_Yvar = gene_drive_objectives(
        pareto_X,
        prop_gg_low=prop_gg_low,
        prop_gg_high=prop_gg_high,
        n_replicates=n_verify_replicates,
        cycles=cycles,
        fixed_temperature=fixed_temperature,
        feeding_type=feeding_type,
        rng=torch.Generator().manual_seed(base_seed),
    )
    verified_Y = verified_Y.to(dtype=torch.double)
    verified_Yvar = verified_Yvar.to(dtype=torch.double)

    if verbose:
        print(f"\nVerification complete ({n_verify_replicates} replicates per point):")
        for i in range(pareto_X.shape[0]):
            f1_std = verified_Yvar[i, 0].item() ** 0.5
            f2_std = verified_Yvar[i, 1].item() ** 0.5
            print(
                f"  [{i}]  "
                f"f1(extinction) = {verified_Y[i, 0].item():.4f} ± {2 * f1_std:.4f}  "
                f"f2(fixation)   = {verified_Y[i, 1].item():.4f} ± {2 * f2_std:.4f}"
            )

    return verified_Y, verified_Yvar


def validate_surrogate(
    model: ModelListGP,
    n_validation_points: int = 64,
    n_replicates: int = 20,
    cycles: int = 3,
    fixed_temperature: int = 20,
    feeding_type: str = "interference",
    prop_gg_low: float = PROP_GG_LOW,
    prop_gg_high: float = PROP_GG_HIGH,
    bounds: Tensor | None = None,
    seed: int = 42,
    verbose: bool = True,
    n_workers: int | None = None,
) -> dict:
    """Validate the GP surrogate against ground-truth IBM evaluations on Sobol points.

    Draws ``n_validation_points`` quasi-random Sobol parameter vectors, queries
    the GP posterior mean (surrogate), then evaluates the true IBM stochastic
    model at the same points using a multiprocess pool (via
    :func:`gene_drive_objectives`).  Reports mean and standard deviation of the
    absolute relative error for each objective.

    Parameters
    ----------
    model:
        Fitted :class:`ModelListGP` returned by :func:`optimise_gene_drive`.
    n_validation_points:
        Number of Sobol-sampled validation points.
    n_replicates:
        IBM replicates per point for the ground-truth evaluation.
    cycles, fixed_temperature, feeding_type:
        Passed directly to :func:`gene_drive_objectives`.
    prop_gg_low, prop_gg_high:
        Release fractions for f1 (extinction) and f2 (fixation).
    bounds:
        Parameter bounds, shape ``(2, d)``.  Defaults to ``PARAM_BOUNDS``.
    seed:
        Seed for the validation Sobol engine and the IBM RNG.
    verbose:
        If True, print the validation summary to stdout.
    n_workers:
        Number of worker processes passed to the ground-truth pool.
        ``None`` uses :func:`cpu_resources.available_cpu_count` (the SLURM-allocated
        core count under SLURM, otherwise the full machine).

    Returns
    -------
    dict
        ``'f1_rel_err_mean'``, ``'f1_rel_err_std'``
            Mean and std of |surrogate − truth| / |truth| for f1.
        ``'f2_rel_err_mean'``, ``'f2_rel_err_std'``
            Mean and std of |surrogate − truth| / |truth| for f2.
        ``'surrogate_Y'``
            GP posterior mean at validation points, shape ``(n, 2)``.
        ``'truth_Y'``
            IBM ground-truth means, shape ``(n, 2)``.
        ``'truth_Yvar'``
            IBM ground-truth variances of the mean, shape ``(n, 2)``.
        ``'X'``
            Sobol validation points in the original parameter space, shape ``(n, d)``.
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    bounds = bounds.to(dtype=torch.double)
    d = bounds.shape[1]

    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    X_unit = sobol.draw(n_validation_points).to(dtype=torch.double)
    X = bounds[0] + X_unit * (bounds[1] - bounds[0])

    model.eval()
    with torch.no_grad():
        surrogate_Y = model.posterior(X).mean  # shape (n, 2)

    true_Y, true_Yvar = gene_drive_objectives(
        X,
        prop_gg_low=prop_gg_low,
        prop_gg_high=prop_gg_high,
        n_replicates=n_replicates,
        cycles=cycles,
        fixed_temperature=fixed_temperature,
        feeding_type=feeding_type,
        rng=torch.Generator().manual_seed(seed),
        n_workers=n_workers,
    )
    true_Y = true_Y.to(dtype=torch.double)

    err = (surrogate_Y - true_Y).abs()
    print(err)

    f1_mean = err[:, 0].mean().item()
    f1_std = err[:, 0].std().item()
    f2_mean = err[:, 1].mean().item()
    f2_std = err[:, 1].std().item()

    if verbose:
        print(
            f"\nSurrogate validation  "
            f"({n_validation_points} Sobol points, {n_replicates} IBM replicates each):"
        )
        print(
            f"  f1 (extinction)  relative error:  mean = {f1_mean:.4f}   std = {f1_std:.4f}"
        )
        print(
            f"  f2 (fixation)    relative error:  mean = {f2_mean:.4f}   std = {f2_std:.4f}"
        )

    return {
        "f1_rel_err_mean": f1_mean,
        "f1_rel_err_std": f1_std,
        "f2_rel_err_mean": f2_mean,
        "f2_rel_err_std": f2_std,
        "surrogate_Y": surrogate_Y,
        "true_Y": true_Y,
        "true_Yvar": true_Yvar,
        "X": X,
    }


def optimise_objectives(
    objective_fn,
    bounds: Tensor,
    ref_point: Tensor,
    n_init: int,
    n_bo_steps: int,
    base_seed: int,
    verbose: bool,
    step_callback: Callable[[int, Tensor, Tensor], None] | None = None,
) -> tuple[Tensor, Tensor, ModelListGP]:
    """Optimise the given multi-objective function using qNEHVI.

    Parameters
    ----------
    step_callback:
        Optional callable invoked at the end of every BO step with arguments
        ``(step, pareto_X, pareto_Y)`` where ``step`` is 1-based.  Useful for
        recording per-step Pareto snapshots without changing the return type.
    """
    dimension = bounds.shape[1]
    sobol = torch.quasirandom.SobolEngine(
        dimension=dimension, scramble=True, seed=base_seed
    )
    X_unit = sobol.draw(n_init).to(dtype=torch.double)
    X_init = bounds[0] + X_unit * (bounds[1] - bounds[0])
    Y_init, Yvar_init = objective_fn(X_init)

    if verbose:
        hv_init = _hypervolume(Y_init, ref_point)
        print(
            f"[init] {n_init} Sobol points evaluated.  "
            f"hypervolume = {hv_init:.4f}  "
            f"best_f1 = {Y_init[:, 0].max().item():.4f}  "
            f"best_f2 = {Y_init[:, 1].max().item():.4f}"
        )

    hv_init = _hypervolume(Y_init, ref_point)
    hv_history: list[tuple[int, float]] = [
        (0, hv_init)
    ]  # (step, hv); step 0 = after init

    train_X = X_init
    train_Y = Y_init
    train_Yvar = Yvar_init
    for step in range(n_bo_steps):
        model = _fit_model(train_X, train_Y, train_Yvar)

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            X_baseline=train_X,
            prune_baseline=True,
        )

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=4,
            num_restarts=40,
            raw_samples=2048,
        )

        new_y, new_yvar = objective_fn(candidate)

        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)
        train_Yvar = torch.cat([train_Yvar, new_yvar], dim=0)

        if step_callback is not None:
            pareto_mask = is_non_dominated(train_Y)
            step_callback(
                step + 1, train_X[pareto_mask].clone(), train_Y[pareto_mask].clone()
            )

        hv = _hypervolume(train_Y, ref_point)
        hv_history.append((step + 1, hv))
        if verbose:
            print(
                f"[step {step + 1:3d}/{n_bo_steps}]  "
                f"candidate = {candidate.squeeze().tolist()}  "
                f"f1 = {new_y[:, 0].squeeze().tolist()}  "
                f"f2 = {new_y[:, 1].squeeze().tolist()}  "
                f"hypervolume = {hv:.4f}"
            )
        if torch.isclose(torch.tensor(hv), torch.tensor(1.0201)):
            print("Early stopping - hypervolume has converged to the maximum.")
            break

    model = _fit_model(train_X, train_Y, train_Yvar)

    # Extract Pareto front
    pareto_mask = is_non_dominated(train_Y)
    pareto_X = train_X[pareto_mask]
    pareto_Y = train_Y[pareto_mask]

    if verbose:
        print(
            f"\nOptimisation complete.  {pareto_mask.sum().item()} Pareto-optimal points found."
        )
        for i in range(pareto_X.shape[0]):
            params = "  ".join(
                f"{name}={pareto_X[i, j].item():.4f}"
                for j, name in enumerate(_PARAM_NAMES[:dimension])
            )
            print(
                f"  [{i}]  {params}  "
                f"f1(extinction)={pareto_Y[i, 0].item():.4f}  "
                f"f2(fixation)={pareto_Y[i, 1].item():.4f}"
            )

        print("\nHypervolume progression (every 10 steps):")
        milestone_steps = {s for s, _ in hv_history if s == 0 or s % 10 == 0}
        milestone_steps.add(hv_history[-1][0])  # always include the final step
        for s, hv in hv_history:
            if s in milestone_steps:
                label = "[init]        " if s == 0 else f"[step {s:3d}/{n_bo_steps}]"
                print(f"  {label}  hypervolume = {hv:.4f}")

    return pareto_X, pareto_Y, model


def _fit_model(
    train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor | None = None
) -> ModelListGP:
    """Fit a ModelListGP (one GP per objective) to the training data."""
    models = [
        SingleTaskGP(
            train_X,
            train_Y[:, i : i + 1],
            train_Yvar=train_Yvar[:, i : i + 1] if train_Yvar is not None else None,
            outcome_transform=Standardize(m=1),
        )
        for i in range(train_Y.shape[1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def _hypervolume(Y: Tensor, ref_point: Tensor) -> float:
    """Compute the dominated hypervolume of *Y* relative to *ref_point*.

    Uses the botorch Hypervolume utility
    """
    hv_calc = Hypervolume(ref_point=ref_point)
    pareto_mask = is_non_dominated(Y)
    return float(hv_calc.compute(Y[pareto_mask]))


def save_model(
    model: ModelListGP,
    n_init: int,
    n_bo_steps: int,
    n_replicates: int,
    prop_gg_low: float,
    prop_gg_high: float,
    directory: str | Path = OPTIMISATION_MODELS_DIR,
    run_timestamp: str | None = None,
) -> Path:
    """Save the fitted ModelListGP to disk using torch.save.

    When *run_timestamp* is supplied (recommended for replicate runs), the
    model is saved into a ``low{low_tag}_high{high_tag}/`` subdirectory of
    *directory*, keeping one directory per threshold pair.  Without it the
    legacy flat layout is used (threshold tags in the filename).

    Parameters
    ----------
    model:
        The fitted ModelListGP returned by :func:`optimise_gene_drive`.
    directory:
        Root destination directory (created if it does not exist).
    n_init, n_bo_steps, n_replicates:
        Run hyper-parameters embedded in the filename for traceability.
    prop_gg_low, prop_gg_high:
        Release fractions for the extinction and fixation objectives.
    run_timestamp:
        ``"%Y%m%d_%H%M%S"`` string generated once per run and shared with
        :func:`save_pareto_frontier_points` so model and CSV carry the same
        label.  If ``None``, a fresh timestamp is generated here and the
        legacy flat layout is used.

    Returns
    -------
    Path
        The path to the saved file.
    """
    import datetime

    directory = Path(directory)
    low_tag = f"{prop_gg_low:.2f}".replace(".", "p")
    high_tag = f"{prop_gg_high:.2f}".replace(".", "p")

    if run_timestamp is not None:
        save_dir = directory / f"low{low_tag}_high{high_tag}"
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"gene_drive_bo_model_init{n_init}_steps{n_bo_steps}_reps{n_replicates}"
            f"_{run_timestamp}.pt"
        )
        path = save_dir / filename
    else:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"gene_drive_bo_model_init{n_init}_steps{n_bo_steps}_reps{n_replicates}"
            f"_low{low_tag}_high{high_tag}_{timestamp}.pt"
        )
        path = directory / filename

    torch.save(model, path)
    print(f"Model saved to {path}")
    return path


def save_pareto_frontier_points(
    pareto_X: Tensor,
    pareto_Y: Tensor,
    verified_pareto_Y: Tensor,
    verified_pareto_Yvar: Tensor,
    filename: str,
    prop_gg_low: float,
    prop_gg_high: float,
    run_timestamp: str | None = None,
) -> None:
    """Save the Pareto frontier points to a CSV file.

    When *run_timestamp* is supplied (recommended for replicate runs), the CSV
    is saved into a ``low{low_tag}_high{high_tag}/`` subdirectory of the
    directory containing *filename*, keeping one directory per threshold pair.
    Without it the legacy flat layout is used (threshold tags in the filename).

    Columns: decision variables, BO-loop objective estimates, then verified
    objective means and variances (from the high-replicate post-hoc evaluation).
    """
    low_tag = f"{prop_gg_low:.2f}".replace(".", "p")
    high_tag = f"{prop_gg_high:.2f}".replace(".", "p")
    p = Path(filename)

    if run_timestamp is not None:
        subdir = p.parent / f"low{low_tag}_high{high_tag}"
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{p.stem}_{run_timestamp}.csv"
    else:
        path = p.with_stem(f"{p.stem}_low{low_tag}_high{high_tag}")

    obj_names = _OBJ_NAMES[: pareto_Y.shape[1]]
    verified_mean_names = [f"{n} [verified mean]" for n in obj_names]
    verified_var_names = [f"{n} [verified var]" for n in obj_names]

    data = (
        torch.cat([pareto_X, pareto_Y, verified_pareto_Y, verified_pareto_Yvar], dim=1)
        .cpu()
        .numpy()
    )

    df = pd.DataFrame(
        data,
        columns=[
            *(_PARAM_NAMES[: pareto_X.shape[1]]),
            *obj_names,
            *verified_mean_names,
            *verified_var_names,
        ],
    )
    df.to_csv(path, index=False)
