"""Variance-based and local sensitivity analysis of a black-box objective.

Uses SALib's Sobol analyser (Saltelli sampling scheme + Jansen estimator) to
compute first-order and total-effect Sobol sensitivity indices, and central
finite differences to estimate the local Jacobian at a Pareto point.

Both analyses operate on a generic objective callable ``objective_fn(X) -> Y``
mapping a ``(n, d)`` batch of parameter vectors to an ``(n, m)`` batch of
objective values.  This decouples the sensitivity analysis from any particular
emulator: callers pass either the underlying (stochastic IBM) objective for
maximum fidelity, or a fitted surrogate's posterior mean for speed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from scipy.stats import norm
from torch import Tensor

from .optimise_gene_drives import _OBJ_NAMES, _PARAM_NAMES, PARAM_BOUNDS


def compute_sobol_indices(
    objective_fn: Callable[[Tensor], Tensor],
    bounds: Tensor | None = None,
    n_samples: int = 2048,
    param_names: list[str] | None = None,
    objective_names: list[str] | None = None,
    calc_second_order: bool = False,
    num_resamples: int = 100,
    conf_level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Estimate Sobol sensitivity indices of an objective using SALib.

    SALib's Saltelli sampling scheme is used to generate the design matrix;
    *objective_fn* is evaluated at every design point; then SALib's Sobol
    analyser computes first-order (*S1*) and total-effect (*ST*) indices
    together with bootstrap confidence intervals.

    Total objective evaluations: ``N * (D + 2)`` when
    ``calc_second_order=False``, or ``N * (2D + 2)`` when ``True``.  When
    *objective_fn* wraps the stochastic IBM this dominates the run time, so
    ``n_samples`` must be chosen far more conservatively than for a cheap GP
    surrogate.  Pass an *objective_fn* whose randomness is fixed (seeded) so
    the response is a deterministic function of the inputs, otherwise Monte
    Carlo noise inflates the estimated output variance and biases the indices.

    Parameters
    ----------
    objective_fn:
        Callable mapping a ``(n, d)`` batch of parameter vectors to an
        ``(n, m)`` batch of objective values.  Pass the underlying IBM
        objective for maximum fidelity, or a surrogate's posterior mean for
        speed.
    bounds:
        Parameter bounds, shape ``(2, d)``.  Defaults to ``PARAM_BOUNDS``
        (all nine parameters in ``[0, 1]``).
    n_samples:
        Base sample size *N* passed to :func:`SALib.sample.sobol.sample`.
        Must be a power of two for best Sobol sequence properties.
    param_names:
        Names for the *d* input parameters.  Defaults to ``_PARAM_NAMES``.
    objective_names:
        Names for the *m* objectives.  Defaults to ``_OBJ_NAMES``.
    calc_second_order:
        If True, also compute second-order interaction indices (doubles the
        number of GP evaluations).
    num_resamples:
        Bootstrap resamples used by SALib to compute confidence intervals.
    conf_level:
        Confidence level for the bootstrap intervals (default 0.95).
    seed:
        Seed passed to both SALib's sampler and its bootstrap resampler.

    Returns
    -------
    dict
        One entry per objective, keyed by objective name.  Each value is the
        SALib result dict containing:

        ``'S1'``, ``'S1_conf'``, ``'S1_se'``
            First-order indices, their SALib bootstrap confidence half-widths,
            and the recovered bootstrap standard errors (``_conf / Z``), shape
            ``(d,)``.
        ``'ST'``, ``'ST_conf'``, ``'ST_se'``
            Total-effect indices, confidence half-widths, and bootstrap standard
            errors, shape ``(d,)``.
        ``'S2'``, ``'S2_conf'``, ``'S2_se'`` *(only when* ``calc_second_order=True`` *)*
            Second-order interaction indices, their confidence half-widths, and
            bootstrap standard errors, shape ``(d, d)``.
        ``'param_names'``
            Copy of the parameter name list for convenience.

        The top-level dict also carries ``'param_names'`` and
        ``'objective_names'`` keys.
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    bounds = bounds.to(dtype=torch.double)

    if param_names is None:
        param_names = _PARAM_NAMES
    if objective_names is None:
        objective_names = _OBJ_NAMES

    problem = {
        "num_vars": bounds.shape[1],
        "names": list(param_names),
        "bounds": bounds.cpu().numpy().T.tolist(),
    }

    X_np = sobol_sample.sample(
        problem,
        n_samples,
        calc_second_order=calc_second_order,
        scramble=True,
        seed=seed,
    )

    X = torch.tensor(X_np, dtype=torch.double)

    with torch.no_grad():
        Y_all = torch.as_tensor(objective_fn(X)).detach().cpu().numpy()
    Y_all = np.asarray(Y_all, dtype=float)
    if Y_all.ndim == 1:
        Y_all = Y_all[:, None]

    results: dict = {
        "param_names": list(param_names),
        "objective_names": list(objective_names),
    }

    # SALib returns ``*_conf`` as a normal-approximation bootstrap interval,
    # i.e. ``Z * bootstrap_se`` with ``Z = norm.ppf(0.5 + conf_level / 2)``.
    # Recover the underlying bootstrap standard errors so callers/plots can
    # draw ±2 sd error bars instead of a normality-assuming 95 % CI.
    z = norm.ppf(0.5 + conf_level / 2.0)

    m = Y_all.shape[1]
    for k in range(m):
        Si = sobol_analyze.analyze(
            problem,
            Y_all[:, k],
            calc_second_order=calc_second_order,
            num_resamples=num_resamples,
            conf_level=conf_level,
            seed=seed,
        )
        Si["param_names"] = list(param_names)
        for base in ("S1", "ST", "S2"):
            conf_key = f"{base}_conf"
            if conf_key in Si:
                Si[f"{base}_se"] = np.asarray(Si[conf_key], dtype=float) / z
        results[objective_names[k]] = Si

    return results


def compute_local_sensitivity(
    objective_fn: Callable[[Tensor], Tensor],
    pareto_X: Tensor,
    pareto_Y: Tensor,
    bounds: Tensor | None = None,
    param_names: list[str] | None = None,
    objective_names: list[str] | None = None,
    step: float = 1e-2,
) -> dict:
    """Local sensitivity of an objective at the best Pareto point.

    Selects the Pareto-optimal point that maximises the mean objective score
    ``(f1 + f2) / 2``, then estimates the Jacobian ``∂f_k / ∂x_j`` at that
    point by **central finite differences** — evaluating *objective_fn* at
    ``x* ± step·e_j`` for each coordinate.  This makes no differentiability
    assumption, so it works directly on the (non-differentiable, stochastic)
    IBM.  All parameters lie in ``[0, 1]``, so the derivatives are directly
    comparable across inputs.

    The whole ``1 + 2d`` finite-difference design (centre + two points per
    coordinate) is submitted to *objective_fn* in a single batched call, so a
    parallel objective evaluates them concurrently.  For a stochastic objective,
    pass an *objective_fn* that uses **common random numbers** (the same fixed
    seed for every row of the batch): the shared noise then cancels in each
    difference, which is essential for a usable gradient estimate.

    Perturbations are clamped to *bounds*.  If a coordinate's step is clamped
    on one side only, the (now asymmetric) central difference would be a biased
    estimate of the derivative *at* ``x*`` — it approximates the derivative at
    the midpoint of the shortened interval instead.  Such a coordinate
    therefore falls back to a first-order one-sided difference against the
    centre point, using the unclamped side: ``(f(x* + h e_j) − f(x*)) / h`` at
    a lower bound and ``(f(x*) − f(x* − h e_j)) / h`` at an upper bound.  In
    the pathological case where *both* sides are clamped (``step`` wider than
    the whole bound range) the asymmetric central difference over the clamped
    interval is used, as there is nothing better available.

    Parameters
    ----------
    objective_fn:
        Callable mapping a ``(n, d)`` batch of parameter vectors to an
        ``(n, m)`` batch of objective values.  Pass the underlying IBM
        objective for maximum fidelity, or a surrogate's posterior mean for
        speed.
    pareto_X:
        Decision variables for each Pareto point, shape ``(n_pareto, d)``.
    pareto_Y:
        Objective values ``[f1, f2]`` for each Pareto point, shape
        ``(n_pareto, 2)``.
    bounds:
        Parameter bounds, shape ``(2, d)``.  Defaults to ``PARAM_BOUNDS``.
        Perturbations are clamped to these bounds.
    param_names:
        Names for the *d* input parameters.  Defaults to ``_PARAM_NAMES``.
    objective_names:
        Names for the *m* objectives.  Defaults to ``_OBJ_NAMES``.
    step:
        Finite-difference half-step ``h`` in the (unit) parameter space.

    Returns
    -------
    dict
        ``'jacobian'``      – ndarray shape ``(m, d)``, partial derivatives
                              ``∂f_k/∂x_j`` at the optimal point.
        ``'x_star'``        – ndarray shape ``(d,)``, the selected Pareto point.
        ``'y_star'``        – ndarray shape ``(m,)``, objectives at ``x_star``.
        ``'best_idx'``      – int, index into ``pareto_X`` of the selected point.
        ``'param_names'``   – list of parameter name strings.
        ``'objective_names'`` – list of objective name strings.
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    bounds = bounds.to(dtype=torch.double)
    if param_names is None:
        param_names = _PARAM_NAMES
    if objective_names is None:
        objective_names = _OBJ_NAMES

    pareto_X = pareto_X.to(dtype=torch.double)
    pareto_Y = pareto_Y.to(dtype=torch.double)

    best_idx = int(pareto_Y.mean(dim=1).argmax().item())
    x_star = pareto_X[best_idx].clone()

    d = x_star.shape[0]
    lb, ub = bounds[0], bounds[1]
    diag = torch.arange(d)

    # Central finite-difference design: the centre point, then x + h·e_j and
    # x − h·e_j for every coordinate, each clamped to the parameter bounds.
    x_plus = x_star.repeat(d, 1)
    x_minus = x_star.repeat(d, 1)
    x_plus[diag, diag] = torch.minimum(x_star + step, ub)
    x_minus[diag, diag] = torch.maximum(x_star - step, lb)

    # Realised half-steps either side of x*, and which of them the bounds cut
    # short.  A one-sided truncation biases the central difference, so those
    # coordinates fall back to a first-order difference below.
    h_plus = x_plus[diag, diag] - x_star  # (d,)
    h_minus = x_star - x_minus[diag, diag]  # (d,)
    tol = step * 1e-9
    clipped_plus = h_plus < step - tol
    clipped_minus = h_minus < step - tol
    backward_only = clipped_plus & ~clipped_minus  # use (f(x*) − f(x⁻)) / h⁻
    forward_only = clipped_minus & ~clipped_plus  # use (f(x⁺) − f(x*)) / h⁺

    design = torch.cat([x_star.unsqueeze(0), x_plus, x_minus], dim=0)  # (1 + 2d, d)

    with torch.no_grad():
        Y = torch.as_tensor(objective_fn(design)).detach().to(dtype=torch.double)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    y_star = Y[0]
    Y_plus = Y[1 : 1 + d]  # (d, m)
    Y_minus = Y[1 + d : 1 + 2 * d]  # (d, m)

    print(design)
    print(y_star)
    print(Y_plus)
    print(Y_minus)

    # Central difference by default; a one-sided first-order difference for any
    # coordinate whose step was clamped on exactly one side.
    numer = Y_plus - Y_minus  # (d, m)
    denom = h_plus + h_minus  # (d,)

    numer = torch.where(backward_only.unsqueeze(1), y_star - Y_minus, numer)
    denom = torch.where(backward_only, h_minus, denom)
    numer = torch.where(forward_only.unsqueeze(1), Y_plus - y_star, numer)
    denom = torch.where(forward_only, h_plus, denom)

    # jacobian[k, j] = ∂f_k/∂x_j at x*
    jacobian = (numer / denom.unsqueeze(1)).T.cpu().numpy()  # (m, d)

    return {
        "jacobian": jacobian,
        "x_star": x_star.cpu().numpy(),
        "y_star": y_star.cpu().numpy(),
        "best_idx": best_idx,
        "param_names": list(param_names),
        "objective_names": list(objective_names),
    }


def print_sobol_indices(results: dict) -> None:
    """Pretty-print the sensitivity index table returned by :func:`compute_sobol_indices`."""
    param_names: list[str] = results["param_names"]
    obj_names: list[str] = results["objective_names"]
    col_w = max(len(n) for n in param_names) + 2

    for obj in obj_names:
        Si = results[obj]
        print(f"\n{'─' * 64}")
        print(f"  {obj}")
        print(f"{'─' * 64}")
        header = (
            f"  {'Parameter':<{col_w}}  {'S1':>8}  {'S1 ±':>8}  {'ST':>8}  {'ST ±':>8}"
        )
        print(header)
        print(f"  {'-' * (col_w + 38)}")
        for j, name in enumerate(param_names):
            print(
                f"  {name:<{col_w}}"
                f"  {Si['S1'][j]:>8.4f}  {Si['S1_conf'][j]:>8.4f}"
                f"  {Si['ST'][j]:>8.4f}  {Si['ST_conf'][j]:>8.4f}"
            )
    print(f"{'─' * 64}\n")
