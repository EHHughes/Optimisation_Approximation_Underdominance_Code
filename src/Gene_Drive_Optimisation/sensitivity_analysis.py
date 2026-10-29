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
import pandas as pd
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


#: Marker written into (and required back out of) the ``analysis`` column of
#: every saved sensitivity CSV.  The two analyses produce different tables from
#: the same script, so the marker is what lets a replot dispatch on a file
#: without guessing from its name.
SOBOL_ANALYSIS_TAG = "sobol_indices"
LOCAL_ANALYSIS_TAG = "local_sensitivity"


def sobol_indices_to_dataframe(results: dict, **metadata) -> pd.DataFrame:
    """Flatten :func:`compute_sobol_indices` output into a tidy dataframe.

    Every Sobol design point costs a full IBM evaluation, so a run of any
    useful size is hours to days of compute; saving the resulting indices means
    a figure can be redrawn -- restyled, relabelled, recoloured -- without ever
    paying that again.  :func:`sobol_indices_from_dataframe` reverses this
    exactly, so the round trip reproduces the figure.

    One row per estimated index: first-order (``S1``) and total-effect (``ST``)
    rows name a single parameter; second-order (``S2``) rows name the pair, and
    appear only for the ``i < j`` entries SALib actually estimates (the rest of
    its matrix is ``NaN`` padding and is not written out).

    Parameters
    ----------
    results:
        Dict returned by :func:`compute_sobol_indices`.
    **metadata:
        Scalars describing the run that produced *results* (sample size,
        replicate count, release thresholds, ...), written as repeated columns.
        They identify the design behind the numbers; nothing reads them back.

    Returns
    -------
    pandas.DataFrame
    """
    param_names = list(results["param_names"])
    d = len(param_names)

    rows: list[dict] = []
    for obj in results["objective_names"]:
        Si = results[obj]
        for kind in ("S1", "ST"):
            for j, name in enumerate(param_names):
                rows.append(
                    {
                        "analysis": SOBOL_ANALYSIS_TAG,
                        "objective": obj,
                        "index": kind,
                        "parameter": name,
                        "parameter_2": None,
                        "value": float(Si[kind][j]),
                        "standard_error": float(Si[f"{kind}_se"][j]),
                        "conf": float(Si[f"{kind}_conf"][j]),
                    }
                )

        if Si.get("S2") is None:
            continue
        S2 = np.asarray(Si["S2"], dtype=float)
        S2_se = np.asarray(Si["S2_se"], dtype=float)
        S2_conf = np.asarray(Si["S2_conf"], dtype=float)
        for i in range(d):
            for j in range(i + 1, d):
                rows.append(
                    {
                        "analysis": SOBOL_ANALYSIS_TAG,
                        "objective": obj,
                        "index": "S2",
                        "parameter": param_names[i],
                        "parameter_2": param_names[j],
                        "value": float(S2[i, j]),
                        "standard_error": float(S2_se[i, j]),
                        "conf": float(S2_conf[i, j]),
                    }
                )

    frame = pd.DataFrame(rows)
    for key, value in metadata.items():
        frame[key] = value
    return frame


def sobol_indices_from_dataframe(frame: pd.DataFrame) -> dict:
    """Rebuild :func:`compute_sobol_indices` output from a saved dataframe.

    The inverse of :func:`sobol_indices_to_dataframe`: returns a dict the
    plotting functions accept unchanged.  ``S2`` is restored as the ``(d, d)``
    matrix SALib produces, ``NaN`` everywhere but the ``i < j`` entries, and is
    omitted entirely when the run did not compute second-order indices -- which
    is what tells :func:`~Gene_Drive_Optimisation.plot_gene_drives.plot_outputs`
    whether to draw the interaction grid.
    """
    if "analysis" not in frame.columns:
        raise ValueError(
            "Dataframe has no 'analysis' column; it was not written by "
            "sobol_indices_to_dataframe()."
        )
    tags = set(frame["analysis"].unique())
    if tags != {SOBOL_ANALYSIS_TAG}:
        raise ValueError(
            f"Expected a dataframe of {SOBOL_ANALYSIS_TAG!r} rows, found {sorted(tags)}."
        )

    objective_names = list(dict.fromkeys(frame["objective"]))
    first = frame[(frame["objective"] == objective_names[0]) & (frame["index"] == "ST")]
    param_names = list(dict.fromkeys(first["parameter"]))
    position = {name: j for j, name in enumerate(param_names)}
    d = len(param_names)

    results: dict = {
        "param_names": param_names,
        "objective_names": objective_names,
    }
    for obj in objective_names:
        rows = frame[frame["objective"] == obj]
        Si: dict = {"param_names": param_names}

        for kind in ("S1", "ST"):
            kind_rows = rows[rows["index"] == kind]
            where = [position[name] for name in kind_rows["parameter"]]
            for column, key in (
                ("value", kind),
                ("standard_error", f"{kind}_se"),
                ("conf", f"{kind}_conf"),
            ):
                values = np.full(d, np.nan)
                values[where] = kind_rows[column].to_numpy(dtype=float)
                Si[key] = values

        pair_rows = rows[rows["index"] == "S2"]
        if not pair_rows.empty:
            i_idx = [position[name] for name in pair_rows["parameter"]]
            j_idx = [position[name] for name in pair_rows["parameter_2"]]
            for column, key in (
                ("value", "S2"),
                ("standard_error", "S2_se"),
                ("conf", "S2_conf"),
            ):
                matrix = np.full((d, d), np.nan)
                matrix[i_idx, j_idx] = pair_rows[column].to_numpy(dtype=float)
                Si[key] = matrix

        results[obj] = Si
    return results


def local_sensitivity_to_dataframe(results: dict, **metadata) -> pd.DataFrame:
    """Flatten :func:`compute_local_sensitivity` output into a tidy dataframe.

    Saved for the same reason as the Sobol indices: the finite-difference
    stencil is ``1 + 2d`` IBM evaluations at many replicates apiece, so the
    figure should never have to be bought twice.
    :func:`local_sensitivity_from_dataframe` reverses it.

    One row per (objective, parameter) pair holding that partial derivative,
    alongside the point it was taken at -- ``x_star`` varies down the
    parameters, ``y_star`` across the objectives, and ``best_idx`` records
    which Pareto row the point came from.

    Parameters
    ----------
    results:
        Dict returned by :func:`compute_local_sensitivity`.
    **metadata:
        Scalars describing the run (step size, replicate count, release
        thresholds, ...), written as repeated columns.

    Returns
    -------
    pandas.DataFrame
    """
    param_names = list(results["param_names"])
    objective_names = list(results["objective_names"])
    jacobian = np.asarray(results["jacobian"], dtype=float)
    x_star = np.asarray(results["x_star"], dtype=float)
    y_star = np.asarray(results["y_star"], dtype=float)

    rows = [
        {
            "analysis": LOCAL_ANALYSIS_TAG,
            "objective": obj,
            "parameter": name,
            "jacobian": float(jacobian[k, j]),
            "x_star": float(x_star[j]),
            "y_star": float(y_star[k]),
            "best_idx": int(results["best_idx"]),
        }
        for k, obj in enumerate(objective_names)
        for j, name in enumerate(param_names)
    ]

    frame = pd.DataFrame(rows)
    for key, value in metadata.items():
        frame[key] = value
    return frame


def local_sensitivity_from_dataframe(frame: pd.DataFrame) -> dict:
    """Rebuild :func:`compute_local_sensitivity` output from a saved dataframe.

    The inverse of :func:`local_sensitivity_to_dataframe`: returns a dict the
    plotting functions accept unchanged.
    """
    if "analysis" not in frame.columns:
        raise ValueError(
            "Dataframe has no 'analysis' column; it was not written by "
            "local_sensitivity_to_dataframe()."
        )
    tags = set(frame["analysis"].unique())
    if tags != {LOCAL_ANALYSIS_TAG}:
        raise ValueError(
            f"Expected a dataframe of {LOCAL_ANALYSIS_TAG!r} rows, found {sorted(tags)}."
        )

    objective_names = list(dict.fromkeys(frame["objective"]))
    param_names = list(
        dict.fromkeys(frame[frame["objective"] == objective_names[0]]["parameter"])
    )
    position = {name: j for j, name in enumerate(param_names)}

    jacobian = np.full((len(objective_names), len(param_names)), np.nan)
    x_star = np.full(len(param_names), np.nan)
    y_star = np.full(len(objective_names), np.nan)
    for k, obj in enumerate(objective_names):
        rows = frame[frame["objective"] == obj]
        where = [position[name] for name in rows["parameter"]]
        jacobian[k, where] = rows["jacobian"].to_numpy(dtype=float)
        x_star[where] = rows["x_star"].to_numpy(dtype=float)
        y_star[k] = float(rows["y_star"].iloc[0])

    return {
        "jacobian": jacobian,
        "x_star": x_star,
        "y_star": y_star,
        "best_idx": int(frame["best_idx"].iloc[0]),
        "param_names": param_names,
        "objective_names": objective_names,
    }
