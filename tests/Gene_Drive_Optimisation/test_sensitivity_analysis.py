"""Tests for surrogate validation and sensitivity analysis.

Three test suites, all built around a single 3-input linear ground-truth model
whose Sobol indices and Jacobian are analytically known:

  TestValidateSurrogate   – validate_surrogate (with gene_drive_objectives mocked)
                            returns the expected structure and plausibly small
                            relative errors; errors decrease with more training data.
  TestGlobalSensitivity   – compute_sobol_indices recovers theoretical total-effect
                            Sobol indices for the linear model.
  TestLocalSensitivity    – compute_local_sensitivity recovers the Jacobian of the
                            linear model at the selected Pareto point.

Linear ground-truth model (3 inputs, 2 outputs)
------------------------------------------------
  f1(x) = 0.9·x0 + 0.3·x1 + 0.1·x2
  f2(x) = 0.1·x0 + 0.8·x1 + 0.2·x2

For xᵢ ~ U[0, 1] independently the exact total-effect Sobol index is
  STᵢ = cᵢ² / Σⱼ cⱼ²
and the Jacobian ∂f_k/∂xⱼ = c_kj everywhere (constant, linear model).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from Gene_Drive_Optimisation.optimise_gene_drives import _fit_model, validate_surrogate
from Gene_Drive_Optimisation.sensitivity_analysis import (
    compute_local_sensitivity,
    compute_sobol_indices,
    local_sensitivity_from_dataframe,
    local_sensitivity_to_dataframe,
    sobol_indices_from_dataframe,
    sobol_indices_to_dataframe,
)

# ---------------------------------------------------------------------------
# Shared ground-truth linear model
# ---------------------------------------------------------------------------

_COEFFS = torch.tensor(
    [
        [0.9, 0.3, 0.1],  # f1: x0 >> x1 > x2
        [0.1, 0.8, 0.2],  # f2: x1 >> x2 > x0
    ],
    dtype=torch.double,
)
_BOUNDS = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
_PARAM_NAMES_3D = ["x0", "x1", "x2"]
_OBJ_NAMES_3D = ["f_1 (low g frequency)", "f_2 (high g frequency)"]


def _linear(X: Tensor) -> Tensor:
    """Evaluate the 3D linear model; returns shape (n, 2)."""
    return X.to(dtype=torch.double) @ _COEFFS.T


def _theoretical_st(obj_idx: int) -> np.ndarray:
    """Exact total-effect Sobol indices for the linear model: STᵢ = cᵢ²/Σcⱼ²."""
    c = _COEFFS[obj_idx].numpy()
    c_sq = c**2
    return c_sq / c_sq.sum()


def _make_gp(n_train: int, seed: int = 0):
    """Fit a GP surrogate to Sobol-sampled points from the 3D linear model."""
    eng = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    X = eng.draw(n_train).to(dtype=torch.double)
    Y = _linear(X)
    Yvar = torch.full_like(Y, 1e-6)
    return _fit_model(X, Y, Yvar)


# ---------------------------------------------------------------------------
# 1. TestValidateSurrogate
# ---------------------------------------------------------------------------


class TestValidateSurrogate:
    """validate_surrogate is exercised end-to-end with gene_drive_objectives
    replaced by the known linear model so no IBM simulation runs.

    Checks:
      - Returned dict has the expected keys and tensor shapes.
      - Relative errors on a well-trained GP are plausibly small (< 0.25).
      - Errors decrease when the GP is trained on more data.
    """

    _N_SMALL = 20
    _N_LARGE = 120
    _N_VAL = 20  # kept small for speed; enough to measure relative error
    _SEED = 7

    @staticmethod
    def _mock_objectives(X: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """Linear-model stand-in for gene_drive_objectives."""
        Y = _linear(X)
        Yvar = torch.full_like(Y, 1e-6)
        return Y, Yvar

    def _run_validate(self, model) -> dict:
        with patch(
            "Gene_Drive_Optimisation.optimise_gene_drives.gene_drive_objectives",
            side_effect=self._mock_objectives,
        ):
            return validate_surrogate(
                model,
                n_validation_points=self._N_VAL,
                n_replicates=1,
                cycles=1,
                bounds=_BOUNDS,
                seed=self._SEED,
                verbose=False,
            )

    @pytest.fixture(scope="class")
    def small_model(self):
        return _make_gp(self._N_SMALL, seed=self._SEED)

    @pytest.fixture(scope="class")
    def large_model(self):
        return _make_gp(self._N_LARGE, seed=self._SEED)

    @pytest.fixture(scope="class")
    def small_results(self, small_model):
        return self._run_validate(small_model)

    @pytest.fixture(scope="class")
    def large_results(self, large_model):
        return self._run_validate(large_model)

    # --- structure --------------------------------------------------------

    def test_scalar_keys_present(self, small_results):
        for key in (
            "f1_rel_err_mean",
            "f1_rel_err_std",
            "f2_rel_err_mean",
            "f2_rel_err_std",
        ):
            assert key in small_results, f"Missing key '{key}'"

    def test_tensor_keys_present(self, small_results):
        for key in ("surrogate_Y", "true_Y", "true_Yvar", "X"):
            assert key in small_results, f"Missing key '{key}'"

    def test_output_shapes(self, small_results):
        n = self._N_VAL
        assert small_results["surrogate_Y"].shape == (n, 2)
        assert small_results["true_Y"].shape == (n, 2)
        assert small_results["true_Yvar"].shape == (n, 2)
        assert small_results["X"].shape == (n, 3)

    def test_errors_are_finite_and_non_negative(self, large_results):
        for key in ("f1_rel_err_mean", "f2_rel_err_mean"):
            val = large_results[key]
            assert np.isfinite(val), f"{key} is not finite"
            assert val >= 0.0, f"{key} is negative"

    # --- plausibility -----------------------------------------------------

    def test_well_trained_model_has_small_f1_error(self, large_results):
        assert large_results["f1_rel_err_mean"] < 0.25, (
            f"f1 relative error {large_results['f1_rel_err_mean']:.3f} is unexpectedly large "
            f"for a GP trained on {self._N_LARGE} points from a linear model"
        )

    def test_well_trained_model_has_small_f2_error(self, large_results):
        assert large_results["f2_rel_err_mean"] < 0.25, (
            f"f2 relative error {large_results['f2_rel_err_mean']:.3f} is unexpectedly large "
            f"for a GP trained on {self._N_LARGE} points from a linear model"
        )

    # --- trend: more training data → lower error -------------------------

    def test_more_data_reduces_f1_error(self, small_results, large_results):
        assert large_results["f1_rel_err_mean"] < small_results["f1_rel_err_mean"], (
            f"f1 error did not improve with more data: "
            f"N={self._N_SMALL} → {small_results['f1_rel_err_mean']:.4f}, "
            f"N={self._N_LARGE} → {large_results['f1_rel_err_mean']:.4f}"
        )

    def test_more_data_reduces_f2_error(self, small_results, large_results):
        assert large_results["f2_rel_err_mean"] < small_results["f2_rel_err_mean"], (
            f"f2 error did not improve with more data: "
            f"N={self._N_SMALL} → {small_results['f2_rel_err_mean']:.4f}, "
            f"N={self._N_LARGE} → {large_results['f2_rel_err_mean']:.4f}"
        )


# ---------------------------------------------------------------------------
# 2. TestGlobalSensitivity
# ---------------------------------------------------------------------------


class TestGlobalSensitivity:
    """compute_sobol_indices recovers the theoretical total-effect Sobol
    indices of the 3D linear model to within _ABS_TOL.

    For xᵢ ~ U[0, 1] and f = Σᵢ cᵢxᵢ (no interactions), the exact result is
    STᵢ = cᵢ² / Σⱼ cⱼ²  and  S1ᵢ = STᵢ.
    """

    _N_SAMPLES = 1024  # SALib base sample count; total evals = N*(D+2) = 5120
    _SEED = 3
    _ABS_TOL = 0.10  # tolerance on |ST_estimated − ST_theoretical|

    @pytest.fixture(scope="class")
    def sobol_results(self):
        return compute_sobol_indices(
            _linear,
            bounds=_BOUNDS,
            n_samples=self._N_SAMPLES,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
            seed=self._SEED,
        )

    @pytest.fixture(scope="class")
    def sobol_results_s2(self):
        return compute_sobol_indices(
            _linear,
            bounds=_BOUNDS,
            n_samples=self._N_SAMPLES,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
            calc_second_order=True,
            seed=self._SEED,
        )

    # --- structure --------------------------------------------------------

    def test_all_objective_keys_present(self, sobol_results):
        for name in _OBJ_NAMES_3D:
            assert name in sobol_results, f"Missing objective key '{name}'"

    def test_param_and_objective_name_lists_correct(self, sobol_results):
        assert sobol_results["param_names"] == _PARAM_NAMES_3D
        assert sobol_results["objective_names"] == _OBJ_NAMES_3D

    def test_each_objective_has_st_and_s1_arrays(self, sobol_results):
        for name in _OBJ_NAMES_3D:
            for field in ("ST", "ST_conf", "S1", "S1_conf"):
                assert (
                    field in sobol_results[name]
                ), f"'{name}' result missing field '{field}'"

    # --- correctness ------------------------------------------------------

    @pytest.mark.parametrize("obj_idx", [0, 1])
    def test_dominant_input_has_largest_st(self, sobol_results, obj_idx):
        """The input with the largest coefficient should have the largest ST."""
        obj_name = _OBJ_NAMES_3D[obj_idx]
        st = np.array(sobol_results[obj_name]["ST"])
        expected_dominant = int(_COEFFS[obj_idx].abs().argmax().item())
        estimated_dominant = int(st.argmax())
        assert estimated_dominant == expected_dominant, (
            f"{obj_name}: dominant input should be x{expected_dominant}, "
            f"got x{estimated_dominant}  (ST = {np.round(st, 3)})"
        )

    @pytest.mark.parametrize("obj_idx", [0, 1])
    def test_st_values_close_to_theoretical(self, sobol_results, obj_idx):
        """Each ST index should be within _ABS_TOL of its analytical value."""
        obj_name = _OBJ_NAMES_3D[obj_idx]
        st_est = np.array(sobol_results[obj_name]["ST"])
        st_true = _theoretical_st(obj_idx)
        np.testing.assert_allclose(
            st_est,
            st_true,
            atol=self._ABS_TOL,
            err_msg=(
                f"{obj_name}: estimated ST {np.round(st_est, 3)} not close to "
                f"theoretical {np.round(st_true, 3)}"
            ),
        )

    @pytest.mark.parametrize("obj_idx", [0, 1])
    def test_weak_input_has_smaller_st_than_dominant(self, sobol_results, obj_idx):
        """The weakest input (index 2 for f1, index 0 for f2) should have
        substantially lower ST than the dominant input."""
        obj_name = _OBJ_NAMES_3D[obj_idx]
        st = np.array(sobol_results[obj_name]["ST"])
        # The theoretical weakest input has coefficient ratio ≥ 9:1 vs dominant
        weakest = int(_COEFFS[obj_idx].abs().argmin().item())
        dominant = int(_COEFFS[obj_idx].abs().argmax().item())
        assert st[weakest] < st[dominant], (
            f"{obj_name}: weakest input x{weakest} (ST={st[weakest]:.3f}) should "
            f"have lower ST than dominant x{dominant} (ST={st[dominant]:.3f})"
        )

    # --- second-order indices (S2) ----------------------------------------

    def test_each_objective_has_s2_arrays(self, sobol_results_s2):
        for name in _OBJ_NAMES_3D:
            for field in ("S2", "S2_conf"):
                assert (
                    field in sobol_results_s2[name]
                ), f"'{name}' result missing field '{field}'"

    def test_s2_shape(self, sobol_results_s2):
        n_params = len(_PARAM_NAMES_3D)
        for name in _OBJ_NAMES_3D:
            s2 = np.array(sobol_results_s2[name]["S2"])
            assert s2.shape == (
                n_params,
                n_params,
            ), f"{name}: expected S2 shape ({n_params}, {n_params}), got {s2.shape}"

    @pytest.mark.parametrize("obj_idx", [0, 1])
    def test_s2_near_zero_for_linear_model(self, sobol_results_s2, obj_idx):
        """For a linear model with independent inputs, all pairwise
        second-order interaction indices should be zero.

        SALib fills only the strict upper triangle of S2; the diagonal and
        lower triangle are NaN and excluded from the check.
        """
        obj_name = _OBJ_NAMES_3D[obj_idx]
        s2 = np.array(sobol_results_s2[obj_name]["S2"])
        upper = s2[np.triu_indices_from(s2, k=1)]
        np.testing.assert_allclose(
            upper,
            0.0,
            atol=self._ABS_TOL,
            err_msg=(
                f"{obj_name}: S2 indices {np.round(upper, 3)} should be ~0 "
                f"for a linear model with no pairwise interactions"
            ),
        )


# ---------------------------------------------------------------------------
# 3. TestLocalSensitivity
# ---------------------------------------------------------------------------


class TestLocalSensitivity:
    """compute_local_sensitivity recovers the Jacobian of the 3D linear model.

    For f_k(x) = Σⱼ c_kj·xⱼ the true partial derivative ∂f_k/∂xⱼ = c_kj
    everywhere.  Central finite differences of the exact linear objective
    recover these coefficients exactly (up to floating point), so the gradient
    matches to well within _ABS_TOL.
    """

    _SEED = 5
    _ABS_TOL = 0.10

    @pytest.fixture(scope="class")
    def pareto_data(self):
        """Three fake Pareto points with objectives evaluated by the linear model."""
        rng = torch.Generator().manual_seed(self._SEED)
        X = torch.rand(3, 3, dtype=torch.double, generator=rng)
        Y = _linear(X)
        return X, Y

    @pytest.fixture(scope="class")
    def local_results(self, pareto_data):
        pareto_X, pareto_Y = pareto_data
        return compute_local_sensitivity(
            _linear,
            pareto_X,
            pareto_Y,
            bounds=_BOUNDS,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
        )

    # --- structure --------------------------------------------------------

    def test_all_keys_present(self, local_results):
        for key in (
            "jacobian",
            "x_star",
            "y_star",
            "best_idx",
            "param_names",
            "objective_names",
        ):
            assert key in local_results, f"Missing key '{key}'"

    def test_jacobian_shape(self, local_results):
        assert local_results["jacobian"].shape == (
            2,
            3,
        ), f"Expected Jacobian shape (2, 3), got {local_results['jacobian'].shape}"

    def test_x_star_and_y_star_shapes(self, local_results):
        assert local_results["x_star"].shape == (3,)
        assert local_results["y_star"].shape == (2,)

    def test_name_lists_propagated(self, local_results):
        assert local_results["param_names"] == _PARAM_NAMES_3D
        assert local_results["objective_names"] == _OBJ_NAMES_3D

    # --- selection logic --------------------------------------------------

    def test_best_idx_is_valid_index(self, local_results, pareto_data):
        pareto_X, _ = pareto_data
        assert 0 <= local_results["best_idx"] < pareto_X.shape[0]

    def test_x_star_matches_selected_pareto_point(self, local_results, pareto_data):
        pareto_X, _ = pareto_data
        idx = local_results["best_idx"]
        np.testing.assert_array_almost_equal(
            local_results["x_star"],
            pareto_X[idx].numpy(),
            decimal=10,
        )

    def test_best_idx_maximises_mean_objective(self, local_results, pareto_data):
        """Selected point should be the argmax of (f1 + f2) / 2."""
        _, pareto_Y = pareto_data
        expected_idx = int(pareto_Y.mean(dim=1).argmax().item())
        assert local_results["best_idx"] == expected_idx

    # --- gradient accuracy -----------------------------------------------

    @pytest.mark.parametrize("obj_idx", [0, 1])
    def test_jacobian_close_to_linear_coefficients(self, local_results, obj_idx):
        """∂f_k/∂xⱼ should match the true coefficient c_kj to within _ABS_TOL."""
        estimated = local_results["jacobian"][obj_idx]
        true_coeffs = _COEFFS[obj_idx].numpy()
        np.testing.assert_allclose(
            estimated,
            true_coeffs,
            atol=self._ABS_TOL,
            err_msg=(
                f"Jacobian row {obj_idx}: got {np.round(estimated, 3)}, "
                f"expected {np.round(true_coeffs, 3)}"
            ),
        )

    @pytest.mark.parametrize("obj_idx", [0, 1])
    def test_dominant_gradient_is_positive(self, local_results, obj_idx):
        """The gradient w.r.t. the input with the largest positive coefficient
        should itself be positive."""
        dominant = int(_COEFFS[obj_idx].argmax().item())
        grad = local_results["jacobian"][obj_idx, dominant]
        assert grad > 0.0, (
            f"Objective {obj_idx}: gradient for dominant input x{dominant} "
            f"should be positive, got {grad:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. TestLocalSensitivityAtBounds
# ---------------------------------------------------------------------------


def _quadratic(X: Tensor) -> Tensor:
    """Curved ground truth f_k(x) = Σⱼ c_kj·xⱼ²; returns shape (n, 2).

    Curvature makes the one-sided and the (clamped) asymmetric central
    difference give different answers, so the fallback is observable.
    """
    return (X.to(dtype=torch.double) ** 2) @ _COEFFS.T


class TestLocalSensitivityAtBounds:
    """Coordinates whose step is clamped on one side use a one-sided difference.

    A step clamped on one side only leaves an interval that is not centred on
    x*, so the central difference over it estimates the derivative at the
    interval midpoint rather than at x*.  Those coordinates must instead fall
    back to a first-order difference against the centre point, taken on the
    side the bounds left intact.
    """

    _STEP = 0.1
    # x0 clamped above, x1 clamped below, x2 interior (full central difference).
    _X_STAR = torch.tensor([[0.98, 0.02, 0.5]], dtype=torch.double)

    @pytest.fixture(scope="class")
    def local_results(self):
        return compute_local_sensitivity(
            _quadratic,
            self._X_STAR,
            _quadratic(self._X_STAR),
            bounds=_BOUNDS,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
            step=self._STEP,
        )

    @pytest.mark.parametrize("obj_idx", [0, 1])
    def test_matches_expected_difference_scheme(self, local_results, obj_idx):
        """Backward for x0, forward for x1, central for x2."""
        c = _COEFFS[obj_idx].numpy()
        x = self._X_STAR[0].numpy()
        h = self._STEP
        expected = np.array(
            [
                c[0] * (x[0] ** 2 - (x[0] - h) ** 2) / h,  # clamped above
                c[1] * ((x[1] + h) ** 2 - x[1] ** 2) / h,  # clamped below
                c[2] * ((x[2] + h) ** 2 - (x[2] - h) ** 2) / (2 * h),  # interior
            ]
        )
        np.testing.assert_allclose(
            local_results["jacobian"][obj_idx],
            expected,
            rtol=1e-10,
            err_msg=f"Jacobian row {obj_idx} does not match the expected scheme",
        )

    def test_clamped_coordinate_differs_from_asymmetric_central(self, local_results):
        """Sanity check that the fallback actually changed the estimate."""
        c = _COEFFS[0, 0].item()
        x0 = self._X_STAR[0, 0].item()
        h_plus = 1.0 - x0
        asymmetric = c * ((x0 + h_plus) ** 2 - (x0 - self._STEP) ** 2) / (h_plus + self._STEP)
        assert not np.isclose(local_results["jacobian"][0, 0], asymmetric, rtol=1e-6), (
            "Clamped coordinate still uses the asymmetric central difference"
        )

    def test_point_on_the_bound_is_still_finite(self):
        """x* sitting exactly on a bound leaves one side with zero width."""
        x_star = torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.double)
        results = compute_local_sensitivity(
            _quadratic,
            x_star,
            _quadratic(x_star),
            bounds=_BOUNDS,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
            step=self._STEP,
        )
        assert np.all(np.isfinite(results["jacobian"])), results["jacobian"]


class TestResultSerialisation:
    """Saved results reload into exactly the dicts the plotting code expects.

    A Sobol run costs one IBM evaluation per design point, so its indices are
    written to CSV and the figures redrawn from the file rather than re-earned
    (``sensitivity_checks.py --from-csv``).  That is only safe if the round
    trip is lossless, which is what these tests pin down.
    """

    _N_SAMPLES = 64  # only the round trip is under test, not the estimates
    _SEED = 11

    @pytest.fixture(scope="class")
    def sobol_results(self):
        return compute_sobol_indices(
            _linear,
            bounds=_BOUNDS,
            n_samples=self._N_SAMPLES,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
            calc_second_order=True,
            seed=self._SEED,
        )

    @pytest.fixture(scope="class")
    def local_results(self):
        rng = torch.Generator().manual_seed(self._SEED)
        X = torch.rand(3, 3, dtype=torch.double, generator=rng)
        return compute_local_sensitivity(
            _linear,
            X,
            _linear(X),
            bounds=_BOUNDS,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
        )

    # --- Sobol indices ----------------------------------------------------

    def test_sobol_round_trip_preserves_every_index(self, sobol_results):
        frame = sobol_indices_to_dataframe(sobol_results)
        restored = sobol_indices_from_dataframe(frame)

        assert restored["param_names"] == _PARAM_NAMES_3D
        assert restored["objective_names"] == _OBJ_NAMES_3D
        for obj in _OBJ_NAMES_3D:
            for key in ("S1", "S1_se", "S1_conf", "ST", "ST_se", "ST_conf"):
                np.testing.assert_allclose(restored[obj][key], sobol_results[obj][key])

    def test_sobol_round_trip_preserves_the_interaction_matrix(self, sobol_results):
        """``S2`` comes back as SALib's upper-triangular matrix, NaNs and all."""
        restored = sobol_indices_from_dataframe(
            sobol_indices_to_dataframe(sobol_results)
        )
        for obj in _OBJ_NAMES_3D:
            for key in ("S2", "S2_se", "S2_conf"):
                np.testing.assert_allclose(
                    restored[obj][key], sobol_results[obj][key]
                )
            lower = np.tril_indices(len(_PARAM_NAMES_3D))
            assert np.all(np.isnan(restored[obj]["S2"][lower]))

    def test_first_order_only_run_restores_without_an_s2_key(self):
        """No ``S2`` key is what tells plot_outputs to skip the interaction grid."""
        results = compute_sobol_indices(
            _linear,
            bounds=_BOUNDS,
            n_samples=self._N_SAMPLES,
            param_names=_PARAM_NAMES_3D,
            objective_names=_OBJ_NAMES_3D,
            calc_second_order=False,
            seed=self._SEED,
        )
        restored = sobol_indices_from_dataframe(sobol_indices_to_dataframe(results))
        for obj in _OBJ_NAMES_3D:
            assert restored[obj].get("S2") is None

    def test_metadata_is_written_as_repeated_columns(self, sobol_results):
        frame = sobol_indices_to_dataframe(
            sobol_results, prop_gg_low=0.3, prop_gg_high=0.5, n_samples=64
        )
        assert (frame["prop_gg_low"] == 0.3).all()
        assert (frame["prop_gg_high"] == 0.5).all()
        assert (frame["n_samples"] == 64).all()

    def test_sobol_survives_a_real_csv(self, sobol_results, tmp_path):
        """Via the file itself, so dtypes and the empty ``parameter_2`` round trip."""
        csv_path = tmp_path / "sobol.csv"
        sobol_indices_to_dataframe(sobol_results).to_csv(csv_path, index=False)
        restored = sobol_indices_from_dataframe(pd.read_csv(csv_path))
        for obj in _OBJ_NAMES_3D:
            np.testing.assert_allclose(restored[obj]["ST"], sobol_results[obj]["ST"])
            np.testing.assert_allclose(restored[obj]["S2"], sobol_results[obj]["S2"])

    def test_local_results_are_rejected_by_the_sobol_reader(self, local_results):
        frame = local_sensitivity_to_dataframe(local_results)
        with pytest.raises(ValueError, match="sobol_indices"):
            sobol_indices_from_dataframe(frame)

    # --- local sensitivity ------------------------------------------------

    def test_local_round_trip_preserves_the_jacobian_and_its_point(
        self, local_results
    ):
        frame = local_sensitivity_to_dataframe(local_results)
        restored = local_sensitivity_from_dataframe(frame)

        for key in ("jacobian", "x_star", "y_star"):
            np.testing.assert_allclose(restored[key], local_results[key])
        assert restored["best_idx"] == local_results["best_idx"]
        assert restored["param_names"] == _PARAM_NAMES_3D
        assert restored["objective_names"] == _OBJ_NAMES_3D

    def test_local_survives_a_real_csv(self, local_results, tmp_path):
        csv_path = tmp_path / "local.csv"
        local_sensitivity_to_dataframe(local_results).to_csv(csv_path, index=False)
        restored = local_sensitivity_from_dataframe(pd.read_csv(csv_path))
        np.testing.assert_allclose(restored["jacobian"], local_results["jacobian"])

    def test_sobol_results_are_rejected_by_the_local_reader(self, sobol_results):
        frame = sobol_indices_to_dataframe(sobol_results)
        with pytest.raises(ValueError, match="local_sensitivity"):
            local_sensitivity_from_dataframe(frame)

    def test_a_table_without_the_analysis_marker_is_rejected(self, sobol_results):
        frame = sobol_indices_to_dataframe(sobol_results).drop(columns=["analysis"])
        with pytest.raises(ValueError, match="analysis"):
            sobol_indices_from_dataframe(frame)
