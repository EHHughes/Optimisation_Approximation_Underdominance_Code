"""Tests for Gene_Drive_Optimisation module.

Three test suites:
  TestOptimiseObjectives   – generic BO loop on the tractable (1/a, a) bi-objective
  TestHypervolume          – _hypervolume produces analytically-exact values
  TestBuildGenotypePenalties – _build_genotype_penalties maps array → dict correctly
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.multi_objective.pareto import is_non_dominated
from Gene_Drive_Optimisation.gene_drive_objective import (
    _build_genotype_penalties,
    evaluate_gene_drive,
    gene_drive_objectives,
)
from Gene_Drive_Optimisation.optimise_gene_drives import (
    _OBJ_NAMES,
    _PARAM_NAMES,
    _hypervolume,
    optimise_objectives,
    save_pareto_frontier_points,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inverse_identity_objectives(
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Toy bi-objective: f1 = 1/a, f2 = a  (both live in the Pareto front).

    Domain: a in [0.1, 1.0]  =>  f1 in [1, 10],  f2 in [0.1, 1.0].
    The entire curve is Pareto-optimal; no point dominates another.
    """
    a = X[:, 0].clamp(min=1e-6)
    f1 = 1.0 / a
    f2 = a
    Y = torch.stack([f1, f2], dim=1).to(dtype=X.dtype)
    Yvar = torch.full_like(Y, 1e-6)
    return Y, Yvar


_1D_BOUNDS = torch.tensor([[0.1], [1.0]], dtype=torch.double)
_REF_ZERO = torch.tensor([0.0, 0.0], dtype=torch.double)


# ---------------------------------------------------------------------------
# 1. optimise_objectives on the (1/a, a) problem
# ---------------------------------------------------------------------------


class TestOptimiseObjectives:
    """optimise_objectives should run end-to-end and return a valid Pareto front
    when given the analytically tractable (1/a, a) bi-objective."""

    @pytest.fixture(scope="class")
    def optimisation_result(self):
        pareto_X, pareto_Y, model = optimise_objectives(
            objective_fn=_inverse_identity_objectives,
            bounds=_1D_BOUNDS,
            ref_point=_REF_ZERO,
            n_init=5,
            n_bo_steps=3,
            base_seed=0,
            verbose=False,
        )
        return pareto_X, pareto_Y, model

    def test_returns_nonempty_pareto_front(self, optimisation_result):
        _, pareto_Y, _ = optimisation_result
        assert pareto_Y.shape[0] >= 1, "Pareto front must contain at least one point"

    def test_pareto_Y_has_two_objectives(self, optimisation_result):
        _, pareto_Y, _ = optimisation_result
        assert pareto_Y.shape[1] == 2

    def test_pareto_X_dimension_matches_bounds(self, optimisation_result):
        pareto_X, _, _ = optimisation_result
        assert pareto_X.shape[1] == _1D_BOUNDS.shape[1]

    def test_pareto_front_is_non_dominated(self, optimisation_result):
        _, pareto_Y, _ = optimisation_result
        dominated_mask = ~is_non_dominated(pareto_Y)
        assert not dominated_mask.any(), "Returned pareto_Y contains dominated points"

    def test_pareto_X_within_bounds(self, optimisation_result):
        pareto_X, _, _ = optimisation_result
        assert (pareto_X >= _1D_BOUNDS[0]).all(), "pareto_X below lower bound"
        assert (pareto_X <= _1D_BOUNDS[1]).all(), "pareto_X above upper bound"

    def test_hypervolume_positive(self, optimisation_result):
        _, pareto_Y, _ = optimisation_result
        hv = _hypervolume(pareto_Y, _REF_ZERO)
        assert hv > 0.0, "Hypervolume of a non-trivial Pareto front should be positive"

    def test_f1_and_f2_values_consistent_with_objective(self, optimisation_result):
        """For every Pareto point, re-evaluating the objective should give the
        same f1 and f2 values (deterministic objective)."""
        pareto_X, pareto_Y, _ = optimisation_result
        recomputed_Y, _ = _inverse_identity_objectives(pareto_X)
        assert torch.allclose(pareto_Y, recomputed_Y, atol=1e-5), (
            "Stored pareto_Y does not match re-evaluation of pareto_X"
        )


# ---------------------------------------------------------------------------
# 2. optimise_objectives on the Branin-Currin benchmark
# ---------------------------------------------------------------------------


class TestBraninCurrinOptimisation:
    """optimise_objectives should meaningfully improve hypervolume over a
    Sobol-only baseline on the standard Branin-Currin bi-objective benchmark.

    BraninCurrin (negate=True) is a well-known 2-input, 2-output maximisation
    problem supplied by BoTorch.  Its built-in reference point and unit-cube
    bounds are used throughout.  The same base_seed is passed to both runs so
    the Sobol initialisation is identical and any HV difference is purely due
    to the 5 BO steps.
    """

    _BC = BraninCurrin(negate=True)
    _BOUNDS = _BC.bounds.to(dtype=torch.double)
    _REF = _BC.ref_point.to(dtype=torch.double)
    _SEED = 42

    @staticmethod
    def _bc_objective(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Y = TestBraninCurrinOptimisation._BC(X).to(dtype=torch.double)
        Yvar = torch.full_like(Y, 1e-6)
        return Y, Yvar

    @pytest.fixture(scope="class")
    def sobol_result(self):
        _, pareto_Y, _ = optimise_objectives(
            objective_fn=self._bc_objective,
            bounds=self._BOUNDS,
            ref_point=self._REF,
            n_init=8,
            n_bo_steps=0,
            base_seed=self._SEED,
            verbose=False,
        )
        return pareto_Y

    @pytest.fixture(scope="class")
    def bo_result(self):
        _, pareto_Y, _ = optimise_objectives(
            objective_fn=self._bc_objective,
            bounds=self._BOUNDS,
            ref_point=self._REF,
            n_init=8,
            n_bo_steps=5,
            base_seed=self._SEED,
            verbose=False,
        )
        return pareto_Y

    def test_bo_hypervolume_exceeds_sobol_baseline(self, sobol_result, bo_result):
        """After 5 BO iterations the dominated hypervolume must be at least
        1.5× the Sobol-only baseline (empirically it is ~6.6× at seed 42)."""
        hv_sobol = _hypervolume(sobol_result, self._REF)
        hv_bo = _hypervolume(bo_result, self._REF)
        assert hv_bo > 1.5 * hv_sobol, (
            f"BO HV ({hv_bo:.4f}) should be >1.5× Sobol HV ({hv_sobol:.4f})"
        )

    def test_bo_pareto_front_is_non_dominated(self, bo_result):
        dominated_mask = ~is_non_dominated(bo_result)
        assert not dominated_mask.any()

    def test_bo_pareto_Y_within_expected_range(self, bo_result):
        """Both objectives should lie in the achievable range of BraninCurrin."""
        assert (bo_result[:, 0] >= -300.0).all(), "f1 below minimum achievable value"
        assert (bo_result[:, 0] <= 0.0).all(), "f1 should be non-positive (negated)"
        assert (bo_result[:, 1] >= -15.0).all(), "f2 below minimum achievable value"
        assert (bo_result[:, 1] <= 0.0).all(), "f2 should be non-positive (negated)"


# ---------------------------------------------------------------------------
# 3. _hypervolume – analytically tractable cases
# ---------------------------------------------------------------------------


class TestHypervolume:
    """Verify _hypervolume returns exact values for hand-calculated cases
    derived from the (1/a, a) curve."""

    def test_single_point(self):
        """Single non-dominated point [2.0, 0.5] vs ref [0, 0].

        HV = (2.0 - 0) * (0.5 - 0) = 1.0  (a simple rectangle).
        """
        Y = torch.tensor([[2.0, 0.5]], dtype=torch.double)
        ref = torch.tensor([0.0, 0.0], dtype=torch.double)
        assert _hypervolume(Y, ref) == pytest.approx(1.0, abs=1e-9)

    def test_two_non_dominated_points(self):
        """Two points [2.0, 0.5] (a=0.5) and [1.0, 1.0] (a=1) vs ref [0, 0].

        The dominated region is the union of:
          [0, 2] × [0, 0.5]  (area 1.0)
          [0, 1] × [0, 1.0]  (area 1.0)
        minus their intersection [0,1] × [0, 0.5] (area 0.5).
        HV = 1.0 + 1.0 - 0.5 = 1.5
        """
        Y = torch.tensor([[2.0, 0.5], [1.0, 1.0]], dtype=torch.double)
        ref = torch.tensor([0.0, 0.0], dtype=torch.double)
        assert _hypervolume(Y, ref) == pytest.approx(1.5, abs=1e-9)

    def test_dominated_point_does_not_change_hypervolume(self):
        """Adding a dominated point should leave the hypervolume unchanged."""
        Y_clean = torch.tensor([[2.0, 0.5], [1.0, 1.0]], dtype=torch.double)
        # [1.5, 0.3] is dominated by [2.0, 0.5]  (2 >= 1.5 AND 0.5 >= 0.3)
        Y_with_dominated = torch.tensor(
            [[2.0, 0.5], [1.0, 1.0], [1.5, 0.3]], dtype=torch.double
        )
        ref = torch.tensor([0.0, 0.0], dtype=torch.double)
        assert _hypervolume(Y_with_dominated, ref) == pytest.approx(
            _hypervolume(Y_clean, ref), abs=1e-9
        )

    def test_non_zero_ref_point(self):
        """Single point [2.0, 0.5] vs ref [0.5, 0.1].

        HV = (2.0 - 0.5) * (0.5 - 0.1) = 1.5 * 0.4 = 0.6
        """
        Y = torch.tensor([[2.0, 0.5]], dtype=torch.double)
        ref = torch.tensor([0.5, 0.1], dtype=torch.double)
        assert _hypervolume(Y, ref) == pytest.approx(1.5 * 0.4, abs=1e-9)

    def test_hypervolume_increases_with_additional_non_dominated_point(self):
        """Adding a new non-dominated point must increase the hypervolume."""
        Y_one = torch.tensor([[2.0, 0.5]], dtype=torch.double)
        Y_two = torch.tensor([[2.0, 0.5], [1.0, 1.0]], dtype=torch.double)
        ref = torch.tensor([0.0, 0.0], dtype=torch.double)
        assert _hypervolume(Y_two, ref) > _hypervolume(Y_one, ref)


# ---------------------------------------------------------------------------
# 3. evaluate_gene_drive – randomness / variance sanity checks
# ---------------------------------------------------------------------------


class TestEvaluateGeneDriveRandomness:
    """Verify that evaluate_gene_drive produces genuinely varied output.

    Low-variance symptoms could indicate that all replicates are using the
    same RNG state (e.g. SeedSequence objects passed where integers are
    expected, or seeds never forwarded to the simulation).

    _PROP_LOW (5 %) is intentionally the low-release value used in the
    optimisation scenario.  It is kept for cross-call variance tests but NOT
    used for the within-call std test: with ~5 GG founders and only a few
    cycles they reliably go extinct in every replicate, so std is genuinely
    zero and that tells us nothing about seeding.  _PROP_MID (30 %) is used
    instead — enough founders to survive and produce variable adult counts.
    """

    _PENALTIES = np.ones((4, 2)) * 0.9  # mild fitness cost on WG and GG
    _CONVERSION = 0.8
    _PROP_MID = 0.50
    _N_REPLICATES = 2
    _CYCLES = 2

    def _call(self, prop: float, base_seed: int) -> tuple[float, float]:
        return evaluate_gene_drive(
            prop_eggs_genetically_modified=prop,
            conversion_rate=self._CONVERSION,
            penalties_array=self._PENALTIES,
            n_replicates=self._N_REPLICATES,
            cycles=self._CYCLES,
            base_seed=base_seed,
        )

    def test_different_seeds_give_different_means(self):
        """Ten independent calls with distinct seeds must not all return the
        same mean G-frequency – that would indicate the RNG is not wired up."""
        means = [self._call(self._PROP_MID, seed)[0] for seed in range(5)]
        unique_means = len(set(means))
        assert unique_means > 1, (
            f"All 5 calls returned the same mean ({means[0]:.6f}). "
            "The RNG seed is likely not being passed through to the simulation."
        )

    def test_std_is_nonzero_within_a_single_call(self):
        """The std returned by a single call must be > 0, confirming that
        individual replicates within one call are not all identical.

        Uses _PROP_MID (50 %) so that ~50 GG founders reliably survive to
        adulthood; stochastic variation in survival then produces different
        G-frequencies across the 5 replicates.
        """
        _, std = self._call(self._PROP_MID, base_seed=0)
        assert std > 0.0, (
            f"std == {std}; all {self._N_REPLICATES} replicates produced "
            "identical G-frequencies, suggesting they share the same RNG state."
        )

    def test_spread_across_repeated_calls_exceeds_threshold(self):
        """The range of means across 5 independent calls should be non-trivial.

        Uses _PROP_MID so that stochastic variation in survival is detectable.
        """
        means = [self._call(self._PROP_MID, seed)[0] for seed in range(5)]
        spread = max(means) - min(means)
        assert spread > 1e-4, (
            f"Range of means across 5 calls is only {spread:.2e}. "
            "This is suspiciously low and may indicate collapsed randomness."
        )

    def test_same_seed_is_reproducible(self):
        """Two calls with the same seed must return the same (mean, std) pair,
        confirming that seeds are deterministically controlling the output."""
        result_a = self._call(self._PROP_MID, base_seed=42)
        result_b = self._call(self._PROP_MID, base_seed=42)
        assert result_a[0] == pytest.approx(result_b[0], abs=1e-12), (
            "Same seed produced different means – non-deterministic behaviour."
        )
        assert result_a[1] == pytest.approx(result_b[1], abs=1e-12), (
            "Same seed produced different stds – non-deterministic behaviour."
        )


# ---------------------------------------------------------------------------
# 4. _build_genotype_penalties
# ---------------------------------------------------------------------------


class TestBuildGenotypePenalties:
    """_build_genotype_penalties maps a (4, 2) penalties array to a dict
    with WW, WG, GG keys and a_fac, alpha_fac, beta_fac, g_fac sub-keys."""

    @pytest.fixture
    def sample_penalties(self):
        return np.array(
            [
                [0.8, 0.6],  # row 0: a_fac      WG, GG
                [0.9, 0.7],  # row 1: alpha_fac   WG, GG
                [0.95, 0.85],  # row 2: beta_fac    WG, GG
                [0.75, 0.5],  # row 3: g_fac       WG, GG
            ]
        )

    def test_ww_genotype_is_always_identity(self, sample_penalties):
        result = _build_genotype_penalties(sample_penalties)
        ww = result["WW"]
        assert ww["a_fac"] == 1
        assert ww["alpha_fac"] == 1
        assert ww["beta_fac"] == 1.0
        assert ww["g_fac"] == 1.0

    def test_wg_maps_column_0(self, sample_penalties):
        result = _build_genotype_penalties(sample_penalties)
        wg = result["WG"]
        assert wg["a_fac"] == pytest.approx(0.8)
        assert wg["alpha_fac"] == pytest.approx(0.9)
        assert wg["beta_fac"] == pytest.approx(0.95)
        assert wg["g_fac"] == pytest.approx(0.75)

    def test_gg_maps_column_1(self, sample_penalties):
        result = _build_genotype_penalties(sample_penalties)
        gg = result["GG"]
        assert gg["a_fac"] == pytest.approx(0.6)
        assert gg["alpha_fac"] == pytest.approx(0.7)
        assert gg["beta_fac"] == pytest.approx(0.85)
        assert gg["g_fac"] == pytest.approx(0.5)

    def test_all_genotype_keys_present(self, sample_penalties):
        result = _build_genotype_penalties(sample_penalties)
        expected_keys = {"WW", "WG", "GG"}
        assert set(result.keys()) == expected_keys

    def test_all_factor_keys_present_for_each_genotype(self, sample_penalties):
        result = _build_genotype_penalties(sample_penalties)
        expected_factors = {"a_fac", "alpha_fac", "beta_fac", "g_fac"}
        for genotype, factors in result.items():
            assert set(factors.keys()) == expected_factors, (
                f"Missing factor keys for genotype {genotype}"
            )

    def test_identity_penalties_all_ones(self):
        """When all penalties are 1.0, every genotype should be identical to WW."""
        identity = np.ones((4, 2))
        result = _build_genotype_penalties(identity)
        for genotype in ("WW", "WG", "GG"):
            for factor in ("a_fac", "alpha_fac", "beta_fac", "g_fac"):
                assert result[genotype][factor] == pytest.approx(1.0), (
                    f"{genotype}.{factor} should be 1.0 for identity penalties"
                )

    def test_zero_penalties(self):
        """Zero penalties should propagate correctly to WG and GG."""
        zeros = np.zeros((4, 2))
        result = _build_genotype_penalties(zeros)
        for factor in ("a_fac", "alpha_fac", "beta_fac", "g_fac"):
            assert result["WG"][factor] == pytest.approx(0.0)
            assert result["GG"][factor] == pytest.approx(0.0)
        # WW is always identity
        for factor in ("a_fac", "alpha_fac", "beta_fac", "g_fac"):
            assert result["WW"][factor] in (1, 1.0)


# ---------------------------------------------------------------------------
# 5. penalties_array reshape – regression test for the row[1:].reshape fix
# ---------------------------------------------------------------------------


class _SyncPool:
    """Minimal pool replacement that runs map() synchronously in the main process.

    Used by TestPenaltiesArrayReshape to keep evaluate_gene_drive patchable:
    multiprocess.Pool normally spawns subprocesses, so unittest.mock patches
    are invisible to the worker.  Replacing Pool with this shim makes
    _eval_single run in the same process where the patch is active.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, fn, iterable):
        return list(map(fn, iterable))


class TestPenaltiesArrayReshape:
    """Regression test for the penalties_array reshape bug in _eval_single.

    The pre-fix code used row[1:].reshape([-1, 2]), which fills the (4, 2)
    matrix row-by-row and interleaves WG and GG factors:
        col 0: [wg_a, wg_beta, gg_a,    gg_beta  ]  <- mixed
        col 1: [wg_alpha, wg_g, gg_alpha, gg_g   ]  <- mixed

    The fix uses row[1:].reshape([2, -1]).T, which groups the four WG factors
    into one row and the four GG factors into another before transposing:
        col 0: [wg_a, wg_alpha, wg_beta, wg_g]   <- all WG
        col 1: [gg_a, gg_alpha, gg_beta, gg_g]   <- all GG
    """

    # Use all-distinct values so any interleaving is immediately detectable.
    _WG = [0.1, 0.2, 0.3, 0.4]  # wg_a, wg_alpha, wg_beta, wg_g
    _GG = [0.5, 0.6, 0.7, 0.8]  # gg_a, gg_alpha, gg_beta, gg_g
    _CONVERSION = 0.9

    @pytest.fixture
    def captured_penalties(self):
        """Run gene_drive_objectives with a mocked evaluate_gene_drive (no IBM
        simulation) and return the penalties_array it received."""
        X = torch.tensor([[self._CONVERSION] + self._WG + self._GG], dtype=torch.double)
        arrays: list[np.ndarray] = []

        def _mock_eval(
            prop_eggs_genetically_modified, conversion_rate, penalties_array, **kwargs
        ):
            arrays.append(penalties_array.copy())
            return 0.5, 0.0

        with (
            patch(
                "Gene_Drive_Optimisation.gene_drive_objective.evaluate_gene_drive",
                side_effect=_mock_eval,
            ),
            patch(
                "Gene_Drive_Optimisation.gene_drive_objective.multiprocess.Pool",
                new=lambda *a, **kw: _SyncPool(),
            ),
        ):
            gene_drive_objectives(
                X, prop_gg_low=0.4, prop_gg_high=0.6, n_replicates=1, cycles=1
            )

        assert arrays, "evaluate_gene_drive was never called"
        return arrays[0]

    def test_penalties_array_has_shape_4_by_2(self, captured_penalties):
        assert captured_penalties.shape == (4, 2), (
            f"Expected (4, 2), got {captured_penalties.shape}"
        )

    def test_column_0_contains_wg_factors(self, captured_penalties):
        """Column 0 must hold [wg_a, wg_alpha, wg_beta, wg_g] in that order."""
        np.testing.assert_array_almost_equal(
            captured_penalties[:, 0],
            self._WG,
            err_msg="Column 0 should be WG factors; got interleaved or GG values instead",
        )

    def test_column_1_contains_gg_factors(self, captured_penalties):
        """Column 1 must hold [gg_a, gg_alpha, gg_beta, gg_g] in that order."""
        np.testing.assert_array_almost_equal(
            captured_penalties[:, 1],
            self._GG,
            err_msg="Column 1 should be GG factors; got interleaved or WG values instead",
        )

    def test_buggy_reshape_would_have_failed(self, captured_penalties):
        """Explicitly document the pre-fix symptom: the old reshape([-1, 2])
        would have placed wg_alpha (0.2) in GG's a_fac slot (col 1, row 0).
        After the fix that slot must contain gg_a (0.5)."""
        gg_a = self._GG[0]  # 0.5
        assert captured_penalties[0, 1] == pytest.approx(gg_a), (
            f"GG a_fac (penalties_array[0, 1]) should be {gg_a} (gg_a); "
            f"got {captured_penalties[0, 1]}. "
            "The old reshape([-1, 2]) bug would have put wg_alpha here instead."
        )


# ---------------------------------------------------------------------------
# 6. save_pareto_frontier_points – CSV column names and values
# ---------------------------------------------------------------------------


class TestSaveParetoFrontierPoints:
    """save_pareto_frontier_points writes a CSV with: decision variables,
    BO-loop objective estimates, verified objective means, and verified
    objective variances — all with correct column names and values."""

    # One Pareto point using all 9 parameter dimensions and both objectives.
    # Values are all-distinct so any column transposition is immediately detectable.
    _PARAM_VALUES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    _OBJ_VALUES = [0.05, 0.95]
    _VERIFIED_MEAN_VALUES = [0.06, 0.94]
    _VERIFIED_VAR_VALUES = [0.001, 0.002]

    _PROP_GG_LOW = 0.3
    _PROP_GG_HIGH = 0.5

    @pytest.fixture
    def saved_csv(self, tmp_path):
        pareto_X = torch.tensor([self._PARAM_VALUES], dtype=torch.double)
        pareto_Y = torch.tensor([self._OBJ_VALUES], dtype=torch.double)
        verified_Y = torch.tensor([self._VERIFIED_MEAN_VALUES], dtype=torch.double)
        verified_Yvar = torch.tensor([self._VERIFIED_VAR_VALUES], dtype=torch.double)
        path = tmp_path / "pareto.csv"
        save_pareto_frontier_points(
            pareto_X,
            pareto_Y,
            verified_Y,
            verified_Yvar,
            str(path),
            prop_gg_low=self._PROP_GG_LOW,
            prop_gg_high=self._PROP_GG_HIGH,
        )
        low_tag = f"{self._PROP_GG_LOW:.2f}".replace(".", "p")
        high_tag = f"{self._PROP_GG_HIGH:.2f}".replace(".", "p")
        actual_path = tmp_path / f"pareto_low{low_tag}_high{high_tag}.csv"
        return pd.read_csv(actual_path)

    def test_column_names_match_param_obj_verified(self, saved_csv):
        """Columns: params, BO obj, verified means, verified variances."""
        verified_mean_names = [f"{n} [verified mean]" for n in _OBJ_NAMES]
        verified_var_names = [f"{n} [verified var]" for n in _OBJ_NAMES]
        expected = (
            list(_PARAM_NAMES)
            + list(_OBJ_NAMES)
            + verified_mean_names
            + verified_var_names
        )
        assert list(saved_csv.columns) == expected

    def test_no_index_column(self, saved_csv):
        """index=False means no 'Unnamed: 0' or similar column."""
        assert not any(col.startswith("Unnamed") for col in saved_csv.columns)

    def test_one_row_per_pareto_point(self, saved_csv):
        assert len(saved_csv) == 1

    def test_param_values_round_trip(self, saved_csv):
        """Each parameter column holds the value supplied in pareto_X."""
        for name, expected in zip(_PARAM_NAMES, self._PARAM_VALUES):
            assert saved_csv[name].iloc[0] == pytest.approx(expected)

    def test_bo_objective_values_round_trip(self, saved_csv):
        """Each BO objective column holds the value supplied in pareto_Y."""
        for name, expected in zip(_OBJ_NAMES, self._OBJ_VALUES):
            assert saved_csv[name].iloc[0] == pytest.approx(expected)

    def test_verified_mean_values_round_trip(self, saved_csv):
        """Each verified mean column holds the value supplied in verified_Y."""
        for name, expected in zip(_OBJ_NAMES, self._VERIFIED_MEAN_VALUES):
            assert saved_csv[f"{name} [verified mean]"].iloc[0] == pytest.approx(
                expected
            )

    def test_verified_var_values_round_trip(self, saved_csv):
        """Each verified variance column holds the value supplied in verified_Yvar."""
        for name, expected in zip(_OBJ_NAMES, self._VERIFIED_VAR_VALUES):
            assert saved_csv[f"{name} [verified var]"].iloc[0] == pytest.approx(
                expected
            )

    def test_timestamp_creates_subdirectory(self, tmp_path):
        """Passing run_timestamp saves into a threshold subdirectory, not flat."""
        pareto_X = torch.tensor([self._PARAM_VALUES], dtype=torch.double)
        pareto_Y = torch.tensor([self._OBJ_VALUES], dtype=torch.double)
        verified_Y = torch.tensor([self._VERIFIED_MEAN_VALUES], dtype=torch.double)
        verified_Yvar = torch.tensor([self._VERIFIED_VAR_VALUES], dtype=torch.double)
        save_pareto_frontier_points(
            pareto_X, pareto_Y, verified_Y, verified_Yvar,
            str(tmp_path / "pareto.csv"),
            prop_gg_low=self._PROP_GG_LOW,
            prop_gg_high=self._PROP_GG_HIGH,
            run_timestamp="20260529_141200",
        )
        low_tag = f"{self._PROP_GG_LOW:.2f}".replace(".", "p")
        high_tag = f"{self._PROP_GG_HIGH:.2f}".replace(".", "p")
        expected = tmp_path / f"low{low_tag}_high{high_tag}" / "pareto_20260529_141200.csv"
        assert expected.exists(), f"Expected file at {expected}"
        assert not (tmp_path / f"pareto_low{low_tag}_high{high_tag}.csv").exists()

    def test_no_timestamp_legacy_path(self, tmp_path):
        """Omitting run_timestamp produces the legacy flat file (no subdirectory)."""
        pareto_X = torch.tensor([self._PARAM_VALUES], dtype=torch.double)
        pareto_Y = torch.tensor([self._OBJ_VALUES], dtype=torch.double)
        verified_Y = torch.tensor([self._VERIFIED_MEAN_VALUES], dtype=torch.double)
        verified_Yvar = torch.tensor([self._VERIFIED_VAR_VALUES], dtype=torch.double)
        save_pareto_frontier_points(
            pareto_X, pareto_Y, verified_Y, verified_Yvar,
            str(tmp_path / "pareto.csv"),
            prop_gg_low=self._PROP_GG_LOW,
            prop_gg_high=self._PROP_GG_HIGH,
        )
        low_tag = f"{self._PROP_GG_LOW:.2f}".replace(".", "p")
        high_tag = f"{self._PROP_GG_HIGH:.2f}".replace(".", "p")
        expected = tmp_path / f"pareto_low{low_tag}_high{high_tag}.csv"
        assert expected.exists(), f"Expected legacy file at {expected}"
        assert not (tmp_path / f"low{low_tag}_high{high_tag}").exists()
