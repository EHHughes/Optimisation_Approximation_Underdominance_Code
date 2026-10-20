"""Tests for the agent-based larval model to ensure Python matches R implementation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from IBM_Implementations.agent_based_larval_model import run_simulation


class TestRPythonSynchronization:
    """Tests to ensure Python simulation output matches R implementation.

    These tests are critical for maintaining consistency between the R and Python
    implementations of the agent-based model. The R outputs serve as the reference.

    To regenerate reference files after intentional changes to both R and Python:
    1. Run IBM_Model_6.4.Rmd with the matching parameters
    2. Copy the output CSV to tests/fixtures/
    """

    @pytest.fixture
    def fixtures_path(self) -> Path:
        """Path to test fixtures directory."""
        return Path(__file__).parent / "fixtures"

    def test_stage_pop_matches_r_output_temp20_cycles5_exploitation(
        self, fixtures_path: Path
    ):
        """Test that stage_pop matches R output for temperature=20, cycles=5.

        This is the primary synchronization test. It runs the Python simulation
        with the exact same parameters used in R (seed=1234, temperature=20,
        interference feeding, 5 cycles) and verifies the stage populations match.

        R simulation parameters (from IBM_Model_6.4.Rmd):
        - seed: 1234
        - temperature: 20
        - competition: EXPLOITATION
        - cycles: 5
        - time_steps: 10
        - initial_food_val: 2700 (multiplied by N_WINNERS=4 for interference)
        - refill_food_val: 75 (multiplied by N_WINNERS=4)
        """
        # Run Python simulation with matching parameters
        state = run_simulation(
            seed=1234,
            cycles=5,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,  # Use SyncRNG for R compatibility
            feeding_type="exploitative",
            random_mating=False,
        )

        # Load expected R output
        expected_df = pd.read_csv(
            fixtures_path / "r_stage_df_temp20_cycles5_exploitation.csv"
        )

        # Extract stage_pop from Python state
        actual_df = state["stage_pop"].reset_index(drop=True).copy()
        actual_df["Cycle"] = np.arange(1, len(actual_df) + 1)

        # Convert to same dtypes for comparison
        actual_df = actual_df.astype(
            {"Eggs": int, "Larvae": int, "Pupae": int, "Adults": int, "Cycle": int}
        )

        # Compare each column
        for col in ["Eggs", "Larvae", "Pupae", "Adults"]:
            np.testing.assert_array_equal(
                actual_df[col].values,
                expected_df[col].values,
                err_msg=f"Mismatch in {col} column between Python and R",
            )

    def test_stage_pop_matches_r_output_temp20_cycles5_interference(
        self, fixtures_path: Path
    ):
        """Test that stage_pop matches R output for temperature=20, cycles=5.

        This is the primary synchronization test. It runs the Python simulation
        with the exact same parameters used in R (seed=1234, temperature=20,
        interference feeding, 5 cycles) and verifies the stage populations match.

        R simulation parameters (from IBM_Model_6.4.Rmd):
        - seed: 1234
        - temperature: 20
        - competition: INTERFERENCE
        - cycles: 5
        - time_steps: 10
        - initial_food_val: 2700 (multiplied by N_WINNERS=4 for interference)
        - refill_food_val: 75 (multiplied by N_WINNERS=4)
        """
        # Run Python simulation with matching parameters
        state = run_simulation(
            seed=1234,
            cycles=5,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,  # Use SyncRNG for R compatibility
            feeding_type="interference",
            random_mating=False,
        )

        # Load expected R output
        expected_df = pd.read_csv(
            fixtures_path / "r_stage_df_temp20_cycles5_interference.csv"
        )

        # Extract stage_pop from Python state
        actual_df = state["stage_pop"].reset_index(drop=True).copy()
        actual_df["Cycle"] = np.arange(1, len(actual_df) + 1)

        # Convert to same dtypes for comparison
        actual_df = actual_df.astype(
            {"Eggs": int, "Larvae": int, "Pupae": int, "Adults": int, "Cycle": int}
        )

        # Compare each column
        for col in ["Eggs", "Larvae", "Pupae", "Adults"]:
            np.testing.assert_array_equal(
                actual_df[col].values,
                expected_df[col].values,
                err_msg=f"Mismatch in {col} column between Python and R",
            )

    def test_stage_pop_eggs_survival_rate(self):
        """Test that egg-to-larvae survival rate is correctly applied.

        At temperature 20C, survival rate should be 0.9, so:
        100 eggs -> 90 larvae (floor(100 * 0.9) = 90)
        """
        state = run_simulation(
            seed=1234,
            cycles=1,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,
            random_mating=False,
        )

        eggs = int(state["stage_pop"].iloc[0]["Eggs"])
        larvae = int(state["stage_pop"].iloc[0]["Larvae"])

        assert eggs == 100, f"Expected 100 eggs, got {eggs}"
        assert larvae == 90, f"Expected 90 larvae (100 * 0.9), got {larvae}"

    def test_deterministic_with_sync_rng(self):
        """Test that simulations with same seed and sync_rng produce identical results."""
        state1 = run_simulation(
            seed=42,
            cycles=2,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,
            random_mating=False,
        )
        state2 = run_simulation(
            seed=42,
            cycles=2,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,
            random_mating=False,
        )

        # stage_pop should be identical
        pd.testing.assert_frame_equal(
            state1["stage_pop"],
            state2["stage_pop"],
            check_dtype=False,
        )


class TestSimulationBasics:
    """Basic tests for simulation functionality."""

    def test_run_simulation_returns_expected_keys(self):
        """Test that run_simulation returns a dict with all expected keys."""
        state = run_simulation(
            seed=1234,
            cycles=1,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,
            random_mating=False,
        )

        expected_keys = [
            "constants",
            "stage_pop",
            "stage_pop_including_genotypes",
            "pupated_all",
            "deceased_all",
            "temperature_matrix",
            "food_matrix",
            "W_matrix",
            "R_matrix",
            "pupation_stats",
            "larval_time_series",
        ]

        for key in expected_keys:
            assert key in state, f"Missing key: {key}"

    def test_larval_time_series_has_correct_rows(self):
        """Test that larval_time_series has correct number of rows.

        Default time_steps=300, so cycles * 300 rows expected.
        """
        cycles = 2
        state = run_simulation(
            seed=1234,
            cycles=cycles,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,
            random_mating=False,
        )

        # Get actual time_steps from constants (default is 300)
        time_steps = state["constants"]["time_steps"]
        expected_rows = cycles * time_steps
        actual_rows = len(state["larval_time_series"])

        assert (
            actual_rows == expected_rows
        ), f"Expected {expected_rows} rows ({cycles} cycles * {time_steps} time_steps), got {actual_rows}"

    def test_food_not_refilled_at_hour_1(self):
        """Test that food is NOT refilled at hour 1 (bug fix verification).

        This test ensures the food refill timing bug is fixed:
        - R refills at hour 24, 48, etc. (hour %% 24 == 0)
        - Python should also refill at hour 24, 48, etc. (hour % 24 == 0)
        - NOT at hour 1 ((hour - 1) % 24 == 0 was the bug)

        Uses exploitative feeding (default) for this test since it's about
        the refill timing logic, not the specific food values.
        """
        state = run_simulation(
            seed=1234,
            cycles=1,
            map_size=10,
            fixed_temperature=20,
            sync_rng=True,
            random_mating=False,
        )

        # Get actual initial food from constants (varies by feeding type)
        initial_food_per_cell = state["constants"]["initial_food"]
        map_size = state["constants"]["map_size"]
        expected_initial_total = initial_food_per_cell * map_size * map_size

        # Get food at hour 1 (index 0 in food_matrix)
        hour1_food = state["food_matrix"][0, 0]

        # Should be at or below initial (some consumption may have occurred)
        # but NOT above initial (which would indicate erroneous refill)
        assert (
            hour1_food <= expected_initial_total
        ), f"Food at hour 1 ({hour1_food}) exceeds initial ({expected_initial_total}) - refill bug?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
