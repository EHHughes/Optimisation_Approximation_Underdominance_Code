"""Tests that ``conversion_rate`` actually reaches the eggs in every pair-generating path.

These paths used to forward ``conversion_rate`` to ``run_ibm_from_eggs``, where
it could never fire: that function hardcodes a single cycle, and the IBM applies
conversion only *between* cycles.  Data generated with a non-zero rate therefore
contained no conversion at all.  Conversion now happens where the IBM itself
does it -- as the eggs are produced -- so these tests pin that down at the level
of the recorded egg counts, which is what the surrogates are trained on.
"""

import numpy as np
import pytest
from IBM_Implementations.helpers.mating import convert_heterozygote_eggs
from IBM_Implementations.input_output_pairs import (
    _run_single_trial,
    generate_pairs_from_fixed_adults,
    generate_pairs_from_fixed_eggs,
)

#: Argument tuple for ``_run_single_trial``, kept small so a trial is quick:
#: (max_adults, pure_genotype_fraction, eggs_per_female, pond_carrying_capacity,
#:  fixed_temperature, feeding_type, trial_seed, genotype_penalties, conversion_rate)
TRIAL_ARGS = (60, 0.0, 10, 400, 20, "interference", 5, None)


class TestConvertHeterozygoteEggs:
    """Unit tests for the shared conversion step."""

    def test_full_rate_converts_every_heterozygote(self):
        eggs_WG, eggs_GG = convert_heterozygote_eggs(120, 30, 1.0)
        assert (eggs_WG, eggs_GG) == (0, 150)

    def test_conversion_conserves_total_eggs(self):
        np.random.seed(0)
        eggs_WG, eggs_GG = convert_heterozygote_eggs(120, 30, 0.5)
        assert eggs_WG + eggs_GG == 150
        assert 0 < eggs_WG < 120

    def test_zero_rate_draws_no_random_numbers(self):
        """A conversion-free run must be bitwise identical to one made before the
        drive existed, so the zero-rate branch may not touch the RNG."""
        np.random.seed(0)
        before = np.random.random()

        np.random.seed(0)
        assert convert_heterozygote_eggs(500, 100, 0.0) == (500, 100)
        after = np.random.random()

        assert before == after


class TestRunSingleTrial:
    """The worker behind ``generate_input_output_pairs`` (the training data)."""

    def test_full_conversion_leaves_no_heterozygote_eggs(self):
        record = _run_single_trial(TRIAL_ARGS + (1.0,) + (False,))

        assert record["input_eggs_WG"] == 0
        assert record["input_eggs_GG"] > 0

    def test_conversion_moves_heterozygotes_into_drive_homozygotes(self):
        """Same seed, so the mating draw is identical and the only difference is
        the drive: every heterozygote egg must reappear as a drive homozygote."""
        no_drive = _run_single_trial(TRIAL_ARGS + (0.0,) + (False,))
        full_drive = _run_single_trial(TRIAL_ARGS + (1.0,) + (False,))

        assert no_drive["input_eggs_WG"] > 0, "test needs heterozygotes to convert"
        assert no_drive["input_eggs_WW"] == full_drive["input_eggs_WW"]
        assert full_drive["input_eggs_GG"] == (
            no_drive["input_eggs_GG"] + no_drive["input_eggs_WG"]
        )

    def test_recorded_eggs_are_the_cohort_the_ibm_ran(self):
        """Output adults can only come from egg classes that were present, so a
        fully converted cohort cannot yield heterozygote adults."""
        record = _run_single_trial(TRIAL_ARGS + (1.0,) + (False,))

        assert record["output_adults_WG"] == 0


class TestGeneratePairsFromFixedAdults:
    def test_full_conversion_leaves_no_heterozygote_eggs(self):
        df = generate_pairs_from_fixed_adults(
            30,
            30,
            0,
            n_trials=2,
            eggs_per_female=8,
            pond_carrying_capacity=300,
            seed=4,
            conversion_rate=1.0,
        )

        assert (df["input_eggs_WG"] == 0).all()
        assert (df["input_eggs_GG"] > 0).all()


class TestGeneratePairsFromFixedEggs:
    """This path has no mating step, so the drive acts on the supplied cohort."""

    def test_full_conversion_leaves_no_heterozygote_eggs(self):
        df = generate_pairs_from_fixed_eggs(
            100, 100, 0, n_trials=2, seed=4, conversion_rate=1.0
        )

        assert (df["input_eggs_WG"] == 0).all()
        assert (df["input_eggs_GG"] == 100).all()

    def test_zero_conversion_records_the_supplied_cohort(self):
        df = generate_pairs_from_fixed_eggs(
            100, 100, 0, n_trials=2, seed=4, conversion_rate=0.0
        )

        assert (df["input_eggs_WG"] == 100).all()
        assert (df["input_eggs_GG"] == 0).all()


def test_run_ibm_from_eggs_rejects_a_conversion_rate():
    """The parameter was a trap -- it could never fire from a single-cycle run --
    so passing one must fail loudly rather than be silently ignored."""
    from IBM_Implementations.input_output_pairs import run_ibm_from_eggs

    with pytest.raises(TypeError):
        run_ibm_from_eggs(100, 0, 0, 20, "interference", conversion_rate=1.0)
