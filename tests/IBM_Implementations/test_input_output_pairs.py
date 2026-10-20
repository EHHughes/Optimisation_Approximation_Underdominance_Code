"""Regression tests for input_output_pairs.py.

These tests pin the current behaviour of every public function so that
refactoring (e.g. to support flexible gene-drive designs) cannot silently
change results. They are *not* tests of biological correctness — that is
established by the R-synchronisation suite. Their sole job is to catch
unintended changes.

Fast tests (no IBM) are in TestBreedAdultsToEggs.
Slow tests (full IBM cycle) are marked with @pytest.mark.slow and kept to
the minimum necessary for confidence.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from Gene_Drive_Optimisation.gene_drive_objective import (
    PARAM_NAMES,
    load_gene_drive_parameters,
    rank_pareto_points,
)
from IBM_Implementations.input_output_pairs import (
    breed_adults_to_eggs,
    generate_input_output_pairs,
    generate_pairs_from_fixed_adults,
    generate_pairs_from_fixed_eggs,
    run_ibm_from_eggs,
    write_input_output_pairs,
)
from paths import input_output_pairs_path

# ---------------------------------------------------------------------------
# breed_adults_to_eggs
# ---------------------------------------------------------------------------


class TestBreedAdultsToEggs:
    """Unit tests for breed_adults_to_eggs.

    These tests run in milliseconds because they never touch the IBM.
    """

    def test_zero_adults_returns_zero_eggs(self):
        """If there are no adults at all, no eggs can be produced.

        This guards the early-return branch at the top of the function and
        ensures that a population-extinction event does not generate phantom
        offspring.
        """
        result = breed_adults_to_eggs(0, 0, 0)
        assert result == (0, 0, 0)

    def test_pure_WW_produces_only_WW_eggs(self):
        """A population with only WW adults cannot carry the G allele.

        Because neither parent has a G allele, every offspring must be WW
        regardless of mating draws or sex assignment. This is a hard genetic
        invariant, not a statistical one.
        """
        eggs_WW, eggs_WG, eggs_GG = breed_adults_to_eggs(200, 0, 0, eggs_per_female=30)
        assert eggs_WG == 0, "WW-only adults cannot produce WG eggs"
        assert eggs_GG == 0, "WW-only adults cannot produce GG eggs"
        assert eggs_WW > 0, "WW-only adults should produce WW eggs"

    def test_pure_GG_produces_only_GG_eggs(self):
        """A population with only GG adults cannot carry the W allele.

        Symmetric to the WW case: every offspring must be GG.
        """
        eggs_WW, eggs_WG, eggs_GG = breed_adults_to_eggs(0, 0, 200, eggs_per_female=30)
        assert eggs_WW == 0, "GG-only adults cannot produce WW eggs"
        assert eggs_WG == 0, "GG-only adults cannot produce WG eggs"
        assert eggs_GG > 0, "GG-only adults should produce GG eggs"

    def test_total_eggs_bounded_by_females_times_eggs_per_female(self):
        """Total eggs must not exceed total_females * eggs_per_female.

        Females are assigned by binomial split, so total_females ≤ total_adults.
        Total eggs = total_females * eggs_per_female (before capacity cap).
        """
        total_adults = 100
        eggs_per_female = 20
        eggs_WW, eggs_WG, eggs_GG = breed_adults_to_eggs(
            total_adults, 0, 0, eggs_per_female=eggs_per_female
        )
        max_possible = total_adults * eggs_per_female  # upper bound (all female)
        assert eggs_WW + eggs_WG + eggs_GG <= max_possible

    def test_carrying_capacity_caps_total_eggs(self):
        """When pond_carrying_capacity is finite, total eggs must not exceed it.

        This guards the capacity-clipping branch that randomly subsamples the
        offspring list down to the cap.
        """
        cap = 50
        eggs_WW, eggs_WG, eggs_GG = breed_adults_to_eggs(
            200, 0, 0, eggs_per_female=30, pond_carrying_capacity=cap
        )
        assert eggs_WW + eggs_WG + eggs_GG <= cap

    def test_gene_drive_converts_WG_to_GG(self):
        """A positive heterozygote_conversion_rate must reduce WG and increase GG.

        With a conversion rate of 1.0 every WG offspring is converted to GG,
        so the final WG count must be zero and GG must equal the number of
        heterozygotes that would have been produced without conversion.
        """
        # Use a cross that guarantees heterozygotes: 100 WW × 100 GG adults.
        # After sex assignment some WW will be female and some GG will be male
        # (or vice versa), producing WG offspring.  With conversion_rate=1 all
        # WG become GG.
        np.random.seed(42)
        eggs_WW_no_drive, eggs_WG_no_drive, eggs_GG_no_drive = breed_adults_to_eggs(
            100, 0, 100, eggs_per_female=10, heterozygote_conversion_rate=0.0
        )

        np.random.seed(42)
        eggs_WW_drive, eggs_WG_drive, eggs_GG_drive = breed_adults_to_eggs(
            100, 0, 100, eggs_per_female=10, heterozygote_conversion_rate=1.0
        )

        assert eggs_WG_drive == 0, "All WG should be converted when rate=1"
        assert eggs_GG_drive == eggs_GG_no_drive + eggs_WG_no_drive, (
            "Converted WG eggs should be added to GG count"
        )
        # WW count is unaffected
        assert eggs_WW_drive == eggs_WW_no_drive

    def test_no_gene_drive_with_zero_conversion_rate(self):
        """With conversion_rate=0, genotype counts must be identical to the baseline.

        Ensures the gene-drive branch is not entered when it should not be.
        """
        np.random.seed(7)
        result_no_drive = breed_adults_to_eggs(
            50, 50, 50, eggs_per_female=10, heterozygote_conversion_rate=0.0
        )
        np.random.seed(7)
        result_zero_rate = breed_adults_to_eggs(
            50, 50, 50, eggs_per_female=10, heterozygote_conversion_rate=0.0
        )
        assert result_no_drive == result_zero_rate

    def test_reproducibility_with_seed(self):
        """Identical seeds must yield identical egg counts.

        This verifies that numpy's global RNG is the sole source of
        randomness and that the function has no hidden state.
        """
        np.random.seed(123)
        result1 = breed_adults_to_eggs(100, 50, 50, eggs_per_female=20)
        np.random.seed(123)
        result2 = breed_adults_to_eggs(100, 50, 50, eggs_per_female=20)
        assert result1 == result2

    def test_returns_non_negative_integers(self):
        """All returned egg counts must be non-negative integers."""
        np.random.seed(0)
        eggs_WW, eggs_WG, eggs_GG = breed_adults_to_eggs(40, 30, 30, eggs_per_female=15)
        for val, name in [(eggs_WW, "WW"), (eggs_WG, "WG"), (eggs_GG, "GG")]:
            assert isinstance(val, int), f"eggs_{name} should be int, got {type(val)}"
            assert val >= 0, f"eggs_{name} must be non-negative"

    def test_genotype_allele_normalisation(self):
        """Offspring genotypes must be stored as WG, never GW.

        The function sorts allele pairs so 'GW' becomes 'WG'.  If this
        normalisation is lost the counts of heterozygotes would be zero even
        when heterozygotes exist.
        """
        np.random.seed(5)
        # Mixed population guarantees heterozygotes will be produced
        eggs_WW, eggs_WG, eggs_GG = breed_adults_to_eggs(
            100, 0, 100, eggs_per_female=20
        )
        # With many adults and 20 eggs/female there must be some heterozygotes
        # (extremely unlikely to be zero with seed=5)
        assert eggs_WG > 0, (
            "Expected heterozygote eggs from WW×GG cross; "
            "check allele normalisation (GW → WG)"
        )


# ---------------------------------------------------------------------------
# run_ibm_from_eggs
# ---------------------------------------------------------------------------


class TestRunIbmFromEggs:
    """Integration tests for run_ibm_from_eggs.

    Each test runs one full IBM cycle and is therefore slower than the pure
    genetics tests above.  Keep the egg counts small enough to remain fast.
    """

    def test_zero_eggs_returns_none(self):
        """When all egg counts are zero the function must return None.

        Callers depend on this sentinel to skip empty generations.
        """
        result = run_ibm_from_eggs(
            0, 0, 0, fixed_temperature=20, feeding_type="interference"
        )
        assert result is None

    def test_returns_three_non_negative_integers(self):
        """Normal inputs must return a 3-tuple of non-negative integers."""
        result = run_ibm_from_eggs(
            100, 0, 0, fixed_temperature=20, feeding_type="interference"
        )
        assert result is not None
        assert len(result) == 3
        for val in result:
            assert isinstance(val, int), f"Expected int, got {type(val)}"
            assert val >= 0

    def test_output_adults_do_not_exceed_input_eggs(self):
        """Biological constraint: adults out ≤ eggs in.

        Each adult started as exactly one egg; survival is ≤ 100 %.
        """
        total_eggs = 200
        result = run_ibm_from_eggs(
            total_eggs, 0, 0, fixed_temperature=20, feeding_type="interference"
        )
        assert result is not None
        assert sum(result) <= total_eggs

    def test_pure_WW_eggs_produce_only_WW_adults(self):
        """WW eggs carry no G allele; no GG or WG adults should emerge."""
        result = run_ibm_from_eggs(
            150, 0, 0, fixed_temperature=20, feeding_type="interference"
        )
        assert result is not None
        adults_WW, adults_WG, adults_GG = result
        assert adults_WG == 0, "WW-only eggs cannot produce WG adults"
        assert adults_GG == 0, "WW-only eggs cannot produce GG adults"

    def test_pure_GG_eggs_produce_only_GG_adults(self):
        """GG eggs carry no W allele; no WW or WG adults should emerge."""
        result = run_ibm_from_eggs(
            0, 0, 150, fixed_temperature=20, feeding_type="interference"
        )
        assert result is not None
        adults_WW, adults_WG, adults_GG = result
        assert adults_WW == 0, "GG-only eggs cannot produce WW adults"
        assert adults_WG == 0, "GG-only eggs cannot produce WG adults"

    def test_exploitative_feeding_also_returns_valid_result(self):
        """Both feeding types must return a valid result tuple."""
        result = run_ibm_from_eggs(
            100, 0, 0, fixed_temperature=20, feeding_type="exploitative"
        )
        assert result is not None
        assert len(result) == 3


# ---------------------------------------------------------------------------
# generate_input_output_pairs
# ---------------------------------------------------------------------------


class TestGenerateInputOutputPairs:
    """Tests for the top-level pair-generation function."""

    def test_returns_dataframe_with_genetics_columns(self):
        """Return value must be a DataFrame containing genotype-level columns."""
        with tempfile.TemporaryDirectory() as tmp:
            df = generate_input_output_pairs(
                n_trials=2, fixed_temperature=20, seed=1, output_dir=tmp
            )
        expected_cols = {
            "input_adults_WW",
            "input_adults_WG",
            "input_adults_GG",
            "output_adults_WW",
            "output_adults_WG",
            "output_adults_GG",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_row_count_at_most_n_trials(self):
        """Result cannot have more rows than n_trials (some trials may return None)."""
        n = 2
        with tempfile.TemporaryDirectory() as tmp:
            df = generate_input_output_pairs(
                n_trials=n, fixed_temperature=20, seed=2, output_dir=tmp
            )
        assert len(df) <= n

    def test_seeded_runs_are_reproducible(self):
        """Two calls with the same seed must return identical DataFrames."""
        with tempfile.TemporaryDirectory() as tmp1:
            df1 = generate_input_output_pairs(
                n_trials=2, fixed_temperature=20, seed=99, output_dir=tmp1
            )
        with tempfile.TemporaryDirectory() as tmp2:
            df2 = generate_input_output_pairs(
                n_trials=2, fixed_temperature=20, seed=99, output_dir=tmp2
            )
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
        )

    def test_csv_files_are_created(self):
        """Both the plain and the genetics CSV must be written to output_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            generate_input_output_pairs(
                n_trials=2, fixed_temperature=20, seed=3, output_dir=tmp
            )
            files = list(Path(tmp).glob("*.csv"))
        assert len(files) == 2, f"Expected 2 CSV files, found {len(files)}: {files}"

    def test_test_flag_creates_test_set_filenames(self):
        """When test=True the CSV filenames must carry a 'test=True' marker."""
        with tempfile.TemporaryDirectory() as tmp:
            generate_input_output_pairs(
                n_trials=2, fixed_temperature=20, seed=4, output_dir=tmp, test=True
            )
            files = list(Path(tmp).glob("*.csv"))
        assert all("test=True" in f.name for f in files), (
            f"test=True should produce test=True filenames, got: {[f.name for f in files]}"
        )

    def test_append_grows_csv_row_count(self):
        """Calling with append_to_existing=True must increase the row count."""
        with tempfile.TemporaryDirectory() as tmp:
            generate_input_output_pairs(
                n_trials=2,
                fixed_temperature=20,
                seed=5,
                output_dir=tmp,
                append_to_existing=False,
            )
            # Find the plain (non-genetics) CSV
            csv_files = list(Path(tmp).glob("input_output_pairs_seed=*.csv"))
            # Exclude the genetics file
            plain_csv = [f for f in csv_files if "genetics" not in f.name][0]
            rows_before = len(pd.read_csv(plain_csv))

            generate_input_output_pairs(
                n_trials=2,
                fixed_temperature=20,
                seed=5,
                output_dir=tmp,
                append_to_existing=True,
            )
            rows_after = len(pd.read_csv(plain_csv))

        assert rows_after > rows_before, "append_to_existing=True should grow the CSV"

    def test_output_counts_are_non_negative(self):
        """All adult counts in the returned DataFrame must be ≥ 0."""
        with tempfile.TemporaryDirectory() as tmp:
            df = generate_input_output_pairs(
                n_trials=8, fixed_temperature=20, seed=6, output_dir=tmp
            )
        for col in ["output_adults_WW", "output_adults_WG", "output_adults_GG"]:
            assert (df[col] >= 0).all(), f"Negative values found in {col}"


# ---------------------------------------------------------------------------
# generate_pairs_from_fixed_adults
# ---------------------------------------------------------------------------


class TestGeneratePairsFromFixedAdults:
    """Tests for generate_pairs_from_fixed_adults."""

    def test_input_adults_are_fixed_across_all_rows(self):
        """Every row must report the same fixed adult input counts.

        This is the defining contract of the function: the input population
        does not vary across trials.
        """
        adults = (80, 20, 0)
        df = generate_pairs_from_fixed_adults(
            *adults, n_trials=2, fixed_temperature=20, seed=7
        )
        assert (df["input_adults_WW"] == adults[0]).all()
        assert (df["input_adults_WG"] == adults[1]).all()
        assert (df["input_adults_GG"] == adults[2]).all()

    def test_returns_dataframe_with_expected_columns(self):
        """Return value must be a DataFrame with both input and output columns."""
        df = generate_pairs_from_fixed_adults(
            50, 0, 0, n_trials=2, fixed_temperature=20, seed=8
        )
        expected_cols = {
            "input_adults_WW",
            "input_adults_WG",
            "input_adults_GG",
            "input_eggs_WW",
            "input_eggs_WG",
            "input_eggs_GG",
            "output_adults_WW",
            "output_adults_WG",
            "output_adults_GG",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_seeded_runs_are_reproducible(self):
        """Two calls with the same seed must return identical DataFrames."""
        kwargs = dict(
            adults_WW=60,
            adults_WG=0,
            adults_GG=0,
            n_trials=2,
            fixed_temperature=20,
            seed=9,
        )
        df1 = generate_pairs_from_fixed_adults(**kwargs)
        df2 = generate_pairs_from_fixed_adults(**kwargs)
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
        )

    def test_save_csv_false_does_not_create_file(self):
        """save_csv=False must not write any CSV files."""
        with tempfile.TemporaryDirectory() as tmp:
            generate_pairs_from_fixed_adults(
                50,
                0,
                0,
                n_trials=2,
                fixed_temperature=20,
                seed=10,
                save_csv=False,
                output_dir=tmp,
            )
            assert list(Path(tmp).glob("*.csv")) == [], (
                "save_csv=False should not create any CSV files"
            )

    def test_save_csv_true_creates_file(self):
        """save_csv=True must write a CSV file to output_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            generate_pairs_from_fixed_adults(
                50,
                0,
                0,
                n_trials=2,
                fixed_temperature=20,
                seed=11,
                save_csv=True,
                output_dir=tmp,
            )
            assert len(list(Path(tmp).glob("*.csv"))) == 1


# ---------------------------------------------------------------------------
# generate_pairs_from_fixed_eggs
# ---------------------------------------------------------------------------


class TestGeneratePairsFromFixedEggs:
    """Tests for generate_pairs_from_fixed_eggs."""

    def test_input_eggs_are_fixed_across_all_rows(self):
        """Every row must report the same fixed egg input counts."""
        eggs = (200, 50, 0)
        df = generate_pairs_from_fixed_eggs(
            *eggs, n_trials=2, fixed_temperature=20, seed=12
        )
        assert (df["input_eggs_WW"] == eggs[0]).all()
        assert (df["input_eggs_WG"] == eggs[1]).all()
        assert (df["input_eggs_GG"] == eggs[2]).all()

    def test_returns_dataframe_with_expected_columns(self):
        """Return value must be a DataFrame with egg inputs and adult outputs."""
        df = generate_pairs_from_fixed_eggs(
            100, 0, 0, n_trials=1, fixed_temperature=20, seed=13
        )
        expected_cols = {
            "input_eggs_WW",
            "input_eggs_WG",
            "input_eggs_GG",
            "output_adults_WW",
            "output_adults_WG",
            "output_adults_GG",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_seeded_runs_are_reproducible(self):
        """Two calls with the same seed must return identical DataFrames."""
        kwargs = dict(
            eggs_WW=100, eggs_WG=0, eggs_GG=0, n_trials=2, fixed_temperature=20, seed=14
        )
        df1 = generate_pairs_from_fixed_eggs(**kwargs)
        df2 = generate_pairs_from_fixed_eggs(**kwargs)
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
        )

    def test_output_stochasticity_across_identical_inputs(self):
        """Repeated IBM runs with identical egg inputs should not all give the same output.

        This verifies that the stochastic IBM is actually running independently
        each trial rather than caching or returning a constant.
        """
        df = generate_pairs_from_fixed_eggs(
            500,
            0,
            500,
            n_trials=5,
            fixed_temperature=20,
            seed=np.random.randint(int(1e6)),
        )
        # With 20 independent stochastic runs it would be extremely unusual
        # for every output to be identical
        unique_totals = (
            df["output_adults_WW"] + df["output_adults_WG"] + df["output_adults_GG"]
        ).nunique()
        assert unique_totals > 1, (
            "All 20 IBM runs produced identical adult counts — "
            "stochasticity may be broken"
        )


# ---------------------------------------------------------------------------
# write_input_output_pairs
# ---------------------------------------------------------------------------


class TestWriteInputOutputPairs:
    """Unit tests for write_input_output_pairs."""

    def _make_dfs(self):
        """Return minimal plain and genetics DataFrames for testing."""
        genetics_df = pd.DataFrame(
            {
                "input_adults_WW": [10],
                "input_adults_WG": [5],
                "input_adults_GG": [0],
                "output_adults_WW": [8],
                "output_adults_WG": [3],
                "output_adults_GG": [0],
            }
        )
        plain_df = genetics_df[
            ["input_adults_WW", "input_adults_WG", "input_adults_GG"]
        ].copy()
        return plain_df, genetics_df

    def test_creates_two_csv_files(self):
        """Both the plain and the genetics CSV must be written."""
        plain_df, genetics_df = self._make_dfs()
        with tempfile.TemporaryDirectory() as tmp:
            write_input_output_pairs(plain_df, genetics_df, output_dir=tmp, seed=1)
            files = list(Path(tmp).glob("*.csv"))
        assert len(files) == 2

    def test_test_flag_produces_test_set_filenames(self):
        """test=True must embed a 'test=True' marker in both filenames."""
        plain_df, genetics_df = self._make_dfs()
        with tempfile.TemporaryDirectory() as tmp:
            write_input_output_pairs(
                plain_df, genetics_df, output_dir=tmp, seed=2, test=True
            )
            files = list(Path(tmp).glob("*.csv"))
        assert all("test=True" in f.name for f in files)

    def test_append_grows_row_count(self):
        """append_to_existing=True must increase the row count of existing files."""
        plain_df, genetics_df = self._make_dfs()
        with tempfile.TemporaryDirectory() as tmp:
            write_input_output_pairs(plain_df, genetics_df, output_dir=tmp, seed=3)
            plain_csv = next(Path(tmp).glob("input_output_pairs_seed=*.csv"))
            rows_before = len(pd.read_csv(plain_csv))

            write_input_output_pairs(
                plain_df, genetics_df, output_dir=tmp, seed=3, append_to_existing=True
            )
            rows_after = len(pd.read_csv(plain_csv))

        assert rows_after > rows_before

    def test_csv_content_matches_dataframe(self):
        """The written CSV must round-trip back to the original DataFrame."""
        plain_df, genetics_df = self._make_dfs()
        with tempfile.TemporaryDirectory() as tmp:
            write_input_output_pairs(plain_df, genetics_df, output_dir=tmp, seed=4)
            genetics_csv = next(Path(tmp).glob("*with_genetics*.csv"))
            loaded = pd.read_csv(genetics_csv)
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True),
            genetics_df.reset_index(drop=True),
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# Custom gene-drive parameter sets (Pareto-frontier points)
# ---------------------------------------------------------------------------


def _write_pareto_csv(
    path: Path,
    n_rows: int = 3,
    *,
    verified_scores: list[float] | None = None,
    surrogate_scores: list[float] | None = None,
) -> np.ndarray:
    """Write a synthetic Pareto-frontier CSV and return its parameter values.

    Mirrors the real layout: the nine parameter columns first, then the
    objective and verification columns the loader must ignore.  Both objectives
    of a row are given the same value, so a row's mean score is exactly its
    entry in *verified_scores* (or *surrogate_scores*); the verified columns are
    written only when *verified_scores* is given, so a caller can also build a
    CSV that predates them.
    """
    values = np.arange(n_rows * len(PARAM_NAMES), dtype=float).reshape(
        n_rows, len(PARAM_NAMES)
    ) / 100.0
    dataframe = pd.DataFrame(values, columns=PARAM_NAMES)
    surrogate = [0.5] * n_rows if surrogate_scores is None else surrogate_scores
    dataframe["f_1 (low g frequency)"] = surrogate
    dataframe["f_2 (high g frequency)"] = surrogate
    if verified_scores is not None:
        dataframe["f_1 (low g frequency) [verified mean]"] = verified_scores
        dataframe["f_2 (high g frequency) [verified mean]"] = verified_scores
        dataframe["f_1 (low g frequency) [verified var]"] = 0.01
        dataframe["f_2 (high g frequency) [verified var]"] = 0.01
    dataframe.to_csv(path, index=False)
    return values


class TestInputOutputPairsPathParamTag:
    """The shared naming scheme must stay backwards compatible."""

    def test_no_tag_leaves_filename_unchanged(self):
        """param_tag=None must reproduce the pre-existing filenames exactly."""
        assert (
            input_output_pairs_path(27, with_genetics=True, param_tag=None).name
            == "input_output_pairs_with_genetics_seed=27.csv"
        )
        assert (
            input_output_pairs_path(61, test=True, param_tag=None).name
            == "input_output_pairs_seed=61_test=True.csv"
        )

    def test_tag_is_embedded_before_the_test_marker(self):
        """A tag must appear as _params=<tag>, ahead of any test=True marker."""
        name = input_output_pairs_path(
            61, with_genetics=True, test=True, param_tag="low0p50_high0p70_row0"
        ).name
        assert (
            name
            == "input_output_pairs_with_genetics_seed=61_params=low0p50_high0p70_row0_test=True.csv"
        )


class TestWriteInputOutputPairsParamTag:
    """write_input_output_pairs must honour param_tag for both files."""

    def test_tagged_filenames_are_written(self):
        genetics_df = pd.DataFrame(
            {
                "input_adults_WW": [10],
                "input_adults_WG": [5],
                "input_adults_GG": [0],
                "output_adults_WW": [8],
                "output_adults_WG": [3],
                "output_adults_GG": [0],
            }
        )
        plain_df = genetics_df[
            ["input_adults_WW", "input_adults_WG", "input_adults_GG"]
        ].copy()

        with tempfile.TemporaryDirectory() as tmp:
            write_input_output_pairs(
                plain_df, genetics_df, output_dir=tmp, seed=5, param_tag="mytag"
            )
            files = sorted(f.name for f in Path(tmp).glob("*.csv"))

        assert files == [
            "input_output_pairs_seed=5_params=mytag.csv",
            "input_output_pairs_with_genetics_seed=5_params=mytag.csv",
        ]


class TestLoadGeneDriveParameters:
    """Unit tests for reading a gene-drive design out of a Pareto CSV."""

    def test_row_maps_to_the_expected_penalties(self):
        """Column order must match the objective's own reshape of X."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gene_drive_pareto_frontier_points_20260612_084001.csv"
            values = _write_pareto_csv(csv_path)

            penalties, conversion_rate, _ = load_gene_drive_parameters(csv_path, 1)

        row = values[1]
        assert conversion_rate == pytest.approx(row[0])
        assert penalties["WW"] == {
            "a_fac": 1,
            "alpha_fac": 1,
            "beta_fac": 1.0,
            "g_fac": 1.0,
        }
        for offset, param in enumerate(["a_fac", "alpha_fac", "beta_fac", "g_fac"]):
            assert penalties["WG"][param] == pytest.approx(row[1 + offset])
            assert penalties["GG"][param] == pytest.approx(row[5 + offset])

    def test_objective_columns_are_ignored(self):
        """Selection is by name, so trailing objective columns must not leak in."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gene_drive_pareto_frontier_points_20260612_084001.csv"
            values = _write_pareto_csv(csv_path)
            # Reorder so the objective columns sit *between* the parameters.
            dataframe = pd.read_csv(csv_path)
            dataframe = dataframe[
                PARAM_NAMES[:5]
                + ["f_1 (low g frequency)", "f_2 (high g frequency)"]
                + PARAM_NAMES[5:]
            ]
            dataframe.to_csv(csv_path, index=False)

            penalties, _, _ = load_gene_drive_parameters(csv_path, 0)

        assert penalties["GG"]["a_fac"] == pytest.approx(values[0][5])

    def test_tag_uses_threshold_directory_and_timestamp(self):
        """The tag must identify the run directory, timestamp and row."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "low0p50_high0p70"
            run_dir.mkdir()
            csv_path = run_dir / "gene_drive_pareto_frontier_points_20260612_084001.csv"
            _write_pareto_csv(csv_path)

            _, _, tag = load_gene_drive_parameters(csv_path, 2)

        assert tag == "low0p50_high0p70_20260612_084001_row2"

    def test_tag_falls_back_for_legacy_flat_layout(self):
        """Legacy CSVs with the thresholds in the filename must still tag."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = (
                Path(tmp)
                / "gene_drive_pareto_frontier_points_low0p30_high0p50.csv"
            )
            _write_pareto_csv(csv_path)

            _, _, tag = load_gene_drive_parameters(csv_path, 0)

        assert tag == "low0p30_high0p50_row0"

    def test_out_of_range_row_raises_index_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "gene_drive_pareto_frontier_points_20260612_084001.csv"
            _write_pareto_csv(csv_path, n_rows=3)

            with pytest.raises(IndexError, match="out of range"):
                load_gene_drive_parameters(csv_path, 3)

    def test_missing_columns_raise_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "not_a_pareto_file.csv"
            pd.DataFrame({"conversion_rate": [0.0]}).to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="missing the gene-drive parameter"):
                load_gene_drive_parameters(csv_path, 0)


class TestRankParetoPoints:
    """Ranking the pooled Pareto points of a directory of optimisation runs."""

    @staticmethod
    def _write_two_replicates(directory: Path) -> tuple[Path, Path]:
        """Two replicate CSVs whose best points sit in different files."""
        first = directory / "gene_drive_pareto_frontier_points_20260609_101608.csv"
        second = directory / "gene_drive_pareto_frontier_points_20260612_084001.csv"
        _write_pareto_csv(first, n_rows=3, verified_scores=[0.10, 0.90, 0.50])
        _write_pareto_csv(second, n_rows=3, verified_scores=[0.70, 0.20, 0.80])
        return first, second

    def test_points_are_pooled_across_replicates_and_ordered_by_score(self):
        """The best drive of the directory may live in any of its runs."""
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self._write_two_replicates(Path(tmp))

            ranked = rank_pareto_points(Path(tmp), top_n=4)

        assert [(path, row) for path, row, _ in ranked] == [
            (first, 1),
            (second, 2),
            (second, 0),
            (first, 2),
        ]
        assert [score for _, _, score in ranked] == pytest.approx(
            [0.90, 0.80, 0.70, 0.50]
        )

    def test_rows_index_back_into_their_own_file(self):
        """The returned (path, row) pairs must be usable as loader arguments."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_two_replicates(Path(tmp))

            best_path, best_row, _ = rank_pareto_points(Path(tmp), top_n=1)[0]
            _, conversion_rate, tag = load_gene_drive_parameters(best_path, best_row)
            expected = pd.read_csv(best_path)["conversion_rate"].iloc[best_row]

        assert conversion_rate == pytest.approx(expected)
        assert tag.endswith(f"_row{best_row}")

    def test_top_n_is_capped_at_the_points_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_two_replicates(Path(tmp))

            ranked = rank_pareto_points(Path(tmp), top_n=50)

        assert len(ranked) == 6

    def test_surrogate_objectives_can_be_ranked_instead(self):
        """--surrogate-style ranking must read the raw objective columns."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = (
                Path(tmp) / "gene_drive_pareto_frontier_points_20260612_084001.csv"
            )
            # Verified and surrogate scores disagree, so the choice shows up in
            # which row comes first.
            _write_pareto_csv(
                csv_path,
                n_rows=3,
                verified_scores=[0.9, 0.1, 0.1],
                surrogate_scores=[0.1, 0.1, 0.9],
            )

            verified = rank_pareto_points(Path(tmp), top_n=1)
            surrogate = rank_pareto_points(Path(tmp), top_n=1, use_verified=False)

        assert verified[0][1] == 0
        assert surrogate[0][1] == 2

    def test_ties_are_broken_deterministically(self):
        """Equal scores must not leave the selection at the mercy of glob order."""
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "gene_drive_pareto_frontier_points_20260609_101608.csv"
            second = Path(tmp) / "gene_drive_pareto_frontier_points_20260612_084001.csv"
            _write_pareto_csv(first, n_rows=2, verified_scores=[0.5, 0.5])
            _write_pareto_csv(second, n_rows=2, verified_scores=[0.5, 0.5])

            ranked = rank_pareto_points(Path(tmp), top_n=4)

        assert [(path, row) for path, row, _ in ranked] == [
            (first, 0),
            (first, 1),
            (second, 0),
            (second, 1),
        ]

    def test_missing_verified_columns_raise_value_error(self):
        """A pre-verification CSV must fail loudly, not rank by surrogate values."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = (
                Path(tmp) / "gene_drive_pareto_frontier_points_20260612_084001.csv"
            )
            _write_pareto_csv(csv_path)

            with pytest.raises(ValueError, match="missing the objective column"):
                rank_pareto_points(Path(tmp))

    def test_directory_without_pareto_files_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "input_output_pairs_seed=27.csv").write_text("nothing\n")

            with pytest.raises(FileNotFoundError, match="No gene_drive_pareto"):
                rank_pareto_points(Path(tmp))

    def test_non_positive_top_n_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_two_replicates(Path(tmp))

            with pytest.raises(ValueError, match="top_n must be at least 1"):
                rank_pareto_points(Path(tmp), top_n=0)


# ---------------------------------------------------------------------------
# Golden-file regression test
# ---------------------------------------------------------------------------


class TestHomozygotesOnlySampling:
    """``homozygotes_only=True`` samples the WW/GG face only.

    Underdominance architectures make the heterozygote non-viable, so the
    dynamics never visit a state with WG adults and the default three-genotype
    sampling spends most of its trials outside the relevant region.

    The nine-column schema is deliberately unchanged in this mode: the
    dataloaders slice positionally, and the recorded ``input_eggs_WG`` /
    ``output_adults_WG`` are what makes
    ``IBM_Emulation.dataloaders.report_heterozygote_residual`` possible.
    """

    @pytest.fixture(scope="class")
    def homozygote_pairs(self, tmp_path_factory) -> pd.DataFrame:
        tmp = tmp_path_factory.mktemp("homozygotes_only")
        return generate_input_output_pairs(
            n_trials=12,
            fixed_temperature=20,
            seed=11,
            output_dir=str(tmp),
            homozygotes_only=True,
        )

    def test_no_heterozygote_input_adults(self, homozygote_pairs):
        assert (homozygote_pairs["input_adults_WG"] == 0).all()

    def test_both_homozygotes_vary(self, homozygote_pairs):
        """A Dirichlet([1, 1]) split, not a collapse onto one genotype."""
        assert homozygote_pairs["input_adults_WW"].nunique() > 1
        assert homozygote_pairs["input_adults_GG"].nunique() > 1

    def test_totals_still_span_a_range_of_population_sizes(self, homozygote_pairs):
        totals = (
            homozygote_pairs["input_adults_WW"] + homozygote_pairs["input_adults_GG"]
        )
        assert totals.min() >= 1
        assert totals.nunique() > 1

    def test_schema_is_the_nine_standard_columns(self, homozygote_pairs):
        assert list(homozygote_pairs.columns) == [
            "input_adults_WW",
            "input_adults_WG",
            "input_adults_GG",
            "input_eggs_WW",
            "input_eggs_WG",
            "input_eggs_GG",
            "output_adults_WW",
            "output_adults_WG",
            "output_adults_GG",
        ]

    def test_heterozygote_eggs_are_still_recorded(self, homozygote_pairs):
        """Every WW x GG cross lays WG eggs; the file must not hide them.

        This is the whole basis of the residual diagnostic: a surrogate that
        asserts WG = 0 is only sound if those eggs die, and that can only be
        checked if the counts are kept.
        """
        mixed = homozygote_pairs[
            (homozygote_pairs["input_adults_WW"] > 0)
            & (homozygote_pairs["input_adults_GG"] > 0)
        ]
        assert not mixed.empty, "sampling produced no mixed WW/GG populations"
        assert (mixed["input_eggs_WG"] > 0).any()

    def test_filenames_carry_the_homozygotes_marker(self, tmp_path):
        generate_input_output_pairs(
            n_trials=2,
            fixed_temperature=20,
            seed=12,
            output_dir=str(tmp_path),
            homozygotes_only=True,
        )
        written = sorted(path.name for path in tmp_path.glob("*.csv"))
        # The marker sits after the seed, alongside _params= and _test=True
        # (see paths.input_output_pairs_path).
        assert written == [
            "input_output_pairs_seed=12_homozygotes=True.csv",
            "input_output_pairs_with_genetics_seed=12_homozygotes=True.csv",
        ]

    def test_default_sampling_still_produces_heterozygotes(self, tmp_path):
        """The opt-in must not have changed the default behaviour.

        ``TestGenerateInputOutputPairsGoldenFile`` is the bit-exact guard; this
        is the cheap statement of intent beside it.
        """
        pairs = generate_input_output_pairs(
            n_trials=12,
            fixed_temperature=20,
            seed=11,
            output_dir=str(tmp_path),
        )
        assert (pairs["input_adults_WG"] > 0).any()


class TestGenerateInputOutputPairsGoldenFile:
    """Bit-exact comparison against a saved fixture.

                The fixture (fixtures/generate_input_output_pairs_seed42_n4.csv) was
                produced by a single authoritative run of generate_input_output_pairs
                with seed=42 and n_trials=4.  Any change to the genetics, mating, IBM
                integration, or seed-derivation logic will cause this test to fail,
                making it the single most sensitive regression guard in the suite.

                To intentionally update the fixture after a deliberate behavioural change:
    python -c "
    import tempfile
    from IBM_Implementations.input_output_pairs import generate_input_output_pairs
    with tempfile.TemporaryDirectory() as tmp:
        df = generate_input_output_pairs(n_trials=4, fixed_temperature=20,
                                            seed=42, output_dir=tmp)
    df.to_csv('generate_input_output_pairs_seed42_n4.csv',
                index=False)
            "
                then commit the updated fixture alongside the code change.
    """

    @pytest.fixture
    def fixtures_path(self) -> Path:
        return Path(__file__).parent / "fixtures"

    def test_output_matches_golden_file(self, fixtures_path: Path):
        """Full pipeline output must be bit-exact against the saved fixture.

        Covers: per-trial seed derivation from master seed, adult sampling,
        breed_adults_to_eggs, run_ibm_from_eggs, and DataFrame assembly.
        """
        expected = pd.read_csv(
            fixtures_path / "generate_input_output_pairs_seed42_n4.csv"
        )

        with tempfile.TemporaryDirectory() as tmp:
            actual = generate_input_output_pairs(
                n_trials=4,
                fixed_temperature=20,
                seed=42,
                output_dir=tmp,
            )

        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
