"""Tests for gene-drive releases into a single equilibrated population.

The driver is exercised against a stand-in for the IBM cycle: a real cycle
costs about fifteen seconds, and a run is dozens of them, which would make
these tests minutes long without testing anything the IBM's own tests do not
already cover.  What matters here is the wiring -- that the release lands at
the frequency it was asked for, that replicates are paired across frequencies,
that the recorded counts are internally consistent, and that the figures are
built from them.

The one thing the stand-in cannot check is that the new stage-count runner
agrees with the adult-count one it now backs, so that is tested against the
real IBM and marked slow.
"""

import numpy as np
import pandas as pd
import pytest

import plot_style
import simulate_single_location_release as release_script
from Gene_Drive_Optimisation import single_location_releases
from Gene_Drive_Optimisation.plot_single_location_releases import (
    plot_release_diagnostics,
    plot_release_trajectories,
    plot_threshold_diagnostics,
)
from Gene_Drive_Optimisation.single_location_releases import (
    PHASE_BURN_IN,
    PHASE_POST_RELEASE,
    PHASE_RELEASE,
    PHASE_SEED,
    add_derived_columns,
    apply_release,
    classify_outcomes,
    equilibrium_summary,
    simulate_single_location_releases,
)
from IBM_Implementations.input_output_pairs import (
    GENOTYPES,
    run_ibm_cycle_from_eggs,
    run_ibm_from_eggs,
)


def _fake_cycle(
    eggs_WW, eggs_WG, eggs_GG, temperature, feeding_type, genotype_penalties=None
):
    """Stand-in for one IBM cycle: cheap, density-dependent, drive-sensitive.

    Mirrors the real signature and the real invariants -- larvae split into
    exactly the three fates, and survival falls as the cohort grows -- while
    giving GG an advantage and WG a penalty, so an underdominant drive has
    something to do.  Draws from numpy's global RNG, as the IBM does, so an
    unseeded replicate would show up as a repeated result.
    """
    total = eggs_WW + eggs_WG + eggs_GG
    if total == 0:
        return None

    advantage = {"WW": 1.0, "WG": 0.3, "GG": 1.4}
    counts = {"WW": eggs_WW, "WG": eggs_WG, "GG": eggs_GG}
    stage_counts = {}
    for genotype, eggs in counts.items():
        larvae = int(eggs * 0.9)
        survival = min(0.95, advantage[genotype] * 120.0 / max(total, 1))
        pupae = int(np.random.binomial(larvae, survival))
        starved = int((larvae - pupae) * 0.7)
        stage_counts[f"Eggs_{genotype}"] = eggs
        stage_counts[f"Larvae_{genotype}"] = larvae
        stage_counts[f"Pupae_{genotype}"] = pupae
        stage_counts[f"Adults_{genotype}"] = pupae
        stage_counts[f"Starved_{genotype}"] = starved
        stage_counts[f"TimedOut_{genotype}"] = larvae - pupae - starved
    return stage_counts


@pytest.fixture
def fake_ibm(monkeypatch):
    """Swap the IBM cycle for the stand-in for the duration of a test.

    Worker processes are forked, so they inherit the patched module global.
    """
    monkeypatch.setattr(single_location_releases, "run_ibm_cycle_from_eggs", _fake_cycle)


@pytest.fixture
def release_run(fake_ibm):
    """A small, seeded run shared by the tests that only read its output."""
    return simulate_single_location_releases(
        release_frequencies=(0.1, 0.3, 0.5),
        conversion_rate=0.4,
        n_replicates=3,
        n_burn_in_generations=3,
        n_generations=4,
        seed=4321,
        n_workers=1,
    )


class TestApplyRelease:
    """The release has to land at the frequency it was asked for."""

    @pytest.mark.parametrize("frequency", [0.0, 0.1, 0.25, 0.5, 0.9])
    def test_additive_release_hits_the_requested_frequency(self, frequency):
        released = apply_release(np.array([100, 0, 0]), frequency)
        total = released.sum()
        assert released[2] / total == pytest.approx(frequency, abs=0.01)

    @pytest.mark.parametrize("frequency", [0.1, 0.3, 0.5])
    def test_additive_release_leaves_residents_untouched(self, frequency):
        residents = np.array([70, 5, 0])
        released = apply_release(residents, frequency)
        assert released[0] == residents[0]
        assert released[1] == residents[1]
        assert released[2] > 0

    @pytest.mark.parametrize("frequency", [0.1, 0.3, 0.5])
    def test_replacement_release_holds_the_total_fixed(self, frequency):
        residents = np.array([70, 6, 0])
        released = apply_release(residents, frequency, mode="replacement")
        assert released.sum() == residents.sum()
        assert released[2] / released.sum() == pytest.approx(frequency, abs=0.02)

    def test_empty_population_cannot_receive_a_release(self):
        assert apply_release(np.zeros(3), 0.5).tolist() == [0, 0, 0]

    @pytest.mark.parametrize("frequency", [-0.1, 1.0, 1.5])
    def test_frequency_outside_the_unit_interval_is_rejected(self, frequency):
        with pytest.raises(ValueError, match=r"outside \[0, 1\)"):
            apply_release(np.array([10, 0, 0]), frequency)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="additive"):
            apply_release(np.array([10, 0, 0]), 0.5, mode="sideways")


class TestSimulationStructure:
    """What the driver records, and how the branches relate to each other."""

    def test_every_phase_and_generation_is_present(self, release_run):
        generations = sorted(release_run["generation"].unique())
        # 3 burn-in generations, their seeding state, the release, and 4 after.
        assert generations == [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert set(release_run["phase"]) == {
            PHASE_SEED,
            PHASE_BURN_IN,
            PHASE_RELEASE,
            PHASE_POST_RELEASE,
        }
        assert set(release_run["genotype"]) == set(GENOTYPES)

    def test_burn_in_is_shared_across_release_frequencies(self, release_run):
        """The comparison is paired: one burn-in per replicate, not per branch.

        If it were not, a difference between frequencies could just as well be
        the wild-type background each release happened to land in.
        """
        burn_in = release_run[release_run["phase"] == PHASE_BURN_IN]
        for replicate, replicate_rows in burn_in.groupby("replicate"):
            by_frequency = [
                frequency_rows.sort_values(["generation", "genotype"])[
                    "adults"
                ].to_numpy()
                for _, frequency_rows in replicate_rows.groupby("release_frequency")
            ]
            for other in by_frequency[1:]:
                assert np.array_equal(by_frequency[0], other), (
                    f"replicate {replicate} burned in differently per frequency"
                )

    def test_replicates_are_independent(self, release_run):
        burn_in = release_run[release_run["phase"] == PHASE_BURN_IN]
        per_replicate = [
            replicate_rows.sort_values(["generation", "genotype"])["adults"].to_numpy()
            for _, replicate_rows in burn_in.groupby("replicate")
        ]
        assert not np.array_equal(per_replicate[0], per_replicate[1])

    def test_released_state_carries_the_requested_gg_frequency(self, release_run):
        derived = add_derived_columns(release_run)
        released = derived[
            (derived["generation"] == 0) & (derived["genotype"] == "GG")
        ]
        assert not released.empty
        for _, row in released.iterrows():
            assert row["genotype_frequency"] == pytest.approx(
                row["release_frequency"], abs=0.02
            )

    def test_burn_in_is_wild_type_only(self, release_run):
        burn_in = release_run[
            release_run["phase"].isin([PHASE_SEED, PHASE_BURN_IN])
            & release_run["genotype"].isin(["WG", "GG"])
        ]
        assert (burn_in["adults"] == 0).all()

    def test_assembled_states_have_no_stage_counts(self, release_run):
        """No eggs were laid to reach them, so a survival rate would be a lie."""
        assembled = release_run[
            release_run["phase"].isin([PHASE_SEED, PHASE_RELEASE])
        ]
        assert assembled["eggs"].isna().all()
        assert assembled["larvae"].isna().all()
        assert assembled["adults"].notna().all()

    def test_larval_fates_account_for_every_larva(self, release_run):
        simulated = release_run.dropna(subset=["larvae"])
        assert not simulated.empty
        assert (
            simulated["larvae"]
            == simulated["starved"] + simulated["timed_out"] + simulated["pupae"]
        ).all()

    def test_stages_never_grow_along_the_life_cycle(self, release_run):
        simulated = release_run.dropna(subset=["larvae"])
        assert (simulated["larvae"] <= simulated["eggs"]).all()
        assert (simulated["pupae"] <= simulated["larvae"]).all()
        assert (simulated["adults"] <= simulated["pupae"]).all()

    def test_metadata_travels_with_the_counts(self, release_run):
        assert (release_run["conversion_rate"] == 0.4).all()
        assert (release_run["seed"] == 4321).all()
        assert (release_run["param_tag"] == "default").all()

    def test_a_run_is_reproducible_from_its_seed(self, fake_ibm):
        kwargs = dict(
            release_frequencies=(0.2,),
            n_replicates=2,
            n_burn_in_generations=2,
            n_generations=3,
            seed=99,
            n_workers=1,
        )
        first = simulate_single_location_releases(**kwargs)
        second = simulate_single_location_releases(**kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_a_generated_seed_is_recorded(self, fake_ibm):
        """An unseeded run still has to be reproducible after the fact."""
        dataframe = simulate_single_location_releases(
            release_frequencies=(0.2,),
            n_replicates=1,
            n_burn_in_generations=2,
            n_generations=2,
            n_workers=1,
        )
        assert dataframe["seed"].nunique() == 1
        assert int(dataframe["seed"].iloc[0]) > 0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"release_frequencies": ()}, "at least one frequency"),
            ({"release_frequencies": (1.2,)}, r"outside \[0, 1\)"),
            ({"n_replicates": 0}, "n_replicates"),
            ({"n_burn_in_generations": 0}, "n_burn_in_generations"),
            ({"n_generations": 0}, "n_generations"),
        ],
    )
    def test_nonsensical_arguments_are_rejected(self, fake_ibm, kwargs, match):
        defaults = dict(
            release_frequencies=(0.2,),
            n_replicates=1,
            n_burn_in_generations=1,
            n_generations=1,
            seed=1,
            n_workers=1,
        )
        with pytest.raises(ValueError, match=match):
            simulate_single_location_releases(**{**defaults, **kwargs})

    def test_csv_round_trip_preserves_the_derived_quantities(self, release_run, tmp_path):
        """``--from-csv`` has to rebuild exactly the figures the run drew."""
        path = tmp_path / "run.csv"
        release_run.to_csv(path, index=False)
        reloaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(
            add_derived_columns(release_run),
            add_derived_columns(reloaded),
            check_dtype=False,
        )


class TestDerivedQuantities:
    """The quantities the figures read, on a table built by hand."""

    @staticmethod
    def _frame(adults_by_genotype: dict[str, int]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "replicate": 0,
                    "release_frequency": 0.3,
                    "generation": 1,
                    "phase": PHASE_POST_RELEASE,
                    "genotype": genotype,
                    "eggs": 100,
                    "larvae": 90,
                    "starved": 30,
                    "timed_out": 20,
                    "pupae": 40,
                    "adults": adults,
                }
                for genotype, adults in adults_by_genotype.items()
            ]
        )

    def test_genotype_frequencies_sum_to_one(self, release_run):
        derived = add_derived_columns(release_run)
        sums = derived.groupby(["replicate", "release_frequency", "generation"])[
            "genotype_frequency"
        ].sum()
        assert np.allclose(sums.dropna(), 1.0)

    @pytest.mark.parametrize(
        "adults, expected",
        [
            ({"WW": 10, "WG": 0, "GG": 0}, 0.0),
            ({"WW": 0, "WG": 0, "GG": 10}, 1.0),
            ({"WW": 5, "WG": 0, "GG": 5}, 0.5),
            ({"WW": 0, "WG": 10, "GG": 0}, 0.5),
            ({"WW": 2, "WG": 4, "GG": 4}, 0.6),
        ],
    )
    def test_allele_frequency_counts_alleles_not_individuals(self, adults, expected):
        derived = add_derived_columns(self._frame(adults))
        assert derived["allele_frequency"].iloc[0] == pytest.approx(expected)

    def test_survival_and_fate_shares(self):
        derived = add_derived_columns(self._frame({"WW": 40, "WG": 40, "GG": 40}))
        row = derived.iloc[0]
        assert row["egg_to_adult_survival"] == pytest.approx(0.4)
        assert row["larva_to_pupa_survival"] == pytest.approx(40 / 90)
        assert row["starved_share"] == pytest.approx(30 / 90)
        assert row["timed_out_share"] == pytest.approx(20 / 90)
        assert row["pupated_share"] == pytest.approx(40 / 90)

    def test_an_extinct_population_gives_no_frequencies(self):
        derived = add_derived_columns(self._frame({"WW": 0, "WG": 0, "GG": 0}))
        assert derived["genotype_frequency"].isna().all()
        assert derived["allele_frequency"].isna().all()

    def test_derived_columns_leave_the_counts_alone(self, release_run):
        derived = add_derived_columns(release_run)
        pd.testing.assert_frame_equal(derived[release_run.columns], release_run)


class TestOutcomeClassification:
    """A mean allele frequency cannot describe runs that end at opposite ends."""

    @staticmethod
    def _final_frame(adults_by_genotype: dict[str, int]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "replicate": 0,
                    "release_frequency": 0.3,
                    "generation": 1,
                    "phase": PHASE_POST_RELEASE,
                    "genotype": genotype,
                    "eggs": 10,
                    "larvae": 9,
                    "starved": 3,
                    "timed_out": 2,
                    "pupae": 4,
                    "adults": adults,
                }
                for genotype, adults in adults_by_genotype.items()
            ]
        )

    @pytest.mark.parametrize(
        "adults, expected",
        [
            ({"WW": 0, "WG": 0, "GG": 0}, "extinct"),
            ({"WW": 100, "WG": 0, "GG": 0}, "lost"),
            ({"WW": 0, "WG": 0, "GG": 100}, "fixed"),
            ({"WW": 50, "WG": 0, "GG": 50}, "polymorphic"),
        ],
    )
    def test_each_end_state_is_labelled(self, adults, expected):
        outcomes = classify_outcomes(self._final_frame(adults))
        assert outcomes["outcome"].tolist() == [expected]

    def test_only_the_final_generation_is_classified(self, release_run):
        outcomes = classify_outcomes(release_run)
        assert len(outcomes) == release_run["release_frequency"].nunique() * (
            release_run["replicate"].nunique()
        )


class TestEquilibriumSummary:
    """The burn-in report is the evidence that the release was into equilibrium."""

    @staticmethod
    def _records(totals):
        return [
            {
                **{f"Adults_{genotype}": 0 for genotype in GENOTYPES},
                "Adults_WW": total,
                "generation": index,
            }
            for index, total in enumerate(totals)
        ]

    def test_a_settled_population_is_reported_as_settled(self):
        summary = equilibrium_summary(self._records([200, 90, 78, 80, 76]))
        assert summary["settled"]
        assert summary["mean"] == pytest.approx(78.0)

    def test_a_population_still_trending_is_flagged(self):
        summary = equilibrium_summary(self._records([200, 160, 120, 80, 40]))
        assert not summary["settled"]

    def test_ordinary_scatter_is_not_mistaken_for_a_trend(self):
        """The wild-type equilibrium here swings by tens of per cent."""
        summary = equilibrium_summary(self._records([70, 111, 68, 79, 77]))
        assert summary["settled"]


class TestFigures:
    """The figures have to be built from the table, and land where they are told."""

    @pytest.fixture(autouse=True)
    def _plain_fonts(self, monkeypatch):
        monkeypatch.setitem(plot_style.HOUSE_RC, "text.usetex", False)

    def test_trajectory_figure_is_written(self, release_run, tmp_path):
        path = plot_release_trajectories(release_run, tmp_path / "traj.png", dpi=50)
        assert path.exists() and path.stat().st_size > 0

    def test_diagnostics_figure_is_written(self, release_run, tmp_path):
        path = plot_release_diagnostics(release_run, tmp_path / "diag.png", dpi=50)
        assert path.exists() and path.stat().st_size > 0

    def test_threshold_figure_is_written(self, release_run, tmp_path):
        path = plot_threshold_diagnostics(release_run, tmp_path / "thresh.png", dpi=50)
        assert path.exists() and path.stat().st_size > 0

    def test_figures_are_created_in_a_missing_directory(self, release_run, tmp_path):
        path = plot_release_trajectories(
            release_run, tmp_path / "nested" / "dir" / "traj.png", dpi=50
        )
        assert path.exists()

    def test_figure_paths_record_the_run(self):
        paths = release_script.figure_paths("low0p30_high0p50_row0", 4321)
        assert set(paths) == {"trajectories", "diagnostics", "threshold"}
        for path in paths.values():
            assert "low0p30_high0p50_row0" in path.name
            assert "seed=4321" in path.name

    def test_default_parameters_are_named_in_the_figure_paths(self):
        paths = release_script.figure_paths(None, 7)
        assert "params=default" in paths["trajectories"].name


class TestStageCountRunner:
    """The new runner against the adult-only one it now backs."""

    def test_an_empty_cohort_has_nothing_to_simulate(self):
        assert run_ibm_cycle_from_eggs(0, 0, 0, 20, "interference") is None
        assert run_ibm_from_eggs(0, 0, 0, 20, "interference") is None

    @pytest.mark.slow
    def test_stage_counts_agree_with_the_adult_counts(self):
        """``run_ibm_from_eggs`` delegates here, so the two cannot disagree.

        Seeded identically either side, since both draw from the global RNG.
        """
        eggs = (40, 20, 20)

        np.random.seed(17)
        stage_counts = run_ibm_cycle_from_eggs(*eggs, 20, "interference")

        np.random.seed(17)
        adults = run_ibm_from_eggs(*eggs, 20, "interference")

        assert adults == (
            stage_counts["Adults_WW"],
            stage_counts["Adults_WG"],
            stage_counts["Adults_GG"],
        )

    @pytest.mark.slow
    def test_the_real_ibm_accounts_for_every_larva(self):
        np.random.seed(23)
        stage_counts = run_ibm_cycle_from_eggs(60, 30, 30, 20, "interference")

        for genotype in GENOTYPES:
            larvae = stage_counts[f"Larvae_{genotype}"]
            assert larvae == (
                stage_counts[f"Starved_{genotype}"]
                + stage_counts[f"TimedOut_{genotype}"]
                + stage_counts[f"Pupae_{genotype}"]
            )
            assert stage_counts[f"Adults_{genotype}"] <= stage_counts[f"Pupae_{genotype}"]
            assert larvae <= stage_counts[f"Eggs_{genotype}"]
