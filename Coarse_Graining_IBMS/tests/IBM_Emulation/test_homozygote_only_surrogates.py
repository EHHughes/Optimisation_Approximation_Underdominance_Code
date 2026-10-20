"""Tests for the homozygote-only (WW/GG) surrogate mode.

Underdominance architectures make the heterozygote non-viable, so the dynamics
never visit a state with WG adults.  The pipeline can then sample only the
WW/GG face and fit a genuine two-genotype surrogate -- but every downstream
caller is built on a three-genotype ``(N, 3) -> (N, 3)`` contract, so the
two-genotype net is wrapped back up before it leaves the loader.

These tests pin the three things that mode depends on: the column select/embed
pair, the wrapper's contract, and the propagation of the ``_homozygotes=True``
marker through every filename layer.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import paths
from IBM_Emulation.dataloaders import (
    HOMOZYGOTE_INDICES,
    embed_homozygotes,
    get_test_train_dataloaders,
    get_test_train_datasets,
    report_heterozygote_residual,
    select_homozygotes,
    warn_if_heterozygote_inputs_present,
)
from IBM_Emulation.surrogates import (
    HomozygoteSurrogate,
    SimpleFeedforwardNN,
    neural_net_model_path,
)
from IBM_Emulation.variational_approximations import maf_model_path

TRAIN_SEED = 81
TEST_SEED = 82
PARAM_TAG = "low0p30_high0p50_20260601_101511_row0"


# ---------------------------------------------------------------------------
# Column select / embed
# ---------------------------------------------------------------------------


class TestSelectAndEmbed:
    """``(..., 3) -> (..., 2) -> (..., 3)``, for numpy and torch alike."""

    def test_homozygote_indices_are_ww_and_gg(self):
        assert HOMOZYGOTE_INDICES == (0, 2)

    def test_select_drops_the_heterozygote_column_numpy(self):
        array = np.array([[1.0, 99.0, 3.0], [4.0, 98.0, 6.0]])
        np.testing.assert_array_equal(
            select_homozygotes(array), np.array([[1.0, 3.0], [4.0, 6.0]])
        )

    def test_select_drops_the_heterozygote_column_torch(self):
        tensor = torch.tensor([[1.0, 99.0, 3.0], [4.0, 98.0, 6.0]])
        torch.testing.assert_close(
            select_homozygotes(tensor), torch.tensor([[1.0, 3.0], [4.0, 6.0]])
        )

    def test_embed_restores_width_with_exactly_zero_heterozygotes_numpy(self):
        embedded = embed_homozygotes(np.array([[1.0, 3.0], [4.0, 6.0]]))
        assert embedded.shape == (2, 3)
        np.testing.assert_array_equal(embedded[:, 1], np.zeros(2))
        np.testing.assert_array_equal(embedded[:, [0, 2]], [[1.0, 3.0], [4.0, 6.0]])

    def test_embed_restores_width_with_exactly_zero_heterozygotes_torch(self):
        embedded = embed_homozygotes(torch.tensor([[1.0, 3.0], [4.0, 6.0]]))
        assert embedded.shape == (2, 3)
        assert bool((embedded[:, 1] == 0).all())

    @pytest.mark.parametrize("to_tensor", [False, True])
    def test_round_trip_preserves_the_homozygotes_and_zeroes_wg(self, to_tensor):
        array = np.array([[7.0, 99.0, 11.0]])
        if to_tensor:
            array = torch.tensor(array)
        result = embed_homozygotes(select_homozygotes(array))
        expected = np.array([[7.0, 0.0, 11.0]])
        np.testing.assert_allclose(np.asarray(result), expected)

    def test_embed_handles_a_single_unbatched_row(self):
        """The MAF mean and the per-location IBM map are both shape ``(n,)``."""
        np.testing.assert_array_equal(
            embed_homozygotes(np.array([5.0, 9.0])), np.array([5.0, 0.0, 9.0])
        )

    def test_embed_keeps_gradients_flowing(self):
        """The wrapper sits mid-graph during training, so it must not detach."""
        leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        embed_homozygotes(leaf).sum().backward()
        assert leaf.grad is not None


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


class TestHomozygoteSurrogate:
    """A 2-in/2-out net must present itself as the pipeline's 3-genotype map."""

    @pytest.fixture
    def wrapped(self):
        torch.manual_seed(0)
        return HomozygoteSurrogate(SimpleFeedforwardNN(2, 8, 1, 2))

    def test_maps_three_genotypes_to_three(self, wrapped):
        assert wrapped(torch.rand(6, 3)).shape == (6, 3)

    def test_heterozygote_output_is_always_exactly_zero(self, wrapped):
        output = wrapped(torch.rand(32, 3) * 500)
        assert bool((output[:, 1] == 0).all())

    def test_homozygote_outputs_match_the_inner_net(self, wrapped):
        inputs = torch.rand(4, 3)
        expected = wrapped.net(inputs[:, [0, 2]])
        output = wrapped(inputs)
        torch.testing.assert_close(output[:, [0, 2]], expected)

    def test_heterozygote_input_is_ignored(self, wrapped):
        """WG adults cannot exist under this architecture, so WG input is noise."""
        base = torch.rand(4, 3)
        perturbed = base.clone()
        perturbed[:, 1] += 250.0
        torch.testing.assert_close(wrapped(base), wrapped(perturbed))

    def test_is_a_module_so_eval_and_parameters_still_work(self, wrapped):
        # Existing callers do net.eval() / net.cpu() on what the loader returns.
        assert wrapped.eval() is wrapped
        assert wrapped.cpu() is wrapped
        assert list(wrapped.parameters())

class TestPlotTestSetPredictionsWithEmbed:
    """The figure path must accept a two-genotype dataset and a wrapped net."""

    def test_two_genotype_dataset_plots_through_the_wrapped_net(self, tmp_path):
        """Regression: the forward pass must see the widened tensor.

        ``plot_test_set_predictions`` reads the dataset's own ``inputs``, which
        are two columns wide here, but the net it is handed is the wrapped
        three-genotype map.  Widening after the forward pass rather than before
        it crashed, and only at the very end of a full training run.
        """
        from IBM_Emulation.dataloaders import InputOutputDataset
        from IBM_Emulation.surrogates import plot_test_set_predictions

        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        dataset = InputOutputDataset(
            rng.integers(1, 100, size=(16, 2)),
            rng.integers(1, 100, size=(16, 2)),
            (1, 2),
            scalings=np.array([100.0, 100.0]),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        save_path = tmp_path / "predictions.png"

        plot_test_set_predictions(
            HomozygoteSurrogate(SimpleFeedforwardNN(2, 8, 1, 2)),
            loader,
            np.array([100.0, 100.0]),
            title="t",
            save_path=str(save_path),
            embed=embed_homozygotes,
        )

        assert save_path.exists()


# ---------------------------------------------------------------------------
# The residual diagnostic
# ---------------------------------------------------------------------------


def _pairs_frame(
    output_heterozygotes: int, input_heterozygotes: int = 0, n_rows: int = 4
) -> pd.DataFrame:
    """A genetics-aware frame with a known WG share, for the diagnostics."""
    return pd.DataFrame(
        {
            "input_adults_WW": [50] * n_rows,
            "input_adults_WG": [input_heterozygotes] * n_rows,
            "input_adults_GG": [50] * n_rows,
            "input_eggs_WW": [0] * n_rows,
            "input_eggs_WG": [0] * n_rows,
            "input_eggs_GG": [0] * n_rows,
            "output_adults_WW": [40] * n_rows,
            "output_adults_WG": [output_heterozygotes] * n_rows,
            "output_adults_GG": [60] * n_rows,
        }
    )


class TestHeterozygoteResidual:
    """How much output mass does forcing WG to zero discard?"""

    def test_zero_residual_reports_zero_and_does_not_warn(self, recwarn):
        result = report_heterozygote_residual(_pairs_frame(0), label="t")
        assert result["output_share"] == 0.0
        assert result["input_share"] == 0.0
        assert not [w for w in recwarn if w.category is UserWarning]

    def test_share_is_wg_over_total_output_adults(self):
        # 10 WG against 40 WW + 10 WG + 60 GG = 110 per row.
        result = report_heterozygote_residual(
            _pairs_frame(10), threshold=1.0, label="t"
        )
        assert result["output_share"] == pytest.approx(10 / 110)
        assert result["output_heterozygotes"] == 40.0

    def test_warns_above_the_threshold(self):
        with pytest.warns(UserWarning, match="discards"):
            report_heterozygote_residual(
                _pairs_frame(10), threshold=0.01, label="t"
            )

    def test_does_not_warn_below_the_threshold(self, recwarn):
        report_heterozygote_residual(_pairs_frame(10), threshold=0.5, label="t")
        assert not [w for w in recwarn if w.category is UserWarning]

    def test_rejects_an_aggregated_frame(self):
        aggregated = pd.DataFrame({"input_adults": [1], "output_adults": [1]})
        with pytest.raises(KeyError, match="genetics-aware"):
            report_heterozygote_residual(aggregated)

    def test_heterozygote_inputs_are_flagged_separately(self):
        """Dropping a WG input column the data varies is its own mistake."""
        with pytest.warns(UserWarning, match="WG input adults"):
            assert warn_if_heterozygote_inputs_present(
                _pairs_frame(0, input_heterozygotes=7), label="t"
            )

    def test_no_flag_when_there_are_no_heterozygote_inputs(self, recwarn):
        assert not warn_if_heterozygote_inputs_present(_pairs_frame(0), label="t")
        assert not [w for w in recwarn if w.category is UserWarning]


# ---------------------------------------------------------------------------
# Marker propagation
# ---------------------------------------------------------------------------


@pytest.fixture
def homozygote_only_data(tmp_path, monkeypatch):
    """Lay out pair files that exist ONLY under the ``_homozygotes=True`` name.

    Pointing ``IBM_OUTPUT_DIR`` at a tmp dir means a loader that drops the
    marker finds nothing at all, so these tests fail loudly rather than
    silently training on whatever real data happens to be checked out. Same
    shape of guard as ``test_dataloaders.TestParameterSetDataIsFound``.
    """
    monkeypatch.setattr(paths, "IBM_OUTPUT_DIR", tmp_path)

    for seed, n_rows, test in ((TRAIN_SEED, 40, False), (TEST_SEED, 10, True)):
        path = paths.input_output_pairs_path(
            seed, with_genetics=True, test=test, homozygotes_only=True
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _pairs_frame(0, n_rows=n_rows).to_csv(path, index=False)


class TestHomozygoteDataIsFound:
    """The loaders must honour the marker and return two-column datasets."""

    def test_dataloaders_find_the_marked_data(self, homozygote_only_data):
        train_loader, test_loader, scalings = get_test_train_dataloaders(
            with_genetics=True,
            data_seed=TRAIN_SEED,
            test_data_seed=TEST_SEED,
            homozygotes_only=True,
            seed=1,
            batch_size=8,
        )

        assert len(train_loader.dataset) == 40
        assert len(test_loader.dataset) == 10
        assert scalings.shape == (2,)

    def test_dataloaders_without_the_flag_cannot_find_it(self, homozygote_only_data):
        with pytest.raises(FileNotFoundError):
            get_test_train_dataloaders(
                with_genetics=True,
                data_seed=TRAIN_SEED,
                test_data_seed=TEST_SEED,
                seed=1,
            )

    def test_datasets_are_two_columns_wide(self, homozygote_only_data):
        trainset, testset, _ = get_test_train_datasets(
            torch.device("cpu"), (TEST_SEED, TRAIN_SEED), homozygotes_only=True
        )

        assert trainset.dataset_shape == (1, 2)
        assert trainset.inputs.shape[1] == 2
        assert trainset.outputs.shape[1] == 2
        assert len(trainset) == 40
        assert len(testset) == 10

    def test_datasets_without_the_flag_cannot_find_it(self, homozygote_only_data):
        with pytest.raises(FileNotFoundError):
            get_test_train_datasets(torch.device("cpu"), (TEST_SEED, TRAIN_SEED))

    def test_scalings_ignore_the_dropped_heterozygote_column(
        self, tmp_path, monkeypatch
    ):
        """A huge WG count must not inflate the two-genotype normalisation."""
        monkeypatch.setattr(paths, "IBM_OUTPUT_DIR", tmp_path)
        frame = _pairs_frame(0, input_heterozygotes=10_000, n_rows=6)
        path = paths.input_output_pairs_path(
            TRAIN_SEED, with_genetics=True, homozygotes_only=True
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

        with pytest.warns(UserWarning, match="WG input adults"):
            _train, _test, scalings = get_test_train_dataloaders(
                with_genetics=True,
                data_seed=TRAIN_SEED,
                homozygotes_only=True,
                seed=1,
                batch_size=4,
            )

        assert scalings[0] == 50  # the WW/GG max, not the 10 000 WG


class TestModelPathsCarryTheMarker:
    """A two-genotype surrogate must not overwrite the three-genotype one."""

    def test_neural_net_path_includes_the_marker(self):
        marked = neural_net_model_path(True, 999, homozygotes_only=True)
        assert "_homozygotes=True" in marked.name
        assert marked.suffix == ".pt"

    def test_maf_path_includes_the_marker(self):
        assert "_homozygotes=True" in maf_model_path(999, homozygotes_only=True).name

    def test_marker_composes_with_the_param_tag(self):
        marked = neural_net_model_path(
            True, 999, param_tag=PARAM_TAG, homozygotes_only=True
        )
        assert f"_params={PARAM_TAG}" in marked.name
        assert "_homozygotes=True" in marked.name

    def test_default_filenames_are_unchanged(self):
        """Existing models and data must keep resolving to their old names."""
        assert (
            neural_net_model_path(True, 999).name
            == "surrogate_model_genetics=True_seed=999.pt"
        )
        assert maf_model_path(999).name == "maf_model_seed=999.pt"
        assert (
            paths.input_output_pairs_path(27, with_genetics=True).name
            == "input_output_pairs_with_genetics_seed=27.csv"
        )
