"""Tests that the surrogate training path can find parameter-set datasets.

Data generated for a non-default gene-drive parameter set (``generate_training_
data.py --pareto-file/--pareto-row``) carries a ``_params=<tag>`` marker in its
filename.  Every layer between the ``train_approximations`` entry point and
``paths.input_output_pairs_path`` has to pass that tag along, or the default-
parameter filename is looked for and training dies with ``FileNotFoundError``.
These tests pin the tag down at each layer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import paths
from IBM_Emulation.dataloaders import (
    get_test_train_dataloaders,
    get_test_train_datasets,
)
from IBM_Emulation.surrogates import neural_net_model_path
from IBM_Emulation.variational_approximations import maf_model_path

PARAM_TAG = "low0p30_high0p50_20260601_101511_row0"
TRAIN_SEED = 46
TEST_SEED = 203


def _write_pairs(path: Path, n_rows: int) -> None:
    """Write a genetics-aware input-output pair CSV of the expected shape.

    The genetics-aware loader reads the three input genotype counts from the
    first columns and the three outputs from column 6 onwards, so the file needs
    at least nine columns for the slicing to line up.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.integers(1, 100, size=(n_rows, 9)),
        columns=[
            "ww_in",
            "wg_in",
            "gg_in",
            "spare_1",
            "spare_2",
            "spare_3",
            "ww_out",
            "wg_out",
            "gg_out",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture
def param_set_data(tmp_path, monkeypatch):
    """Lay out train/test pair files that exist ONLY under the ``_params=`` name.

    Pointing ``IBM_OUTPUT_DIR`` at a tmp dir means a loader that drops the tag
    finds nothing at all, so these tests fail loudly rather than silently
    training on whatever real data happens to be checked out.
    """
    monkeypatch.setattr(paths, "IBM_OUTPUT_DIR", tmp_path)

    train_path = paths.input_output_pairs_path(
        TRAIN_SEED, with_genetics=True, param_tag=PARAM_TAG
    )
    test_path = paths.input_output_pairs_path(
        TEST_SEED, with_genetics=True, test=True, param_tag=PARAM_TAG
    )
    _write_pairs(train_path, n_rows=40)
    _write_pairs(test_path, n_rows=10)
    return train_path, test_path


class TestParameterSetDataIsFound:
    """The loaders must honour ``param_tag`` for both the train and test file."""

    def test_dataloaders_load_param_set_data(self, param_set_data):
        train_loader, test_loader, scalings = get_test_train_dataloaders(
            with_genetics=True,
            data_seed=TRAIN_SEED,
            test_data_seed=TEST_SEED,
            param_tag=PARAM_TAG,
            seed=1,
            batch_size=8,
        )

        assert len(train_loader.dataset) == 40
        assert len(test_loader.dataset) == 10
        assert scalings.shape == (2,)

    def test_dataloaders_without_tag_cannot_find_param_set_data(self, param_set_data):
        """The original failure: no tag means the default filename is looked for."""
        with pytest.raises(FileNotFoundError):
            get_test_train_dataloaders(
                with_genetics=True,
                data_seed=TRAIN_SEED,
                test_data_seed=TEST_SEED,
                seed=1,
            )

    def test_dataloaders_split_training_data_with_tag(self, param_set_data):
        """With no held-out test seed the tagged training file is split in two."""
        train_loader, test_loader, _ = get_test_train_dataloaders(
            with_genetics=True,
            data_seed=TRAIN_SEED,
            param_tag=PARAM_TAG,
            train_fraction=0.9,
            seed=1,
            batch_size=8,
        )

        assert len(train_loader.dataset) == 36
        assert len(test_loader.dataset) == 4

    def test_datasets_load_param_set_data(self, param_set_data):
        trainset, testset, _ = get_test_train_datasets(
            torch.device("cpu"), (TEST_SEED, TRAIN_SEED), param_tag=PARAM_TAG
        )

        assert len(trainset) == 40
        assert len(testset) == 10

    def test_datasets_without_tag_cannot_find_param_set_data(self, param_set_data):
        with pytest.raises(FileNotFoundError):
            get_test_train_datasets(torch.device("cpu"), (TEST_SEED, TRAIN_SEED))


class TestModelPathsCarryTheTag:
    """A surrogate for an optimised drive must not overwrite the default one."""

    def test_neural_net_path_includes_tag(self):
        tagged = neural_net_model_path(True, 999, param_tag=PARAM_TAG)
        assert f"_params={PARAM_TAG}" in tagged.name
        assert tagged.suffix == ".pt"

    def test_maf_path_includes_tag(self):
        tagged = maf_model_path(999, param_tag=PARAM_TAG)
        assert f"_params={PARAM_TAG}" in tagged.name
        assert tagged.suffix == ".pt"

    @pytest.mark.parametrize("seed", [None, 999])
    def test_untagged_model_names_are_unchanged(self, seed):
        """Default-parameter filenames stay exactly what they always were."""
        assert "_params=" not in neural_net_model_path(True, seed).name
        assert "_params=" not in maf_model_path(seed).name

    def test_tagged_and_untagged_paths_differ(self):
        assert neural_net_model_path(True, 999) != neural_net_model_path(
            True, 999, param_tag=PARAM_TAG
        )
        assert maf_model_path(999) != maf_model_path(999, param_tag=PARAM_TAG)
