# Optimisation and Approximation of Underdominance Gene Drives — reference code

The code and data in this repository can be used to reproduce Hughes & Bonsall 2026 (soon to be submitted). The agent-based larval model is also potentially of wider interest (although a more efficient refactored model is coming soon!). 

## Layout

```
.
├── setup.py, pytest.ini, requirements.txt
├── src/      # source; the Python path is rooted here (see src/README.md)
├── tests/    # pytest suite
└── data/     # input data, generated data, trained models, figures
```

## Installation

This repo should be relatively straightforward to install and should run on most modern versions of python (data and figures were generated using python 3.11). Follow the script below to install (the final step can be omitted unless you intend to edit, modify, or otherwise build upon the code).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes the src/ packages importable from outside src/
```

The trained surrogates are stored with [git LFS](https://git-lfs.com); clone
with LFS available, or run `git lfs pull` afterwards, or the `.pt` files will be stubs and scripts calling these will fail.

## Running

All entry-point scripts are run from `src/`:

```bash
python generate_training_data.py            # IBM input-output pairs for surrogate training
python train_approximations.py              # train the NN and MAF surrogates reproducibly
python validate_approximations.py           # validate both against a held-out test set
python simulate_nonspatial.py               # multigeneration IBM vs surrogate comparison
python optimise_gene_drive.py               # multi-objective BO; --prop-gg-low / --prop-gg-high
python sensitivity_checks.py                # surrogate validation, Sobol, local sensitivity
python compare_pareto_replicates.py         # Pareto frontiers/parameters across BO replicates
python compare_across_thresholds.py         # optimal designs across release-threshold windows
python simulate_single_location_release.py  # release a drive into an equilibrated population
python conference_figures.py                # conference-slide figures
```

Tests (from the repository root):

```bash
pytest                 # full suite, as CI runs it — includes the slow IBM-driven tests
pytest -m "not slow"   # fast local iteration
pytest -m slow         # only the slow tests
```

The `slow` marker exists so the tests that drive the real IBM can be skipped. If you are running this on a machine with limited resources you should consider this, as they can be quite resource intensive (particularly those which exercise the input-output pairs code, which aggressively spawn multiple processes).

## Data

Under `data/`:

- `ziqian_ibm_runs/` — stage counts from Ziqian's R IBM; committed *input* data.
- `ibm_output/` — IBM input-output pair CSVs (named by seed), the
  `low<lo>_high<hi>/` Pareto-frontier replicates for each release-threshold
  window, and `single_location_releases/` raw per-generation stage counts.
- `models/surrogates/` — the trained NN and MAF surrogates for seed
  `1180468651`, which are the ones behind the manuscript's figures.
- `models/optimisation/` — BoTorch model checkpoints for every BO run.
- `images/` — the figures these scripts produce.

The surrogates trained for individual Pareto points (filenames carrying a
`_params=` marker) are **not** included, to keep the repository to a sensible
size. They can be regenerated with `train_approximations.py`, passing the
`--pareto-file`/`--pareto-row` and `--homozygotes-only` flags recorded in the
filenames of the corresponding figures in `images/`.

## Deliberate divergences from the snapshot commit

Everything here follows from dropping the spatial package and the unused Ray
Tune hyperparameter search; nothing else was changed.

- `src/conference_figures.py` — figures 4 (2D MAF release) and 5 (dispersal
  kernel) were spatial and are removed. The remaining figures keep their
  original numbers.
- `tests/IBM_Emulation/test_homozygote_only_surrogates.py` — the single test
  exercising `HomozygoteSurrogate` inside `SpatialSimulation` is removed; the
  rest of the file is unchanged.
- `src/README.md` and `src/IBM_Emulation/README.md` — spatial and Ray Tune
  entries removed.
- The Ray Tune hyperparameter search is removed: it was exploratory and no
  result in the manuscript depends on it. Gone are `src/fit_neural_net.py`, the
  nine search helpers in `src/IBM_Emulation/surrogates.py`
  (`tune_hyperparameters`, `get_param_space_and_config`,
  `train_with_tunable_hyperparameters`, `_config_n_features`,
  `load_and_split_dataset`, `train_for_one_epoch`, `compute_val_loss`,
  `checkpoint_and_report`, `display_output_and_save_best_model`),
  `TestTuneHyperparametersEndToEnd` in `tests/IBM_Emulation/test_surrogates.py`,
  and `ray` plus its 54 transitive dependencies from `requirements.txt`. The
  surrogates behind the manuscript's figures were trained by
  `train_approximations.py`, which is untouched. The `slow` marker stays: the
  IBM-driven tests still use it.
- The working repository's `Coarse_Graining_IBMS/` directory level is removed:
  `src/`, `tests/` and `data/` sit at the repository root here. The level
  carried no meaning once the project stood alone, and it put every path in the
  documentation one step away from what a reader actually types. `pytest.ini`
  (`testpaths`, `pythonpath`), `setup.py` (`package_dir`) and the "run from"
  docstrings follow the new locations; `src/paths.py` needed no change, since
  it already resolved `data/` as a sibling of `src/`.
- `setup.py` — renamed distribution (see "Installation") and adjusted docstring.
- Dropped: the superseded `src/IBM_Implementations/legacy/` implementations, the
  `legacy/` data and figure directories, a stale duplicate `pyproject.toml` that
  conflicted with the root `setup.py`/`pytest.ini`, the committed `egg-info`
  build artefacts, and the HPC directory (an Apptainer recipe and Slurm script
  for the optimisation runs, plus the 699 MB built image).
- A handful of explanatory comments in `IBM_Emulation/` and
  `Gene_Drive_Optimisation/` still mention `SpatialSimulation` when explaining
  why an interface has the shape it does. They are left verbatim rather than
  reworded: the interface contract they describe is real, and the class itself
  lives in the companion spatial repository.

## AI Usage

This code was developed with assistance from Claude (Opus 4.6-5.5). All AI outputs were reviewed by the authors, and they take full responsibility for the contents of this repository.
