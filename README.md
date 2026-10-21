# Optimisation and Approximation of Underdominance Gene Drives — Reference Code

The code and data in this repository can be used to reproduce Hughes & Bonsall 2026 (soon to be submitted). The agent-based larval model is also potentially of wider interest (although a more efficient refactored model is coming soon!). Almost all of the code is written in `python`, although a version of the model without genetics is available in `R`.

## Layout

```
.
├── setup.py, pytest.ini, requirements.txt
├── src/      # source; the Python path is rooted here (see src/README.md)
├── tests/    # pytest suite
└── data/     # input data, generated data, trained models, figures
```

## Installation

This repo should be relatively straightforward to install and should run on most modern versions of python (data and figures were generated using python 3.11 on linux). Follow the script below to install (the final step can be omitted unless you intend to edit, modify, or otherwise build upon the code). 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes the installation editable
```

If you're not intending to use the surrogate modelling code, you may be able to skip installing `pytorch` (which is otherwise a heavy, slow download) but I have not debugged this thoroughly, so caveat emptor. 

The trained surrogates are stored with [git LFS](https://git-lfs.com); clone
with LFS available, or run `git lfs pull` afterwards, or the `.pt` files will be stubs and scripts calling these will fail.

## Running

All entry-point scripts are run from `src/`:

```bash
python optimise_gene_drive.py               # multi-objective BO; --prop-gg-low / --prop-gg-high
python sensitivity_checks.py                # Sobol, local sensitivity
python compare_pareto_replicates.py         # Pareto frontiers/parameters across BO replicates
python compare_across_thresholds.py         # optimal designs across release-threshold windows
python generate_training_data.py            # IBM input-output pairs for surrogate training
python train_approximations.py              # train the NN and MAF surrogates reproducibly
python validate_approximations.py           # validate both against a held-out test set
python simulate_nonspatial.py               # multigeneration IBM vs surrogate comparison
```

If you wish to reproduce the publication figures exactly, you'll need to check the seeds used to generate each figure, which will generally be saved in the figure name (check `data/figures/etc` to find these).

The other top-level scripts in `src` are utils which handle path generation and plotting, so can largely be ignored unless you want to override those defaults.

In case you want to modify or extend this code, tests can be found in the `tests` folder, and can be run as follows (from the repository root):

```bash
pytest                 # full suite, as CI runs it — includes the slow IBM-driven tests
pytest -m "not slow"   # fast local iteration
pytest -m slow         # only the slow tests
```

The `slow` marker exists so the tests that run the IBM can be skipped. If you are running this on a machine with limited resources you should consider this, as they can be quite resource intensive (particularly those which exercise the input-output pairs code, which aggressively spawn multiple processes).

If you wish to run the `R` version of the model, it can be found in `src/IBM_Implementations/`.

## Data

Under `data/`:

- `ibm_output/` — IBM input-output pair CSVs (named by seed), the
  `low<lo>_high<hi>/` Pareto-frontier replicates for each release-threshold
  window, and `single_location_releases/` raw per-generation stage counts.
- `models/surrogates/` — the trained NN and MAF surrogates for seed
  `1180468651`, which are the ones behind the manuscript's figures.
- `models/optimisation/` — BoTorch model checkpoints for every BO run.
- `images/` — images used in the manuscript.

## AI Usage

This code was developed with assistance from Claude (Opus 4.6-5.5). All AI outputs were reviewed by the authors, and they take full responsibility for the contents of this repository.
