# Optimisation and Approximation of Underdominance Gene Drives — reference code

Frozen reference code for the first paper of this project: fitting neural-network
surrogates to an agent-based individual-based model (IBM) of mosquito larvae, and
using those surrogates to optimise underdominance gene-drive parameters by
multi-objective Bayesian optimisation.

This repository is the reproducibility artefact that accompanies the manuscript.
It is **frozen**: it is not where development happens, and it is not expected to
track the working repository it was taken from.

## Provenance

Snapshot of the `Coarse_Graining_IBMS` project in the working thesis repository
(`EHHughes/Thesis`) at commit `a1bf66346319e17931e0609b2eaa72ab842473c7`
(2026-10-15), taken on 2026-10-20.

The snapshot covers the pipeline **up to, but not including, the spatial
simulations**, which belong to the second paper and are released separately.
Everything carried over is byte-identical to that commit apart from the
divergences listed under "Deliberate divergences" below.

## Scope

Included — the three sequential stages of the non-spatial pipeline:

1. **IBM** (`src/IBM_Implementations/`) — a Python re-implementation of Ziqian's
   R larval model (the original `IBM_Model_6.4.Rmd` is included for reference),
   with temperature-dependent parameters, per-timestep larval dynamics,
   Mendelian mating plus gene-drive conversion, life-stage transitions and
   statistics logging. `sync_rng_random_functions.py` provides SyncRNG-seeded
   wrappers used to cross-validate against the R code.
2. **Surrogate** (`src/IBM_Emulation/`) — a feedforward neural net mapping
   `(WW, WG, GG)` adults to `(WW, WG, GG)` adults after one cycle, plus a
   normalising-flow alternative (masked autoregressive flow via `zuko`) that
   captures the output distribution rather than just its mean.
3. **Optimisation** (`src/Gene_Drive_Optimisation/`) — multi-objective Bayesian
   optimisation (qNEHVI via BoTorch/GPyTorch) over nine gene-drive parameters,
   trading extinction at a low release fraction against fixation at a high one;
   Sobol and local sensitivity analysis; and the single-location release
   experiments.

Excluded — the spatial work, i.e. the `Spatial_Simulations` package, the
`simulate_spatial_model*.py` entry points and `calculate_analytical_wavespeed.py`.

## Layout

```
.
├── setup.py, pytest.ini, requirements.txt
└── Coarse_Graining_IBMS/
    ├── src/      # source; the Python path is rooted here (see src/README.md)
    ├── tests/    # pytest suite
    ├── hpc/      # Apptainer recipe + Slurm script for the optimisation runs
    └── data/     # input data, generated data, trained models, figures
```

The `Coarse_Graining_IBMS/` level is kept so that every path, docstring and
"run from `Coarse_Graining_IBMS/src/`" instruction reads exactly as it does in
the working repository.

## Installation

Install into **its own** virtual environment: this distribution and the working
repository provide the same top-level package names (`IBM_Emulation`, `paths`, …)
and cannot be installed editable side by side.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # required so Ray Tune worker subprocesses can import the modules
```

The trained surrogates are stored with [git LFS](https://git-lfs.com); clone
with LFS available, or run `git lfs pull` afterwards, or the `.pt` files will be
pointer stubs.

## Running

All entry-point scripts are run from `Coarse_Graining_IBMS/src/`:

```bash
python generate_training_data.py            # IBM input-output pairs for surrogate training
python fit_neural_net.py                    # Ray Tune hyperparameter search
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
pytest                 # full suite, as CI runs it — includes the slow Ray Tune tests
pytest -m "not slow"   # fast local iteration
pytest -m slow         # only the slow tests
```

The `slow` marker exists so the Ray Tune end-to-end tests can be skipped
*locally*. It must not be excluded by default, or CI coverage drops silently.

## Data

Under `Coarse_Graining_IBMS/data/`:

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

Everything here follows from dropping the spatial package; nothing else was
changed.

- `src/conference_figures.py` — figures 4 (2D MAF release) and 5 (dispersal
  kernel) were spatial and are removed. The remaining figures keep their
  original numbers.
- `tests/IBM_Emulation/test_homozygote_only_surrogates.py` — the single test
  exercising `HomozygoteSurrogate` inside `SpatialSimulation` is removed; the
  rest of the file is unchanged.
- `src/README.md` and `src/IBM_Emulation/README.md` — spatial entries removed.
- `setup.py` — renamed distribution (see "Installation") and adjusted docstring.
- Dropped: the superseded `src/IBM_Implementations/legacy/` implementations, the
  `legacy/` data and figure directories, a stale duplicate
  `Coarse_Graining_IBMS/pyproject.toml` that conflicted with the root
  `setup.py`/`pytest.ini`, the committed `egg-info` build artefacts, and the
  built Apptainer image `hpc/optimise_gene_drive.sif` (699 MB; rebuild it from
  `hpc/optimise_gene_drive.def` with `hpc/build_container.sh`).
- A handful of explanatory comments in `IBM_Emulation/` and
  `Gene_Drive_Optimisation/` still mention `SpatialSimulation` when explaining
  why an interface has the shape it does. They are left verbatim rather than
  reworded: the interface contract they describe is real, and the class itself
  lives in the companion spatial repository.

## Key design patterns

- Scripts run as `__main__` are thin wrappers; the real logic lives in the
  submodule functions.
- Genotypes are tracked as three counts: WW (wild-type homozygote), WG
  (heterozygote) and GG (drive homozygote). Genotype penalties (`wg_*_fac`,
  `gg_*_fac`) scale the five larval parameters (`a`, `alpha`, `beta`, `g`, `c`).
- All surrogate inputs and outputs are normalised by their per-column maxima
  (`scalings`); callers must pass `scalings` alongside a loaded surrogate.
- `src/paths.py` is the single source of truth for output directories; no module
  constructs paths locally.
- The IBM uses `SyncRNG` for reproducible cross-validation against the R code,
  and ordinary numpy RNG otherwise.
