## src

All source code for the coarse-graining project. The Python path is rooted here, and every entry-point script is run from this directory.

### Directory layout

```
src/
├── Gene_Drive_Optimisation/   # multi-objective BO for gene-drive parameters
├── IBM_Emulation/             # neural-network and normalising-flow surrogates of the IBM
├── IBM_Implementations/       # Python (and original R) IBM implementations
└── *.py                       # top-level entry-point scripts (see below)
```

Each subdirectory has its own README with module-level detail. Generated data, trained models and figures live in `../data/`, whose locations are defined in `paths.py`.

### Entry-point scripts

- **`generate_training_data.py`** — runs the non-spatial IBM to generate the `(adults in → adults out)` input-output pairs the surrogates are trained on.
- **`fit_neural_net.py`** — runs Ray Tune hyperparameter search for the IBM surrogate neural net.
- **`train_approximations.py`** — trains both surrogates (feedforward NN and MAF normalising flow) reproducibly under a single seed.
- **`validate_approximations.py`** — validates both trained surrogates against a held-out test set.
- **`simulate_nonspatial.py`** — compares multigeneration IBM and surrogate adult trajectories.
- **`optimise_gene_drive.py`** — runs the gene-drive multi-objective Bayesian optimisation (qNEHVI) and saves Pareto-front figures.
- **`sensitivity_checks.py`** — loads a saved BO model and Pareto-front CSV, then runs surrogate validation against the IBM, Sobol global sensitivity analysis, and local sensitivity analysis around the Pareto points, saving summary figures.
- **`compare_pareto_replicates.py`** — compares Pareto frontiers and parameters across optimisation replicates.
- **`compare_across_thresholds.py`** — compares the optimal designs across release-threshold windows.
- **`simulate_single_location_release.py`** — releases a drive into one equilibrated wild-type population and records every per-genotype stage count.
- **`conference_figures.py`** — generates the conference-slide figures.

Shared helpers: `paths.py` (all output locations), `plot_style.py` (the house figure style) and `cpu_resources.py`.

### Submodules

- **`Gene_Drive_Optimisation/`** — multi-objective Bayesian optimisation (qNEHVI) for underdominance gene-drive parameters, the sensitivity analyses, and the single-location release experiments; see its own README for details.
- **`IBM_Emulation/`** — code to train and load feedforward neural-network and normalising-flow surrogates of the IBM; see its own README for details.
- **`IBM_Implementations/`** — Python implementations of Ziqian's larval IBM (original R code also included), plus input-output pair generation.
