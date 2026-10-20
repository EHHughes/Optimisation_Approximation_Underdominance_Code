## Gene_Drive_Optimisation

Multi-objective Bayesian optimisation for underdominance gene-drive release strategies, using the IBM (see `IBM_Implementations`) as the objective function.

### Problem

We optimise 9 continuous decision variables simultaneously:

| Variable | Description | Range |
|---|---|---|
| `conversion_rate` | WG → GG gene-drive conversion probability per generation | [0, 1] |
| `wg_a_fac` | Multiplicative penalty on *a* for WG heterozygotes | [0, 1] |
| `wg_alpha_fac` | Multiplicative penalty on *alpha* for WG heterozygotes | [0, 1] |
| `wg_beta_fac` | Multiplicative penalty on *beta* for WG heterozygotes | [0, 1] |
| `wg_g_fac` | Multiplicative penalty on *g* for WG heterozygotes | [0, 1] |
| `gg_a_fac` | Multiplicative penalty on *a* for GG homozygotes | [0, 1] |
| `gg_alpha_fac` | Multiplicative penalty on *alpha* for GG homozygotes | [0, 1] |
| `gg_beta_fac` | Multiplicative penalty on *beta* for GG homozygotes | [0, 1] |
| `gg_g_fac` | Multiplicative penalty on *g* for GG homozygotes | [0, 1] |

Two objectives are maximised jointly:

- **f1 (extinction)** = `1 − mean_GG_freq` at a low release fraction (`PROP_GG_LOW = 0.4`). Maximised when the drive goes to extinction below the underdominance threshold.
- **f2 (fixation)** = `mean_GG_freq` at a high release fraction (`PROP_GG_HIGH = 0.6`). Maximised when the drive reaches fixation above the threshold.

Each objective is averaged over `n_replicates` stochastic IBM runs and evaluated at the final simulated cycle.

### Files

- **`gene_drive_objective.py`** — IBM evaluation functions.
  - `evaluate_gene_drive`: runs `n_replicates` IBM simulations for a single candidate and returns `(mean_GG_freq, std_GG_freq)`.
  - `gene_drive_objectives`: vectorised entry point for BoTorch; accepts a `(batch_size, 9)` tensor and returns `(means, variances)` tensors.
- **`optimise_gene_drives.py`** — BoTorch optimisation loop and plotting utilities.
  - `optimise_gene_drive`: high-level entry point; runs Sobol initialisation followed by `qLogNoisyExpectedHypervolumeImprovement` (qNEHVI) BO steps and returns the Pareto front.
  - `optimise_objectives`: generic multi-objective BO loop decoupled from the IBM objective — accepts any `objective_fn(X) → (Y, Yvar)` callable, making it testable independently.
  - `plot_pareto_grid`: N×N pairwise scatter-plot matrix of Pareto-optimal decision variables (and optionally objectives), with optional per-step history overlay.
  - `plot_objectives_grid`: pairwise scatter-plot matrix of Pareto-front objective values only.
  - `plot_outputs`: top-level figure-generation function called by `sensitivity_checks.py`; combines the verified Pareto front, Sobol indices, validation results, and local sensitivity into a single summary output.

### Entry point

`../optimise_gene_drive.py` is a thin script that calls `optimise_gene_drive` and saves the resulting Pareto-front figures to disk. Run from `src/`:

```bash
python optimise_gene_drive.py
```

### Tests

`tests/test_gene_drive_optimisation.py` covers:

- **`TestOptimiseObjectives`** — verifies the generic BO loop on the analytically tractable `(1/a, a)` bi-objective.
- **`TestBraninCurrinOptimisation`** — verifies that 5 BO steps improve hypervolume over a Sobol-only baseline on the standard Branin-Currin benchmark.
- **`TestHypervolume`** — unit tests for `_hypervolume` against hand-calculated values.
- **`TestBuildGenotypePenalties`** — unit tests for `_build_genotype_penalties`.
