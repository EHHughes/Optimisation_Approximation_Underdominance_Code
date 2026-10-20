# HPC container: gene-drive optimisation

Run `optimise_gene_drive.py` (multi-objective Bayesian optimisation over the underdominance
gene-drive parameters) inside a Singularity/Apptainer container on the research cluster.

## Contents

| File | Purpose |
|------|---------|
| `optimise_gene_drive.def` | Singularity/Apptainer image definition (CPU-only, Python 3.11). |
| `build_container.sh` | Builds `optimise_gene_drive.sif` from the `.def`. |
| `run_optimise_gene_drive.sbatch` | SLURM job: takes the two thresholds, binds output, runs the container. |
| `optimise_gene_drive.sif` | The built image (tracked by Git LFS; not present until built). |

The objective drives the IBM directly, so the container needs **no input data files** — only
the baked code and Python dependencies. The 73–102 MB surrogate `.pt` models are *not* required
and are excluded from the image.

## 1. Build the image (on a machine with apptainer/singularity)

```bash
bash Coarse_Graining_IBMS/hpc/build_container.sh
```

Requires `apptainer`/`singularity` with `--fakeroot` (rootless build). Produces
`Coarse_Graining_IBMS/hpc/optimise_gene_drive.sif`.

## 2. Publish via Git LFS

`*.sif` is routed through Git LFS (see repo-root `.gitattributes`, alongside the existing
`*.pt` rule).

```bash
git lfs install            # once per clone
git add Coarse_Graining_IBMS/hpc/optimise_gene_drive.sif
git commit -m "Add gene-drive optimisation Singularity image"
git push
```

## 3. Fetch on the cluster

```bash
git lfs install
git lfs pull               # materialises optimise_gene_drive.sif
```

## 4. Submit a job

```bash
sbatch Coarse_Graining_IBMS/hpc/run_optimise_gene_drive.sbatch <prop_gg_low> <prop_gg_high> [base_seed]
# e.g.
sbatch Coarse_Graining_IBMS/hpc/run_optimise_gene_drive.sbatch 0.3 0.5
```

Thresholds must satisfy `0 < low < high < 1`. Logs go to `hpc/logs/mosq-gd-opt_<jobid>.out`.

### Outputs

By default outputs land in the repo's data tree, threshold-namespaced, exactly like a local run:

```
Coarse_Graining_IBMS/data/ibm_output/low<low>_high<high>/gene_drive_pareto_frontier_points_*.csv
Coarse_Graining_IBMS/data/models/optimisation/low<low>_high<high>/gene_drive_bo_model_*.pt
```

Point the output elsewhere (e.g. scratch/project space) with `OUTPUT_DIR`:

```bash
OUTPUT_DIR=/scratch/$USER/mosq sbatch Coarse_Graining_IBMS/hpc/run_optimise_gene_drive.sbatch 0.3 0.5
```

Override the image path with `SIF=/path/to/image.sif` if it isn't beside the sbatch script.

## Notes

- **Edit the `#SBATCH` directives** in `run_optimise_gene_drive.sbatch` to match your cluster
  (partition, account, time limit, cores). The job requests a fixed core count on a shared node
  (`--cpus-per-task=8`) rather than a whole node (`--exclusive`), which allocates faster.
  `multiprocess.Pool()` is sized by `cpu_resources.available_cpu_count()`, which reads the
  SLURM-allocated core count (forwarded into the container as `SLURM_CPUS_PER_TASK`) under SLURM
  and falls back to the full machine otherwise — so it never oversubscribes a shared node. Bump
  `--cpus-per-task` to use more cores per job.
- **Rebuild after code changes.** The image pins a snapshot of `Coarse_Graining_IBMS/src`. After
  editing the source, re-run `build_container.sh` and re-commit the `.sif`.
- **Writable `data/` is required.** `paths.py` creates its output directories at import time,
  so the container only runs when a writable host directory is bound over
  `/opt/Mosquitos_Bonsall/Coarse_Graining_IBMS/data` (the sbatch script always does this). A bare
  `singularity run optimise_gene_drive.sif …` without the bind fails with
  `Read-only file system`.
- **Smoke test** (local — note the bind):
  ```bash
  TMP=$(mktemp -d)
  singularity exec --bind "$TMP":/opt/Mosquitos_Bonsall/Coarse_Graining_IBMS/data \
      optimise_gene_drive.sif \
      python -c "import torch, botorch, gpytorch, multiprocess; from paths import IBM_OUTPUT_DIR; print(torch.__version__, IBM_OUTPUT_DIR)"
  ```
