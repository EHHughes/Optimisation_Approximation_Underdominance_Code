#!/bin/bash
# Submit the gene-drive Bayesian-optimisation job inside the Singularity container.
#
#   sbatch run_optimise_gene_drive.sbatch <prop_gg_low> <prop_gg_high> [base_seed]
#
# Outputs land in the repo's Coarse_Graining_IBMS/data/ tree by default (override with
# OUTPUT_DIR). Adjust the #SBATCH directives below to match your cluster (partition,
# account, time, core count). We request a fixed number of cores on a shared node
# (--cpus-per-task) rather than the whole node (--exclusive): the in-container code
# sizes its multiprocess.Pool() from cpu_resources.available_cpu_count(), which reads
# the SLURM-allocated core count forwarded below, so it never oversubscribes.

# job and output location
#SBATCH --job-name=mosq-gd-opt
#SBATCH -o ./reports/output.%a.out # STDOUT

# resources
#SBATCH --time=02:00:00
#SBATCH --clusters=all
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

LOW="${1:?usage: sbatch run_optimise_gene_drive.sbatch <prop_gg_low> <prop_gg_high> [base_seed]}"
HIGH="${2:?provide prop_gg_high}"
SEED="${3:-}"

HPC_DIR=$DATA/Thesis/Mosquitos_Bonsall/Coarse_Graining_IBMS/hpc/
SIF="${SIF:-$HPC_DIR/optimise_gene_drive.sif}"
# Default: the repo's real data dir, so outputs land in data/ibm_output + data/models
# exactly like a local run. Override via env for scratch/project space.
OUTPUT_DIR="${OUTPUT_DIR:-$HPC_DIR/../data}"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
RUNNER="$(command -v apptainer || command -v singularity)"

ARGS=(--prop-gg-low "$LOW" --prop-gg-high "$HIGH")
[ -n "$SEED" ] && ARGS+=(--base-seed "$SEED")

# --cleanenv strips the host environment, so forward the SLURM core-allocation
# vars into the container (apptainer/singularity honour the *ENV_ prefixes even
# with --cleanenv). available_cpu_count() inside the container reads these to size
# its worker pool to the cores SLURM granted us.
CORES="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-}}"
for prefix in APPTAINERENV SINGULARITYENV; do
    export "${prefix}_SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    export "${prefix}_SLURM_CPUS_PER_TASK=${CORES}"
done

echo "Image:   $SIF"
echo "Output:  $OUTPUT_DIR"
echo "Args:    ${ARGS[*]}"
echo "Cores:   ${CORES:-<unset>}"

"$RUNNER" run --cleanenv \
    --bind "$OUTPUT_DIR":/opt/Mosquitos_Bonsall/Coarse_Graining_IBMS/data \
    "$SIF" "${ARGS[@]}"
