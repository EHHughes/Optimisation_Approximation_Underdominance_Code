#!/usr/bin/env bash
# Build optimise_gene_drive.sif from optimise_gene_drive.def.
# Run from anywhere; the build is executed from the repo root so the .def's
# relative %files paths (Coarse_Graining_IBMS/src, requirements.txt) resolve.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEF="$REPO_ROOT/Coarse_Graining_IBMS/hpc/optimise_gene_drive.def"
SIF="$REPO_ROOT/Coarse_Graining_IBMS/hpc/optimise_gene_drive.sif"

BUILDER="$(command -v apptainer || command -v singularity || true)"
if [ -z "$BUILDER" ]; then
    echo "ERROR: neither apptainer nor singularity found on PATH." >&2
    exit 1
fi

cd "$REPO_ROOT"
echo "Building image with: $BUILDER"
"$BUILDER" build --fakeroot --force "$SIF" "$DEF"

echo
echo "Built $SIF"
echo "Publish via Git LFS:"
echo "  git lfs install   # once per clone"
echo "  git add $SIF"
echo "  git commit -m 'Add gene-drive optimisation Singularity image'"
echo "  git push"
