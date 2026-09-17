#!/bin/bash
#SBATCH --job-name=rewire-mfass
#SBATCH --output=rewire-%j.log
set -euo pipefail
: "${REWIRE_IMAGE:?Set absolute path to a verified SIF}"
: "${REWIRE_PREPARED:?Set absolute path to prepared MFASS directory}"
: "${REWIRE_OUTPUT_ROOT:?Set absolute output directory}"
mkdir -p "$REWIRE_OUTPUT_ROOT"
export APPTAINERENV_REWIRE_SIF_SHA256="$(sha256sum "$REWIRE_IMAGE" | cut -d' ' -f1)"
export APPTAINERENV_REWIRE_CONTAINER_RUNTIME="apptainer $(apptainer version)"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
apptainer exec --cleanenv --containall \
  --bind "$REWIRE_PREPARED:/input:ro,$REWIRE_OUTPUT_ROOT:/output:rw" \
  "$REWIRE_IMAGE" rewirebench run --prepared /input \
  --adapter rewirebench.adapters.mfass:KmerBaseline \
  --model-name 'Corrected k-mer baseline' --output "/output/run-${SLURM_JOB_ID}"
