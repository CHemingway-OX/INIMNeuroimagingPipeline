#!/bin/bash
#SBATCH --job-name=inim-report
#SBATCH --partition=jobs-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source hpc/config.local.sh
: "${BIDS_DIR:?Set BIDS_DIR}" "${FS_LICENSE_DIR:?Set FS_LICENSE_DIR}"
export STRUCT_BIDS_RUN_ID="${SLURM_JOB_ID}_report"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
pixi run --locked struct --bids-dir "$BIDS_DIR" --fs-license-dir "$FS_LICENSE_DIR" \
    --skip-fastsurfer --skip-lst --skip-lit "$@"
