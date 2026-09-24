#!/bin/bash
#SBATCH --job-name=inim-struct
#SBATCH --partition=jobs-gpu-long
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
# Submit from the repository root. Add --account on submission if your site requires it.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source hpc/config.local.sh
: "${BIDS_DIR:?Set BIDS_DIR}" "${FS_LICENSE_DIR:?Set FS_LICENSE_DIR}"
if [[ $# -lt 1 ]]; then echo "Usage: sbatch hpc/struct.sbatch SUBJECT [pipeline options]" >&2; exit 2; fi
subject="$1"
shift
export STRUCT_BIDS_RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_sub-${subject#sub-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
# Any site module loads should precede config.local.sh so it restores cache settings.
pixi run --locked struct --bids-dir "$BIDS_DIR" --fs-license-dir "$FS_LICENSE_DIR" \
    --subjects "$subject" --threads "$SLURM_CPUS_PER_TASK" --workers 1 \
    --long-parallel-seg 1 --long-parallel-surf 1 --long-threads-surf "$SLURM_CPUS_PER_TASK" \
    --skip-qc --skip-summary "$@"
