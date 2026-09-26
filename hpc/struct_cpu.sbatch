#!/bin/bash
#SBATCH --account=core-nrad-liebig
#SBATCH --job-name=inim-struct-cpu
#SBATCH --partition=jobs-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
# CPU part of the GPU/CPU split: FastSurfer surfaces and LST-AI annotation.
# Run after hpc/struct.sbatch ... --stage gpu succeeded for the same subject:
#   sbatch --dependency=afterok:<gpu job> hpc/struct_cpu.sbatch SUBJECT [pipeline options]
# Pass the same pipeline options as for the GPU job (for example --long).
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source hpc/config.local.sh
source hpc/subject.sh
: "${BIDS_DIR:?Set BIDS_DIR}" "${FS_LICENSE_DIR:?Set FS_LICENSE_DIR}"
resolve_subject "$@"
for arg in "${PIPELINE_ARGS[@]}"; do
    [[ "$arg" == --stage* ]] && { echo "struct_cpu.sbatch sets --stage cpu itself" >&2; exit 2; }
done
export STRUCT_BIDS_RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_sub-${SUBJECT#sub-}_cpu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
# Two time points build their surfaces in parallel, each with half of the CPUs.
surf_threads=$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK / 2 : 1 ))
pixi run --locked struct --bids-dir "$BIDS_DIR" --fs-license-dir "$FS_LICENSE_DIR" \
    --subjects "$SUBJECT" --threads "$SLURM_CPUS_PER_TASK" --workers 1 \
    --long-parallel-seg 1 --long-parallel-surf 2 --long-threads-surf "$surf_threads" \
    --skip-qc --skip-summary "${PIPELINE_ARGS[@]}" --stage cpu
