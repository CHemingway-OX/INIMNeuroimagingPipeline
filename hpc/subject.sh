# Sourced by the sbatch templates. resolve_subject "$@" sets SUBJECT and PIPELINE_ARGS:
#   SUBJECT [pipeline options]                                  single job
#   --subjects-file FILE [pipeline options]   (array job)       task N takes line N;
#                                                               blank lines and # comments are ignored
resolve_subject() {
    local usage="Usage: sbatch TEMPLATE SUBJECT [pipeline options] | sbatch --array=1-N TEMPLATE --subjects-file FILE [pipeline options]"
    if [[ $# -lt 1 ]]; then echo "$usage" >&2; exit 2; fi
    if [[ "$1" == "--subjects-file" ]]; then
        [[ $# -ge 2 && -f "$2" ]] || { echo "Subjects file not found: ${2:-}" >&2; exit 2; }
        : "${SLURM_ARRAY_TASK_ID:?--subjects-file requires an array job (sbatch --array=1-N)}"
        SUBJECT="$(grep -vE '^[[:space:]]*(#|$)' "$2" | sed -n "${SLURM_ARRAY_TASK_ID}p" | tr -d '[:space:]')"
        [[ -n "$SUBJECT" ]] || { echo "No subject for array task ${SLURM_ARRAY_TASK_ID} in $2" >&2; exit 2; }
        shift 2
    else
        SUBJECT="$1"
        shift
    fi
    PIPELINE_ARGS=("$@")
    echo "Subject: ${SUBJECT} (job ${SLURM_JOB_ID:-none}, array task ${SLURM_ARRAY_TASK_ID:-none}, node $(hostname))"
}
