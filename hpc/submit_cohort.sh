#!/bin/bash
# Submit one throttled array task per subject, then one dataset-wide report job.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  hpc/submit_cohort.sh (--subjects-file FILE | --all) [options] [-- pipeline options]

Options:
  --subjects-file FILE  One subject per line (001 or sub-001); # comments allowed.
  --all                 Every sub-* directory in BIDS_DIR.
  --max-parallel K      Array tasks running at once. Default: 2, maximum: 4
                        (jobs-gpu QoS allows 4 A100 per user; the cluster has 8).
  --time HH:MM:SS       Walltime per subject. Default: from hpc/struct.sbatch.
  --no-report           Do not submit the report job.
  --dry-run             Print the sbatch commands without submitting.
  -h, --help            Show this help.

Everything after -- is passed to struct_bids.sh, e.g. -- --long --skip-lit.
EOF
}

SUBJECTS_FILE=""
ALL=0
MAX_PARALLEL=2
TIME=""
REPORT=1
DRY_RUN=0
PIPELINE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subjects-file) SUBJECTS_FILE="$2"; shift 2 ;;
        --all) ALL=1; shift ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --no-report) REPORT=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PIPELINE_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source hpc/config.local.sh
: "${BIDS_DIR:?Set BIDS_DIR in hpc/config.local.sh}"

if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-parallel must be a positive integer" >&2; exit 2
fi
if (( MAX_PARALLEL > 4 )); then
    echo "--max-parallel ${MAX_PARALLEL} exceeds the per-user limit of 4 A100 GPUs; using 4." >&2
    MAX_PARALLEL=4
fi
if [[ -n "$SUBJECTS_FILE" && "$ALL" -eq 1 ]] || [[ -z "$SUBJECTS_FILE" && "$ALL" -eq 0 ]]; then
    echo "Pass exactly one of --subjects-file or --all." >&2; usage >&2; exit 2
fi

subjects=()
if [[ "$ALL" -eq 1 ]]; then
    for dir in "$BIDS_DIR"/sub-*; do
        [[ -d "$dir" ]] && subjects+=("$(basename "$dir")")
    done
else
    [[ -f "$SUBJECTS_FILE" ]] || { echo "Subjects file not found: $SUBJECTS_FILE" >&2; exit 2; }
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line//[[:space:]]/}"
        [[ -n "$line" ]] && subjects+=("sub-${line#sub-}")
    done < "$SUBJECTS_FILE"
fi

# Validate and de-duplicate while keeping order.
declare -A seen=()
valid=()
for subject in "${subjects[@]}"; do
    [[ -n "${seen[$subject]:-}" ]] && continue
    seen[$subject]=1
    if [[ ! -d "$BIDS_DIR/$subject" ]]; then
        echo "Subject not found in BIDS_DIR: $subject" >&2; exit 1
    fi
    valid+=("$subject")
done
count="${#valid[@]}"
(( count > 0 )) || { echo "No subjects to submit." >&2; exit 1; }
max_array="$(scontrol show config 2>/dev/null | awk '/^MaxArraySize/ {print $3}')"
if [[ -n "$max_array" ]] && (( count >= max_array )); then
    echo "${count} subjects exceed MaxArraySize ${max_array}; split the list." >&2; exit 1
fi

# Freeze the list: array task N reads line N, so later edits must not shift subjects.
mkdir -p logs
frozen="$PWD/logs/cohort_$(date +%Y%m%d_%H%M%S).txt"
if [[ "$DRY_RUN" -eq 1 ]]; then
    frozen="${frozen%.txt}.dry-run.txt"
fi
printf '%s\n' "${valid[@]}" > "$frozen"

struct_cmd=(sbatch --parsable --array="1-${count}%${MAX_PARALLEL}")
[[ -n "$TIME" ]] && struct_cmd+=(--time="$TIME")
struct_cmd+=(hpc/struct.sbatch --subjects-file "$frozen" "${PIPELINE_ARGS[@]}")

report_args=()
for arg in "${PIPELINE_ARGS[@]}"; do
    [[ "$arg" == "--long" ]] && report_args+=(--long)
done

echo "Subjects (${count}): ${valid[*]}"
echo "Frozen list: $frozen"
echo "Parallel tasks: ${MAX_PARALLEL}"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ${struct_cmd[*]}"
    [[ "$REPORT" -eq 1 ]] && echo "[dry-run] sbatch --parsable --dependency=afterany:<array id> hpc/report.sbatch ${report_args[*]}"
    exit 0
fi

array_id="$("${struct_cmd[@]}")"
echo "Array job: ${array_id} (logs/inim-struct-${array_id}_<task>.out)"
if [[ "$REPORT" -eq 1 ]]; then
    # afterany: report whatever finished, even if single subjects failed.
    report_id="$(sbatch --parsable --dependency="afterany:${array_id}" hpc/report.sbatch "${report_args[@]}")"
    echo "Report job: ${report_id} (starts after all array tasks end)"
fi
echo "Monitor: squeue -u \$USER; failed tasks: sacct -j ${array_id} --state=FAILED,TIMEOUT,OUT_OF_MEMORY -X"
