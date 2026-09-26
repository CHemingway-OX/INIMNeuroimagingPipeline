#!/bin/bash
# Submit a cohort: per subject a GPU job, then a CPU job (GPU/CPU split), then one report job.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  hpc/submit_cohort.sh (--subjects-file FILE | --subjects IDS | --all) [options] [-- pipeline options]

Subjects:
  --subjects-file FILE    One subject per line (001 or sub-001); # comments allowed.
  --subjects IDS          Comma-separated, e.g. 394,sub-395.
  --all                   Every sub-* directory in BIDS_DIR.

Options:
  --max-parallel K        GPU tasks running at once. Default: 2, maximum: 4
                          (jobs-gpu QoS allows 4 A100 per user; the cluster has 8).
  --max-parallel-cpu K    CPU tasks running at once (8 CPUs each). Default: 8, maximum: 32.
  --gpu-time HH:MM:SS     Walltime per GPU task. Default: from hpc/struct.sbatch.
  --cpu-time HH:MM:SS     Walltime per CPU task. Default: from hpc/struct_cpu.sbatch.
  --no-split              One GPU job per subject for all stages (holds the GPU during
                          CPU-only surface reconstruction; only for debugging).
  --no-report             Do not submit the report job.
  --dry-run               Print the sbatch commands without submitting.
  -h, --help              Show this help.

Everything after -- is passed to struct_bids.sh in every job, e.g. -- --long --skip-lit.
EOF
}

SUBJECTS_FILE=""
SUBJECTS_CSV=""
ALL=0
MAX_PARALLEL=2
MAX_PARALLEL_CPU=8
GPU_TIME=""
CPU_TIME=""
SPLIT=1
REPORT=1
DRY_RUN=0
PIPELINE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subjects-file) SUBJECTS_FILE="$2"; shift 2 ;;
        --subjects) SUBJECTS_CSV="$2"; shift 2 ;;
        --all) ALL=1; shift ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --max-parallel-cpu) MAX_PARALLEL_CPU="$2"; shift 2 ;;
        --gpu-time) GPU_TIME="$2"; shift 2 ;;
        --cpu-time) CPU_TIME="$2"; shift 2 ;;
        --no-split) SPLIT=0; shift ;;
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

limit() {  # limit NAME VALUE MAX
    if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--$1 must be a positive integer" >&2; exit 2
    fi
    if (( $2 > $3 )); then
        echo "--$1 $2 exceeds $3; using $3." >&2
        echo "$3"
    else
        echo "$2"
    fi
}
MAX_PARALLEL="$(limit max-parallel "$MAX_PARALLEL" 4)"
MAX_PARALLEL_CPU="$(limit max-parallel-cpu "$MAX_PARALLEL_CPU" 32)"

sources=0
[[ -n "$SUBJECTS_FILE" ]] && sources=$((sources + 1))
[[ -n "$SUBJECTS_CSV" ]] && sources=$((sources + 1))
[[ "$ALL" -eq 1 ]] && sources=$((sources + 1))
if (( sources != 1 )); then
    echo "Pass exactly one of --subjects-file, --subjects or --all." >&2; usage >&2; exit 2
fi
for arg in "${PIPELINE_ARGS[@]}"; do
    if [[ "$arg" == --stage* || "$arg" == --subjects* ]]; then
        echo "Do not pass $arg as a pipeline option; this script sets it per job." >&2; exit 2
    fi
done

subjects=()
if [[ "$ALL" -eq 1 ]]; then
    for dir in "$BIDS_DIR"/sub-*; do
        [[ -d "$dir" ]] && subjects+=("$(basename "$dir")")
    done
elif [[ -n "$SUBJECTS_CSV" ]]; then
    IFS=',' read -ra items <<< "$SUBJECTS_CSV"
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -n "$item" ]] && subjects+=("sub-${item#sub-}")
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
[[ "$DRY_RUN" -eq 1 ]] && frozen="${frozen%.txt}.dry-run.txt"
printf '%s\n' "${valid[@]}" > "$frozen"

gpu_cmd=(sbatch --parsable --array="1-${count}%${MAX_PARALLEL}")
[[ -n "$GPU_TIME" ]] && gpu_cmd+=(--time="$GPU_TIME")
gpu_cmd+=(hpc/struct.sbatch --subjects-file "$frozen")
[[ "$SPLIT" -eq 1 ]] && gpu_cmd+=(--stage gpu)
gpu_cmd+=("${PIPELINE_ARGS[@]}")

# aftercorr: CPU task N starts once GPU task N succeeded; tasks whose GPU part failed are
# cancelled instead of waiting forever (kill-on-invalid-dep).
cpu_cmd=(sbatch --parsable --array="1-${count}%${MAX_PARALLEL_CPU}" --kill-on-invalid-dep=yes)
[[ -n "$CPU_TIME" ]] && cpu_cmd+=(--time="$CPU_TIME")

report_args=()
for arg in "${PIPELINE_ARGS[@]}"; do
    [[ "$arg" == "--long" ]] && report_args+=(--long)
done

echo "Subjects (${count}): ${valid[*]}"
echo "Frozen list: $frozen"
if [[ "$SPLIT" -eq 1 ]]; then
    echo "GPU tasks at once: ${MAX_PARALLEL}; CPU tasks at once: ${MAX_PARALLEL_CPU}"
else
    echo "GPU tasks at once: ${MAX_PARALLEL} (no split)"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ${gpu_cmd[*]}"
    last="<gpu array id>"
    if [[ "$SPLIT" -eq 1 ]]; then
        echo "[dry-run] ${cpu_cmd[*]} --dependency=aftercorr:<gpu array id> hpc/struct_cpu.sbatch --subjects-file $frozen ${PIPELINE_ARGS[*]}"
        last="<cpu array id>"
    fi
    [[ "$REPORT" -eq 1 ]] && echo "[dry-run] sbatch --parsable --dependency=afterany:${last} hpc/report.sbatch ${report_args[*]}"
    exit 0
fi

gpu_id="$("${gpu_cmd[@]}")"
echo "GPU array: ${gpu_id} (logs/inim-struct-${gpu_id}_<task>.out)"
last_id="$gpu_id"
if [[ "$SPLIT" -eq 1 ]]; then
    cpu_id="$("${cpu_cmd[@]}" --dependency="aftercorr:${gpu_id}" hpc/struct_cpu.sbatch \
        --subjects-file "$frozen" "${PIPELINE_ARGS[@]}")"
    echo "CPU array: ${cpu_id} (logs/inim-struct-cpu-${cpu_id}_<task>.out)"
    last_id="$cpu_id"
fi
if [[ "$REPORT" -eq 1 ]]; then
    # afterany: report whatever finished, even if single subjects failed.
    report_id="$(sbatch --parsable --dependency="afterany:${last_id}" hpc/report.sbatch "${report_args[@]}")"
    echo "Report job: ${report_id} (starts after all tasks end)"
fi
job_ids="$gpu_id"
[[ "$last_id" != "$gpu_id" ]] && job_ids+=",${last_id}"
echo "Monitor: squeue -u \$USER; failed tasks: sacct -j ${job_ids} --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED -X"
