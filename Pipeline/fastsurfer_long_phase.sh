#!/bin/bash
# Run one resource phase of FastSurfer 2.4's long_fastsurfer.sh inside the container.
# The commands and their order mirror long_fastsurfer.sh, which refuses --seg_only/--surf_only:
#   gpu: 1 prepare template (co-registration, brain masks), 2 base segmentation, 4 long segmentation
#   cpu: 3 base surface reconstruction, 5 long surface reconstruction
# Usage: fastsurfer_long_phase.sh gpu|cpu --tid TID --tpids ID... --sd DIR [--t1s T1...]
#        [--parallel_seg N] [--parallel_surf N] [run_fastsurfer.sh options...]
set -euo pipefail

phase="${1:-}"
shift || true
[[ "$phase" == gpu || "$phase" == cpu ]] || { echo "First argument must be gpu or cpu" >&2; exit 2; }

tid=""
sd=""
tpids=()
t1s=()
brun_flags=()
fastsurfer_args=()
python="python3.10 -s"  # long_fastsurfer.sh default: no user-directory packages
while [[ $# -gt 0 ]]; do
    key="$1"
    shift
    case "$key" in
        --tid) tid="$1"; shift ;;
        --sd) sd="$1"; shift ;;
        --py) python="$1"; shift ;;
        --tpids) while [[ $# -gt 0 && "$1" != -* ]]; do tpids+=("$1"); shift; done ;;
        --t1s) while [[ $# -gt 0 && "$1" != -* ]]; do t1s+=("$1"); shift; done ;;
        --parallel|--parallel_seg|--parallel_surf) brun_flags+=("$key" "$1"); shift ;;
        *) fastsurfer_args+=("$key") ;;
    esac
done
[[ -n "$tid" && -n "$sd" && "${#tpids[@]}" -gt 0 ]] || { echo "--tid, --sd and --tpids are required" >&2; exit 2; }
if [[ "$phase" == gpu && "${#t1s[@]}" -ne "${#tpids[@]}" ]]; then
    echo "The gpu phase needs one --t1s entry per --tpids entry" >&2; exit 2
fi

FASTSURFER_HOME="${FASTSURFER_HOME:-/fastsurfer}"
export SUBJECTS_DIR="$sd"
time_points=()
for tpid in "${tpids[@]}"; do
    time_points+=("${tpid}=from-base")
done

run() {
    echo "=== $(date --iso-8601=seconds) $*"
    "$@"
}

if [[ "$phase" == gpu ]]; then
    run "$FASTSURFER_HOME/recon_surf/long_prepare_template.sh" --tid "$tid" --t1s "${t1s[@]}" \
        --tpids "${tpids[@]}" --py "$python" "${fastsurfer_args[@]}"
    run "$FASTSURFER_HOME/run_fastsurfer.sh" --sid "$tid" --sd "$sd" --base --seg_only \
        --py "$python" "${fastsurfer_args[@]}"
    run "$FASTSURFER_HOME/brun_fastsurfer.sh" --subjects "${time_points[@]}" --sd "$sd" --seg_only \
        --long "$tid" "${brun_flags[@]}" "${fastsurfer_args[@]}"
else
    run "$FASTSURFER_HOME/run_fastsurfer.sh" --sid "$tid" --sd "$sd" --surf_only --base \
        --py "$python" "${fastsurfer_args[@]}"
    run "$FASTSURFER_HOME/brun_fastsurfer.sh" --subjects "${time_points[@]}" --sd "$sd" --surf_only \
        --long "$tid" "${brun_flags[@]}" "${fastsurfer_args[@]}"
fi
