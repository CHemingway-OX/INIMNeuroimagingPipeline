#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FS_LICENSE_DIR="${FS_LICENSE_DIR:-${HOME}/freesurfer}"
THREADS=8
WORKERS=1
LONG_PARALLEL_SEG=1
LONG_PARALLEL_SURF=1
LONG_THREADS_SURF=4
SYSTEM="GH"
CLIP_MIN="0.5"
CLIP_MAX="99.5"
LESION_THRESHOLD=0
DILATE=1
SUBJECTS_CSV=""
FOREGROUND=0
INTERNAL_RUN=0
SKIP_FASTSURFER=0
SKIP_LST=0
SKIP_LIT=0
SKIP_QC=0
SKIP_SUMMARY=0
LONG_FASTSURFER=0
LONG_TMP_ROOT=""

FASTSURFER_NAME="fastsurfer_v2.4.2_docker"
FASTSURFER_LONG_NAME="fastsurfer_v2.4.2_docker_long"
LST_NAME="lst-ai-v1.2.0_docker"
LIT_NAME="FS-LIT2"
STRUCT_NAME="structural_pipeline"

BIDS_DIR=""
ORIGINAL_ARGS=("$@")

usage() {
    cat <<'EOF'
Usage:
  ./Pipeline/struct_bids.sh --bids-dir /path/to/bids [options]
  ./Pipeline/struct_bids.sh /path/to/bids [options]

Options:
  --bids-dir PATH          BIDS root directory.
  --fs-license-dir PATH    Directory containing FreeSurfer license.txt.
  --python BIN             Python interpreter to use. Default: python3
  --threads N              Threads passed to FastSurfer and LST-AI. Default: 8
  --workers N              Worker count passed to LST-AI. Default: 1
  --long-parallel-seg N    Longitudinal FastSurfer segmentation jobs. Default: 1
  --long-parallel-surf N   Longitudinal FastSurfer surface jobs. Default: 1
  --long-threads-surf N    Threads per longitudinal surface job. Default: 4
  --system NAME            Processing system tag for existing wrappers. Default: GH
  --clip-min VALUE         LST-AI clipping minimum. Default: 0.5
  --clip-max VALUE         LST-AI clipping maximum. Default: 99.5
  --lesion-threshold N     LST-AI lesion threshold. Default: 0
  --dilate N               FS-LIT mask dilation. Default: 1
  --subjects IDS          Comma-separated subject IDs, e.g. 001,002,sub-010
  --long                   Run FastSurfer with its longitudinal stream.
  --cpu                    Run wrappers in CPU mode where supported.
  --container-runtime NAME docker (default), singularity or apptainer; also CONTAINER_RUNTIME.
  --foreground             Keep the pipeline attached to the current shell.
  --skip-fastsurfer        Skip FastSurfer.
  --skip-lst               Skip LST-AI.
  --skip-lit               Skip FS-LIT.
  --skip-qc                Skip QC HTML generation.
  --skip-summary           Skip metrics CSV generation.
  -h, --help               Show this help.
EOF
}

CPU_FLAG=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)
            BIDS_DIR="$2"
            shift 2
            ;;
        --bids-dir=*)
            BIDS_DIR="${1#*=}"
            shift
            ;;
        --fs-license-dir)
            FS_LICENSE_DIR="$2"
            shift 2
            ;;
        --fs-license-dir=*)
            FS_LICENSE_DIR="${1#*=}"
            shift
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --python=*)
            PYTHON_BIN="${1#*=}"
            shift
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --threads=*)
            THREADS="${1#*=}"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --workers=*)
            WORKERS="${1#*=}"
            shift
            ;;
        --long-parallel-seg)
            LONG_PARALLEL_SEG="$2"
            shift 2
            ;;
        --long-parallel-seg=*)
            LONG_PARALLEL_SEG="${1#*=}"
            shift
            ;;
        --long-parallel-surf)
            LONG_PARALLEL_SURF="$2"
            shift 2
            ;;
        --long-parallel-surf=*)
            LONG_PARALLEL_SURF="${1#*=}"
            shift
            ;;
        --long-threads-surf)
            LONG_THREADS_SURF="$2"
            shift 2
            ;;
        --long-threads-surf=*)
            LONG_THREADS_SURF="${1#*=}"
            shift
            ;;
        --system)
            SYSTEM="$2"
            shift 2
            ;;
        --system=*)
            SYSTEM="${1#*=}"
            shift
            ;;
        --clip-min)
            CLIP_MIN="$2"
            shift 2
            ;;
        --clip-min=*)
            CLIP_MIN="${1#*=}"
            shift
            ;;
        --clip-max)
            CLIP_MAX="$2"
            shift 2
            ;;
        --clip-max=*)
            CLIP_MAX="${1#*=}"
            shift
            ;;
        --lesion-threshold)
            LESION_THRESHOLD="$2"
            shift 2
            ;;
        --lesion-threshold=*)
            LESION_THRESHOLD="${1#*=}"
            shift
            ;;
        --dilate)
            DILATE="$2"
            shift 2
            ;;
        --dilate=*)
            DILATE="${1#*=}"
            shift
            ;;
        --subjects)
            SUBJECTS_CSV="$2"
            shift 2
            ;;
        --subjects=*)
            SUBJECTS_CSV="${1#*=}"
            shift
            ;;
        --long)
            LONG_FASTSURFER=1
            shift
            ;;
        --cpu)
            CPU_FLAG=1
            shift
            ;;
        --container-runtime)
            export CONTAINER_RUNTIME="$2"
            shift 2
            ;;
        --foreground)
            FOREGROUND=1
            shift
            ;;
        --skip-fastsurfer)
            SKIP_FASTSURFER=1
            shift
            ;;
        --skip-lst)
            SKIP_LST=1
            shift
            ;;
        --skip-lit)
            SKIP_LIT=1
            shift
            ;;
        --skip-qc)
            SKIP_QC=1
            shift
            ;;
        --skip-summary)
            SKIP_SUMMARY=1
            shift
            ;;
        --internal-run)
            INTERNAL_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "${BIDS_DIR}" ]]; then
                BIDS_DIR="$1"
                shift
            else
                echo "Unexpected positional argument: $1" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

[[ -n "${BIDS_DIR}" ]] || { echo "BIDS directory is required." >&2; usage; exit 1; }
BIDS_DIR="${BIDS_DIR%/}"
[[ -d "${BIDS_DIR}" ]] || { echo "BIDS directory not found: ${BIDS_DIR}" >&2; exit 1; }
[[ -d "${FS_LICENSE_DIR}" ]] || { echo "FreeSurfer license directory not found: ${FS_LICENSE_DIR}" >&2; exit 1; }
[[ -f "${FS_LICENSE_DIR}/license.txt" ]] || { echo "Missing ${FS_LICENSE_DIR}/license.txt" >&2; exit 1; }
BIDS_DIR="$(cd "${BIDS_DIR}" && pwd)"
FS_LICENSE_DIR="$(cd "${FS_LICENSE_DIR}" && pwd)"
CONTAINER_RUNNER="${REPO_ROOT}/utils/container_runtime.py"
if [[ "${CPU_FLAG}" -eq 1 && "${SKIP_LIT}" -eq 0 ]]; then
    echo "FS-LIT requires GPU support in this pipeline. Use --skip-lit with --cpu." >&2
    exit 1
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then FOREGROUND=1; fi
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python interpreter not found: ${PYTHON_BIN}" >&2; exit 1; }

PIPELINE_ROOT="${BIDS_DIR}/derivatives/${STRUCT_NAME}"
LOG_DIR="${PIPELINE_ROOT}/logs"
QC_DIR="${PIPELINE_ROOT}/qc"
METRICS_DIR="${PIPELINE_ROOT}/metrics"
mkdir -p "${LOG_DIR}" "${QC_DIR}" "${METRICS_DIR}"

RUN_ID="${STRUCT_BIDS_RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_$$}"
RUN_LOG="${STRUCT_BIDS_RUN_LOG:-${LOG_DIR}/struct_bids_${RUN_ID}.log}"
STATUS_FILE="${STRUCT_BIDS_STATUS_FILE:-${LOG_DIR}/struct_bids_${RUN_ID}.status}"

notify_user() {
    local title="$1"
    local body="$2"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "$title" "$body" >/dev/null 2>&1 || true
        return
    fi
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display notification \"${body}\" with title \"${title}\"" >/dev/null 2>&1 || true
        return
    fi
}

if [[ "${INTERNAL_RUN}" -eq 0 && "${FOREGROUND}" -eq 0 ]]; then
    env \
        STRUCT_BIDS_RUN_ID="${RUN_ID}" \
        STRUCT_BIDS_RUN_LOG="${RUN_LOG}" \
        STRUCT_BIDS_STATUS_FILE="${STATUS_FILE}" \
        nohup bash "$0" --internal-run "${ORIGINAL_ARGS[@]}" >"${RUN_LOG}" 2>&1 < /dev/null &
    echo "Structural pipeline started in background."
    echo "PID: $!"
    echo "Log: ${RUN_LOG}"
    exit 0
fi

if [[ "${INTERNAL_RUN}" -eq 0 && "${FOREGROUND}" -eq 1 ]]; then
    exec > >(tee -a "${RUN_LOG}") 2>&1
fi

echo "Run ID: ${RUN_ID}"
echo "BIDS directory: ${BIDS_DIR}"
if [[ -n "${SUBJECTS_CSV}" ]]; then
    echo "Subjects: ${SUBJECTS_CSV}"
fi
if [[ "${LONG_FASTSURFER}" -eq 1 ]]; then
    echo "FastSurfer mode: longitudinal"
    echo "FastSurfer longitudinal parallelism: seg=${LONG_PARALLEL_SEG}, surf=${LONG_PARALLEL_SURF}, threads_surf=${LONG_THREADS_SURF}"
fi
echo "Run log: ${RUN_LOG}"
echo "Started: $(date --iso-8601=seconds)"

FASTSURFER_DIR="${BIDS_DIR}/derivatives/${FASTSURFER_NAME}"
FASTSURFER_LONG_DIR="${BIDS_DIR}/derivatives/${FASTSURFER_LONG_NAME}"
FASTSURFER_ACTIVE_DIR="${FASTSURFER_DIR}"
if [[ "${LONG_FASTSURFER}" -eq 1 ]]; then
    FASTSURFER_ACTIVE_DIR="${FASTSURFER_LONG_DIR}"
fi
LST_DIR="${BIDS_DIR}/derivatives/${LST_NAME}"
LIT_DIR="${BIDS_DIR}/derivatives/${LIT_NAME}"
echo "FastSurfer derivatives: ${FASTSURFER_ACTIVE_DIR}"
export FASTSURFER_OUTPUT_DIR="${FASTSURFER_ACTIVE_DIR}"
if [[ "${CONTAINER_RUNTIME:-docker}" != docker && -z "${ASEGSTATS2TABLE:-}" ]]; then
    export ASEGSTATS2TABLE="${REPO_ROOT}/scripts/asegstats2table"
fi

FASTSURFER_SCRIPT="${REPO_ROOT}/run_scripts/run_fastsurfer_docker.py"
LST_SCRIPT="${REPO_ROOT}/run_scripts/run_lst_docker.py"
LIT_SCRIPT="${REPO_ROOT}/run_scripts/run_lesioninpainting.py"
QC_SCRIPT="${REPO_ROOT}/Pipeline/generate_structural_qc.py"
METRICS_SCRIPT="${REPO_ROOT}/Pipeline/collect_structural_metrics.py"

for script_path in "${FASTSURFER_SCRIPT}" "${LST_SCRIPT}" "${LIT_SCRIPT}" "${QC_SCRIPT}" "${METRICS_SCRIPT}"; do
    [[ -f "${script_path}" ]] || { echo "Required script not found: ${script_path}" >&2; exit 1; }
done

CPU_ARGS=()
if [[ "${CPU_FLAG}" -eq 1 ]]; then
    CPU_ARGS+=(--cpu)
fi

SUBJECT_ARGS=()
if [[ -n "${SUBJECTS_CSV}" ]]; then
    SUBJECT_ARGS+=(--subjects "${SUBJECTS_CSV}")
fi

run_step() {
    local label="$1"
    shift
    echo
    echo "=== ${label} ==="
    echo "Command: $*"
    "$@"
    echo "=== ${label} complete ==="
}

cleanup_long_tmp() {
    if [[ -n "${LONG_TMP_ROOT}" && -d "${LONG_TMP_ROOT}" ]]; then
        case "${LONG_TMP_ROOT}" in
            "${PIPELINE_ROOT}/tmp/fastsurfer_long_"*)
                if ! rm -rf -- "${LONG_TMP_ROOT}"; then
                    echo "Failed to remove temp directory: ${LONG_TMP_ROOT}" >&2
                fi
                ;;
            *)
                echo "Refusing to remove unexpected temp directory: ${LONG_TMP_ROOT}" >&2
                ;;
        esac
    fi
}

finish() {
    local exit_code="$1"
    local finished_at
    finished_at="$(date --iso-8601=seconds)"
    cleanup_long_tmp
    if [[ "${exit_code}" -eq 0 ]]; then
        echo "SUCCESS ${finished_at}" > "${STATUS_FILE}"
        echo "Pipeline finished successfully at ${finished_at}"
        notify_user "struct_bids finished" "Structural pipeline completed for $(basename "${BIDS_DIR}")"
    else
        echo "FAILED ${finished_at}" > "${STATUS_FILE}"
        echo "Pipeline failed at ${finished_at}"
        notify_user "struct_bids failed" "Structural pipeline failed for $(basename "${BIDS_DIR}")"
    fi
}

trap 'finish $?' EXIT

subject_requested() {
    local subject_label="${1#sub-}"
    local item

    if [[ -z "${SUBJECTS_CSV}" ]]; then
        return 0
    fi

    IFS=',' read -ra requested_subjects <<< "${SUBJECTS_CSV}"
    for item in "${requested_subjects[@]}"; do
        item="${item//[[:space:]]/}"
        item="${item#sub-}"
        if [[ "${item}" == "${subject_label}" ]]; then
            return 0
        fi
    done
    return 1
}

find_primary_t1w() {
    local anat_dir="$1"
    local candidate
    local lower
    local t1w_candidates=()

    for candidate in "${anat_dir}"/*_T1w.nii.gz "${anat_dir}"/*_T1w.nii; do
        [[ -e "${candidate}" ]] || continue
        lower="${candidate,,}"
        if [[ "${lower}" == *gadolinium* ]]; then
            continue
        fi
        t1w_candidates+=("${candidate}")
    done

    if [[ "${#t1w_candidates[@]}" -eq 0 ]]; then
        return 1
    fi

    if [[ "${#t1w_candidates[@]}" -gt 1 ]]; then
        echo "Multiple T1w candidates in ${anat_dir}; using ${t1w_candidates[0]}" >&2
    fi
    printf '%s\n' "${t1w_candidates[0]}"
}

fastsurfer_long_timepoint_dir() {
    local subject_label="$1"
    local session_label="$2"
    local tid="${subject_label}_template"
    local prefix="${subject_label}_${session_label}"
    local plain_dir="${FASTSURFER_LONG_DIR}/longitudinal_work/${subject_label}/${prefix}"
    local freesurfer_style_dir="${FASTSURFER_LONG_DIR}/longitudinal_work/${subject_label}/${prefix}.long.${tid}"

    if [[ -d "${plain_dir}" ]]; then
        printf '%s\n' "${plain_dir}"
    elif [[ -d "${freesurfer_style_dir}" ]]; then
        printf '%s\n' "${freesurfer_style_dir}"
    else
        printf '%s\n' "${plain_dir}"
    fi
}

fastsurfer_timepoint_complete() {
    local subject_label="$1"
    local session_label="$2"
    local subject_dir
    subject_dir="$(fastsurfer_long_timepoint_dir "${subject_label}" "${session_label}")"

    [[ -f "${subject_dir}/mri/aparc.DKTatlas+aseg.deep.mgz" \
        && ( -f "${subject_dir}/surf/lh.pial" || -f "${subject_dir}/surf/lh.pial.T1" ) \
        && ( -f "${subject_dir}/surf/rh.pial" || -f "${subject_dir}/surf/rh.pial.T1" ) ]]
}

ensure_longitudinal_output_link() {
    local subject_label="$1"
    local session_label="$2"
    local prefix="${subject_label}_${session_label}"
    local target
    target="$(fastsurfer_long_timepoint_dir "${subject_label}" "${session_label}")"
    local link_parent="${FASTSURFER_LONG_DIR}/${subject_label}/${session_label}"
    local link_path="${link_parent}/${prefix}"

    if [[ ! -d "${target}" ]]; then
        echo "Expected longitudinal FastSurfer output not found: ${target}" >&2
        return 1
    fi

    mkdir -p "${link_parent}"
    if [[ -L "${link_path}" ]]; then
        ln -sfn "${target}" "${link_path}"
    elif [[ -e "${link_path}" ]]; then
        echo "Existing non-symlink FastSurfer output remains at ${link_path}" >&2
        echo "Longitudinal output for this time point is at ${target}" >&2
    else
        ln -s "${target}" "${link_path}"
    fi
}

run_fastsurfer_longitudinal() {
    local subject_dir
    local subject_label
    local session_dir
    local session_label
    local anat_dir
    local t1w
    local tpid
    local conformed_t1w
    local conformed_dir
    local conformed_name
    local input_rel
    local subject_work_dir
    local tid
    local all_complete
    local i
    local subject_failed
    local processed_subjects=0
    local session_labels=()
    local tpids=()
    local host_t1s=()
    local container_t1s=()
    local conform_command=()
    local docker_command=()

    command -v "${CONTAINER_RUNTIME:-docker}" >/dev/null 2>&1 || { echo "Container runtime not found: ${CONTAINER_RUNTIME:-docker}" >&2; exit 1; }

    mkdir -p "${FASTSURFER_LONG_DIR}/longitudinal_work"
    LONG_TMP_ROOT="${PIPELINE_ROOT}/tmp/fastsurfer_long_${RUN_ID}"
    mkdir -p "${LONG_TMP_ROOT}"

    for subject_dir in "${BIDS_DIR}"/sub-*; do
        [[ -d "${subject_dir}" ]] || continue
        subject_label="$(basename "${subject_dir}")"
        subject_requested "${subject_label}" || continue
        tid="${subject_label}_template"

        session_labels=()
        tpids=()
        host_t1s=()
        container_t1s=()
        all_complete=1

        for session_dir in "${subject_dir}"/ses-*; do
            [[ -d "${session_dir}" ]] || continue
            session_label="$(basename "${session_dir}")"
            anat_dir="${session_dir}/anat"
            [[ -d "${anat_dir}" ]] || continue

            if ! t1w="$(find_primary_t1w "${anat_dir}")"; then
                echo "${subject_label}_${session_label}: no non-gadolinium T1w image found; skipping this time point." >&2
                continue
            fi

            tpid="${subject_label}_${session_label}"
            session_labels+=("${session_label}")
            tpids+=("${tpid}")
            host_t1s+=("${t1w}")

            if ! fastsurfer_timepoint_complete "${subject_label}" "${session_label}"; then
                all_complete=0
            fi
        done

        if [[ "${#tpids[@]}" -eq 0 ]]; then
            echo "${subject_label}: no longitudinal FastSurfer inputs found; skipping subject." >&2
            continue
        fi

        if [[ "${all_complete}" -eq 1 ]]; then
            echo "${subject_label}: requested longitudinal FastSurfer outputs already exist; skipping subject."
            for session_label in "${session_labels[@]}"; do
                ensure_longitudinal_output_link "${subject_label}" "${session_label}"
            done
            continue
        fi

        for i in "${!tpids[@]}"; do
            tpid="${tpids[i]}"
            t1w="${host_t1s[i]}"

            conformed_dir="${LONG_TMP_ROOT}/${subject_label}"
            conformed_name="${tpid}_desc-conform_T1w.nii.gz"
            conformed_t1w="${conformed_dir}/${conformed_name}"
            mkdir -p "${conformed_dir}"

            if [[ ! -f "${conformed_t1w}" ]]; then
                echo "${tpid}: conforming T1w to temporary 1 mm isotropic image."
                if [[ "${t1w}" != "${BIDS_DIR}/"* ]]; then
                    echo "${tpid}: T1w path is not inside the BIDS directory: ${t1w}" >&2
                    exit 1
                fi
                input_rel="${t1w#"${BIDS_DIR}/"}"
                conform_command=(
                    "${PYTHON_BIN}" "${CONTAINER_RUNNER}" --tool fastsurfer
                    --bind "${BIDS_DIR}:/input:ro"
                    --bind "${conformed_dir}:/output"
                    --bind "${FS_LICENSE_DIR}:/fs_license:ro"
                    -- /bin/bash
                    -lc 'if ! command -v mri_convert >/dev/null 2>&1; then for setup in /usr/local/freesurfer/SetUpFreeSurfer.sh /opt/freesurfer/SetUpFreeSurfer.sh /freesurfer/SetUpFreeSurfer.sh; do [[ -f "$setup" ]] && source "$setup" && break; done; fi; export FS_LICENSE=/fs_license/license.txt; mri_convert --conform "$1" "$2"'
                    _
                    "/input/${input_rel}"
                    "/output/${conformed_name}"
                )
                printf 'Command:'
                printf ' %q' "${conform_command[@]}"
                printf '\n'
                "${conform_command[@]}"
            fi
            container_t1s+=("/data/${conformed_name}")
        done

        subject_work_dir="${FASTSURFER_LONG_DIR}/longitudinal_work/${subject_label}"
        mkdir -p "${subject_work_dir}"

        docker_command=("${PYTHON_BIN}" "${CONTAINER_RUNNER}" --tool fastsurfer --workdir /fastsurfer)
        if [[ "${CPU_FLAG}" -eq 0 ]]; then
            docker_command+=(--gpu)
        fi
        docker_command+=(
            --bind "${LONG_TMP_ROOT}/${subject_label}:/data:ro"
            --bind "${subject_work_dir}:/output"
            --bind "${FS_LICENSE_DIR}:/fs_license:ro"
            -- /fastsurfer/long_fastsurfer.sh
            --fs_license /fs_license/license.txt
            --tid "${tid}"
            --t1s
        )
        docker_command+=("${container_t1s[@]}")
        docker_command+=(--tpids)
        docker_command+=("${tpids[@]}")
        docker_command+=(
            --sd /output
            --3T
            --threads "${THREADS}"
            --parallel_seg "${LONG_PARALLEL_SEG}"
            --parallel_surf "${LONG_PARALLEL_SURF}"
            --threads_surf "${LONG_THREADS_SURF}"
        )
        if [[ "${CPU_FLAG}" -eq 1 ]]; then
            docker_command+=(--cpu)
        fi

        echo "${subject_label}: running FastSurfer longitudinal stream for ${#tpids[@]} time point(s)."
        printf 'Command:'
        printf ' %q' "${docker_command[@]}"
        printf '\n'
        "${docker_command[@]}"

        subject_failed=0
        for session_label in "${session_labels[@]}"; do
            if fastsurfer_timepoint_complete "${subject_label}" "${session_label}"; then
                ensure_longitudinal_output_link "${subject_label}" "${session_label}"
            else
                echo "${subject_label}_${session_label}: longitudinal FastSurfer outputs are incomplete after container run." >&2
                echo "Expected mri/aparc.DKTatlas+aseg.deep.mgz plus pial surfaces in $(fastsurfer_long_timepoint_dir "${subject_label}" "${session_label}")" >&2
                subject_failed=1
            fi
        done
        if [[ "${subject_failed}" -ne 0 ]]; then
            exit 1
        fi
        processed_subjects=$((processed_subjects + 1))
    done

    echo "FastSurfer longitudinal subjects processed: ${processed_subjects}"
}

if [[ "${SKIP_FASTSURFER}" -eq 0 ]]; then
    if [[ "${LONG_FASTSURFER}" -eq 1 ]]; then
        run_step "FastSurfer longitudinal" run_fastsurfer_longitudinal
    else
        run_step \
            "FastSurfer" \
            "${PYTHON_BIN}" "${FASTSURFER_SCRIPT}" \
            -i "${BIDS_DIR}" \
            --fs_license "${FS_LICENSE_DIR}" \
            -t "${THREADS}" \
            --system "${SYSTEM}" \
            "${SUBJECT_ARGS[@]}" \
            "${CPU_ARGS[@]}"
    fi
fi

if [[ "${SKIP_LST}" -eq 0 ]]; then
    run_step \
        "LST-AI" \
        "${PYTHON_BIN}" "${LST_SCRIPT}" \
        -i "${BIDS_DIR}" \
        -n "${WORKERS}" \
        -t "${THREADS}" \
        --system "${SYSTEM}" \
        --probability_map \
        --clipping "${CLIP_MIN}" "${CLIP_MAX}" \
        --lesion_threshold "${LESION_THRESHOLD}" \
        "${SUBJECT_ARGS[@]}" \
        "${CPU_ARGS[@]}"
fi

if [[ "${SKIP_LIT}" -eq 0 ]]; then
    run_step \
        "FS-LIT lesion inpainting" \
        "${PYTHON_BIN}" "${LIT_SCRIPT}" \
        -i "${BIDS_DIR}" \
        -d "${LST_DIR}" \
        "${SUBJECT_ARGS[@]}" \
        --dilate "${DILATE}"
fi

if [[ "${SKIP_SUMMARY}" -eq 0 ]]; then
    run_step \
        "Metric summary CSV" \
        "${PYTHON_BIN}" "${METRICS_SCRIPT}" \
        --bids-dir "${BIDS_DIR}" \
        --fastsurfer-dir "${FASTSURFER_ACTIVE_DIR}" \
        --lst-dir "${LST_DIR}" \
        --lit-dir "${LIT_DIR}" \
        "${SUBJECT_ARGS[@]}" \
        --output-dir "${METRICS_DIR}"
fi

if [[ "${SKIP_QC}" -eq 0 ]]; then
    run_step \
        "QC HTML report" \
        "${PYTHON_BIN}" "${QC_SCRIPT}" \
        --bids-dir "${BIDS_DIR}" \
        --fastsurfer-dir "${FASTSURFER_ACTIVE_DIR}" \
        --lst-dir "${LST_DIR}" \
        --lit-dir "${LIT_DIR}" \
        "${SUBJECT_ARGS[@]}" \
        --output-dir "${QC_DIR}"
fi
