# INIM Neuroimaging Pipeline

BIDS structural MRI processing: FastSurfer, LST-AI, FS-LIT inpainting, metrics and
HTML QC. Inputs follow `sub-*/ses-*/anat/`; derivatives stay below the BIDS root.
The pipeline supports Docker, SingularityCE and Apptainer. The supplied HPC
configuration targets SingularityCE 4.2 and the core-nrad-liebig Slurm partitions.

## Python environment

Linux x86-64, Python **3.11**, with the exact dependency builds in `pixi.lock`.
The manifest declares NumPy >=1.26,<2, SciPy >=1.11,<1.16, pandas >=2.1,<3,
nibabel >=5.2,<6, matplotlib-base >=3.8,<4, scikit-image >=0.22,<0.26 and
antspyx ==0.6.3 (imported as `ants`). Pixi resolves their native and Python
dependencies. GPU application libraries are supplied by the containers.

```bash
export PIXI_CACHE_DIR=/data/core-nrad-liebig/chemingw/INIM_NIPipeline
pixi install --locked
pixi run --locked check
pixi run --locked test
```

Commit the manifest and lockfile, not `.pixi/`. No dependency changes are needed
for the Singularity backend. A lockfile-format warning from newer Pixi is harmless.

## HPC setup (no GPU allocation needed)

From the updated repository on the HPC:

```bash
cp hpc/config.example.sh hpc/config.local.sh
# Edit BIDS_DIR and FS_LICENSE_DIR in hpc/config.local.sh.
# FS_LICENSE_DIR must contain your FreeSurfer license.txt.
source hpc/config.local.sh
mkdir -p "$PIXI_CACHE_DIR" "$SINGULARITY_CACHEDIR" "$CONTAINER_DIR" logs
pixi install --locked
pixi run --locked check
pixi run --locked test
pixi run --locked prepare-images
```

Image preparation downloads large images; run on a network-enabled node where
site policy allows container pulls. It does not request a GPU or submit a job.
Allow ample disk space for both the conversion cache and final images. If the
site requires conversion temporary files on project/scratch storage, set
`SINGULARITY_TMPDIR` to an existing writable directory with sufficient space.

The image preparation task pulls these versions by verified registry digest:

| Tool | Version | Default SIF filename |
| --- | --- | --- |
| FastSurfer | cuda-v2.4.2 | fastsurfer-2.4.2.sif |
| LST-AI | v1.2.0 | lst-ai-1.2.0.sif |
| FS-LIT | 0.5.0 | lit-0.5.0.sif |

Digests are recorded in `utils/container_runtime.py`. Preparation writes a JSON
source record and SHA-256 checksum beside each SIF and refuses to reuse an image
with a mismatched source record. Use `FASTSURFER_SIF`, `LST_SIF` or `LIT_SIF` to
use existing local images. `*_DOCKER_IMAGE` variables override registry sources
for preparation and Docker execution; prefer digests for reproducibility.

## Submit one subject when ready

No GPU diagnostic job is required before submission. From the repository root:

```bash
mkdir -p logs
sbatch hpc/struct.sbatch 001
# Longitudinal alternative: all available sessions of this subject together.
# sbatch hpc/struct.sbatch 001 --long
```

The processing template requests `jobs-gpu`, one full A100, eight CPUs,
32 GiB RAM and 2 hours. These are pilot resource requests, not measured needs.
Longitudinal FastSurfer alone took about 1:45 h for two sessions, so a full run
usually needs a longer `--time` (jobs-gpu allows 8 hours, `jobs-gpu-long` seven
days). Pass `--time` on submission after a pilot. Add
`--account=...` or other site-required submission options if necessary.
The default `testing` partition has only 15 minutes, so templates explicitly
choose a partition. No jobs are submitted by installation, image preparation,
environment checks or tests.

## Submit a cohort (GPU/CPU split)

`hpc/submit_cohort.sh` submits per subject a short GPU job and a CPU job, then
one dataset-wide report job:

```bash
hpc/submit_cohort.sh --subjects-file cohort.txt -- --long
hpc/submit_cohort.sh --subjects 394,395 --cpu-time 06:00:00 -- --long
hpc/submit_cohort.sh --all --dry-run          # preview; every sub-* in BIDS_DIR
```

| Stage (`struct_bids.sh --stage`) | Template | Work |
| --- | --- | --- |
| `gpu` | `hpc/struct.sbatch` (jobs-gpu, 1 A100) | FastSurfer segmentation, LST-AI segmentation, FS-LIT |
| `cpu` | `hpc/struct_cpu.sbatch` (jobs-cpu, 8 CPUs) | FastSurfer surfaces, LST-AI annotation |

Surface reconstruction takes most of the FastSurfer time but never uses the GPU
(for sub-394 with two sessions: about 8 min segmentation versus 1:40 h
surfaces), so the split frees the A100 after the segmentation. The LST-AI
annotation needs `aparc.DKTatlas+aseg.mapped.mgz`, which FastSurfer creates
with the surfaces, so it runs in the CPU job. Longitudinal processing runs the
steps of FastSurfer's `long_fastsurfer.sh` in the same order through
`Pipeline/fastsurfer_long_phase.sh`, because `long_fastsurfer.sh` itself cannot
be split. `--stage all` (the default) keeps the single-job behaviour; use
`--no-split` in `submit_cohort.sh` for that.

CPU task N starts only after GPU task N succeeded (`aftercorr`); CPU tasks whose
GPU task failed are cancelled. The report job starts after all CPU tasks ended
(`afterany`), so it covers the subjects that succeeded.

The subject list (one ID per line, `001` or `sub-001`, `#` comments allowed) is
validated against `BIDS_DIR` and frozen to `logs/cohort_<timestamp>.txt`, because
array task N reads line N. `--max-parallel` throttles concurrent GPU tasks; the
jobs-gpu QoS allows at most 4 A100 per user and the cluster has 8 in total, so
keep the default of 2 while others are queued. `--max-parallel-cpu` (default 8,
64 CPUs) throttles CPU tasks. List failed tasks with
`sacct -j <gpu id>,<cpu id> --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED -X`
and resubmit only those subjects; finished stages are skipped on rerun.

### Slowly expanding lesions (SEL)

Add `--SEL` to a longitudinal run (`-- --long --SEL`) for the deformation-based
SEL analysis ported from TWIN_MRI. It runs in the CPU job and needs scan dates in
`sub-<ID>/sub-<ID>_sessions.tsv` (`session_id`, `acq_time`). Method, inputs,
limitations and the standalone chronic lesion volumetry are described in
`Pipeline/longitudinal_lesion_research.md`. It is a research adaptation without
calibration: two visits yield candidates only.

## Process a cohort that lives on a workstation

`hpc/cohort_driver.py` runs on the workstation that holds the BIDS cohort (for
example WSL2) and streams it through the HPC, so HPC storage holds only a few
subjects at a time. The HPC never needs to reach the workstation. Per subject
it uploads T1w/FLAIR images, their JSON sidecars and `sub-<ID>_sessions.tsv`,
submits GPU job -> CPU job -> per-subject report, fetches the derivatives when the
report succeeded, verifies every file by SHA-256 and only then deletes the
subject from the HPC. SEL deformation fields (`syn_*Warp.nii.gz`) are not
fetched.

The workstation needs Python 3.8+, ssh with a key for `core-mgm` and a copy of
the script (for example this repository). Start it in `tmux` or `screen`:

```bash
python3 hpc/cohort_driver.py \
  --local-bids /mnt/e/COHORT/BIDS --local-out /mnt/e/COHORT/hpc_results \
  --remote-bids /data2/core-nrad-liebig/chemingw/INIM_NIPipeline/staging/COHORT \
  --all -- --long --SEL
python3 hpc/cohort_driver.py ... --status          # progress, job ids, failures
python3 hpc/cohort_driver.py ... --retry-failed    # queue failed subjects again
```

* `--remote-bids` must be a dedicated staging directory: the driver marks it
  and refuses a non-empty directory it did not create, because it deletes
  subjects there after fetching.
* `--local-out` receives `derivatives/...` as on the HPC plus
  `derivatives/structural_pipeline/metrics/cohort_structural_metrics.csv`
  (merged per-subject metrics) and per-subject QC in `qc/sub-<ID>/`. Driver state,
  log and Slurm logs are in `--local-out/.cohort_driver/`.
* `--max-gpu` (default 2) bounds subjects waiting for or using a GPU,
  `--max-on-hpc` (default 6) subjects with data on the HPC. Failed subjects keep
  their HPC data for inspection; after `--max-failed` (default 3) failures no
  new subjects start.
* Defaults `--gpu-time 04:00:00` and `--cpu-time 06:00:00` suit 4-5 sessions per
  subject (LIT takes about 15 min and LST-AI about 6 min per session).
* Stopping and restarting is safe: submitted subjects are picked up again, and
  unreachable ssh (VPN) is retried at the next poll.

Jobs inherit the cluster GPU allocation: the runtime forwards
`CUDA_VISIBLE_DEVICES` through Singularity's clean environment and uses `--nv`
only for GPU stages. Do not hard-code GPU indices. The pipeline runs in the
foreground under Slurm, and logs/status filenames include job identifiers.
Each container call is independent, so there are no fixed Docker instance names.

The processing job skips shared dataset-wide reports. After processing succeeds,
submit a CPU reporting job, replacing 12345 with the processing job ID:

```bash
sbatch --dependency=afterok:12345 hpc/report.sbatch --subjects 001
# Add --long here too if processing used --long.
```

For a completed cohort, omit `--subjects` to generate dataset-wide reports.
Run only one report job at a time because the summary CSV and HTML index are
shared. The report job uses FreeSurfer's `asegstats2table` inside the FastSurfer
image, so a separate host FreeSurfer install is unnecessary for Singularity.
An explicit `ASEGSTATS2TABLE` setting can still select a host executable.

## Local execution and runtime choices

```bash
source hpc/config.local.sh
pixi run --locked struct --bids-dir "$BIDS_DIR" \
  --fs-license-dir "$FS_LICENSE_DIR" --subjects 001 --threads 8
```

Run processing only inside an allocation on the HPC. The above is an invocation
example, not an instruction to process images on the login node.
`CONTAINER_RUNTIME` can be `singularity`, `apptainer`, or `docker` (the default
outside the HPC config). `--container-runtime singularity` also selects it.
FastSurfer and LST-AI support CPU mode; use `--cpu --skip-lit` because FS-LIT is
GPU-enabled in this pipeline. The longitudinal surface default is one parallel
job; budget its thread count against the Slurm allocation.

FS-LIT now uses code and weights packaged in its image; a separate checkout is
not required by default. Explicit `LIT_REPO` overrides must contain `run_lit.sh`,
`lit/inpaint_image.py`, and all three model weights under `weights/`. Such an
override replaces `/inpainting` and must match the selected image. The earlier
`LIT_IMAGE` variable is superseded by `LIT_DOCKER_IMAGE` / `LIT_SIF`.

Historical derivative folder names retain `_docker` for compatibility with
existing QC and metrics paths, even when processing uses Singularity.
Existing FastSurfer outputs will be reused: do not mix prior `latest` image
results with this pinned version without deciding how to handle that version
change. Use a fresh BIDS derivative location/cohort copy for the first pilot.

## Validation and provenance

`pixi run check` imports dependencies and checks entry-point help and Bash syntax.
`pixi run test` uses fake container executables to exercise GPU environment
handling, mounts, failure propagation, cross-sectional and longitudinal
orchestration, restart behavior, FS-LIT invocation and container-based metrics.
These tests require neither Singularity nor a GPU and never submit Slurm jobs.

Real SIF conversion, the site's GPU/driver integration and scientific outputs
still require validation on the HPC. A passing local test is not an end-to-end
imaging validation. Full container processing has not been run as part of this
conversion.

`SOURCE_PROVENANCE.json` records the original TWIN_MRI working-tree hashes and
Git HEAD before the extraction and subsequent portability edits. No datasets,
licenses or model weights are stored in this repository. The original TWIN_MRI
repository is unchanged.

References: [SingularityCE 4.2](https://docs.sylabs.io/guides/4.2/user-guide/),
[FastSurfer](https://github.com/Deep-MI/FastSurfer),
[Slurm GPU resources](https://slurm.schedmd.com/gres.html).
