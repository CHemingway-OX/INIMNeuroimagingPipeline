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

The processing template requests `jobs-gpu-long`, one full A100, eight CPUs,
32 GiB RAM and 24 hours. These are pilot resource requests, not measured needs.
The partition permits up to seven days; adjust `--time` after a pilot. Add
`--account=...` or other site-required submission options if necessary.
The default `testing` partition has only 15 minutes, so templates explicitly
choose a partition. No jobs are submitted by installation, image preparation,
environment checks or tests.

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
