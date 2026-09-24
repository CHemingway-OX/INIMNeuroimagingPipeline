# INIM Neuroimaging Pipeline

Structural MRI pipeline extracted from TWIN_MRI: FastSurfer, LST-AI, FS-LIT lesion
inpainting, structural metrics and HTML QC. Entry point: `Pipeline/struct_bids.sh`.
Inputs follow `sub-*/ses-*/anat/`; derivatives are written below the BIDS root.
Use `--system GH` (default) or `--system BMC` according to the existing filename
conventions. This option is a dataset convention, not a cluster selection.

## Pixi environment

Target: Linux x86-64 (`linux-64`), Python **3.11**. The environment deliberately
uses NumPy 1.x and SciPy <1.16 for this initial dependency baseline. This does
not assert that other versions cannot work. Python 3.11 is the supported version
for this extraction; the original project did not provide a dependency lock.

| Library | Manifest constraint | Purpose |
| --- | --- | --- |
| Python | 3.11.* | Pipeline wrappers and reporting |
| NumPy | >=1.26,<2 | Image arrays |
| SciPy | >=1.11,<1.16 | Image processing and resampling |
| pandas | >=2.1,<3 | Metrics tables |
| nibabel | >=5.2,<6 | NIfTI and FreeSurfer image I/O |
| matplotlib-base | >=3.8,<4 | Headless QC figures (`matplotlib` import) |
| scikit-image | >=0.22,<0.26 | Connected components and morphology |
| antspyx (PyPI) | ==0.6.3 | ANTs image transforms (`ants` import) |

Pixi resolves transitive dependencies, including ANTsPy's additional Python
libraries and native runtime libraries. PyTorch/CUDA application dependencies
belong to the processing containers, not this host Python environment.

On a login node with network access, from this repository:

```bash
pixi install
pixi run check
pixi run struct --help
```

Commit `pixi.toml` and the generated `pixi.lock`; never commit `.pixi/`.
For an existing lockfile, use `pixi install --locked`. Prepare the environment
before submitting compute jobs; compute nodes need access to the repository and
its environment, but should not need to download packages at job startup.

## External software and configuration

**The current processing backend is Docker. Apptainer/Singularity support has
not yet been implemented. Installing Pixi alone does not make the full pipeline
runnable on an Apptainer-only cluster.**

The processing wrappers currently use:

- `deepmi/fastsurfer:latest` (also for the longitudinal stream).
- `jqmcginnis/lst-ai:v1.2.0`.
- `deepmi/lit:<version from the LIT checkout>`, or `LIT_IMAGE` if set.

These images and their model weights must be available to the container runtime.
FastSurfer output directory names contain `v2.4.2`, but the actual image is still
`latest`; pin and validate image versions before production processing.

Requirements outside Pixi:

- Bash and GNU command-line utilities on the Linux host.
- Docker access for the current implementation, plus NVIDIA container support
  and a GPU allocation. `--cpu` is not sufficient to remove all GPU requests.
- FreeSurfer `license.txt`; set `FS_LICENSE_DIR` or pass `--fs-license-dir`.
  The portable fallback is `$HOME/freesurfer`.
- A FreeSurfer installation/module exposing `asegstats2table` for metric
  collection. Source `SetUpFreeSurfer.sh` as appropriate, or set `ASEGSTATS2TABLE`
  to its executable path. The license alone does not provide this program.
- A separate FS-LIT source checkout, mounted into the LIT container at
  `/inpainting`. Set `LIT_REPO` to its absolute path; the fallback is a sibling
  `LIT` directory next to this repository. This external project was not copied.
  The original local checkout reported version 0.5.0. Set `LIT_IMAGE` explicitly
  if needed and use a compatible source checkout/image pair.

Example once the external requirements are available:

```bash
export LIT_REPO=/project/software/LIT
export FS_LICENSE_DIR=/project/licenses/freesurfer
pixi run struct --bids-dir /project/data/BIDS --subjects 001 --threads 8
```

The `struct` Pixi task includes `--foreground`, which is required under Slurm.
Use absolute paths for data, license and LIT locations.

## Slurm deployment notes

Submit through `sbatch` using the site's account, partition, memory, CPU and GPU
settings. Inside the batch job, change to this repository and run, for example:

```bash
export STRUCT_BIDS_RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
pixi run --locked struct --bids-dir /project/data/BIDS \
  --subjects 001 --threads "$SLURM_CPUS_PER_TASK" --workers 1 \
  --skip-qc --skip-summary
```

This is only a launch pattern: it still requires Docker or a future Apptainer
backend. The current Docker wrappers use fixed container names, so concurrent
jobs on the same Docker daemon can collide. Some container/worker failures are
not propagated reliably yet; verify outputs and fix failure propagation before
large array runs. Keep all sessions of a subject together for `--long`; budget
`--long-parallel-surf` times `--long-threads-surf` against the CPU allocation.

Generate dataset-wide metrics and QC in one separate job after processing;
parallel subject jobs otherwise overwrite the shared CSV and HTML index.
The reporting-only invocation is:

```bash
pixi run --locked struct --bids-dir /project/data/BIDS \
  --skip-fastsurfer --skip-lst --skip-lit
```

Supply the license directory even for this invocation, since the shell driver
currently checks it unconditionally. Add `--long` if processing used longitudinal
FastSurfer. Before relying on Slurm `afterok` dependencies, fix the failure
propagation noted above.

## Source provenance and validation

`SOURCE_PROVENANCE.json` records the original Git HEAD and source-file SHA-256
hashes. Files were copied from the working tree, including existing uncommitted
changes. No imaging datasets, derived outputs, licenses or model weights were
copied. The original TWIN_MRI working tree was not changed.

Changes in this extraction: portable FreeSurfer license default, configurable
`LIT_REPO` and `LIT_IMAGE`, an early check for the LIT checkout, and Pixi setup.
The existing processing algorithms are retained.

`pixi run check` checks dependency imports, Python entry-point help and Bash
syntax without launching containers or processing scans. A successful check
is not end-to-end validation of container processing or scientific results.

References: [Pixi manifest](https://pixi.prefix.dev/v0.62.2/reference/pixi_manifest/)
and [ANTsPy package](https://pypi.org/project/antspyx/0.6.3/).
