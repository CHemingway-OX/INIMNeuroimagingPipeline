# Standalone longitudinal lesion research tools

Ported from TWIN_MRI commit `11b3628` (see `SOURCE_PROVENANCE.json`). HPC changes:
scan dates from BIDS `sessions.tsv`, the SEL stage runs in the CPU job, a fixed
SEL thread count, and locks left by ended Slurm jobs are detected.

These scripts remain usable as standalone tools. The deformation-based SEL
analysis is also available as an optional stage in `struct_bids.sh`. They are
testable research adaptations, not validated reproductions of the original
Elliott or Klistorner implementations.

## Programs

* `sel_deformation_analysis.py`: multimodal deformation, annualized Jacobian
  expansion, spatial candidates, temporal and inward-shell scores. Two visits
  produce candidates only. Classification requires assessable temporal/spatial
  scores, an explicit matching calibration, and known enhancement status.
* `chronic_lesion_volume_analysis.py`: native-mask volumes, rigidly aligned
  correspondence groups, split/merge handling, and reviewed chronic-tissue
  summaries for consecutive intervals. Raw volume changes are always exported.

Both require Python, NumPy, SciPy, nibabel, ANTsPy (`antspyx`) and matplotlib.
Tested locally with ANTsPy 0.6.1. ANTs is imported after the thread count is set.

## Inputs and dates

Supply a CSV or TSV manifest with one subject per invocation and these columns:

| Column | Meaning |
| --- | --- |
| subject, session | Unique subject/session identifiers |
| date or days | ISO scan date or actual elapsed days, provided for all visits |
| t1, flair | 3D native images; the second channel is explicitly a FLAIR adaptation |
| lesion | Binary lesion mask, or probability image with `--mask-kind probability` |
| brain_mask | Binary brain mask on the FLAIR grid |
| enhancement_mask | Optional reviewed binary enhancement mask on the FLAIR grid; an all-zero mask means known absent enhancement, omission means unknown |
| chronic_mask | Optional reviewed mask of tissue established to be chronic at that visit, on the FLAIR grid |

Paths can be absolute or relative to the manifest. Visits are sorted by time.
Duplicate sessions/times, nonfinite images, nonbinary masks, inconsistent mask
geometry and lesions outside the brain mask fail validation.

If dates/days are omitted, T1 and FLAIR JSON sidecars are examined for
`AcquisitionDateTime`, `AcquisitionDate`, `StudyDate`, or `SeriesDate`. Conflicting
or missing dates fail explicitly. Time of day, file modification times, scanner
software dates and session numbers are never treated as scan dates. Years are
elapsed days divided by 365.25.

Anonymized sidecars often keep only `AcquisitionTime`, which is insufficient. The
pipeline therefore reads scan dates from the BIDS sessions file (see below).
Scan dates are personal data: keep them in the BIDS dataset, not in this
repository.

## Run from the repository root

```bash
pixi run python Pipeline/sel_deformation_analysis.py --manifest /path/to/sub-XXX_longitudinal.tsv --output /path/to/sel_out --dry-run
pixi run python Pipeline/chronic_lesion_volume_analysis.py --manifest /path/to/sub-XXX_longitudinal.tsv --output /path/to/volume_out --dry-run
```

Remove `--dry-run` to run each analysis. The default mask kind is binary,
matching the final LST-AI binary segmentations. For probability inputs,
explicitly choose `--mask-kind probability --threshold 0.5`. Different
segmentation choices are different experiments, not interchangeable inputs to a
published protocol. Run real analyses in a Slurm job, not on the login node.

## Structural pipeline integration

Add both `--long` and `--SEL`. SEL needs only the LST-AI lesion and FLAIR brain
masks and ANTs, so it runs in the CPU stage (or `--stage all`); the GPU stage
skips it with a note. Submit a cohort with:

```bash
hpc/submit_cohort.sh --subjects-file cohort.txt -- --long --SEL
```

The pipeline builds one manifest per subject from raw T1w and FLAIR images and
the `lst-ai-v1.2.0_docker` lesion and FLAIR brain masks. Scan dates come from
`sub-<ID>/sub-<ID>_sessions.tsv` (BIDS), columns `session_id` and `acq_time`:

```text
session_id	acq_time
ses-1	2024-01-02T10:15:00
ses-2	2025-01-03
```

`acq_time` must be ISO 8601 (`YYYY-MM-DD`, optionally with a time); other formats
fail. The date part is used. Without the file, the session row or a value
(`n/a`), the date is left to the JSON sidecar inference above. For a single
subject with dates maintained elsewhere, supply a manifest explicitly:

```bash
pixi run --locked struct --bids-dir /path/to/bids --subjects 394 --long --SEL \
  --stage cpu --sel-manifest /path/to/sub-394_longitudinal.tsv
```

Outputs are written to
`derivatives/structural_pipeline/sel_deformation/sub-<ID>`; automatically
generated manifests are retained in the adjacent `manifests` directory. An
existing output with matching provenance is resumed and its cached registration
stages are verified. A changed manifest, input, setting, or implementation fails
provenance validation and requires a new output directory. The pipeline passes
`--threads 8` (`SEL_THREADS` overrides it) independently of the job's CPU count,
because threads are part of the provenance.

Add `--resume` to reuse cached stages with identical inputs, settings, source
code and package versions. Input and cached file hashes are checked. Changed
settings require a new output directory. Each subject/pair owns its transform
files. `complete.json` reports execution completion, not scientific validation.
Do not run concurrent processes into one output directory.

A run holds `.running/owner.json` (host, PID, Slurm job). Slurm stops timed-out
jobs with SIGKILL while ANTs is running, so the lock can remain. A later run
removes it when the owning Slurm job is no longer queued or running, or when the
owning process on the same host has exited; otherwise it stops and names the
owner. A lock without `owner.json` is never removed automatically.

`--alignment already-aligned` bypasses registration only for verified aligned
data or synthetic testing. It is not a default shortcut for clinical images.
For SELs it bypasses affine alignment, but still computes nonlinear deformation.

## Deformation analysis details

1. N4-correct FLAIR, rigidly align to T1, and N4-correct T1 using the mapped mask.
2. Estimate a brain-masked affine for each baseline/follow-up pair. Construct
   physical half-transforms, including the affine's fixed center, on a shared
   1 mm grid covering both scans.
3. Use simultaneous T1 and FLAIR CC metrics with `SyNOnly`, identity initial
   transform, gradient step 0.7, flow sigma 2, total sigma 0, CC radius 2,
   iterations 40/20/0. These are explicit experimental settings, not a claim of
   the original study's complete registration protocol. Supported CLI parameters
   allow sensitivity runs; the effective argument dictionary is saved per pair.
4. Compute the geometric Jacobian determinant of the physical pull map from
   baseline-half to follow-up-half coordinates. Synthetic field tests verify
   that physical expansion gives J > 1. Affine/global scaling is excluded.
   Return the scalar map to the baseline T1 1 mm grid and intersect scan support.
5. Annual expansion is `100 * (J - 1) / elapsed_years`. It is not log(J), and
   not a compound annual growth rate. Outside valid support J is set to 1;
   those voxels are excluded from detection and the support mask is exported.
6. Seed at 12.5%/year and grow within baseline lesions at 4%/year, preserving
   identities of 18-connected seeds with priority growth. Discard candidates
   smaller than `--min-candidate-mm3` (default 10 mm3 on the 1 mm grid).
   This physical size choice differs from 10 voxels on the original study grid.
7. Fit mean cumulative percentage expansion against elapsed years through the
   origin. Export the slope and mean squared fractional residual. At least
   two follow-ups are required for this temporal score. Compute spatial scores
   from the slope of mean annual expansion against physical inward distance in
   equal-width shells; fewer than three occupied shells is unassessable.

`--calibration calibration.json` supports an explicitly supplied, frozen
calibration for this exact score model. It must contain:

```json
{
  "score_model": "inward_mm_shell_slope_and_normalized_origin_fit_mse_v1",
  "reference": "Description of the calibration cohort and protocol",
  "concentricity_mean": 0.0,
  "concentricity_sd": 1.0,
  "constancy_mean": 0.0,
  "constancy_sd": 1.0,
  "threshold": 0.0
}
```

The numbers above are schema examples, NOT estimated or published constants.
Do not use them for a scientific result. The implemented combined score is
`z(concentricity) - z(constancy_error)`. No single-subject calibration is fitted.
Even with a supplied calibration, outputs are named `adapted_score_pass`, never
"published high-confidence SELs". Original shell definitions, scoring details,
calibration and classification equivalence have not been established.

Outputs include candidate and adapted-score-pass masks, candidate CSV, all
Jacobian/rate/support maps, temporal and shell profiles in JSON, the first 12
candidate temporal curves, registration and candidate QC PNGs, transforms and
provenance. Candidate-region volume is baseline tissue selected by the method;
it is NOT the volume of newly added tissue.

## Direct volumetry and review

Connected components use 26-connectivity. Visits are matched using rigidly
aligned component overlap, with flagged proximity rescue for unmatched follow-up
components (`--match-distance-mm`, default 2). Splits/merges form groups rather
than being double-counted. Matching precedes the 50 mm3 group filter to avoid
calling lesions new when they grow across the size threshold.

Volumes come from native masks and the absolute affine determinant. Aligned
mask volumes are exported separately to expose interpolation losses. Rigid
matching, rather than local affine mask normalization, is an explicit adaptation;
proximity and lost-component flags require inspection. Components completely
lost in resampling are retained as flagged records.

The default +/-10% stable interval is an unannualized historical tolerance,
NOT a repeatability estimate validated for LST-AI. It is configurable. The script
reports signed changes, percentage changes and annualized rates independently
of the stability label. Each CSV row is an interval correspondence group, not a
guaranteed persistent lesion ID through splits and merges.

Chronic eligibility is intentionally separate. The script creates
`review_template.csv` with `include`, `exclude`, or `pending` decisions. Matched
groups require review for new confluent activity, and also report enhancement,
chronicity and possible post-acute shrinkage flags. Unknown enhancement is not
equivalent to an all-zero reviewed enhancement mask. First-observed age is
conservative; baseline lesion age is unknown unless a chronic mask/review
establishes it. A default age of 365.25 days is configurable.

Review the overlays and complete a separate review CSV, then run with
`--review /path/to/review.csv` into a NEW output directory. Include/exclude
decisions require a reason and are bound to input/settings/source hashes.
Inclusion adjudicates all remaining uncertainty for the whole group. New,
disappeared, enhancing or resampling-lost groups cannot be included as matched
chronic tissue. This version does not automatically delineate and subtract a
new confluent subregion: exclude the group if that activity is present. That is
a conservative adaptation, not the published Klistorner custom classifier.

`lesion_groups.csv` always contains raw changes. `interval_summary.csv` and
`summary.json` distinguish the reviewed subtotal from all native lesion volume.
Cumulative reviewed change is withheld if an interval has pending groups.
Cumulative changes sum actual interval differences, not annualized rates.
`native_volume_trajectory.png` is explicitly total native burden; it should not
be interpreted as chronic expansion.

## Validation and remaining integration gates

```bash
pixi run python -m unittest tests.test_lesion_longitudinal -v
```

Tests cover centered affine decomposition, analytic physical Jacobians and
pull direction, physical radial shells, candidate identity, time normalization,
split/merge and proximity matching, cache integrity, date inference, empty
lesions, review gating, and both complete command-line workflows on phantoms.
The normal affine/nonlinear registration path is exercised separately.

Before clinical research use: inspect real-data registration and segmentation
QC, test scan-rescan false expansion, assess
scanner/sequence changes, compare against reviewed reference cases, and obtain
or independently validate scoring calibration and acute/confluent exclusion
rules. No patient-level result or original-method equivalence is established by
passing these software tests. Enabling the pipeline stage remains explicit.

## Primary method references

* Elliott et al. (2019): https://doi.org/10.1177/1352458518814117
* Nakamura et al. (2024), independent ANTs SEL implementation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11190892/
* Klistorner et al. (2021): https://doi.org/10.1177/1352458520974357
* Klistorner et al. (2022): https://doi.org/10.1177/13524585221080667
* Klistorner et al. (2025): https://doi.org/10.1212/NXI.0000000000200377
