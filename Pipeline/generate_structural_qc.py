#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))

import ants
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np
from scipy import ndimage


META_FIELDS = (
    "Manufacturer",
    "ManufacturersModelName",
    "MagneticFieldStrength",
    "SeriesDescription",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "FlipAngle",
    "SliceThickness",
)


def discover_sessions(bids_dir: Path) -> list[tuple[str, str]]:
    sessions: set[tuple[str, str]] = set()
    for t1_path in bids_dir.glob("sub-*/ses-*/anat/*_T1w.nii.gz"):
        sub = t1_path.parts[-4].replace("sub-", "")
        ses = t1_path.parts[-3].replace("ses-", "")
        sessions.add((sub, ses))
    return sorted(sessions)


def parse_subject_ids(subjects_arg: str | None) -> set[str] | None:
    if not subjects_arg:
        return None
    subject_ids = set()
    for item in subjects_arg.split(","):
        item = item.strip()
        if not item:
            continue
        subject_ids.add(item.replace("sub-", ""))
    return subject_ids or None


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_readable_image(*paths: Path, issues: list[str]) -> Path | None:
    """Select a usable volume, recording broken outputs before trying fallbacks."""
    for path in paths:
        if not path.exists():
            continue
        try:
            image = nib.load(str(path))
            if len(image.shape) != 3 or any(size == 0 for size in image.shape):
                raise ValueError(f"Expected a nonempty 3D image, got shape {image.shape}")
            # Loading the header alone does not detect truncated voxel data.
            np.asanyarray(image.dataobj)
        except (nib.filebasedimages.ImageFileError, OSError, EOFError, ValueError) as exc:
            message = f"Unreadable image: {path} ({type(exc).__name__}: {exc})"
            issues.append(message)
            print(f"WARNING: {message}", file=sys.stderr)
            continue
        return path
    return None


def load_nifti(path: Path) -> nib.spatialimages.SpatialImage:
    image = nib.load(str(path))
    return nib.as_closest_canonical(image)


def read_image(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = load_nifti(path)
    return image.get_fdata(), image.header.get_zooms()[:3]


def interpolation_order(interpolation: str) -> int:
    return 0 if interpolation == "nearest" else 1


def load_isotropic_nifti(
    path: Path,
    interpolation: str = "continuous",
    voxel_size: float = 1.0,
) -> nib.spatialimages.SpatialImage:
    image = load_nifti(path)
    return resample_to_output(
        image,
        voxel_sizes=(voxel_size, voxel_size, voxel_size),
        order=interpolation_order(interpolation),
    )


def read_isotropic_image(
    path: Path,
    interpolation: str = "continuous",
    voxel_size: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = load_isotropic_nifti(path, interpolation=interpolation, voxel_size=voxel_size)
    return image.get_fdata(), image.header.get_zooms()[:3]


def read_isotropic_like(
    moving_path: Path,
    reference_path: Path,
    interpolation: str = "continuous",
    voxel_size: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    moving = load_nifti(moving_path)
    reference = load_isotropic_nifti(reference_path, interpolation="continuous", voxel_size=voxel_size)
    resampled = resample_from_to(moving, reference, order=interpolation_order(interpolation))
    return resampled.get_fdata(), reference.header.get_zooms()[:3]


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    positive = slice_2d[np.isfinite(slice_2d)]
    if positive.size == 0:
        return np.zeros_like(slice_2d, dtype=float)
    p_low, p_high = np.percentile(positive, [1, 99])
    if p_high <= p_low:
        return np.zeros_like(slice_2d, dtype=float)
    out = (slice_2d - p_low) / (p_high - p_low)
    return np.clip(out, 0, 1)


def choose_mid_slices(image: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    source = mask if mask is not None and mask.shape == image.shape and np.any(mask > 0) else image
    coords = np.argwhere(source > 0)
    if coords.size == 0:
        return {
            "sagittal": image.shape[0] // 2,
            "coronal": image.shape[1] // 2,
            "axial": image.shape[2] // 2,
        }
    return {
        "sagittal": int(np.median(coords[:, 0])),
        "coronal": int(np.median(coords[:, 1])),
        "axial": int(np.median(coords[:, 2])),
    }


def choose_peak_mask_slices(mask: np.ndarray | None, fallback_image: np.ndarray) -> dict[str, int]:
    if mask is None or mask.shape != fallback_image.shape or not np.any(mask > 0):
        return choose_mid_slices(fallback_image)

    binary = mask > 0
    peak_slices: dict[str, int] = {}
    for view, counts in (
        ("sagittal", binary.sum(axis=(1, 2))),
        ("coronal", binary.sum(axis=(0, 2))),
        ("axial", binary.sum(axis=(0, 1))),
    ):
        max_count = counts.max()
        peak_indices = np.flatnonzero(counts == max_count)
        peak_slices[view] = int(peak_indices[len(peak_indices) // 2])
    return peak_slices


def extract_slice(
    image: np.ndarray,
    view: str,
    index: int,
    bbox: tuple[int, int, int, int, int, int] | None = None,
) -> np.ndarray:
    if view == "coronal":
        index = int(np.clip(index, 0, image.shape[1] - 1))
        slice_2d = image[:, index, :]
        if bbox is not None:
            x_min, x_max, _, _, z_min, z_max = bbox
            slice_2d = slice_2d[x_min : x_max + 1, z_min : z_max + 1]
    elif view == "axial":
        index = int(np.clip(index, 0, image.shape[2] - 1))
        slice_2d = image[:, :, index]
        if bbox is not None:
            x_min, x_max, y_min, y_max, _, _ = bbox
            slice_2d = slice_2d[x_min : x_max + 1, y_min : y_max + 1]
    elif view == "sagittal":
        index = int(np.clip(index, 0, image.shape[0] - 1))
        slice_2d = image[index, :, :]
        if bbox is not None:
            _, _, y_min, y_max, z_min, z_max = bbox
            slice_2d = slice_2d[y_min : y_max + 1, z_min : z_max + 1]
    else:
        raise ValueError(f"Unsupported view {view}")
    return np.rot90(slice_2d)


def parcellation_edges(seg_2d: np.ndarray) -> np.ndarray:
    seg_2d = seg_2d.astype(int)
    edges = np.zeros_like(seg_2d, dtype=bool)
    edges[:-1, :] |= seg_2d[:-1, :] != seg_2d[1:, :]
    edges[:, :-1] |= seg_2d[:, :-1] != seg_2d[:, 1:]
    edges &= seg_2d > 0
    return edges


def compute_mask_components(mask: np.ndarray, zooms: tuple[float, float, float]) -> dict[str, object]:
    binary = mask > 0
    labeled, n_labels = ndimage.label(binary)
    voxel_volume_mm3 = float(np.prod(zooms))

    if n_labels == 0:
        return {
            "count": 0,
            "total_volume_mm3": 0.0,
            "largest_volume_mm3": 0.0,
            "center": None,
            "bbox": None,
        }

    component_sizes = ndimage.sum(binary, labeled, index=np.arange(1, n_labels + 1))
    largest_idx = int(np.argmax(component_sizes)) + 1
    total_voxels = float(binary.sum())
    largest_voxels = float(component_sizes[largest_idx - 1])
    coords = np.argwhere(labeled == largest_idx)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = tuple(int(round(v)) for v in coords.mean(axis=0))

    return {
        "count": int(n_labels),
        "total_volume_mm3": total_voxels * voxel_volume_mm3,
        "largest_volume_mm3": largest_voxels * voxel_volume_mm3,
        "center": center,
        "bbox": tuple(int(v) for v in (*mins, *maxs)),
    }


def expand_bbox(
    bbox: tuple[int, int, int, int, int, int] | None,
    shape: tuple[int, int, int],
    pad: int = 12,
) -> tuple[int, int, int, int, int, int]:
    if bbox is None:
        return (0, shape[0] - 1, 0, shape[1] - 1, 0, shape[2] - 1)

    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    return (
        max(0, x_min - pad),
        min(shape[0] - 1, x_max + pad),
        max(0, y_min - pad),
        min(shape[1] - 1, y_max + pad),
        max(0, z_min - pad),
        min(shape[2] - 1, z_max + pad),
    )


def render_missing_figure(title: str, message: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_base(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(normalize_slice(image), cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_raw_or_stripped(title: str, t1: np.ndarray | None, flair: np.ndarray | None, t1_slices: dict[str, int], flair_slices: dict[str, int], out_path: Path) -> None:
    if t1 is None or flair is None:
        render_missing_figure(title, "Missing T1w or FLAIR image for this QC step.", out_path)
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = [
        ("T1w axial", t1, "axial", t1_slices["axial"]),
        ("T1w coronal", t1, "coronal", t1_slices["coronal"]),
        ("FLAIR axial", flair, "axial", flair_slices["axial"]),
        ("FLAIR coronal", flair, "coronal", flair_slices["coronal"]),
    ]
    for ax, (panel_title, volume, view, index) in zip(axes.ravel(), panels):
        draw_base(ax, extract_slice(volume, view, index), f"{panel_title} | slice {index}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_fastsurfer_overlay(title: str, t1: np.ndarray | None, seg: np.ndarray | None, t1_slices: dict[str, int], out_path: Path) -> None:
    if t1 is None or seg is None:
        render_missing_figure(title, "Missing T1w image or FastSurfer parcellation.", out_path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, view in zip(axes, ("axial", "coronal")):
        index = t1_slices[view]
        image_slice = extract_slice(t1, view, index)
        seg_slice = extract_slice(seg, view, index)
        draw_base(ax, image_slice, f"T1w {view} | slice {index}")
        edges = parcellation_edges(seg_slice)
        if np.any(edges):
            ax.contour(edges.astype(int), levels=[0.5], colors="yellow", linewidths=0.45)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_lesion_overlay(
    title: str,
    t1: np.ndarray | None,
    flair: np.ndarray | None,
    lesion_t1: np.ndarray | None,
    lesion_flair: np.ndarray | None,
    t1_slices: dict[str, int],
    flair_slices: dict[str, int],
    out_path: Path,
    lesion_summary: dict[str, object] | None = None,
    t1_bbox: tuple[int, int, int, int, int, int] | None = None,
    flair_bbox: tuple[int, int, int, int, int, int] | None = None,
) -> None:
    if t1 is None or flair is None or lesion_flair is None:
        render_missing_figure(title, "Missing T1w, FLAIR, or annotated lesion mask.", out_path)
        return

    lesion_cmap = ListedColormap(["#00000000", "#ffb000", "#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        ("T1w axial", t1, lesion_t1, "axial", t1_slices["axial"]),
        ("T1w coronal", t1, lesion_t1, "coronal", t1_slices["coronal"]),
        ("T1w sagittal", t1, lesion_t1, "sagittal", t1_slices["sagittal"]),
        ("FLAIR axial", flair, lesion_flair, "axial", flair_slices["axial"]),
        ("FLAIR coronal", flair, lesion_flair, "coronal", flair_slices["coronal"]),
        ("FLAIR sagittal", flair, lesion_flair, "sagittal", flair_slices["sagittal"]),
    ]
    for ax, (panel_title, image, overlay, view, index) in zip(axes.ravel(), panels):
        bbox = t1_bbox if "T1w" in panel_title else flair_bbox
        image_slice = extract_slice(image, view, index, bbox)
        overlay_slice = extract_slice(overlay, view, index, bbox) if overlay is not None else None
        draw_base(ax, image_slice, f"{panel_title} | slice {index}")
        if overlay_slice is not None:
            overlay_mask = np.ma.masked_where(overlay_slice <= 0, overlay_slice)
            if np.any(overlay_slice > 0):
                ax.imshow(overlay_mask, cmap=lesion_cmap, alpha=0.45, interpolation="nearest", vmin=0, vmax=5)

    summary_line = ""
    if lesion_summary is not None:
        summary_line = (
            f" | lesions={lesion_summary['count']} "
            f"| total_volume_ml={lesion_summary['total_volume_mm3'] / 1000.0:.2f} "
            f"| largest_volume_ml={lesion_summary['largest_volume_mm3'] / 1000.0:.2f}"
        )
    fig.suptitle(f"{title}{summary_line}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_inpainting(
    title: str,
    inpainted_t1: np.ndarray | None,
    flair: np.ndarray | None,
    t1_slices: dict[str, int],
    flair_slices: dict[str, int],
    out_path: Path,
) -> None:
    if inpainted_t1 is None or flair is None:
        render_missing_figure(title, "Missing inpainted T1w or FLAIR image.", out_path)
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = [
        ("Inpainted T1w axial", inpainted_t1, "axial", t1_slices["axial"]),
        ("Inpainted T1w coronal", inpainted_t1, "coronal", t1_slices["coronal"]),
        ("FLAIR axial (unchanged by FS-LIT)", flair, "axial", flair_slices["axial"]),
        ("FLAIR coronal (unchanged by FS-LIT)", flair, "coronal", flair_slices["coronal"]),
    ]
    for ax, (panel_title, volume, view, index) in zip(axes.ravel(), panels):
        draw_base(ax, extract_slice(volume, view, index), f"{panel_title} | slice {index}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def summarize_metadata(image_path: Path, json_path: Path) -> dict[str, object]:
    data = read_json(json_path)
    if image_path.exists():
        image = nib.load(str(image_path))
        data = {
            **data,
            "shape": "x".join(str(dim) for dim in image.shape),
            "voxel_size_mm": " x ".join(f"{zoom:.3f}" for zoom in image.header.get_zooms()[:3]),
        }
    return {field: data[field] for field in ("shape", "voxel_size_mm", *META_FIELDS) if field in data}


def load_affine_transform(txt_path: Path, ants_path: Path) -> Path:
    matrix = np.loadtxt(txt_path)
    transform = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    transform.set_parameters(list(matrix[:3, :3].ravel()) + list(matrix[:3, 3]))
    ants.write_transform(transform, str(ants_path))
    return ants_path


def transform_lesion_to_t1(
    lesion_flair_path: Path,
    t1_path: Path,
    flair_to_mni_txt: Path,
    t1_to_mni_txt: Path,
    cache_dir: Path,
    cache_name: str,
) -> Path | None:
    if not all(path.exists() for path in (lesion_flair_path, t1_path, flair_to_mni_txt, t1_to_mni_txt)):
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / cache_name
    if out_path.exists():
        return out_path

    flair_ants = ants.image_read(str(lesion_flair_path))
    t1_ants = ants.image_read(str(t1_path))
    flair_to_mni_ants = load_affine_transform(flair_to_mni_txt, cache_dir / f"{cache_name}.flair_to_mni.mat")
    t1_to_mni_ants = load_affine_transform(t1_to_mni_txt, cache_dir / f"{cache_name}.t1_to_mni.mat")
    warped = ants.apply_transforms(
        fixed=t1_ants,
        moving=flair_ants,
        transformlist=[str(flair_to_mni_ants), str(t1_to_mni_ants)],
        whichtoinvert=[False, True],
        interpolator="nearestNeighbor",
    )
    ants.image_write(warped, str(out_path))
    return out_path


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def path_details(path: Path | None) -> str:
    if path is None:
        return "not resolved"
    if not path.exists():
        return f"missing: {path}"
    modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return f"{path} | modified={modified}"


def rel(path: Path, start: Path) -> str:
    return os.path.relpath(path, start)


def session_context(bids_dir: Path, fastsurfer_dir: Path, lst_dir: Path, lit_dir: Path, sub: str, ses: str, *, image_issues: list[str] | None = None) -> dict[str, Path | None]:
    if image_issues is None:
        image_issues = []
    prefix = f"sub-{sub}_ses-{ses}"
    raw_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "anat"
    lst_temp = lst_dir / f"sub-{sub}" / f"ses-{ses}" / "temp"
    fs_subject = fastsurfer_dir / f"sub-{sub}" / f"ses-{ses}" / prefix
    lit_volumes = lit_dir / f"sub-{sub}" / f"ses-{ses}" / "anat" / "inpainting_volumes"

    return {
        "raw_t1": raw_dir / f"{prefix}_T1w.nii.gz",
        "raw_flair": raw_dir / f"{prefix}_FLAIR.nii.gz",
        "raw_t1_json": raw_dir / f"{prefix}_T1w.json",
        "raw_flair_json": raw_dir / f"{prefix}_FLAIR.json",
        "stripped_t1": lst_temp / f"{prefix}_space-t1w_desc-stripped_T1w.nii.gz",
        "stripped_flair": first_existing(
            lst_temp / f"{prefix}_space-flair_desc-stripped_FLAIR.nii.gz",
            lst_temp / f"{prefix}_space-FLAIR_desc-stripped_FLAIR.nii.gz",
        ),
        "t1_brainmask": lst_temp / f"{prefix}_space-t1w_brainmask.nii.gz",
        "flair_brainmask": first_existing(
            lst_temp / f"{prefix}_space-flair_brainmask.nii.gz",
            lst_temp / f"{prefix}_space-FLAIR_brainmask.nii.gz",
        ),
        "fastsurfer_seg": first_readable_image(
            fs_subject / "mri" / "aparc.DKTatlas+aseg.mapped.mgz",
            fs_subject / "mri" / "aparc.DKTatlas+aseg.deep.mgz",
            issues=image_issues,
        ),
        "fastsurfer_t1": first_readable_image(
            fs_subject / "mri" / "T1.mgz",
            fs_subject / "mri" / "orig.mgz",
            issues=image_issues,
        ),
        "lesion_annotated_flair": lst_dir / f"sub-{sub}" / f"ses-{ses}" / "anat" / f"{prefix}_space-FLAIR_desc-annotated_label-lesion_mask.nii.gz",
        "lesion_flair": lst_dir / f"sub-{sub}" / f"ses-{ses}" / "anat" / f"{prefix}_space-FLAIR_label-lesion_mask.nii.gz",
        "lesion_annot_stats": lst_dir / f"sub-{sub}" / f"ses-{ses}" / "anat" / f"{prefix}_annotated_lesion_stats.csv",
        "flair_to_mni": lst_temp / f"{prefix}_affine_flair_to_mni.mat",
        "t1_to_mni": lst_temp / f"{prefix}_affine_t1w_to_mni.mat",
        "inpainted_t1": lit_volumes / f"{prefix}_inpainting_result.nii.gz",
    }


def write_session_html(
    session_dir: Path,
    sub: str,
    ses: str,
    metadata_t1: dict[str, object],
    metadata_flair: dict[str, object],
    processing_rows: list[tuple[str, str, str]],
    figures: list[tuple[str, str, Path]],
) -> None:
    rows = "\n".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value))}</td><td>{html.escape(str(meta_type))}</td></tr>"
        for meta_type, meta_dict in (("T1w", metadata_t1), ("FLAIR", metadata_flair))
        for key, value in meta_dict.items()
    )
    if not rows:
        rows = '<tr><td colspan="3">No JSON metadata found.</td></tr>'

    proc_rows = "\n".join(
        f"<tr><td>{html.escape(step)}</td><td>{html.escape(status)}</td><td>{html.escape(details)}</td></tr>"
        for step, status, details in processing_rows
    )
    fig_rows = "\n".join(
        f"""
        <section class="panel">
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(description)}</p>
          <img src="{html.escape(rel(path, session_dir))}" alt="{html.escape(title)}">
        </section>
        """
        for title, description, path in figures
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Structural QC sub-{sub} ses-{ses}</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; margin: 24px; color: #1e293b; background: #f8fafc; }}
    h1, h2 {{ color: #0f172a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 24px; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e2e8f0; }}
    .panel {{ background: white; border: 1px solid #cbd5e1; padding: 16px; margin: 0 0 20px; }}
    img {{ width: 100%; max-width: 1100px; border: 1px solid #cbd5e1; }}
    a {{ color: #0369a1; }}
  </style>
</head>
<body>
  <h1>Structural MRI QC: sub-{sub} ses-{ses}</h1>
  <p><a href="../index.html">Back to dataset index</a></p>
  <h2>Acquisition Metadata</h2>
  <table>
    <thead><tr><th>Field</th><th>Value</th><th>Modality</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Processing Summary</h2>
  <table>
    <thead><tr><th>Step</th><th>Status</th><th>Details</th></tr></thead>
    <tbody>{proc_rows}</tbody>
  </table>
  {fig_rows}
</body>
</html>
"""
    (session_dir / "index.html").write_text(html_text)


def write_dataset_index(output_dir: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    body = "\n".join(
        f'<tr><td><a href="{html.escape(link)}">{html.escape(session_id)}</a></td><td>{html.escape(fastsurfer)}</td><td>{html.escape(lst)}</td><td>{html.escape(lit)}</td></tr>'
        for session_id, link, fastsurfer, lst, lit in rows
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Structural QC Index</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; margin: 24px; color: #1e293b; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
    th {{ background: #e2e8f0; }}
    a {{ color: #0369a1; }}
  </style>
</head>
<body>
  <h1>Structural MRI QC Index</h1>
  <table>
    <thead><tr><th>Session</th><th>FastSurfer</th><th>LST-AI</th><th>FS-LIT</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structural MRI QC HTML outputs for a BIDS dataset.")
    parser.add_argument("--bids-dir", required=True, type=Path, help="BIDS root directory.")
    parser.add_argument("--fastsurfer-dir", required=True, type=Path, help="FastSurfer derivatives directory.")
    parser.add_argument("--lst-dir", required=True, type=Path, help="LST-AI derivatives directory.")
    parser.add_argument("--lit-dir", required=True, type=Path, help="FS-LIT derivatives directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output QC directory.")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs to include.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows: list[tuple[str, str, str, str, str]] = []
    subject_ids = parse_subject_ids(args.subjects)

    for sub, ses in discover_sessions(args.bids_dir):
        if subject_ids is not None and sub not in subject_ids:
            continue
        prefix = f"sub-{sub}_ses-{ses}"
        session_dir = args.output_dir / f"{prefix}"
        images_dir = session_dir / "images"
        cache_dir = session_dir / "cache"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_issues: list[str] = []
        ctx = session_context(args.bids_dir, args.fastsurfer_dir, args.lst_dir, args.lit_dir, sub, ses, image_issues=image_issues)

        raw_t1 = read_isotropic_image(ctx["raw_t1"])[0] if ctx["raw_t1"] and ctx["raw_t1"].exists() else None
        raw_flair = read_isotropic_image(ctx["raw_flair"])[0] if ctx["raw_flair"] and ctx["raw_flair"].exists() else None
        stripped_t1 = read_isotropic_image(ctx["stripped_t1"])[0] if ctx["stripped_t1"] and ctx["stripped_t1"].exists() else None
        stripped_flair = read_isotropic_image(ctx["stripped_flair"])[0] if ctx["stripped_flair"] and ctx["stripped_flair"].exists() else None
        fastsurfer_t1 = read_isotropic_image(ctx["fastsurfer_t1"])[0] if ctx["fastsurfer_t1"] and ctx["fastsurfer_t1"].exists() else None
        fastsurfer_seg = (
            read_isotropic_like(ctx["fastsurfer_seg"], ctx["fastsurfer_t1"], interpolation="nearest")[0]
            if ctx["fastsurfer_seg"] and ctx["fastsurfer_seg"].exists() and ctx["fastsurfer_t1"] and ctx["fastsurfer_t1"].exists()
            else None
        )
        lesion_flair = (
            read_isotropic_like(ctx["lesion_annotated_flair"], ctx["raw_flair"], interpolation="nearest")[0]
            if ctx["lesion_annotated_flair"] and ctx["lesion_annotated_flair"].exists() and ctx["raw_flair"] and ctx["raw_flair"].exists()
            else None
        )
        inpainted_t1 = (
            read_isotropic_like(ctx["inpainted_t1"], ctx["raw_t1"])[0]
            if ctx["inpainted_t1"] and ctx["inpainted_t1"].exists() and ctx["raw_t1"] and ctx["raw_t1"].exists()
            else None
        )
        t1_mask = (
            read_isotropic_like(ctx["t1_brainmask"], ctx["raw_t1"], interpolation="nearest")[0]
            if ctx["t1_brainmask"] and ctx["t1_brainmask"].exists() and ctx["raw_t1"] and ctx["raw_t1"].exists()
            else None
        )
        flair_mask = (
            read_isotropic_like(ctx["flair_brainmask"], ctx["raw_flair"], interpolation="nearest")[0]
            if ctx["flair_brainmask"] and ctx["flair_brainmask"].exists() and ctx["raw_flair"] and ctx["raw_flair"].exists()
            else None
        )
        stripped_t1_mask = (
            read_isotropic_like(ctx["t1_brainmask"], ctx["stripped_t1"], interpolation="nearest")[0]
            if ctx["t1_brainmask"] and ctx["t1_brainmask"].exists() and ctx["stripped_t1"] and ctx["stripped_t1"].exists()
            else None
        )
        stripped_flair_mask = (
            read_isotropic_like(ctx["flair_brainmask"], ctx["stripped_flair"], interpolation="nearest")[0]
            if ctx["flair_brainmask"] and ctx["flair_brainmask"].exists() and ctx["stripped_flair"] and ctx["stripped_flair"].exists()
            else None
        )

        base_t1_source = raw_t1 if raw_t1 is not None else stripped_t1 if stripped_t1 is not None else np.zeros((2, 2, 2))
        base_flair_source = raw_flair if raw_flair is not None else stripped_flair if stripped_flair is not None else np.zeros((2, 2, 2))
        stripped_t1_source = stripped_t1 if stripped_t1 is not None else base_t1_source
        stripped_flair_source = stripped_flair if stripped_flair is not None else base_flair_source
        base_t1_slices = choose_mid_slices(base_t1_source, t1_mask)
        base_flair_slices = choose_mid_slices(base_flair_source, flair_mask)
        stripped_t1_slices = choose_mid_slices(stripped_t1_source, stripped_t1_mask)
        stripped_flair_slices = choose_mid_slices(stripped_flair_source, stripped_flair_mask)
        fastsurfer_slices = choose_mid_slices(
            fastsurfer_t1 if fastsurfer_t1 is not None else np.zeros((2, 2, 2)),
            fastsurfer_seg if fastsurfer_seg is not None else None,
        )
        lesion_t1_slices = dict(base_t1_slices)
        lesion_flair_slices = dict(base_flair_slices)

        lesion_t1_path = transform_lesion_to_t1(
            ctx["lesion_annotated_flair"],
            ctx["raw_t1"],
            ctx["flair_to_mni"],
            ctx["t1_to_mni"],
            cache_dir,
            f"{prefix}_annotated_lesion_in_t1w.nii.gz",
        ) if all(ctx[key] is not None for key in ("lesion_annotated_flair", "raw_t1", "flair_to_mni", "t1_to_mni")) else None
        lesion_t1 = (
            read_isotropic_like(lesion_t1_path, ctx["raw_t1"], interpolation="nearest")[0]
            if lesion_t1_path and lesion_t1_path.exists() and ctx["raw_t1"] and ctx["raw_t1"].exists()
            else None
        )

        lesion_summary = None
        flair_bbox = None
        t1_bbox = None
        if ctx["lesion_annotated_flair"] and ctx["lesion_annotated_flair"].exists():
            lesion_flair_data, lesion_flair_zooms = read_image(ctx["lesion_annotated_flair"])
            lesion_summary = compute_mask_components(lesion_flair_data, lesion_flair_zooms)
        if lesion_flair is not None and raw_flair is not None:
            lesion_flair_slices = choose_peak_mask_slices(lesion_flair, raw_flair)
            lesion_flair_display_summary = compute_mask_components(lesion_flair, (1.0, 1.0, 1.0))
            flair_bbox = expand_bbox(lesion_flair_display_summary["bbox"], lesion_flair.shape)
        if lesion_t1 is not None and raw_t1 is not None:
            lesion_t1_slices = choose_peak_mask_slices(lesion_t1, raw_t1)
            lesion_t1_summary = compute_mask_components(lesion_t1, (1.0, 1.0, 1.0))
            t1_bbox = expand_bbox(lesion_t1_summary["bbox"], lesion_t1.shape)

        raw_png = images_dir / "01_raw_inputs.png"
        stripped_png = images_dir / "02_brain_extraction.png"
        fastsurfer_png = images_dir / "03_fastsurfer_overlay.png"
        lesion_png = images_dir / "04_lesion_overlay.png"
        inpaint_png = images_dir / "05_inpainting.png"

        render_raw_or_stripped("Raw input images", raw_t1, raw_flair, base_t1_slices, base_flair_slices, raw_png)
        render_raw_or_stripped("Images after brain extraction", stripped_t1, stripped_flair, stripped_t1_slices, stripped_flair_slices, stripped_png)
        render_fastsurfer_overlay("T1.mgz with FastSurfer parcellation overlay", fastsurfer_t1, fastsurfer_seg, fastsurfer_slices, fastsurfer_png)
        render_lesion_overlay(
            "T1w and FLAIR with annotated lesion mask overlay",
            raw_t1,
            raw_flair,
            lesion_t1,
            lesion_flair,
            lesion_t1_slices,
            lesion_flair_slices,
            lesion_png,
            lesion_summary=lesion_summary,
            t1_bbox=t1_bbox,
            flair_bbox=flair_bbox,
        )
        render_inpainting("T1w and FLAIR after lesion inpainting", inpainted_t1, raw_flair, base_t1_slices, base_flair_slices, inpaint_png)

        metadata_t1 = summarize_metadata(ctx["raw_t1"], ctx["raw_t1_json"]) if ctx["raw_t1"] and ctx["raw_t1_json"] else {}
        metadata_flair = summarize_metadata(ctx["raw_flair"], ctx["raw_flair_json"]) if ctx["raw_flair"] and ctx["raw_flair_json"] else {}
        processing_rows = [
            ("FastSurfer parcellation", bool_text(ctx["fastsurfer_seg"] is not None and ctx["fastsurfer_seg"].exists()), path_details(ctx["fastsurfer_seg"])),
            ("FastSurfer T1.mgz", bool_text(ctx["fastsurfer_t1"] is not None and ctx["fastsurfer_t1"].exists()), path_details(ctx["fastsurfer_t1"])),
            ("LST-AI lesion mask", bool_text(ctx["lesion_annotated_flair"] is not None and ctx["lesion_annotated_flair"].exists()), path_details(ctx["lesion_annotated_flair"])),
            ("LST-AI stripped T1w", bool_text(ctx["stripped_t1"] is not None and ctx["stripped_t1"].exists()), path_details(ctx["stripped_t1"])),
            ("LST-AI stripped FLAIR", bool_text(ctx["stripped_flair"] is not None and ctx["stripped_flair"].exists()), path_details(ctx["stripped_flair"])),
            ("FS-LIT inpainted T1w", bool_text(ctx["inpainted_t1"] is not None and ctx["inpainted_t1"].exists()), path_details(ctx["inpainted_t1"])),
            ("FLAIR to MNI affine", bool_text(ctx["flair_to_mni"] is not None and ctx["flair_to_mni"].exists()), path_details(ctx["flair_to_mni"])),
            ("T1w to MNI affine", bool_text(ctx["t1_to_mni"] is not None and ctx["t1_to_mni"].exists()), path_details(ctx["t1_to_mni"])),
        ]
        processing_rows.extend(("FastSurfer input", "warning", issue) for issue in image_issues)
        figures = [
            ("Raw input images", "Mid axial and coronal slices for both T1w and FLAIR, resampled to a 1 mm isotropic display grid.", raw_png),
            ("After brain extraction", "LST-AI stripped T1w and stripped FLAIR outputs, resampled to a 1 mm isotropic display grid.", stripped_png),
            ("FastSurfer parcellation", "T1w with FastSurfer parcellation contour overlay on a 1 mm isotropic display grid.", fastsurfer_png),
            ("Annotated lesion mask", "Annotated lesion mask on the axial, coronal, and sagittal slices with the highest lesion density.", lesion_png),
            ("After lesion inpainting", "Inpainted T1w plus unchanged FLAIR for side-by-side review on a 1 mm isotropic display grid.", inpaint_png),
        ]
        write_session_html(session_dir, sub, ses, metadata_t1, metadata_flair, processing_rows, figures)

        dataset_rows.append(
            (
                prefix,
                f"{prefix}/index.html",
                bool_text(ctx["fastsurfer_seg"] is not None and ctx["fastsurfer_seg"].exists()),
                bool_text(ctx["lesion_annotated_flair"] is not None and ctx["lesion_annotated_flair"].exists()),
                bool_text(ctx["inpainted_t1"] is not None and ctx["inpainted_t1"].exists()),
            )
        )

    write_dataset_index(args.output_dir, dataset_rows)
    print(f"Wrote QC reports for {len(dataset_rows)} sessions to {args.output_dir}")


if __name__ == "__main__":
    main()
