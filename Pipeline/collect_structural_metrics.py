#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pandas as pd


def sanitize(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")


def coerce_value(value: str):
    text = value.strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


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


def parse_measure_line(line: str, stats_name: str) -> dict[str, object]:
    payload = line.replace("# Measure", "", 1).strip()
    parts = [part.strip() for part in payload.split(",")]
    if len(parts) < 4:
        return {}

    measure_name = sanitize(parts[1] or parts[0] or "value")
    value = coerce_value(parts[3])
    return {f"fastsurfer.{stats_name}.measure.{measure_name}": value}


def parse_stats_table(lines: list[str], stats_name: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    headers: list[str] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# Measure"):
            metrics.update(parse_measure_line(line, stats_name))
            continue
        if line.startswith("# ColHeaders"):
            headers = line.replace("# ColHeaders", "", 1).strip().split()
            continue
        if line.startswith("#"):
            continue
        if not headers:
            continue

        values = line.split()
        if len(values) < len(headers):
            continue

        row = dict(zip(headers, values[: len(headers)]))
        row_name = (
            row.get("StructName")
            or row.get("FoldName")
            or row.get("SegId")
            or row.get("Index")
            or f"row_{len(metrics)}"
        )
        row_name = sanitize(str(row_name))

        for header, value in row.items():
            if header in {"Index", "StructName", "FoldName"}:
                continue
            key = f"fastsurfer.{stats_name}.table.{row_name}.{sanitize(header)}"
            metrics[key] = coerce_value(value)

    return metrics


def resolve_asegstats2table(command: str | None = None) -> tuple[str, dict[str, str]]:
    """Find FreeSurfer's executable and configure its bundled Python launcher."""
    explicit = command or os.environ.get("ASEGSTATS2TABLE")
    candidates = [explicit] if explicit else ["asegstats2table"]
    if not explicit:
        if os.environ.get("FREESURFER_HOME"):
            candidates.append(str(Path(os.environ["FREESURFER_HOME"]) / "bin" / "asegstats2table"))
        candidates.append(str(Path.home() / "freesurfer" / "bin" / "asegstats2table"))
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            executable = str(Path(executable).resolve())
            env = os.environ.copy()
            install = Path(executable).parent.parent
            if (install / "bin" / "fspython").is_file():
                env["FREESURFER_HOME"] = str(install)
                env["PATH"] = str(install / "bin") + os.pathsep + env.get("PATH", "")
            return executable, env
    raise RuntimeError(
        "Cannot find FreeSurfer's asegstats2table. Source SetUpFreeSurfer.sh, "
        "set FREESURFER_HOME, or pass --asegstats2table /path/to/bin/asegstats2table "
        "(ASEGSTATS2TABLE also works when running struct_bids.sh)."
    )


def extract_segmentation_volumes(stats_file: Path, tool: tuple[str, dict[str, str]]) -> dict[str, object]:
    executable, env = tool
    # One input per invocation makes the session mapping unambiguous even when
    # FreeSurfer emits row numbers instead of subject IDs for --inputs.
    with tempfile.TemporaryDirectory(prefix="asegstats2table_") as tmp:
        table = Path(tmp) / "volumes.tsv"
        result = subprocess.run(
            [executable, "--inputs", str(stats_file.resolve()), "--meas", "volume",
             "--delimiter", "tab", "--tablefile", str(table)],
            env=env, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise ValueError(
                f"asegstats2table exited {result.returncode}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        with table.open(newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        if len(rows) != 2 or len(rows[0]) < 2 or len(rows[0]) != len(rows[1]):
            raise ValueError("asegstats2table did not produce a single-session metrics table")
        metrics = {}
        for name, value in zip(rows[0][1:], rows[1][1:]):
            key = f"fastsurfer.{sanitize(stats_file.stem)}.volume.{sanitize(name)}"
            if key in metrics:
                raise ValueError(f"Duplicate metric name in asegstats2table output: {name}")
            number = float(value)
            metrics[key] = coerce_value(str(number))
        return metrics


def parse_fastsurfer_metrics(stats_dir: Path, asegstats2table: str | None = None) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if not stats_dir.exists():
        return metrics

    tool = None
    errors = []
    for stats_file in sorted(stats_dir.glob("*.stats")):
        lines = stats_file.read_text().splitlines()
        if not any(line.strip() for line in lines):
            message = f"Empty stats file: {stats_file}"
            errors.append(message)
            print(f"WARNING: {message}", file=sys.stderr)
            continue
        headers = next((line.split()[2:] for line in lines if line.startswith("# ColHeaders")), [])
        if {"SegId", "Volume_mm3", "StructName"}.issubset(headers):
            if tool is None:
                tool = resolve_asegstats2table(asegstats2table)
            try:
                metrics.update(extract_segmentation_volumes(stats_file, tool))
            except (ValueError, OSError) as exc:
                message = f"{stats_file}: {exc}"
                errors.append(message)
                print(f"WARNING: {message}", file=sys.stderr)
        else:
            # Cortical aparc tables and brainvol headers are not aseg-format
            # segmentation tables; retain their existing metrics and naming.
            metrics.update(parse_stats_table(lines, sanitize(stats_file.stem)))
    if errors:
        metrics["fastsurfer.extraction_errors"] = " | ".join(errors)
    return metrics


def iter_lst_tables(lst_dir: Path, sub: str, ses: str) -> Iterator[tuple[str, Path]]:
    anat_dir = lst_dir / f"sub-{sub}" / f"ses-{ses}" / "anat"
    yield "lesion", anat_dir / f"sub-{sub}_ses-{ses}_lesion_stats.csv"
    yield "annotated", anat_dir / f"sub-{sub}_ses-{ses}_annotated_lesion_stats.csv"


def row_identifier(row: pd.Series, fallback: str) -> str:
    for candidate in ("Region", "region", "Label", "label", "Name", "name", "Location", "location"):
        if candidate in row.index and pd.notna(row[candidate]):
            value = sanitize(str(row[candidate]))
            if value:
                return value
    return fallback


def flatten_table(df: pd.DataFrame, prefix: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if df.empty:
        return metrics

    single_row = len(df.index) == 1
    for index, (_, row) in enumerate(df.iterrows()):
        ident = row_identifier(row, f"row_{index}")
        for column, value in row.items():
            if pd.isna(value):
                continue
            if single_row:
                key = f"{prefix}.{sanitize(str(column))}"
            else:
                key = f"{prefix}.{ident}.{sanitize(str(column))}"
            metrics[key] = value.item() if hasattr(value, "item") else value
    return metrics


def parse_lst_metrics(lst_dir: Path, sub: str, ses: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for label, csv_path in iter_lst_tables(lst_dir, sub, ses):
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        metrics.update(flatten_table(df, f"lst_ai.{label}"))
    return metrics


def session_record(
    bids_dir: Path,
    fastsurfer_dir: Path,
    lst_dir: Path,
    lit_dir: Path,
    sub: str,
    ses: str,
    asegstats2table: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "sub": sub,
        "ses": ses,
        "session_id": f"sub-{sub}_ses-{ses}",
    }

    record["raw_t1w"] = str(bids_dir / f"sub-{sub}" / f"ses-{ses}" / "anat" / f"sub-{sub}_ses-{ses}_T1w.nii.gz")
    record["raw_flair"] = str(bids_dir / f"sub-{sub}" / f"ses-{ses}" / "anat" / f"sub-{sub}_ses-{ses}_FLAIR.nii.gz")
    record["fastsurfer_dir"] = str(
        fastsurfer_dir / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}"
    )
    record["lst_dir"] = str(lst_dir / f"sub-{sub}" / f"ses-{ses}")
    record["lit_dir"] = str(lit_dir / f"sub-{sub}" / f"ses-{ses}")

    fs_stats_dir = fastsurfer_dir / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}" / "stats"
    record.update(parse_fastsurfer_metrics(fs_stats_dir, asegstats2table))
    record.update(parse_lst_metrics(lst_dir, sub, ses))
    return record


def write_long_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect structural FastSurfer and LST-AI metrics into one CSV.")
    parser.add_argument("--bids-dir", required=True, type=Path, help="BIDS root directory.")
    parser.add_argument("--fastsurfer-dir", required=True, type=Path, help="FastSurfer derivatives directory.")
    parser.add_argument("--lst-dir", required=True, type=Path, help="LST-AI derivatives directory.")
    parser.add_argument("--lit-dir", required=True, type=Path, help="FS-LIT derivatives directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for CSV output.")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs to include.")
    parser.add_argument("--asegstats2table", help="FreeSurfer asegstats2table executable; defaults to ASEGSTATS2TABLE, PATH, FREESURFER_HOME/bin, or ~/freesurfer/bin.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = parse_subject_ids(args.subjects)
    rows = [
        session_record(args.bids_dir, args.fastsurfer_dir, args.lst_dir, args.lit_dir, sub, ses, args.asegstats2table)
        for sub, ses in discover_sessions(args.bids_dir)
        if subject_ids is None or sub in subject_ids
    ]

    output_csv = args.output_dir / "structural_metrics_summary.csv"
    write_long_csv(output_csv, rows)
    print(f"Wrote {len(rows)} session rows to {output_csv}")


if __name__ == "__main__":
    main()
