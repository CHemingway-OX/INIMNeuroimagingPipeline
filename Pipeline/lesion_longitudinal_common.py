"""Input, geometry and provenance for the standalone lesion research tools."""
import argparse
import csv
from datetime import date, datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
from contextlib import contextmanager

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.linalg import sqrtm


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mask-kind', choices=['binary', 'probability'], default='binary')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--alignment', choices=['register', 'already-aligned'], default='register',
                   help='already-aligned is for verified aligned data / synthetic tests only')
    p.add_argument('--max-lesion-outside-brain-mm3', type=float, default=10.,
                   help='Lesion volume per visit allowed outside the brain mask; it is removed and recorded '
                        'as lesion_outside_brain_mm3. More fails validation. Default: 10')
    return p


def same_grid(a, b):
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4, rtol=0)


def infer_date(row, directory):
    found, sources = set(), []
    for key in ['t1', 'flair']:
        filename = row.get(key, '')
        f = Path(filename)
        f = directory / f if not f.is_absolute() else f
        sidecar = Path(str(f)[:-7] + '.json') if str(f).endswith('.nii.gz') else f.with_suffix('.json')
        if not sidecar.exists():
            continue
        metadata = json.loads(sidecar.read_text())
        for field in ['AcquisitionDateTime', 'AcquisitionDate', 'StudyDate', 'SeriesDate']:
            value = str(metadata.get(field, '')).strip()
            if not value or value == 'n/a':
                continue
            if re.match(r'^\d{8}', value):
                value = value[:4] + '-' + value[4:6] + '-' + value[6:8]
            parsed = date.fromisoformat(value[:10])
            found.add(parsed.isoformat()); sources.append(str(sidecar) + ':' + field)
    if len(found) != 1:
        raise ValueError(f'{row.get("session")}: cannot infer one scan date from sidecars '
                         '(missing or conflicting dates). Supply date or elapsed days; AcquisitionTime alone is insufficient.')
    row['date'] = found.pop()
    row['date_source'] = ';'.join(sources)


def load_manifest(args):
    if args.threads < 1 or not 0 < args.threshold < 1:
        raise ValueError('threads must be positive and threshold must lie in (0, 1)')
    path = args.manifest.resolve()
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t' if path.suffix == '.tsv' else ','))
    if len(rows) < 2:
        raise ValueError('At least two manifest rows are required')
    required = ['subject', 'session', 't1', 'flair', 'lesion', 'brain_mask']
    paths = ['t1', 'flair', 'lesion', 'brain_mask', 'enhancement_mask', 'chronic_mask']
    dated = all(r.get('date', '').strip() for r in rows)
    elapsed = all(r.get('days', '').strip() for r in rows)
    if not dated and not elapsed:
        for r in rows:
            if not r.get('date', '').strip():
                infer_date(r, path.parent)
        dated = True
    seen = set()
    for r in rows:
        if any(not r.get(k, '').strip() for k in required):
            raise ValueError(f'Missing required columns/values: {required}')
        for key in ['subject', 'session']:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', r[key]):
                raise ValueError(f'Unsafe {key}: {r[key]}')
        if r['session'] in seen:
            raise ValueError('Duplicate session')
        seen.add(r['session'])
        r['time'] = float(date.fromisoformat(r['date']).toordinal() if dated else r['days'])
        if not np.isfinite(r['time']):
            raise ValueError('Nonfinite visit time')
        for key in paths:
            value = r.get(key, '').strip()
            if not value:
                r[key] = None
                continue
            f = Path(value)
            f = (path.parent / f).resolve() if not f.is_absolute() else f.resolve()
            img = nib.load(f)
            data = np.asanyarray(img.dataobj)
            if data.ndim != 3 or not np.isfinite(data).all():
                raise ValueError(f'{f}: expected finite 3D image')
            if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-9:
                raise ValueError(f'{f}: invalid affine')
            is_binary = key.endswith('_mask') or (key == 'lesion' and args.mask_kind == 'binary')
            if is_binary and not np.isin(data, [0, 1]).all():
                raise ValueError(f'{f}: expected binary 0/1 mask')
            if key == 'lesion' and args.mask_kind == 'probability' and (data.min() < 0 or data.max() > 1):
                raise ValueError(f'{f}: probabilities must lie in [0,1]')
            r[key] = str(f)
        flair = nib.load(r['flair'])
        for key in ['lesion', 'brain_mask', 'enhancement_mask', 'chronic_mask']:
            if r[key] and not same_grid(flair, nib.load(r[key])):
                raise ValueError(f'{r["session"]}: {key} must be on the FLAIR grid')
        brain = np.asanyarray(nib.load(r['brain_mask']).dataobj) > 0
        lesion = mask_data(r['lesion'], args)
        if not brain.any():
            raise ValueError(f'{r["session"]}: empty brain mask')
        # Independent lesion and brain segmentations disagree by a few voxels at the brain edge.
        outside = float((lesion & ~brain).sum() * abs(np.linalg.det(flair.affine[:3, :3])))
        limit = getattr(args, 'max_lesion_outside_brain_mm3', 0.)
        if outside > limit:
            raise ValueError(f'{r["session"]}: {outside:.1f} mm3 of lesions outside the brain mask '
                             f'(more than {limit:g} mm3); check the lesion and brain masks')
        r['lesion_outside_brain_mm3'] = round(outside, 3)
    if len({r['subject'] for r in rows}) != 1:
        raise ValueError('Use one subject per invocation')
    rows.sort(key=lambda r: r['time'])
    start = rows[0]['time']
    for r in rows:
        r['years'] = (r['time'] - start) / 365.25
    if any(b['time'] <= a['time'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Visit times must be distinct')
    return rows


def mask_data(path, args):
    data = np.asanyarray(nib.load(path).dataobj)
    return data > (args.threshold if args.mask_kind == 'probability' else 0)


def lesion_in_brain(row, args):
    """Lesion mask restricted to the brain mask (both on the FLAIR grid); see load_manifest."""
    return mask_data(row['lesion'], args) & (np.asanyarray(nib.load(row['brain_mask']).dataobj) > 0)


def volume(path, mask):
    return float(mask.sum() * abs(np.linalg.det(nib.load(path).affine[:3, :3])))


def configure_ants(args):
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(args.threads)
    import ants
    return ants


def stale_lock_reason(owner):
    """Why a lock's owner has certainly ended, or None if it may still be running.

    Slurm stops timed-out jobs with SIGKILL while ANTs holds the interpreter, so
    an ended job can leave its lock behind.
    """
    job = owner.get('slurm_job_id')
    if job and job != os.environ.get('SLURM_JOB_ID'):
        try:
            state = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%T'],
                                   capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if state.returncode == 0 and not state.stdout.strip():
            return f'Slurm job {job} is no longer queued or running'
        if state.returncode != 0 and 'Invalid job id' in state.stderr:
            return f'Slurm job {job} has ended'
        return None
    if owner.get('host') == socket.gethostname() and isinstance(owner.get('pid'), int):
        try:
            os.kill(owner['pid'], 0)
        except ProcessLookupError:
            return f'process {owner["pid"]} on {owner["host"]} has exited'
        except PermissionError:
            return None
    return None


def acquire_lock(lock):
    owner_file = lock / 'owner.json'
    owner = {'host': socket.gethostname(), 'pid': os.getpid(),
             'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
             'started': datetime.now().isoformat(timespec='seconds')}
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            previous = json.loads(owner_file.read_text())
        except (OSError, ValueError):
            previous = {}
        reason = stale_lock_reason(previous) if previous else None
        if reason is None:
            raise ValueError(f'{lock} exists: another run or interrupted run '
                             f'(owner: {previous or "unknown"}); inspect before removing')
        print(f'Removing stale lock {lock}: {reason}', flush=True)
        release_lock(lock)
        lock.mkdir()
    write_json(owner_file, owner)


def release_lock(lock):
    for p in lock.iterdir():
        p.unlink()
    lock.rmdir()


@contextmanager
def run_directory(args, rows, method):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.running'
    acquire_lock(lock)
    # SIGTERM (scancel, Slurm time limit) unwinds through finally when Python has control.
    previous_handler = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        import importlib.metadata as metadata
        files = {r[k]: digest(r[k]) for r in rows for k in
                 ['t1', 'flair', 'lesion', 'brain_mask', 'enhancement_mask', 'chronic_mask'] if r[k]}
        for key in ['review', 'calibration']:
            f = getattr(args, key, None)
            if f:
                files[str(f.resolve())] = digest(f)
        source = Path(__file__).parent
        config = {'method': method, 'rows': rows, 'files_sha256': files,
                  'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                if k not in ['resume', 'dry_run']},
                  'source_sha256': {p.name: digest(p) for p in [Path(__file__), source / (method + '.py')]},
                  'versions': {n: metadata.version(n) for n in ['numpy', 'scipy', 'nibabel', 'antspyx']},
                  'method_status': 'research_adaptation_not_validated_reference_reproduction'}
        cfg = root / 'provenance.json'
        if cfg.exists():
            if not args.resume or json.loads(cfg.read_text()) != config:
                raise ValueError('Output exists or configuration changed: use a new output directory; --resume requires identical inputs/settings')
        elif any(p != lock for p in root.iterdir()):
            raise ValueError('Output directory must be empty for a new run')
        else:
            write_json(cfg, config)
        write_json(root / 'complete.json', {'complete': False})
        yield root
        write_json(root / 'complete.json', {'complete': True})
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        release_lock(lock)


def cached(directory, build):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'stage.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        if all(Path(p).exists() and digest(p) == h for p, h in result['hashes'].items()):
            return result['outputs']
        raise ValueError(f'Cached outputs changed: {directory}; use a new output directory')
    outputs = build(directory)
    hashes = {str(p): digest(p) for p in directory.iterdir() if p.is_file()}
    write_json(marker, {'outputs': outputs, 'hashes': hashes})
    return outputs


def ants_like(ants, data, ref):
    return ants.from_numpy(np.asarray(data, dtype=np.float32), origin=ref.origin,
                           spacing=ref.spacing, direction=ref.direction)


def prepare(row, args, root):
    ants = configure_ants(args)
    def build(d):
        flair = ants.image_read(row['flair'])
        t1 = ants.image_read(row['t1'])
        brain = ants.image_read(row['brain_mask'])
        flair = ants.n4_bias_field_correction(flair, mask=brain)
        if args.alignment == 'register':
            reg = ants.registration(fixed=t1, moving=flair, type_of_transform='Rigid',
                                    random_seed=args.seed, outprefix=str(d / 'flair_to_t1_'))
            tx = reg['fwdtransforms']
        else:
            if not same_grid(nib.load(row['t1']), nib.load(row['flair'])):
                raise ValueError('already-aligned requires matching T1/FLAIR grids')
            tx = []
        def warp(img, interp):
            return ants.apply_transforms(fixed=t1, moving=img, transformlist=tx, interpolator=interp)
        b = warp(brain, 'nearestNeighbor')
        t1 = ants.n4_bias_field_correction(t1, mask=b)
        images = {'t1': t1, 'flair': warp(flair, 'linear'), 'brain': b,
                  'lesion': warp(ants_like(ants, lesion_in_brain(row, args), flair), 'nearestNeighbor')}
        for k in ['enhancement_mask', 'chronic_mask']:
            if row[k]:
                images[k] = warp(ants.image_read(row[k]), 'nearestNeighbor')
        outputs = {}
        native_labels, _ = ndimage.label(lesion_in_brain(row, args), ndimage.generate_binary_structure(3, 3))
        images['labels'] = warp(ants_like(ants, native_labels, flair), 'nearestNeighbor')
        for k, img in images.items():
            p = str(d / (k + '.nii.gz'))
            ants.image_write(img, p)
            outputs[k] = p
        return outputs
    return cached(root / 'prepared' / row['session'], build)


def homogeneous(tx):
    m = np.asarray(tx.parameters[:9]).reshape(3, 3)
    t = np.asarray(tx.parameters[9:12])
    c = np.asarray(tx.fixed_parameters[:3])
    out = np.eye(4)
    out[:3, :3] = m
    out[:3, 3] = t + c - m @ c
    return out


def half_matrix(matrix):
    h = sqrtm(matrix)
    if np.max(np.abs(h.imag)) > 1e-7:
        raise ValueError('Affine has no usable real halfway transform')
    h = h.real
    if np.linalg.det(h[:3, :3]) <= 0 or not np.allclose(h @ h, matrix, atol=1e-5):
        raise ValueError('Invalid halfway transform')
    return h


def transform(ants, matrix):
    return ants.create_ants_transform(matrix=matrix[:3, :3], translation=matrix[:3, 3], center=(0, 0, 0))


def midpoint_grid(ants, a, b, h, spacing=1.):
    # ANTs affine matrices are physical pull mappings: midpoint -> source image.
    points = []
    for img, source_to_mid in [(a, h), (b, np.linalg.inv(h))]:
        for index in itertools.product(*[(-1, n) for n in img.shape]):
            p = np.asarray(img.origin) + img.direction @ (np.asarray(index) * img.spacing)
            points.append((source_to_mid @ np.r_[p, 1])[:3])
    origin = np.floor(np.min(points, axis=0) / spacing) * spacing
    shape = np.ceil((np.max(points, axis=0) - origin) / spacing).astype(int) + 1
    if np.prod(shape) > 256_000_000:
        raise ValueError('Implausibly large midpoint grid: inspect image headers/registration')
    return ants.from_numpy(np.zeros(tuple(shape), np.float32), origin=tuple(origin), spacing=(spacing,) * 3)


def qc_overlay(path, background, labels, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = nib.as_closest_canonical(nib.load(background)).get_fdata()
    b = nib.as_closest_canonical(nib.load(labels)).get_fdata()
    if a.shape != b.shape:
        raise ValueError('QC grid mismatch')
    counts = (b > 0).sum(axis=(0, 1))
    slices = np.unique(np.r_[np.argmax(counts), np.linspace(a.shape[2] // 4, 3 * a.shape[2] // 4, 3).astype(int)])
    fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4), squeeze=False)
    positive = a[a > 0]
    vmax = np.percentile(positive, 99) if positive.size else 1
    for ax, z in zip(axes[0], slices):
        ax.imshow(a[:, :, z].T, origin='lower', cmap='gray', vmin=0, vmax=vmax)
        ax.imshow(np.ma.masked_where(b[:, :, z].T == 0, b[:, :, z].T), origin='lower', cmap='tab20', alpha=.6)
        ax.set_title(f'z={z}'); ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def execute(main):
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f'Error: {error}') from error


def qc_registration(path, fixed, moving, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images = [nib.as_closest_canonical(nib.load(p)).get_fdata() for p in [fixed, moving]]
    if images[0].shape != images[1].shape:
        raise ValueError('Registration QC grid mismatch')
    for i, a in enumerate(images):
        vals = a[a > 0]
        high = np.percentile(vals, 99) if vals.size else 1
        images[i] = a / max(high, 1e-9)
    slices = np.linspace(images[0].shape[2] // 3, 2 * images[0].shape[2] // 3, 3).astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for col, z in enumerate(slices):
        for row in range(2):
            axes[row, col].imshow(images[row][:, :, z].T, origin='lower', cmap='gray', vmin=0, vmax=1)
        axes[2, col].imshow((images[1] - images[0])[:, :, z].T, origin='lower', cmap='coolwarm', vmin=-.5, vmax=.5)
        for ax in axes[:, col]: ax.axis('off')
    fig.suptitle(title + '\nReference / aligned follow-up / normalized difference')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_profiles(path, profiles, xlabel, ylabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, x, y in profiles:
        ax.plot(x, y, marker='o', label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if profiles: ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
