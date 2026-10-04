#!/usr/bin/env python3
"""Correct NIfTI orientation of a session scanned head first but registered as feet first.

If 'Feet First Supine' (FFS) was entered at the scanner for a head-first scan, the
DICOM patient coordinates are rotated by 180 degrees about the anterior-posterior
axis: left/right and superior/inferior are swapped, anterior/posterior is not.
dcm2niix converts this faithfully, so the image looks upside down although its
header claims a valid orientation. This rotation (not a mirror) is undone exactly
by negating the x and z world coordinates in the qform and sform; voxel data are
not touched.

Only images whose JSON sidecar says PatientPosition FFS are changed. Without
--write this is a dry run. With --write the originals are first copied to a backup
directory outside the BIDS tree, then header and sidecar are updated (the sidecar
records the correction). Verify left/right afterwards against another session of
the subject, e.g. with a clearly one-sided lesion, before processing the data.

Needs only Python 3.8+ (no nibabel); NIfTI-1 (dcm2niix default) only:
  python3 fix_patient_position.py --bids /mnt/e/COHORT/BIDS --session sub-072_ses-2
  python3 fix_patient_position.py --bids /mnt/e/COHORT/BIDS --session sub-072_ses-2 --write
"""
import argparse
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys

HEADER_SIZE = 348
QFORM = 252           # qform_code, sform_code (int16)
QUATERN = 256         # quatern_b, c, d, qoffset_x, y, z (float32)
SROW = 280            # srow_x, srow_y, srow_z (3 x 4 float32)
TOOL = 'INIMNeuroimagingPipeline hpc/fix_patient_position.py'


def open_image(path, mode):
    return gzip.open(path, mode) if str(path).endswith('.gz') else open(path, mode)


def endian(header):
    for order in '<>':
        if struct.unpack_from(order + 'i', header, 0)[0] == HEADER_SIZE:
            return order
    raise ValueError('not a NIfTI-1 file (sizeof_hdr is not 348; NIfTI-2 is not supported)')


def rotate_quaternion(b, c, d):
    """Quaternion (b, c, d) of the rotation diag(-1, 1, -1) applied after (b, c, d)."""
    a = math.sqrt(max(0.0, 1.0 - b * b - c * c - d * d))
    # (0, 0, 1, 0) * (a, b, c, d): 180 degrees about y, Hamilton product.
    w, x, y, z = -c, d, a, -b
    if w < 0:  # NIfTI stores the quaternion with a non-negative real part
        w, x, y, z = -w, -x, -y, -z
    return x, y, z


def corrected_header(header):
    order = endian(header)
    out = bytearray(header)
    b, c, d, qx, qy, qz = struct.unpack_from(order + '6f', header, QUATERN)
    struct.pack_into(order + '6f', out, QUATERN, *rotate_quaternion(b, c, d), -qx, qy, -qz)
    srow = list(struct.unpack_from(order + '12f', header, SROW))
    srow[0:4] = [-v for v in srow[0:4]]    # x row
    srow[8:12] = [-v for v in srow[8:12]]  # z row
    struct.pack_into(order + '12f', out, SROW, *srow)
    return bytes(out)


def axcodes(header):
    """Orientation letters of the sform (or qform) voxel axes, like nibabel.aff2axcodes."""
    order = endian(header)
    qform_code, sform_code = struct.unpack_from(order + '2h', header, QFORM)
    if sform_code > 0:
        rows = struct.unpack_from(order + '12f', header, SROW)
        matrix = [rows[0:3], rows[4:7], rows[8:11]]
    elif qform_code > 0:
        b, c, d = struct.unpack_from(order + '3f', header, QUATERN)
        a = math.sqrt(max(0.0, 1.0 - b * b - c * c - d * d))
        qfac = -1.0 if struct.unpack_from(order + 'f', header, 76)[0] < 0 else 1.0  # pixdim[0]
        matrix = [[a*a + b*b - c*c - d*d, 2*(b*c - a*d), qfac * 2*(b*d + a*c)],
                  [2*(b*c + a*d), a*a + c*c - b*b - d*d, qfac * 2*(c*d - a*b)],
                  [2*(b*d - a*c), 2*(c*d + a*b), qfac * (a*a + d*d - b*b - c*c)]]
    else:
        return '???'
    labels = (('L', 'R'), ('P', 'A'), ('I', 'S'))
    codes = ''
    for column in range(3):
        values = [matrix[row][column] for row in range(3)]
        axis = max(range(3), key=lambda i: abs(values[i]))
        codes += labels[axis][values[axis] > 0]
    return codes


def rewrite(path, target):
    """Copy path to target with the corrected header; data are streamed unchanged."""
    with open_image(path, 'rb') as source, open_image(target, 'wb') as sink:
        header = source.read(HEADER_SIZE)
        sink.write(corrected_header(header))
        shutil.copyfileobj(source, sink, 1 << 22)


def read_header(path):
    with open_image(path, 'rb') as stream:
        return stream.read(HEADER_SIZE)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bids', type=Path, required=True)
    p.add_argument('--session', required=True, help='e.g. sub-072_ses-2')
    p.add_argument('--backup-dir', type=Path,
                   help='Default: <bids>_patient_position_backup next to the BIDS directory.')
    p.add_argument('--write', action='store_true', help='Change the files. Default: dry run.')
    args = p.parse_args(argv)

    match = re.fullmatch(r'(sub-[A-Za-z0-9]+)_(ses-[A-Za-z0-9]+)', args.session)
    if not match:
        p.error('--session must look like sub-072_ses-2')
    bids = args.bids.resolve()
    anat = bids / match.group(1) / match.group(2) / 'anat'
    images = sorted(anat.glob('*.nii.gz')) + sorted(anat.glob('*.nii'))
    if not images:
        raise SystemExit(f'No NIfTI images in {anat}')

    todo = []
    for image in images:
        sidecar = Path((str(image)[:-7] if image.name.endswith('.nii.gz') else str(image)[:-4]) + '.json')
        position = json.loads(sidecar.read_text()).get('PatientPosition') if sidecar.exists() else None
        header = read_header(image)
        if position != 'FFS':
            print(f'{image.name}: PatientPosition {position}, left unchanged')
            continue
        print(f'{image.name}: PatientPosition FFS, orientation {axcodes(header)} -> {axcodes(corrected_header(header))}')
        todo.append((image, sidecar))
    if not todo:
        print('Nothing to correct.')
        return 0
    if not args.write:
        print('Dry run: nothing written. Add --write to correct these files.')
        return 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = (args.backup_dir or bids.parent / (bids.name + '_patient_position_backup')).resolve() / stamp
    for image, sidecar in todo:
        for original in (image, sidecar):
            target = backup / original.relative_to(bids)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        # Keep the .nii/.nii.gz ending: it decides whether the copy is gzip-compressed.
        tmp = image.with_name(image.name.replace('.nii', '.tmp.nii', 1))
        rewrite(image, tmp)
        tmp.replace(image)
        meta = json.loads(sidecar.read_text())
        meta['PatientPosition'] = 'HFS'
        meta['PatientPositionCorrection'] = {
            'original': 'FFS', 'corrected_to': 'HFS',
            'transform': 'qform/sform rotated 180 degrees about the anterior-posterior axis '
                         '(world x and z negated); voxel data unchanged',
            'reason': 'head-first scan registered as feet first at the scanner',
            'date': datetime.now().isoformat(timespec='seconds'), 'tool': TOOL,
            'backup': str(backup / sidecar.relative_to(bids).parent)}
        sidecar.write_text(json.dumps(meta, indent=2) + '\n')
        print(f'{image.name}: corrected; original in {backup / image.relative_to(bids)}')
    print('Now compare left/right with another session of this subject before processing.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
