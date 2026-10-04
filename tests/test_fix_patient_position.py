"""Header correction for head-first scans registered as feet first, checked with nibabel."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hpc'))
import fix_patient_position as tool

ROTATION = np.diag([-1., 1., -1., 1.])  # 180 degrees about the anterior-posterior axis


def oblique_affine(qfac_negative=False):
    angle = np.deg2rad(17)
    rot = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    zooms = np.diag([1.0, 0.94, 0.94 * (-1 if qfac_negative else 1)])
    affine = np.eye(4)
    affine[:3, :3] = rot @ zooms
    affine[:3, 3] = [-85.9, -104.4, -70.3]
    return affine


class FixPatientPositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bids = Path(self.tmp.name) / 'bids'
        self.anat = self.bids / 'sub-072' / 'ses-2' / 'anat'
        self.anat.mkdir(parents=True)
        self.data = np.arange(6 * 7 * 8, dtype=np.int16).reshape(6, 7, 8)
        self.affines = {}
        for name, position, qfac, ext in (('T1w', 'FFS', False, '.nii.gz'), ('FLAIR', 'FFS', True, '.nii'),
                                          ('T2w', 'HFS', False, '.nii.gz')):
            affine = oblique_affine(qfac)
            img = nib.Nifti1Image(self.data, affine)
            img.set_qform(affine, code=1)
            img.set_sform(affine, code=1)
            nib.save(img, self.anat / f'sub-072_ses-2_{name}{ext}')
            (self.anat / f'sub-072_ses-2_{name}.json').write_text(json.dumps({'PatientPosition': position}))
            self.affines[name] = (img.get_qform(), img.get_sform(), ext)

    def run_tool(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = tool.main(['--bids', str(self.bids), '--session', 'sub-072_ses-2', *extra])
        return code, out.getvalue()

    def test_dry_run_reports_and_changes_nothing(self):
        before = {p.name: p.read_bytes() for p in self.anat.iterdir()}
        code, out = self.run_tool()
        self.assertEqual(code, 0)
        self.assertIn('sub-072_ses-2_T1w.nii.gz: PatientPosition FFS', out)
        self.assertIn('sub-072_ses-2_T2w.nii.gz: PatientPosition HFS, left unchanged', out)
        self.assertEqual({p.name: p.read_bytes() for p in self.anat.iterdir()}, before)

    def test_write_rotates_qform_and_sform_and_keeps_data(self):
        self.run_tool('--write')
        for name in ('T1w', 'FLAIR'):
            qform, sform, ext = self.affines[name]
            img = nib.load(self.anat / f'sub-072_ses-2_{name}{ext}')
            np.testing.assert_allclose(img.get_qform(), ROTATION @ qform, atol=1e-4)
            np.testing.assert_allclose(img.get_sform(), ROTATION @ sform, atol=1e-4)
            np.testing.assert_array_equal(np.asanyarray(img.dataobj), self.data)
            if ext == '.nii.gz':  # still a gzip file, as its name says
                self.assertEqual((self.anat / f'sub-072_ses-2_{name}{ext}').read_bytes()[:2], b'\x1f\x8b')
            meta = json.loads((self.anat / f'sub-072_ses-2_{name}.json').read_text())
            self.assertEqual(meta['PatientPosition'], 'HFS')
            self.assertEqual(meta['PatientPositionCorrection']['original'], 'FFS')
        qform, sform, ext = self.affines['T2w']  # HFS: untouched
        np.testing.assert_allclose(nib.load(self.anat / 'sub-072_ses-2_T2w.nii.gz').get_sform(), sform)
        backups = list(Path(self.tmp.name).glob('bids_patient_position_backup/*/sub-072/ses-2/anat/*'))
        self.assertFalse(list(self.anat.glob('*.tmp*')))
        self.assertEqual(sorted(p.name for p in backups), sorted([
            'sub-072_ses-2_T1w.nii.gz', 'sub-072_ses-2_T1w.json', 'sub-072_ses-2_FLAIR.nii', 'sub-072_ses-2_FLAIR.json']))
        original = nib.load(next(p for p in backups if p.name.endswith('T1w.nii.gz')))
        np.testing.assert_allclose(original.get_sform(), self.affines['T1w'][1])

    def test_second_run_finds_nothing_to_correct(self):
        self.run_tool('--write')
        _, out = self.run_tool('--write')
        self.assertIn('Nothing to correct.', out)

    def test_axcodes_match_nibabel(self):
        for name in ('T1w', 'FLAIR'):
            path = self.anat / f'sub-072_ses-2_{name}{self.affines[name][2]}'
            header = tool.read_header(path)
            self.assertEqual(tool.axcodes(header), ''.join(nib.aff2axcodes(nib.load(path).affine)))
            corrected = ''.join(nib.aff2axcodes(ROTATION @ nib.load(path).affine))
            self.assertEqual(tool.axcodes(tool.corrected_header(header)), corrected)

    def test_quaternion_rotation_matches_matrix(self):
        for qform, _, _ in self.affines.values():
            b, c, d = nib.quaternions.mat2quat(qform[:3, :3] / np.linalg.norm(qform[:3, :3], axis=0))[1:]
            x, y, z = tool.rotate_quaternion(b, c, d)
            expected = ROTATION[:3, :3] @ nib.quaternions.quat2mat([np.sqrt(max(0, 1 - b*b - c*c - d*d)), b, c, d])
            np.testing.assert_allclose(nib.quaternions.quat2mat([np.sqrt(max(0, 1 - x*x - y*y - z*z)), x, y, z]),
                                       expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
