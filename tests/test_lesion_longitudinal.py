"""Synthetic correctness and end-to-end tests; no patient data required."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Pipeline import lesion_longitudinal_common as common
from Pipeline import chronic_lesion_volume_analysis as volume
from Pipeline import sel_deformation_analysis as sel


class MeasurementTests(unittest.TestCase):
    def test_jacobian_units_and_irregular_intervals(self):
        np.testing.assert_allclose(sel.annual_expansion(np.array([1., 1.25, .75]), 2), [0, 12.5, -12.5])
        with self.assertRaises(ValueError): sel.annual_expansion(np.array([0.]), 1)
        with self.assertRaises(ValueError): sel.annual_expansion(np.array([1.]), 0)

    def test_temporal_origin_fit(self):
        score = sel.temporal_score([.5, 2, 3], [5, 20, 30])
        self.assertAlmostEqual(score['slope_percent_per_year'], 10)
        self.assertAlmostEqual(score['constancy_nmse'], 0)
        self.assertGreater(sel.temporal_score([1, 2, 3], [20, 20, 20])['constancy_nmse'], 0)
        self.assertIsNone(sel.temporal_score([2], [20])['constancy_nmse'])

    def test_radial_shells_not_axis_slices(self):
        x = np.indices((25, 25, 25)) - 12
        mask = (x * x).sum(axis=0) < 100
        distance = ndimage.distance_transform_edt(mask, sampling=(1, 1, 2))
        slope, _ = sel.radial_score(mask, distance * 2, (1, 1, 2), 1)
        self.assertAlmostEqual(slope, 2)
        self.assertIsNone(sel.radial_score(np.ones((1, 1, 1), bool), np.ones((1, 1, 1)), (1, 1, 1), 1)[0])

    def test_seed_ids_survive_low_threshold_bridge(self):
        rate = np.zeros((12, 5, 5))
        rate[1:11, 2, 2] = 5
        rate[2, 2, 2] = rate[9, 2, 2] = 15
        labels = sel.candidates(rate > 0, rate, 12.5, 4, 1)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertEqual(np.count_nonzero(labels), 10)
        self.assertEqual(np.count_nonzero(sel.candidates(rate > 0, rate, 12.5, 4, 20)), 0)

    def test_split_merge_and_new_lesions(self):
        a = np.zeros((10, 10, 10), int); b = a.copy()
        a[1:5, 1:5, 1:5] = 1
        b[1:3, 1:5, 1:5] = 1; b[3:5, 1:5, 1:5] = 2
        b[8, 8, 8] = 3
        groups = volume.correspondence(a, b, (1, 1, 1))
        self.assertIn(([1], [1, 2], False), groups)
        self.assertIn(([], [3], False), groups)
        self.assertIn(([1, 2], [1], False), volume.correspondence(b, a, (1, 1, 1)))

    def test_translation_rescue_is_flagged_in_physical_units(self):
        a = np.zeros((10, 5, 5), int); b = a.copy()
        a[2, 2, 2] = 1; b[3, 2, 2] = 2
        self.assertEqual(len(volume.correspondence(a, b, (3, 1, 1), 2)), 2)
        self.assertEqual(volume.correspondence(a, b, (3, 1, 1), 3), [([1], [2], True)])

    def test_native_volume_and_percent(self):
        r = volume.change_metrics(100, 130, 2, 10)
        self.assertEqual(r['volume_class'], 'expanding')
        self.assertEqual(r['change_percent_per_year'], 15)
        self.assertIsNone(volume.change_metrics(0, 10, 1, 10)['change_percent'])
        self.assertEqual(volume.change_metrics(100, 90, 1, 10)['volume_class'], 'stable')

    def test_calibration_does_not_invent_confidence(self):
        t = sel.temporal_score([1, 2], [10, 20])
        self.assertEqual(sel.classify(t, 1., None, False, True)[0], 'candidate_uncalibrated')
        c = dict(concentricity_mean=0, concentricity_sd=1, constancy_mean=0, constancy_sd=1, threshold=0)
        self.assertEqual(sel.classify(t, 1., c, False, False)[0], 'candidate_enhancement_unknown')
        self.assertEqual(sel.classify(t, 1., c, True, True)[0], 'excluded_enhancing')

    def test_centered_affine_square_root(self):
        class Tx:
            parameters = list(np.diag([1.2, 1., .9]).ravel()) + [2, -3, 1]
            fixed_parameters = [80, -50, 30]
        a = common.homogeneous(Tx())
        h = common.half_matrix(a)
        np.testing.assert_allclose(h @ h, a, atol=1e-8)
        c = np.asarray(Tx.fixed_parameters)
        np.testing.assert_allclose((a @ np.r_[c, 1])[:3], c + np.array([2, -3, 1]))

    def test_cache_detects_changed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            def build(root):
                (root / 'out.txt').write_text('first')
                return {'result': str(root / 'out.txt')}
            common.cached(d, build)
            self.assertEqual(common.cached(d, lambda _: self.fail('cache miss'))['result'], str(d / 'out.txt'))
            (d / 'out.txt').write_text('changed')
            with self.assertRaises(ValueError): common.cached(d, build)

    def test_sidecar_dates_do_not_use_acquisition_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(session='ses-1', t1='t1.nii.gz', flair='flair.nii.gz')
            (root / 't1.json').write_text(json.dumps({'AcquisitionTime': '10:40:20'}))
            with self.assertRaisesRegex(ValueError, 'AcquisitionTime'): common.infer_date(row, root)
            (root / 't1.json').write_text(json.dumps({'AcquisitionDateTime': '20200102123456'}))
            common.infer_date(row, root)
            self.assertEqual(row['date'], '2020-01-02')
            (root / 'flair.json').write_text(json.dumps({'StudyDate': '2020-01-03'}))
            with self.assertRaises(ValueError): common.infer_date(row, root)


class AntsGeometryTests(unittest.TestCase):
    def test_analytic_displacement_jacobian_and_pull_direction(self):
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
        import ants
        shape = (24, 24, 24)
        ref = ants.from_numpy(np.zeros(shape, np.float32), spacing=(1, 1.5, 2), origin=(-10, -12, -20))
        scale = np.array([1.05, 1.1, 1.])
        coords = np.moveaxis(np.indices(shape), 0, -1) * np.asarray(ref.spacing) + ref.origin
        field = ants.from_numpy((coords * (scale - 1)).astype(np.float32), spacing=ref.spacing, origin=ref.origin, has_components=True)
        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / 'warp.nii.gz'); ants.image_write(field, p)
            jac = ants.create_jacobian_determinant_image(ref, p, do_log=False, geom=True).numpy()
            np.testing.assert_allclose(jac[3:-3, 3:-3, 3:-3], np.prod(scale), atol=2e-5)
            # A source ramp is sampled at phi(x)=scale*x; this verifies ANTs' pull convention.
            ramp = common.ants_like(ants, coords[..., 0], ref)
            result = ants.apply_transforms(fixed=ref, moving=ramp, transformlist=[p]).numpy()
            np.testing.assert_allclose(result[5:-5, 5:-5, 5:-5], (coords[..., 0] * scale[0])[5:-5, 5:-5, 5:-5], atol=1e-4)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lesion-analysis-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shape = (24, 24, 24)
        grid = np.indices(shape)
        radius = ((grid - 12) ** 2).sum(axis=0)
        brain = radius < 100
        t1 = (50 + 40 * np.exp(-radius / 60) + 5 * np.sin(grid[0])) * brain
        rows = []
        for i, day in enumerate([0, 365.25, 913.125]):
            lesion = np.zeros(shape, np.uint8); lesion[9:14+i, 10:14, 10:14] = 1
            row = dict(subject='sub-test', session=f'ses-{i+1}', days=day)
            for key, data in [('t1', t1), ('flair', t1), ('lesion', lesion), ('brain_mask', brain), ('enhancement_mask', np.zeros(shape))]:
                name = f'{i}_{key}.nii.gz'
                nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), self.root / name)
                row[key] = name
            rows.append(row)
        self.manifest = self.root / 'visits.tsv'
        with self.manifest.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)

    def run_cli(self, name, output, extra=(), check=True):
        command = [sys.executable, '-m', 'Pipeline.' + name, '--manifest', str(self.manifest), '--output', str(output),
                   '--alignment', 'already-aligned', '--threads', '1', *extra]
        result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
        if check and result.returncode:
            self.fail(result.stdout[-2000:] + result.stderr[-3000:])
        return result

    def test_volume_cli_review_resume_and_dates(self):
        output = self.root / 'volume'
        self.run_cli('chronic_lesion_volume_analysis', output)
        with (output / 'lesion_groups.csv').open() as f: rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]['change_mm3']), 16)
        self.assertAlmostEqual(float(rows[1]['change_mm3_per_year']), 16 / 1.5)
        self.assertEqual(rows[0]['decision'], 'pending')
        self.run_cli('chronic_lesion_volume_analysis', output, ['--resume'])
        with (output / 'review_template.csv').open() as f: reviewed = list(csv.DictReader(f))
        for row in reviewed: row.update(decision='include', reason='Synthetic phantom: no acute/confluent activity')
        review = self.root / 'review.csv'; common.write_csv(review, reviewed, list(reviewed[0]))
        out2 = self.root / 'reviewed'
        self.run_cli('chronic_lesion_volume_analysis', out2, ['--review', str(review)])
        summary = json.loads((out2 / 'summary.json').read_text())
        self.assertEqual(summary['intervals'][-1]['cumulative_reviewed_change_mm3'], 32)
        self.assertNotEqual(self.run_cli('chronic_lesion_volume_analysis', output, ['--resume', '--stable-percent', '12'], check=False).returncode, 0)

    def test_sel_cli_identity_has_no_expansion(self):
        output = self.root / 'sel'
        self.run_cli('sel_deformation_analysis', output, ['--iterations', '2', '1', '0'])
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['candidate_count'], 0)
        self.assertTrue(summary['temporal_assessable'])
        self.assertIsNone(summary['calibration'])
        self.run_cli('sel_deformation_analysis', output, ['--iterations', '2', '1', '0', '--resume'])

    def test_manifest_rejects_duplicate_times_and_geometry(self):
        args = argparse.Namespace(manifest=self.manifest, threads=1, threshold=.5, mask_kind='binary')
        rows = common.load_manifest(args)
        self.assertEqual(rows[-1]['years'], 2.5)
        path = self.root / '1_lesion.nii.gz'
        img = nib.load(path); affine = img.affine.copy(); affine[0, 3] = 5
        nib.save(nib.Nifti1Image(img.get_fdata(), affine), path)
        with self.assertRaisesRegex(ValueError, 'FLAIR grid'): common.load_manifest(args)

    def test_normal_registration_path(self):
        output = self.root / 'registered_sel'
        self.run_cli('sel_deformation_analysis', output, ['--alignment', 'register', '--iterations', '2', '1', '0'])
        self.assertTrue(json.loads((output / 'complete.json').read_text())['complete'])
        self.assertEqual(len(list(output.glob('pairs/*/registration_parameters.json'))), 2)

    def test_empty_lesions_and_two_visit_status(self):
        for i in range(3):
            path = self.root / f'{i}_lesion.nii.gz'
            img = nib.load(path)
            nib.save(nib.Nifti1Image(np.zeros(img.shape, np.uint8), img.affine), path)
        output = self.root / 'empty'
        self.run_cli('chronic_lesion_volume_analysis', output)
        with (output / 'lesion_groups.csv').open() as f: self.assertEqual(list(csv.DictReader(f)), [])
        lines = self.manifest.read_text().splitlines()
        self.manifest.write_text('\n'.join(lines[:3]) + '\n')
        result = self.run_cli('sel_deformation_analysis', self.root / 'two', ['--dry-run'])
        self.assertFalse(json.loads(result.stdout)['temporal_assessable'])


class LockTests(unittest.TestCase):
    """Slurm can SIGKILL a run inside ANTs, leaving .running behind."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = argparse.Namespace(output=Path(self.tmp.name) / 'out', resume=False, dry_run=False)
        self.lock = self.args.output / '.running'
        self.env = unittest.mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop('SLURM_JOB_ID', None)

    def leave_lock(self, owner):
        self.lock.mkdir(parents=True)
        common.write_json(self.lock / 'owner.json', owner)

    def enter(self):
        with common.run_directory(self.args, [], 'sel_deformation_analysis') as root:
            self.assertEqual(json.loads((self.lock / 'owner.json').read_text())['pid'], os.getpid())
        self.assertFalse(self.lock.exists())
        self.assertTrue(json.loads((root / 'complete.json').read_text())['complete'])

    def test_lock_of_exited_process_is_removed(self):
        finished = subprocess.run([sys.executable, '-c', 'import os; print(os.getpid())'],
                                  capture_output=True, text=True).stdout.strip()
        self.leave_lock({'host': common.socket.gethostname(), 'pid': int(finished), 'slurm_job_id': None})
        self.enter()

    def test_lock_of_running_process_is_kept(self):
        self.leave_lock({'host': common.socket.gethostname(), 'pid': os.getpid(), 'slurm_job_id': None})
        with self.assertRaisesRegex(ValueError, 'another run or interrupted run'):
            self.enter()
        self.assertTrue(self.lock.exists())

    def test_lock_without_owner_is_kept(self):
        self.lock.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'owner: unknown'):
            self.enter()

    def test_lock_of_ended_slurm_job_is_removed(self):
        self.leave_lock({'host': 'other-node', 'pid': 1, 'slurm_job_id': '123'})
        ended = subprocess.CompletedProcess([], 1, '', 'slurm_load_jobs error: Invalid job id specified')
        with unittest.mock.patch.object(common.subprocess, 'run', return_value=ended):
            self.enter()

    def test_lock_of_queued_slurm_job_is_kept(self):
        self.leave_lock({'host': 'other-node', 'pid': 1, 'slurm_job_id': '123'})
        running = subprocess.CompletedProcess([], 0, 'RUNNING\n', '')
        with unittest.mock.patch.object(common.subprocess, 'run', return_value=running):
            with self.assertRaisesRegex(ValueError, 'another run'):
                self.enter()


if __name__ == '__main__':
    unittest.main()
