"""Workstation driver: transfer, verification and cleanup rules, with ssh faked locally."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hpc'))
import cohort_driver as driver


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        fake_ssh = self.base / 'ssh'
        # Ignores the host and runs the remote command locally. COPYFILE_DISABLE stops
        # macOS tar from adding ._ metadata files (the HPC uses GNU tar).
        fake_ssh.write_text('#!/bin/bash\nexport COPYFILE_DISABLE=1\nexec bash -c "${@: -1}"\n')
        fake_ssh.chmod(0o755)
        bindir = self.base / 'bin'
        bindir.mkdir()
        (bindir / 'python3').symlink_to(sys.executable)
        self.path = os.environ['PATH']
        os.environ['PATH'] = f'{bindir}{os.pathsep}{self.path}'
        self.addCleanup(os.environ.__setitem__, 'PATH', self.path)
        self.hpc = driver.Hpc('host', str(fake_ssh), str(ROOT))
        self.bids = self.base / 'bids'
        for ses in ('ses-1', 'ses-2'):
            anat = self.bids / 'sub-001' / ses / 'anat'
            anat.mkdir(parents=True)
            for name in ('T1w.nii.gz', 'T1w.json', 'FLAIR.nii.gz', 'FLAIR.json', 'T2w.nii.gz'):
                (anat / f'sub-001_{ses}_{name}').write_text(name + ses)
            (self.bids / 'sub-001' / ses / 'dwi').mkdir()
            (self.bids / 'sub-001' / ses / 'dwi' / 'big_dwi.nii.gz').write_text('dwi')
        (self.bids / 'sub-001' / 'sub-001_sessions.tsv').write_text('session_id\tacq_time\n')

    def derivatives(self, root):
        long = root / 'derivatives/fastsurfer_v2.4.2_docker_long'
        tp = long / 'longitudinal_work/sub-001/sub-001_ses-1/mri'
        tp.mkdir(parents=True)
        (tp / 'aparc.DKTatlas+aseg.mapped.mgz').write_text('seg')
        (long / 'sub-001/ses-1').mkdir(parents=True)
        os.symlink('../../longitudinal_work/sub-001/sub-001_ses-1', long / 'sub-001/ses-1/sub-001_ses-1')
        pair = root / 'derivatives/structural_pipeline/sel_deformation/sub-001/pairs/ses-1_to_ses-2'
        pair.mkdir(parents=True)
        for name in ('syn_1Warp.nii.gz', 'syn_1InverseWarp.nii.gz', 'jacobian.nii.gz'):
            (pair / name).write_text(name)
        return long

    def test_upload_selects_anatomy_and_sessions_only(self):
        rels = driver.upload_files(self.bids, 'sub-001')
        self.assertIn('sub-001/sub-001_sessions.tsv', rels)
        self.assertIn('sub-001/ses-2/anat/sub-001_ses-2_FLAIR.json', rels)
        self.assertEqual(len(rels), 9)
        self.assertFalse(any('dwi' in r or 'T2w' in r for r in rels))

    def test_manifest_skips_deformation_fields_and_keeps_links(self):
        self.derivatives(self.bids)
        rels = [p.format(sub='sub-001') for p in driver.DERIVATIVES]
        found = driver.manifest(self.bids, rels, driver.EXCLUDE)
        self.assertFalse(any('Warp' in name for name in found))
        self.assertTrue(any(name.endswith('jacobian.nii.gz') for name in found))
        link = 'derivatives/fastsurfer_v2.4.2_docker_long/sub-001/ses-1/sub-001_ses-1'
        self.assertEqual(found[link], {'link': '../../longitudinal_work/sub-001/sub-001_ses-1'})

    def test_roundtrip_upload_download_verified(self):
        remote = self.base / 'remote'
        rels = driver.upload_files(self.bids, 'sub-001')
        driver.send(self.hpc, self.bids, rels, remote)
        self.assertEqual(self.hpc.helper(op='manifest', root=str(remote), rels=rels),
                         driver.manifest(self.bids, rels))
        self.derivatives(remote)
        wanted = [p.format(sub='sub-001') for p in driver.DERIVATIVES]
        listed = self.hpc.helper(op='manifest', root=str(remote), rels=wanted, exclude=driver.EXCLUDE)
        local = self.base / 'local'
        driver.receive(self.hpc, remote, sorted(listed), local)
        self.assertEqual(driver.manifest(local, sorted(listed)), listed)
        link = local / 'derivatives/fastsurfer_v2.4.2_docker_long/sub-001/ses-1/sub-001_ses-1'
        self.assertTrue((link / 'mri/aparc.DKTatlas+aseg.mapped.mgz').is_file())  # works off the HPC
        self.assertFalse(list(local.rglob('*Warp*')))

    def test_remove_only_touches_the_subject(self):
        long = self.derivatives(self.bids)
        with self.assertRaisesRegex(ValueError, 'Refusing'):
            driver.remove(self.bids, ['derivatives'], 'sub-001')
        with self.assertRaisesRegex(ValueError, 'Refusing'):
            driver.remove(self.bids, ['sub-001/../other'], 'sub-001')
        self.hpc.helper(op='remove', root=str(self.bids), subject='sub-001',
                        rels=['sub-001'] + [p.format(sub='sub-001') for p in driver.DERIVATIVES])
        self.assertFalse((self.bids / 'sub-001').exists())
        self.assertFalse((long / 'sub-001').exists())
        self.assertTrue((long / 'longitudinal_work').is_dir())

    def test_staging_must_be_dedicated(self):
        with self.assertRaisesRegex(driver.RemoteError, 'dedicated staging'):
            self.hpc.helper(op='prepare_staging', root=str(self.bids))
        staging = self.base / 'staging'
        self.hpc.helper(op='prepare_staging', root=str(staging))
        (staging / 'sub-001').mkdir()
        self.hpc.helper(op='prepare_staging', root=str(staging))  # marked: reuse is fine

    def test_evaluate_job_chain(self):
        entry = {'gpu_job': '1', 'cpu_job': '2', 'report_job': '3'}
        self.assertEqual(driver.evaluate(entry, {'1': 'COMPLETED', '2': 'RUNNING', '3': 'PENDING'}), 'active')
        self.assertEqual(driver.evaluate(entry, {'1': 'COMPLETED', '2': 'COMPLETED', '3': 'COMPLETED'}), 'fetch')
        self.assertEqual(driver.evaluate(entry, {'1': 'FAILED', '2': 'CANCELLED', '3': 'CANCELLED'}),
                         'failed: GPU job FAILED, CPU job CANCELLED, report job CANCELLED')
        self.assertEqual(driver.evaluate(entry, {}), 'active')  # sacct lags right after submission
        self.assertEqual(driver.evaluate(dict(entry, report_job=None), {'2': 'COMPLETED'}), 'active')

    def test_merge_metrics_unions_columns(self):
        metrics = self.base / 'out/derivatives/structural_pipeline/metrics'
        for sub, text in (('sub-001', 'a,b\n1,2\n'), ('sub-002', 'a,c\n3,4\n')):
            (metrics / sub).mkdir(parents=True)
            (metrics / sub / 'structural_metrics_summary.csv').write_text(text)
        merged = driver.merge_metrics(self.base / 'out').read_text().splitlines()
        self.assertEqual(merged, ['a,b,c', '1,2,', '3,,4'])

    def test_arguments(self):
        common = ['--local-bids', str(self.bids), '--local-out', str(self.base / 'out'),
                  '--remote-bids', '/staging/x', '--subjects', '1']
        self.assertIsNone(driver.parse_args(common).pipeline_args)
        self.assertEqual(driver.parse_args(common + ['--', '--long']).pipeline_args, ['--long'])
        for bad in (['--', '--stage', 'gpu'], ['--local-out', str(self.bids)]):
            with self.assertRaises(SystemExit):
                driver.parse_args(common + bad)


if __name__ == '__main__':
    unittest.main()
