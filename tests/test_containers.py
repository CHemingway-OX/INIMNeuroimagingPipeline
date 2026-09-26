"""Command construction and integration checks using a fake container executable."""
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'run_scripts'))
from utils.container_runtime import build_command, run_container, IMAGES

FAKE = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['FAKE_LOG'], 'a') as out:
    out.write(json.dumps(args) + '\n')
if os.environ.get('FAKE_FAIL'):
    sys.exit(42)
binds = {}
for i, a in enumerate(args):
    if a == '--bind':
        src, dst, *_ = args[i+1].split(':')
        binds[dst] = Path(src)
def host(path):
    for dst in sorted(binds, key=len, reverse=True):
        if path == dst or path.startswith(dst + '/'):
            return binds[dst] / path[len(dst):].lstrip('/')
    raise ValueError(path)
def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
def outputs(base):
    for rel in ('mri/aparc.DKTatlas+aseg.deep.mgz', 'surf/lh.pial.T1', 'surf/rh.pial.T1'):
        touch(base / rel)
if '/fastsurfer/run_fastsurfer.sh' in args:
    outputs(host(args[args.index('--sd')+1]) / args[args.index('--sid')+1])
elif '/fastsurfer/long_fastsurfer.sh' in args:
    i = args.index('--tpids') + 1
    while i < len(args) and not args[i].startswith('--'):
        outputs(binds['/output'] / args[i])
        i += 1
elif '--tablefile' in args:
    host(args[args.index('--tablefile')+1]).write_text('Measure\tLeft-Thalamus\nsubject\t12.5\n')
elif any('mri_convert --conform' in arg for arg in args):
    touch(host(args[-1]))
"""

class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='inim container test ')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        for image, _ in IMAGES.values():
            (self.base / image).touch()
        binary = self.base / 'singularity'
        binary.write_text(FAKE)
        binary.chmod(0o755)
        self.log = self.base / 'commands.jsonl'
        self.env = dict(os.environ, CONTAINER_RUNTIME='singularity', CONTAINER_DIR=str(self.base),
                        PATH=str(self.base) + os.pathsep + os.environ['PATH'], FAKE_LOG=str(self.log),
                        PYTHON_BIN=sys.executable)
        for name in ('FAKE_FAIL', 'SLURM_JOB_ID', 'LIT_REPO', 'FASTSURFER_OUTPUT_DIR',
                     'STRUCT_BIDS_RUN_LOG', 'STRUCT_BIDS_STATUS_FILE', 'STRUCT_BIDS_RUN_ID',
                     'FASTSURFER_SIF', 'LST_SIF', 'LIT_SIF'):
            self.env.pop(name, None)
        self.env['CUDA_VISIBLE_DEVICES'] = '2,3'
        self.bids = self.base / 'BIDS'
        for ses in ('1', '2'):
            anat = self.bids / 'sub-001' / f'ses-{ses}' / 'anat'
            anat.mkdir(parents=True)
            for contrast in ('T1w', 'FLAIR'):
                (anat / f'sub-001_ses-{ses}_{contrast}.nii.gz').touch()
        self.license = self.base / 'license'
        self.license.mkdir()
        (self.license / 'license.txt').touch()

    def run_cli(self, args, **env):
        return subprocess.run(args, cwd=ROOT, env=dict(self.env, **env), text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_singularity_gpu_visibility_and_spaces(self):
        with patch.dict(os.environ, self.env, clear=True):
            cmd, env = build_command('fastsurfer', ['/script', '/input/a b.nii.gz'],
                                     [f'{self.bids}:/data:ro'], gpu=True)
        self.assertIn('--nv', cmd)
        self.assertIn('--cleanenv', cmd)
        self.assertIn(f'{self.bids}:/data:ro', cmd)
        self.assertEqual(cmd[-1], '/input/a b.nii.gz')
        self.assertEqual(env['SINGULARITYENV_CUDA_VISIBLE_DEVICES'], '2,3')

    def test_cpu_does_not_enable_gpu(self):
        with patch.dict(os.environ, self.env, clear=True):
            cmd, env = build_command('fastsurfer', ['true'])
        self.assertNotIn('--nv', cmd)
        self.assertNotIn('SINGULARITYENV_CUDA_VISIBLE_DEVICES', env)

    def test_apptainer_environment(self):
        with patch.dict(os.environ, dict(self.env, CONTAINER_RUNTIME='apptainer'), clear=True):
            cmd, env = build_command('lst', ['lst'], gpu=True)
        self.assertEqual(cmd[0], 'apptainer')
        self.assertEqual(env['APPTAINERENV_CUDA_VISIBLE_DEVICES'], '2,3')

    def test_missing_sif_fails_before_launch(self):
        with patch.dict(os.environ, dict(self.env, FASTSURFER_SIF='/does/not/exist.sif'), clear=True):
            with self.assertRaises(FileNotFoundError):
                build_command('fastsurfer', ['true'])

    def test_container_error_propagates(self):
        with patch.dict(os.environ, dict(self.env, FAKE_FAIL='1'), clear=True):
            with self.assertRaises(subprocess.CalledProcessError) as exc:
                run_container('lst', ['lst'])
        self.assertEqual(exc.exception.returncode, 42)

    def test_docker_has_no_persistent_names(self):
        with patch.dict(os.environ, dict(self.env, CONTAINER_RUNTIME='docker'), clear=True):
            cmd, _ = build_command('fastsurfer', ['/script', '--cpu'])
        self.assertIn('--rm', cmd)
        self.assertNotIn('--name', cmd)
        self.assertNotIn('--gpus', cmd)
        self.assertIn('--entrypoint', cmd)

    def pipeline(self, *extra, **env):
        return self.run_cli(['bash', 'Pipeline/struct_bids.sh', '--bids-dir', str(self.bids),
                             '--fs-license-dir', str(self.license), '--foreground', '--subjects', '001',
                             '--skip-lst', '--skip-lit', '--skip-qc', '--skip-summary', *extra], **env)

    def test_cross_sectional_pipeline_and_resume(self):
        result = self.pipeline('--cpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 2)
        result = self.pipeline('--cpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_longitudinal_pipeline_and_links(self):
        result = self.pipeline('--long')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertNotIn('--nv', calls[0])  # Conforming is CPU-only.
        self.assertIn('--nv', calls[-1])
        link = self.bids / 'derivatives/fastsurfer_v2.4.2_docker_long/sub-001/ses-1/sub-001_ses-1'
        self.assertTrue(link.is_symlink())
        self.assertTrue((link / 'surf/lh.pial.T1').is_file())

    def test_conform_exports_mounted_license_inside_container(self):
        result = self.pipeline('--long')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = self.calls()[0]
        bootstrap = command[command.index('-lc') + 1]
        # Execute the actual shell bootstrap against a stub that requires the
        # license environment, rather than assuming a bind mount is sufficient.
        converter = self.base / 'mri_convert'
        converter.write_text('#!/bin/bash\n[[ "$FS_LICENSE" == /fs_license/license.txt ]] || exit 14\n')
        converter.chmod(0o755)
        env = dict(self.env, FS_LICENSE='/wrong/host/license.txt')
        result = subprocess.run(['bash', '-lc',
            'export PATH=' + shlex.quote(str(self.base)) + ':"$PATH"; ' + bootstrap,
            '_', '/input/t1.nii.gz', '/output/t1.nii.gz'],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_longitudinal_pipeline_has_failure_status(self):
        result = self.pipeline('--long', FAKE_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        status = next((self.bids / 'derivatives/structural_pipeline/logs').glob('*.status'))
        self.assertTrue(status.read_text().startswith('FAILED'))

    def test_lst_worker_failure_reaches_parent(self):
        result = self.run_cli([sys.executable, 'run_scripts/run_lst_docker.py', '-i', str(self.bids),
                               '--subjects', '001', '-n', '2'], FAKE_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('42', result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.calls()[0][-self.calls()[0][::-1].index('lst')-1], 'lst')

    def test_lit_uses_packaged_code_without_host_checkout(self):
        from run_lesioninpainting import run_lit_container
        out = self.base / 'lit output'
        out.mkdir()
        with patch.dict(os.environ, self.env, clear=True):
            run_lit_container(str(self.base / 't1.nii.gz'), str(self.base / 'mask.nii.gz'), str(out), 1)
        call = self.calls()[0]
        self.assertIn('/inpainting/run_lit.sh', call)
        self.assertIn('--nv', call)
        self.assertFalse(any(arg.endswith(':/inpainting:ro') for arg in call))

    def test_metrics_helper_runs_without_gpu(self):
        stats = self.base / 'aseg.stats'
        stats.touch()
        table = self.base / 'volumes.tsv'
        result = self.run_cli(['bash', 'scripts/asegstats2table', '--inputs', str(stats),
                               '--tablefile', str(table), '--meas', 'volume', '--delimiter', 'tab'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('12.5', table.read_text())
        self.assertNotIn('--nv', self.calls()[0])

if __name__ == '__main__':
    unittest.main()
