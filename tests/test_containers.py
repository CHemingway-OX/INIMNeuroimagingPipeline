"""Command construction and integration checks using a fake container executable."""
import importlib.util
import json
import os
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
SEG = ('mri/aparc.DKTatlas+aseg.deep.mgz',)
SURF = ('surf/lh.pial.T1', 'surf/rh.pial.T1')
def outputs(base, rels=SEG + SURF):
    for rel in rels:
        touch(base / rel)
def values(flag):
    i = args.index(flag) + 1
    out = []
    while i < len(args) and not args[i].startswith('--'):
        out.append(args[i])
        i += 1
    return out
if '/fastsurfer/run_fastsurfer.sh' in args:
    rels = SEG if '--seg_only' in args else SURF if '--surf_only' in args else SEG + SURF
    outputs(host(args[args.index('--sd')+1]) / args[args.index('--sid')+1], rels)
elif '/pipeline/fastsurfer_long_phase.sh' in args:
    phase = args[args.index('/pipeline/fastsurfer_long_phase.sh') + 1]
    rels = SEG if phase == 'gpu' else SURF
    for sid in values('--tid') + values('--tpids'):
        outputs(binds['/output'] / sid, rels)
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
        self.hdbet_home = self.base / 'hdbet home'
        (self.hdbet_home / 'hd-bet_params').mkdir(parents=True)
        for fold in range(5):
            (self.hdbet_home / 'hd-bet_params' / f'{fold}.model').touch()
        self.env['HDBET_HOME'] = str(self.hdbet_home)
        self.bids = self.base / 'BIDS'
        for ses in ('1', '2'):
            anat = self.bids / 'sub-001' / f'ses-{ses}' / 'anat'
            anat.mkdir(parents=True)
            for contrast in ('T1w', 'FLAIR'):
                (anat / f'sub-001_ses-{ses}_{contrast}.nii.gz').touch()
            fs_mri = self.base / 'fastsurfer' / 'sub-001' / f'ses-{ses}' / f'sub-001_ses-{ses}' / 'mri'
            fs_mri.mkdir(parents=True)
            (fs_mri / 'aparc.DKTatlas+aseg.mapped.mgz').touch()
        self.env['FASTSURFER_OUTPUT_DIR'] = str(self.base / 'fastsurfer')
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

    def test_longitudinal_gpu_then_cpu_stage(self):
        result = self.pipeline('--long', '--stage', 'gpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)  # two conform calls, one GPU phase
        gpu_call = calls[-1]
        self.assertIn('--nv', gpu_call)
        self.assertEqual(gpu_call[gpu_call.index('/pipeline/fastsurfer_long_phase.sh') + 1], 'gpu')
        self.assertIn('--t1s', gpu_call)
        link = self.bids / 'derivatives/fastsurfer_v2.4.2_docker_long/sub-001/ses-1/sub-001_ses-1'
        self.assertFalse(link.exists())

        result = self.pipeline('--long', '--stage', 'gpu')  # segmented: nothing to do
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 3)

        result = self.pipeline('--long', '--stage', 'cpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 4)  # no conforming in the cpu stage
        cpu_call = calls[-1]
        self.assertNotIn('--nv', cpu_call)
        self.assertEqual(cpu_call[cpu_call.index('/pipeline/fastsurfer_long_phase.sh') + 1], 'cpu')
        self.assertNotIn('--t1s', cpu_call)
        self.assertTrue(link.is_symlink())
        self.assertTrue((link / 'surf/lh.pial.T1').is_file())

    def test_cpu_stage_requires_gpu_stage(self):
        result = self.pipeline('--long', '--stage', 'cpu')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('run --stage gpu first', result.stdout + result.stderr)
        self.assertFalse(self.log.exists() and self.calls())

    def test_cross_sectional_gpu_then_cpu_stage(self):
        result = self.pipeline('--stage', 'gpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all('--seg_only' in c and '--nv' in c for c in calls))
        result = self.pipeline('--stage', 'cpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()[2:]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all('--surf_only' in c and '--nv' not in c and '--t2' not in c for c in calls))
        result = self.pipeline('--stage', 'cpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 4)

    def test_unknown_stage_is_rejected(self):
        result = self.pipeline('--stage', 'fast')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown --stage', result.stderr)

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
        call = self.calls()[0]
        self.assertIn(f'{self.hdbet_home}:/hdbet_home:ro', call)
        self.assertEqual(call[call.index('--home') + 1], '/hdbet_home')

    def test_lst_fails_early_without_fastsurfer_segmentation(self):
        (self.base / 'fastsurfer' / 'sub-001' / 'ses-2' / 'sub-001_ses-2' / 'mri'
         / 'aparc.DKTatlas+aseg.mapped.mgz').unlink()
        result = self.run_cli([sys.executable, 'run_scripts/run_lst_docker.py', '-i', str(self.bids),
                               '--subjects', '001'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FastSurfer segmentation required', result.stderr)
        self.assertIn('sub-001_ses-2', result.stderr)
        self.assertIn('--long', result.stderr)
        self.assertFalse(self.log.exists() and self.calls())

    def test_lst_reruns_only_missing_annotation(self):
        import run_lst_docker
        deriv = self.bids / 'derivatives' / 'lst-ai-v1.2.0_docker'
        for ses in ('1', '2'):
            anat = deriv / 'sub-001' / f'ses-{ses}' / 'anat'
            temp = deriv / 'sub-001' / f'ses-{ses}' / 'temp'
            anat.mkdir(parents=True)
            temp.mkdir(parents=True)
            for name in ('space-FLAIR_label-lesion_mask', 'space-FLAIR_desc-annotated_label-lesion_mask'):
                (anat / f'sub-001_ses-{ses}_{name}.nii.gz').touch()
            (temp / f'sub-001_ses-{ses}_space-FLAIR_seg-lst_prob.nii.gz').touch()
        (deriv / 'sub-001' / 'ses-2' / 'temp' / 'sub-001_ses-2_space-flair_desc-annotated_fsseg.nii.gz').touch()
        with patch.dict(os.environ, self.env, clear=True), \
                patch.object(run_lst_docker, 'annotate_lesions_fsseg_variablethresh') as annotate:
            run_lst_docker.process_lst_ai([str(self.bids / 'sub-001')], 0, str(deriv), str(self.bids),
                                          [0.5, 99.5], 0)
        self.assertFalse(self.log.exists() and self.calls())
        annotate.assert_called_once()
        prob, fs_seg, out = annotate.call_args.args
        self.assertTrue(prob.endswith('sub-001_ses-1_space-FLAIR_seg-lst_prob.nii.gz'))
        self.assertTrue(fs_seg.startswith(str(self.base / 'fastsurfer')))
        self.assertTrue(out.endswith('sub-001_ses-1_space-flair_desc-annotated_fsseg.nii.gz'))

    def test_lst_skip_annotation_needs_no_fastsurfer(self):
        (self.base / 'fastsurfer' / 'sub-001' / 'ses-1' / 'sub-001_ses-1' / 'mri'
         / 'aparc.DKTatlas+aseg.mapped.mgz').unlink()
        result = self.run_cli([sys.executable, 'run_scripts/run_lst_docker.py', '-i', str(self.bids),
                               '--subjects', '001', '--skip_annotation'])
        self.assertNotIn('FastSurfer segmentation required', result.stderr)
        self.assertGreaterEqual(len(self.calls()), 1)  # LST-AI container started

    def test_lst_annotation_only_requires_lst_outputs(self):
        result = self.run_cli([sys.executable, 'run_scripts/run_lst_docker.py', '-i', str(self.bids),
                               '--subjects', '001', '--annotation_only'], HDBET_HOME='/nonexistent')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('run the LST-AI segmentation (GPU stage) first', result.stderr)
        self.assertNotIn('HD-BET weights missing', result.stderr)
        self.assertFalse(self.log.exists() and self.calls())

    def test_lst_fails_early_without_hdbet_weights(self):
        (self.hdbet_home / 'hd-bet_params' / '3.model').unlink()
        result = self.run_cli([sys.executable, 'run_scripts/run_lst_docker.py', '-i', str(self.bids),
                               '--subjects', '001'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('HD-BET weights missing', result.stderr)
        self.assertFalse(self.log.exists() and self.calls())

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
