#!/usr/bin/env python3
"""Process a BIDS cohort on the HPC in a rolling window, driven from the workstation that holds it.

Per subject: upload T1w/FLAIR images and sub-<ID>_sessions.tsv to a staging BIDS
directory on the HPC, submit the GPU job, the CPU job and a per-subject report,
fetch the derivatives once the report succeeded, verify every file by SHA-256 and
then delete the subject from the HPC. At most --max-on-hpc subjects occupy HPC
storage and at most --max-gpu subjects wait for or use a GPU. The HPC never has
to reach the workstation. Stop and restart at any time: the state is kept in
<local-out>/.cohort_driver/state.json.

The workstation needs Python 3.8+, ssh with a key for the HPC, and this file.

  python3 cohort_driver.py --local-bids /mnt/e/COHORT/BIDS --local-out /mnt/e/COHORT/hpc_results \\
      --remote-bids /data2/core-nrad-liebig/chemingw/INIM_NIPipeline/staging/COHORT --all
"""
import argparse
import csv
from datetime import datetime
import fnmatch
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zlib

# Relative to the staging BIDS root; {sub} is sub-<ID>.
DERIVATIVES = [
    'derivatives/fastsurfer_v2.4.2_docker_long/{sub}',
    'derivatives/fastsurfer_v2.4.2_docker_long/longitudinal_work/{sub}',
    'derivatives/fastsurfer_v2.4.2_docker/{sub}',
    'derivatives/lst-ai-v1.2.0_docker/{sub}',
    'derivatives/FS-LIT2/{sub}',
    'derivatives/structural_pipeline/sel_deformation/{sub}',
    'derivatives/structural_pipeline/sel_deformation/manifests/{sub}_longitudinal.tsv',
    'derivatives/structural_pipeline/metrics/{sub}',
    'derivatives/structural_pipeline/qc/{sub}',
]
LOGS = [
    'derivatives/structural_pipeline/logs/struct_bids_*_{sub}.*',
    'derivatives/structural_pipeline/logs/struct_bids_*_{sub}_cpu.*',
    'derivatives/structural_pipeline/logs/struct_bids_{report}_report.*',
]
# SEL deformation fields are only needed to resume or audit a SEL run on the HPC.
EXCLUDE = ['derivatives/structural_pipeline/sel_deformation/*/pairs/*/syn_*Warp.nii.gz']
UPLOAD_PATTERNS = ['*_T1w.nii', '*_T1w.nii.gz', '*_T1w.json', '*_FLAIR.nii', '*_FLAIR.nii.gz', '*_FLAIR.json']
STAGING_MARKER = '.cohort_driver_staging'
FAILED = {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED',
          'CANCELLED', 'REVOKED'}
FINISHED = FAILED | {'COMPLETED'}


# --- Also executed on the HPC: python3 - --remote '<json>' with this file on stdin. ---

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expand(root, rel):
    if any(c in rel for c in '*?['):
        return sorted(root.glob(rel))
    path = root / rel
    return [path] if os.path.lexists(path) else []


def walk(path):
    """Files and symlinks below path; symlinked directories are listed, not followed."""
    if path.is_symlink() or not path.is_dir():
        yield path
        return
    for dirpath, dirnames, filenames in os.walk(path):
        base = Path(dirpath)
        for name in dirnames:
            if (base / name).is_symlink():
                yield base / name
        for name in filenames:
            yield base / name


def manifest(root, rels, exclude=()):
    """{relative path: {'sha256', 'size'} or {'link'}} for the given paths/globs below root."""
    root = Path(root)
    out = {}
    for rel in rels:
        for path in expand(root, rel):
            for item in walk(path):
                name = item.relative_to(root).as_posix()
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude):
                    continue
                if item.is_symlink():
                    out[name] = {'link': os.readlink(item)}
                else:
                    out[name] = {'sha256': digest(item), 'size': item.stat().st_size}
    return out


def check_subject_path(rel, subject):
    parts = PurePosixPath(rel).parts
    if PurePosixPath(rel).is_absolute() or '..' in parts or not any(subject in p for p in parts):
        raise ValueError(f'Refusing to touch {rel!r}: not a relative path of {subject}')


def remove(root, rels, subject):
    root = Path(root)
    for rel in rels:
        check_subject_path(rel, subject)
        for path in expand(root, rel):
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)


def prepare_staging(root):
    """Create the staging root; refuse a non-empty directory the driver did not create."""
    root = Path(root)
    marker = root / STAGING_MARKER
    if root.exists() and any(root.iterdir()) and not marker.exists():
        raise ValueError(f'{root} is not empty and has no {STAGING_MARKER}: the driver deletes subjects '
                         'there after fetching, so use a dedicated staging directory')
    root.mkdir(parents=True, exist_ok=True)
    marker.touch()


def remote_main(payload):
    op = payload['op']
    try:
        if op == 'manifest':
            result = manifest(payload['root'], payload['rels'], payload.get('exclude', ()))
        elif op == 'remove':
            remove(payload['root'], payload['rels'], payload['subject'])
            result = {}
        elif op == 'prepare_staging':
            prepare_staging(payload['root'])
            result = {}
        else:
            raise ValueError(f'Unknown operation {op}')
    except (OSError, ValueError) as error:
        print(json.dumps({'error': str(error)}))
        return
    print(json.dumps({'result': result}))


# --- Workstation side. ---

class RemoteError(Exception):
    def __init__(self, message, returncode=None):
        super().__init__(message)
        self.returncode = returncode

    @property
    def transient(self):
        return self.returncode == 255  # ssh itself failed (network, VPN)


class Hpc:
    def __init__(self, host, ssh, repo):
        self.host = host
        self.ssh = shlex.split(ssh)
        self.repo = repo

    def command(self, remote_command):
        return [*self.ssh, self.host, remote_command]

    def run(self, remote_command, input=None):
        result = subprocess.run(self.command(remote_command), input=input, capture_output=True, text=True)
        if result.returncode:
            raise RemoteError(f'{remote_command[:120]}: exit {result.returncode}: '
                              f'{(result.stderr or result.stdout).strip()[-800:]}', result.returncode)
        return result.stdout

    def helper(self, **payload):
        out = self.run('python3 - --remote ' + shlex.quote(json.dumps(payload)), input=Path(__file__).read_text())
        answer = json.loads(out.strip().splitlines()[-1])
        if 'error' in answer:
            raise RemoteError(answer['error'])
        return answer['result']

    def in_repo(self, command):
        repo = self.repo
        cd = 'cd ~/' + shlex.quote(repo[2:]) if repo.startswith('~/') else 'cd ' + shlex.quote(repo)
        return self.run(f'{cd} && {command}')


def send(hpc, local_root, rels, remote_root):
    """Stream local files to remote_root as a tar archive."""
    remote_root = shlex.quote(str(remote_root))
    proc = subprocess.Popen(hpc.command(f'mkdir -p {remote_root} && tar -C {remote_root} -xf -'),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for rel in rels:
                tar.add(Path(local_root) / rel, arcname=rel, recursive=False)
    finally:
        proc.stdin.close()
    err = proc.stderr.read().decode(errors='replace')
    proc.stderr.close()
    if proc.wait():
        raise RemoteError(f'upload failed: {err.strip()[-800:]}', proc.returncode)


def receive(hpc, remote_root, rels, local_root):
    """Stream the listed remote files and symlinks into local_root."""
    wanted = set(rels)
    proc = subprocess.Popen(hpc.command(f'tar -C {shlex.quote(str(remote_root))} --null -T - -cf -'),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Feed the name list from a thread: tar starts writing before it has read all names.
    feeder = threading.Thread(target=lambda: (proc.stdin.write(b''.join(r.encode() + b'\0' for r in rels)),
                                              proc.stdin.close()))
    feeder.start()
    local_root = Path(local_root)
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            for member in tar:
                if member.isdir():
                    continue
                if member.name not in wanted:
                    raise RemoteError(f'unexpected archive member {member.name!r}')
                target = local_root / member.name
                if os.path.lexists(target) and (member.issym() or target.is_symlink()):
                    target.unlink()
                if hasattr(tarfile, 'data_filter'):
                    tar.extract(member, local_root, filter='data')
                else:
                    tar.extract(member, local_root)
    finally:
        feeder.join()
        proc.stdout.read()  # drain the end-of-archive padding so tar does not hit a closed pipe
        proc.stdout.close()
        err = proc.stderr.read().decode(errors='replace')
        proc.stderr.close()
        returncode = proc.wait()
    if returncode:
        raise RemoteError(f'download failed: {err.strip()[-800:]}', returncode)


def subject_label(value):
    value = value.strip()
    return value if value.startswith('sub-') else 'sub-' + value


def resolve_subjects(values, known):
    """Map IDs to known subject labels: exact label first, else by number (1, sub-1, 001 -> sub-001)."""
    by_number = {}
    for label in known:
        match = re.fullmatch(r'sub-0*(\d+)', label)
        if match:
            by_number.setdefault(int(match.group(1)), []).append(label)
    resolved, missing = [], []
    for value in values:
        label = subject_label(value)
        match = re.fullmatch(r'sub-0*(\d+)', label)
        # A bare number is matched by value only; 12 is ambiguous with sub-12 and sub-012.
        explicit = label in known and (value.strip().startswith('sub-') or not match)
        candidates = [label] if explicit else by_number.get(int(match.group(1)), []) if match else []
        if len(candidates) == 1:
            resolved.append(candidates[0])
        else:
            missing.append(value.strip() + (' (ambiguous: ' + ', '.join(candidates) + ')' if candidates else ''))
    return list(dict.fromkeys(resolved)), missing


def upload_files(local_bids, sub, excluded=()):
    """T1w/FLAIR images with sidecars and the sessions file, relative to the BIDS root.

    Sessions in excluded (e.g. ['ses-2']) are left out entirely.
    """
    local_bids = Path(local_bids)
    rels = []
    sessions = local_bids / sub / f'{sub}_sessions.tsv'
    if sessions.is_file():
        rels.append(sessions.relative_to(local_bids).as_posix())
    for anat in sorted((local_bids / sub).glob('ses-*/anat')):
        if anat.parent.name in excluded:
            continue
        for pattern in UPLOAD_PATTERNS:
            rels += [p.relative_to(local_bids).as_posix() for p in sorted(anat.glob(pattern))]
    return rels


def broken_gzip(path):
    """Error text if a .gz file does not decompress completely (e.g. a truncated copy), else None."""
    try:
        with gzip.open(path, 'rb') as stream:
            while stream.read(1 << 22):
                pass
    except (OSError, EOFError, zlib.error) as error:
        return str(error)
    return None


def parse_session_list(value, known_subjects):
    """{'sub-072': ['ses-2']} from 'sub-072_ses-2,72_ses-3'; subjects are matched by number."""
    excluded, problems = {}, []
    for item in [i.strip() for i in value.split(',') if i.strip()]:
        match = re.fullmatch(r'(.+?)[_/](ses-[A-Za-z0-9]+)', item)
        if not match:
            problems.append(f'{item} (expected SUBJECT_ses-N, e.g. sub-072_ses-2)')
            continue
        subjects, missing = resolve_subjects([match.group(1)], known_subjects)
        if missing:
            problems.append(f'{item} (unknown subject)')
            continue
        excluded.setdefault(subjects[0], set()).add(match.group(2))
    if problems:
        raise SystemExit('--exclude-sessions/--include-sessions: ' + ', '.join(problems))
    return excluded


def job_states(hpc, job_ids):
    """Slurm state per job id; array tasks are reported under their array id."""
    if not job_ids:
        return {}
    out = hpc.run('sacct -X -n -P --format=JobID,State -j ' + ','.join(sorted(job_ids)))
    states = {}
    for line in out.splitlines():
        if '|' not in line:
            continue
        job, state = line.split('|', 1)
        states[job.split('_')[0]] = (state.split() or ['UNKNOWN'])[0]
    return states


def evaluate(entry, states):
    """'fetch', 'failed: ...' or 'active' for a submitted subject."""
    jobs = [('GPU', entry.get('gpu_job')), ('CPU', entry.get('cpu_job')), ('report', entry.get('report_job'))]
    found = {name: states.get(job, 'UNKNOWN') if job else 'MISSING' for name, job in jobs}
    bad = [f'{name} job {found[name]}' for name, _ in jobs if found[name] in FAILED]
    if bad:
        return 'failed: ' + ', '.join(bad)
    if found['report'] == 'COMPLETED':
        return 'fetch'
    return 'active'


def merge_metrics(local_out):
    """Combine the per-subject metrics CSVs into one cohort CSV."""
    metrics = Path(local_out) / 'derivatives' / 'structural_pipeline' / 'metrics'
    rows, fields = [], []
    for path in sorted(metrics.glob('sub-*/structural_metrics_summary.csv')):
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields += [f for f in reader.fieldnames or [] if f not in fields]
            rows += list(reader)
    if not rows:
        return None
    target = metrics / 'cohort_structural_metrics.csv'
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


class Driver:
    def __init__(self, args):
        self.args = args
        self.hpc = Hpc(args.host, args.ssh, args.remote_repo)
        self.local_bids = args.local_bids.resolve()
        self.local_out = args.local_out.resolve()
        self.remote_bids = args.remote_bids.rstrip('/')
        self.state_dir = self.local_out / '.cohort_driver'
        self.state_file = self.state_dir / 'state.json'
        self.log_file = self.state_dir / 'driver.log'
        self.pipeline_args = args.pipeline_args or ['--long', '--SEL']
        self.state = {'subjects': {}}

    def log(self, message):
        line = f'{datetime.now().isoformat(timespec="seconds")} {message}'
        print(line, flush=True)
        with self.log_file.open('a') as stream:
            stream.write(line + '\n')

    def save(self):
        tmp = self.state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=2) + '\n')
        tmp.replace(self.state_file)

    def load(self, subjects):
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
            if self.state.get('remote_bids') != self.remote_bids:
                raise SystemExit(f'State in {self.state_file} belongs to --remote-bids '
                                 f'{self.state.get("remote_bids")}; use that or a new --local-out')
        self.state['remote_bids'] = self.remote_bids
        for entry in self.state['subjects'].values():
            if (entry.get('status') == 'failed' and entry.get('remote_data') is False
                    and any(m in (entry.get('reason') or '') for m in ('no T1w images', 'sessions.tsv missing'))):
                entry['status'] = 'skipped'
        for sub in subjects:
            self.state['subjects'].setdefault(sub, {'status': 'pending'})

    def entries(self, *statuses):
        return [(s, e) for s, e in self.state['subjects'].items() if e['status'] in statuses]

    # --- one subject ---

    def start(self, sub, entry):
        # Data problems on the workstation are 'skipped': they do not count towards --max-failed.
        if '--SEL' in self.pipeline_args and not (self.local_bids / sub / f'{sub}_sessions.tsv').is_file():
            self.skip(sub, entry, f'{sub}_sessions.tsv missing (scan dates for SEL)')
            return
        rels = upload_files(self.local_bids, sub, self.state.get('excluded_sessions', {}).get(sub, ()))
        if not any('_T1w.' in r for r in rels):
            self.skip(sub, entry, 'no T1w images (ses-*/anat/*_T1w.nii[.gz])')
            return
        # A truncated image is read partially by some tools without an error; never process it.
        for rel in rels:
            error = broken_gzip(self.local_bids / rel) if rel.endswith('.gz') else None
            if error:
                self.skip(sub, entry, f'corrupt file {rel}: {error}')
                return
        self.log(f'{sub}: uploading {len(rels)} files')
        entry['remote_data'] = True
        send(self.hpc, self.local_bids, rels, self.remote_bids)
        local = manifest(self.local_bids, rels)
        remote = self.hpc.helper(op='manifest', root=self.remote_bids, rels=rels)
        if local != remote:
            raise RemoteError(f'{sub}: upload verification failed')
        env = 'BIDS_DIR=' + shlex.quote(self.remote_bids)
        out = self.hpc.in_repo(
            f'{env} hpc/submit_cohort.sh --subjects {sub} --no-report --gpu-time {self.args.gpu_time} '
            f'--cpu-time {self.args.cpu_time} -- ' + ' '.join(map(shlex.quote, self.pipeline_args)))
        ids = {}
        for line in out.splitlines():
            for key, prefix in (('gpu_job', 'GPU array: '), ('cpu_job', 'CPU array: ')):
                if line.startswith(prefix):
                    ids[key] = line[len(prefix):].split()[0]
        if set(ids) != {'gpu_job', 'cpu_job'}:
            raise RemoteError(f'{sub}: could not read job ids from submit_cohort.sh:\n{out}')
        entry.update(status='submitted', submitted=datetime.now().isoformat(timespec='seconds'),
                     report_job=None, **ids)
        try:
            self.submit_report(sub, entry, after_cpu=True)
        except RemoteError as error:  # retried at the next poll; GPU and CPU jobs are queued
            self.log(f'{sub}: report job not submitted yet: {error}')
        self.log(f'{sub}: submitted GPU {entry["gpu_job"]}, CPU {entry["cpu_job"]}, report {entry["report_job"]}')

    def skip(self, sub, entry, reason):
        entry.update(status='skipped', remote_data=False, reason=reason)
        self.log(f'{sub}: SKIPPED before upload: {reason}; fix the data, then restart with --retry-failed')

    def submit_report(self, sub, entry, after_cpu):
        """Per-subject metrics and QC into their own directories (no shared CSV/index)."""
        base = f'{self.remote_bids}/derivatives/structural_pipeline'
        report_args = ['--subjects', sub, '--metrics-dir', f'{base}/metrics/{sub}', '--qc-dir', f'{base}/qc/{sub}']
        if '--long' in self.pipeline_args:
            report_args.append('--long')
        dependency = f'--kill-on-invalid-dep=yes --dependency=afterok:{entry["cpu_job"]} ' if after_cpu else ''
        entry['report_job'] = self.hpc.in_repo(
            f'BIDS_DIR={shlex.quote(self.remote_bids)} sbatch --parsable {dependency}hpc/report.sbatch '
            + ' '.join(map(shlex.quote, report_args))).strip().split(';')[0]

    def fetch(self, sub, entry):
        rels = [p.format(sub=sub) for p in DERIVATIVES]
        rels += [p.format(sub=sub, report=entry['report_job']) for p in LOGS]
        remote = self.hpc.helper(op='manifest', root=self.remote_bids, rels=rels, exclude=EXCLUDE)
        self.log(f'{sub}: fetching {len(remote)} files ({sum(v.get("size", 0) for v in remote.values()) / 1e9:.2f} GB)')
        receive(self.hpc, self.remote_bids, sorted(remote), self.local_out)
        local = manifest(self.local_out, sorted(remote))
        if local != remote:
            differing = sorted(k for k in remote if local.get(k) != remote[k])[:5]
            raise RemoteError(f'{sub}: download verification failed, e.g. {differing}')
        slurm_logs = [f'logs/inim-struct-{entry["gpu_job"]}_1.out', f'logs/inim-struct-cpu-{entry["cpu_job"]}_1.out',
                      f'logs/inim-report-{entry["report_job"]}.out']
        repo_root = self.hpc.in_repo('pwd').strip()
        found = self.hpc.helper(op='manifest', root=repo_root, rels=slurm_logs)
        if found:
            receive(self.hpc, repo_root, sorted(found), self.local_out / '.cohort_driver' / 'slurm')
        # Only now, with every file verified locally, remove the subject from the HPC.
        self.hpc.helper(op='remove', root=self.remote_bids, subject=sub,
                        rels=[sub] + [p.format(sub=sub) for p in DERIVATIVES])
        entry.update(status='done', remote_data=False, fetched=datetime.now().isoformat(timespec='seconds'))
        self.log(f'{sub}: done, verified locally and removed from the HPC')

    # --- scheduling ---

    def poll(self):
        submitted = self.entries('submitted')
        ids = {e[k] for _, e in submitted for k in ('gpu_job', 'cpu_job', 'report_job') if e.get(k)}
        states = job_states(self.hpc, ids)
        for sub, entry in submitted:
            try:
                cpu_state = states.get(entry['cpu_job'], 'UNKNOWN')
                if entry.get('report_job') is None and cpu_state not in FAILED:
                    # A finished CPU job may already be purged from Slurm: no dependency then.
                    self.submit_report(sub, entry, after_cpu=cpu_state != 'COMPLETED')
                    self.log(f'{sub}: submitted report {entry["report_job"]}')
                decision = evaluate(entry, states)
                if decision == 'fetch':
                    self.fetch(sub, entry)
                elif decision.startswith('failed'):
                    entry.update(status='failed', reason=decision[len('failed: '):])
                    self.log(f'{sub}: FAILED ({entry["reason"]}); data stays on the HPC for inspection')
                else:
                    entry['gpu_state'] = states.get(entry['gpu_job'], 'UNKNOWN')
            except RemoteError as error:
                if error.transient:
                    raise
                entry.update(status='failed', reason=str(error))
                self.log(f'{sub}: FAILED: {error}; data stays on the HPC for inspection')
            finally:
                self.save()
        failed = len(self.entries('failed'))
        if failed >= self.args.max_failed:
            return False
        for sub, entry in self.entries('pending'):
            active = self.entries('submitted')
            gpu_busy = sum(1 for _, e in active if e.get('gpu_state', 'UNKNOWN') not in FINISHED)
            on_hpc = len(active) + sum(1 for _, e in self.entries('failed') if e.get('remote_data'))
            if gpu_busy >= self.args.max_gpu or on_hpc >= self.args.max_on_hpc:
                break
            try:
                self.start(sub, entry)
            except RemoteError as error:
                if error.transient:
                    raise
                entry.update(status='failed', reason=str(error))
                self.log(f'{sub}: FAILED to start: {error}')
            if entry['status'] == 'submitted':
                entry['gpu_state'] = 'PENDING'
            self.save()
        return True

    def reset(self, subjects):
        """Queue finished or failed subjects again; results are moved to a backup, not deleted."""
        unknown = [s for s in subjects if s not in self.state['subjects']]
        active = [s for s in subjects if self.state['subjects'].get(s, {}).get('status') == 'submitted']
        if unknown:
            raise SystemExit(f'--reset: not in the selected subjects: {", ".join(unknown)}')
        if active:
            raise SystemExit(f'--reset: jobs are still active for {", ".join(active)}; '
                             'cancel them with scancel first, then run the driver once to record the failure')
        backup = self.state_dir / 'reset_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        for sub in subjects:
            entry = self.state['subjects'][sub]
            rels = [p.format(sub=sub) for p in DERIVATIVES]
            # Leftovers on the HPC (failed runs) would clash with the new provenance, e.g. changed dates.
            self.hpc.helper(op='remove', root=self.remote_bids, subject=sub, rels=[sub] + rels)
            moved = 0
            for rel in rels:
                source = self.local_out / rel
                if os.path.lexists(source):
                    target = backup / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
                    moved += 1
            previous = {k: entry.get(k) for k in ('status', 'reason', 'gpu_job', 'cpu_job', 'report_job') if entry.get(k)}
            self.state['subjects'][sub] = {'status': 'pending', 'reset': datetime.now().isoformat(timespec='seconds'),
                                           'previous': previous}
            self.save()
            self.log(f'{sub}: reset (was {previous.get("status")}); '
                     + (f'{moved} result path(s) moved to {backup}' if moved else 'no local results'))

    def set_excluded_sessions(self, exclude, include):
        """Persist session exclusions; they apply to the next upload of each subject."""
        known = list(self.state['subjects'])
        current = {s: set(v) for s, v in self.state.get('excluded_sessions', {}).items()}
        for sub, sessions in parse_session_list(exclude or '', known).items():
            current.setdefault(sub, set()).update(sessions)
        for sub, sessions in parse_session_list(include or '', known).items():
            current[sub] = current.get(sub, set()) - sessions
        self.state['excluded_sessions'] = {s: sorted(v) for s, v in sorted(current.items()) if v}
        self.save()
        for sub, sessions in self.state['excluded_sessions'].items():
            status = self.state['subjects'][sub]['status']
            note = '' if status in ('pending', 'skipped') else f' (status {status}: add --reset {sub} to process it again without them)'
            self.log(f'{sub}: excluded sessions {", ".join(sessions)}{note}')

    def status(self):
        counts = {}
        for sub, entry in sorted(self.state['subjects'].items()):
            counts[entry['status']] = counts.get(entry['status'], 0) + 1
            if entry['status'] in ('submitted', 'failed', 'skipped'):
                jobs = ' '.join(f'{k[:-4]}={entry[k]}' for k in ('gpu_job', 'cpu_job', 'report_job') if entry.get(k))
                print(f'{sub:>14} {entry["status"]:<9} {jobs} {entry.get("reason", "")}')
        for sub, sessions in self.state.get('excluded_sessions', {}).items():
            print(f'{sub:>14} excluded sessions: {", ".join(sessions)}')
        print(' '.join(f'{k}={v}' for k, v in sorted(counts.items())))

    def run(self):
        admitting = True
        while True:
            try:
                admitting = self.poll() and admitting
            except RemoteError as error:
                if not error.transient:
                    raise
                self.log(f'HPC not reachable, retrying at the next poll: {error}')
            if not self.entries('submitted') and (not admitting or not self.entries('pending')):
                break
            if self.args.once:
                return 0
            time.sleep(self.args.poll)
        merged = merge_metrics(self.local_out)
        if merged:
            self.log(f'Cohort metrics: {merged}')
        failed = self.entries('failed')
        skipped = self.entries('skipped')
        if not admitting:
            self.log(f'Stopped admitting subjects after {len(failed)} failures (--max-failed)')
        self.log(f'Finished: {len(self.entries("done"))} done, {len(failed)} failed, '
                 f'{len(skipped)} skipped, {len(self.entries("pending"))} pending')
        return 1 if failed or skipped else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--local-bids', type=Path, required=True, help='BIDS dataset on the workstation.')
    p.add_argument('--local-out', type=Path, required=True,
                   help='Destination for fetched derivatives (mirrors derivatives/...) and driver state.')
    p.add_argument('--remote-bids', required=True,
                   help='Dedicated staging directory on the HPC; subjects are deleted there after fetching.')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--subjects', help='Comma-separated IDs, e.g. 394,sub-395.')
    group.add_argument('--subjects-file', type=Path, help='One ID per line; # comments allowed.')
    group.add_argument('--all', action='store_true', help='Every sub-* directory in --local-bids.')
    p.add_argument('--host', default='core-mgm')
    p.add_argument('--remote-repo', default='~/code/INIMNeuroimagingPipeline')
    p.add_argument('--ssh', default='ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=60')
    p.add_argument('--max-gpu', type=int, default=2, help='Subjects waiting for or using a GPU. Default: 2')
    p.add_argument('--max-on-hpc', type=int, default=6, help='Subjects with data on the HPC. Default: 6')
    p.add_argument('--max-failed', type=int, default=3, help='Stop admitting subjects after this many failures.')
    # FS-LIT took 12-79 min per session in the pilot; 8 h is the jobs-gpu maximum.
    p.add_argument('--gpu-time', default='08:00:00')
    p.add_argument('--cpu-time', default='06:00:00')
    p.add_argument('--poll', type=int, default=300, help='Seconds between checks. Default: 300')
    p.add_argument('--once', action='store_true', help='Run one scheduling round and exit.')
    p.add_argument('--status', action='store_true', help='Print the state and exit.')
    p.add_argument('--retry-failed', action='store_true', help='Queue failed and skipped subjects again.')
    p.add_argument('--reset', metavar='IDS',
                   help='Process these finished or failed subjects again from scratch (e.g. after '
                        'correcting scan dates): their HPC data is removed and fetched results are moved '
                        'to .cohort_driver/reset_backup/. With --status only resets.')
    p.add_argument('--exclude-sessions', metavar='LIST',
                   help='Leave sessions out of future uploads, e.g. sub-072_ses-2 (unusable images). '
                        'Kept in the state; combine with --reset for subjects already processed.')
    p.add_argument('--include-sessions', metavar='LIST', help='Undo --exclude-sessions for these sessions.')
    p.add_argument('pipeline_args', nargs=argparse.REMAINDER,
                   help='After --: options for struct_bids.sh. Default: --long --SEL')
    args = p.parse_args(argv)
    args.pipeline_args = [a for a in args.pipeline_args if a != '--'] or None
    if args.pipeline_args and any(a.startswith(('--stage', '--subjects')) for a in args.pipeline_args):
        p.error('--stage and --subjects are set per job by the driver')
    if not args.remote_bids.startswith('/'):
        p.error('--remote-bids must be an absolute path')
    if args.local_out.resolve() == args.local_bids.resolve():
        p.error('--local-out must differ from --local-bids')
    if min(args.max_gpu, args.max_on_hpc, args.max_failed, args.poll) < 1:
        p.error('limits and --poll must be positive')
    return args


def select_subjects(args):
    if args.all:
        return sorted(p.name for p in args.local_bids.glob('sub-*') if p.is_dir())
    if args.subjects:
        items = args.subjects.split(',')
    else:
        items = [line.split('#')[0] for line in args.subjects_file.read_text().splitlines()]
    known = [p.name for p in args.local_bids.glob('sub-*') if p.is_dir()]
    subjects, missing = resolve_subjects([i for i in items if i.strip()], known)
    if missing:
        raise SystemExit(f'Not in --local-bids: {", ".join(missing)}')
    return subjects


def main(argv=None):
    args = parse_args(argv)
    driver = Driver(args)
    driver.state_dir.mkdir(parents=True, exist_ok=True)
    driver.load(select_subjects(args))
    changes_state = args.reset or args.exclude_sessions or args.include_sessions
    if args.status and not changes_state:
        driver.status()
        return 0
    lock = driver.state_dir / 'driver.pid'
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            raise SystemExit(f'Another driver is running (PID {lock.read_text().strip()})')
        except (ProcessLookupError, ValueError):
            pass
    lock.write_text(str(os.getpid()))
    try:
        if args.exclude_sessions or args.include_sessions:
            driver.set_excluded_sessions(args.exclude_sessions, args.include_sessions)
        if args.reset:
            subjects, missing = resolve_subjects([s for s in args.reset.split(',') if s.strip()],
                                                 list(driver.state['subjects']))
            if missing:
                raise SystemExit(f'--reset: not in the selected subjects: {", ".join(missing)}')
            driver.reset(subjects)
        if args.status and changes_state:
            driver.status()
            return 0
        if args.retry_failed:
            for _, entry in driver.entries('failed', 'skipped'):
                entry.update(status='pending', reason=None)
        driver.save()
        driver.hpc.helper(op='prepare_staging', root=driver.remote_bids)
        driver.log(f'Driver started: {len(driver.entries("pending"))} pending, remote {driver.remote_bids}, '
                   f'pipeline options {" ".join(driver.pipeline_args)}')
        return driver.run()
    finally:
        lock.unlink()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--remote':
        remote_main(json.loads(sys.argv[2]))
    else:
        sys.exit(main())
