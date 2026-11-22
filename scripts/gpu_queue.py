"""Queue the approved scale_003 job on the project's existing BitaHub machine.

The platform owns GPU allocation; this local controller must remain running to
deploy after allocation and collect results. The platform deadline also applies
while queued, so controller failure cannot extend the authorized power window.
"""

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import time

import gpu

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'runs/s3_17/scale24_queue.json'
PREFLIGHT = ROOT / 'runs/s3_17/scale24_preflight.json'
RESULT = ROOT / 'runs/scale_003'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def check_bundle(path):
    receipt = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if receipt['status'] != 'local_preflight_passed' or receipt['cuda']['status'] != 'passed':
        raise gpu.GPUError('The 24-epoch CUDA preflight must have passed')
    package = receipt['package']
    if path.stat().st_size != package['bytes'] or digest(path) != package['sha256']:
        raise gpu.GPUError('Deployment package differs from the verified preflight')
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if len(members) != len(package['file_hashes']) or {x.name for x in members} != set(package['file_hashes']):
            raise gpu.GPUError('Deployment package file whitelist differs')
        for item in members:
            if not item.isfile() or hashlib.sha256(archive.extractfile(item).read()).hexdigest() != package['file_hashes'][item.name]:
                raise gpu.GPUError('Unsafe or changed deployment member')
            if digest(ROOT / item.name) != package['file_hashes'][item.name]:
                raise gpu.GPUError(f'Active input differs from deployment package: {item.name}')
    return package


def budget_limit(minutes, rate, budget, available):
    if not all(math.isfinite(x) and x > 0 for x in (rate, budget)):
        raise gpu.GPUError('Rate and budget must be finite and positive')
    cost = minutes / 60 * rate
    if not 5 <= minutes <= 300 or cost > min(budget, available):
        raise gpu.GPUError('Requested power window exceeds the approved budget or 300-minute limit')
    return round(cost, 4)


def available_cash(payload):
    accounts = payload['balances']
    cash = [item['cash'] for item in accounts if item.get('productId') == 'heros']
    if len(cash) != 1:
        raise gpu.GPUError('Cannot uniquely determine the production cash balance')
    return float(cash[0]['postedBalance']) - float(cash[0]['reservedAmount'])


def verify_inventory(root, manifest):
    for name, entry in manifest['files'].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            raise gpu.GPUError('Unsafe result manifest path')
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise gpu.GPUError('Result escaped its run directory')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise gpu.GPUError(f'Result hash or size mismatch: {name}')


# Transferred as a command, without credentials. The worker is stored alongside
# its job log on /new; code and inputs in the approved tar remain byte-identical.
REMOTE = r'''
import base64, hashlib, json, os, pathlib, subprocess, sys, tarfile, time
c = json.loads(base64.b64decode(sys.argv[1]))
action = sys.argv[2]
root = pathlib.Path(c['root'])
job = root / 'job'
out = root / 'runs/scale_003'
def emit(value):
    print(json.dumps(value), flush=True)
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def write(path, data):
    p = path.with_suffix('.tmp')
    p.write_text(json.dumps(data), encoding='utf-8')
    p.replace(path)
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def started(pid):
    return pathlib.Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
if action == 'prepare':
    assert not root.exists(), 'Deployment directory already exists'
    bundle = pathlib.Path(c['bundle'])
    assert sha(bundle) == c['sha256'], 'Uploaded package hash mismatch'
    root.mkdir()
    with tarfile.open(bundle) as archive:
        items = archive.getmembers()
        assert len(items) == len(c['files'])
        assert {x.name for x in items} == set(c['files'])
        for item in items:
            assert item.isfile()
            assert hashlib.sha256(archive.extractfile(item).read()).hexdigest() == c['files'][item.name]
        archive.extractall(root, filter='data')
    bundle.unlink()  # This controller's verified transfer copy is no longer needed.
    job.mkdir()
    (root / 'runs').mkdir(exist_ok=True)
    sources = [pathlib.Path(p) for p in c['sources']]
    references = [pathlib.Path(p) for p in c['references']]
    source = next((p for p in sources if (p/'READY.json').is_file() and sha(p/'READY.json') == c['ready']), None)
    reference = None
    for p in references:
        if not (p/'manifest.json').is_file():
            continue
        m = read(p/'manifest.json')
        r = read(p/'checkpoint_roles.json')['diagnostic']
        if m['identity']['ready_sha256'] == c['ready'] and r['epoch'] == 80 and r['sha256'] == c['weight']:
            w = (p/r['path']).resolve()
            assert w.is_relative_to(p.resolve()) and sha(w) == c['weight']
            reference = p
            break
    python = next((pathlib.Path(p) for p in c['pythons'] if pathlib.Path(p).is_file()), None)
    assert source and reference and python, 'Frozen data, R80 or Python environment not found'
    subprocess.run([str(python), '-c', 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; assert "4090" in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))'], check=True, stdout=subprocess.DEVNULL)
    command = [str(python), '-B', '-u', 'scripts/train_scale.py', 'run', '--seed', '29', '--epochs', '24', '--train-only', '--source-root', str(source), '--reference-dir', str(reference), '--output', 'runs/scale_003', '--scratch', '/tmp']
    write(job/'command.json', dict(command=command, cwd=str(root)))
    # Record the actual mount supporting outputs; completion must be validated
    # and downloaded before requesting a normal shutdown.
    mount = subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'TARGET,FSTYPE'], text=True).strip()
    emit(dict(prepared=True, source=str(source), reference=str(reference), python=str(python), mount=mount))
elif action == 'worker':
    command = read(job/'command.json')['command']
    write(job/'status.json', dict(status='running', pid=os.getpid(), started=time.time()))
    try:
        with (job/'train.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            write(job/'child.json', dict(pid=child.pid, start=started(child.pid)))
            code = child.wait()
        write(job/'status.json', dict(status='finished', exit_code=code, finished=time.time()))
    except BaseException as exc:
        write(job/'status.json', dict(status='failed', error=type(exc).__name__, finished=time.time()))
        raise
elif action == 'launch':
    assert not (job/'launched.json').exists(), 'Worker already launched'
    script = base64.b64decode(c['worker'])
    (job/'runner.py').write_bytes(script)
    with (job/'runner.log').open('w', encoding='utf-8') as log:
        worker = subprocess.Popen(['python3', '-u', str(job/'runner.py'), sys.argv[1], 'worker'], cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(pid=worker.pid, start=started(worker.pid))
    write(job/'launched.json', value)
    emit(value)
elif action == 'poll':
    launch = read(job/'launched.json')
    s = read(job/'status.json') if (job/'status.json').exists() else dict(status='starting')
    try:
        alive = started(launch['pid']) == launch['start']
        # An unreaped zombie is no longer a live controller.
        alive = alive and pathlib.Path('/proc', str(launch['pid']), 'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        alive = False
    s['worker_alive'] = alive
    p = out/'progress.json'
    if p.is_file():
        s['progress'] = read(p)
    emit(s)
elif action == 'verify':
    s = read(job/'status.json')
    assert s['status'] == 'finished' and s['exit_code'] == 0, 'Training did not finish successfully'
    m = read(out/'training_manifest.json')
    assert m['status'] == 'training_complete' and not (out/'.active').exists()
    assert m['identity']['epochs'] == 24 and m['identity']['seed'] == 29
    assert not m['final_test_executed']
    for name, item in m['files'].items():
        p = (out/name).resolve()
        assert p.is_relative_to(out.resolve())
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
    for size in (640,960):
        a = read(out/str(size)/'train.json')
        assert a['status'] == 'complete' and len(a['history']) == 24 and a['updates'] == 3000
    emit(dict(status='training_complete', files=len(m['files']), manifest_sha256=sha(out/'training_manifest.json')))
else:
    raise ValueError(action)
'''


def remote(hub, machine, config, action, timeout=60):
    info = hub.connection(machine)
    payload = base64.b64encode(json.dumps(config).encode()).decode()
    script = base64.b64encode(REMOTE.encode()).decode()
    code = f'import base64;exec(base64.b64decode({script!r}))'
    command = shlex.join(['python3', '-c', code, payload, action])
    result = gpu.invoke(['ssh', *info['options'], '-p', str(info['port']),
                         f"{info['user']}@{info['host']}", command], timeout)
    if result.returncode:
        # Remote scripts contain no credentials; keep their actionable failure.
        raise gpu.GPUError(f'Remote {action} failed: {result.stderr[-2000:]}')
    return json.loads(result.stdout)


def copy(hub, machine, source, target, *, download=False, timeout=300):
    info = hub.connection(machine)
    endpoint = f"{info['user']}@{info['host']}:"
    argv = ['scp', *info['options'], '-P', str(info['port'])]
    argv += ['-r', endpoint + source, str(target)] if download else [str(source), endpoint + target]
    result = gpu.invoke(argv, timeout)
    if result.returncode:
        raise gpu.GPUError(f'SCP failed (exit {result.returncode}); results remain on the remote persistent volume')


def queue(hub, minutes, update, *, min_remaining_minutes=10):
    machine = hub.machine()
    if machine['state'] != 'Stopped':
        raise gpu.GPUError('Queue requires Stopped; existing jobs cannot be adopted')
    for field, key in [('powerConfId', 'power_conf_id'), ('sshId', 'ssh_id'), ('startupType', 'startup_type')]:
        if machine.get(field) != hub.config[key]:
            raise gpu.GPUError(f'Changed platform binding: {field}')
    hub.ssh_files()
    previous_job = machine.get('jobId')
    end = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=minutes)
    # Persist intent before the only open request. A transport error is ambiguous:
    # never retry open automatically or close a job whose ownership is unknown.
    update(status='submitting', shutdown_at_utc=end.isoformat(), previous_job_id=machine.get('jobId'))
    hub.call('devmachine', 'open', hub.id, '--startup-type', str(hub.config['startup_type']))
    owned_job = None
    try:
        while True:
            machine = hub.machine()
            if machine['state'] in gpu.STOPPABLE:
                candidate = machine.get('jobId')
                if candidate and candidate != previous_job:
                    owned_job = candidate
                    break
            if machine['state'] not in gpu.BOOTING | {'Stopped', 'Running'} or datetime.now(timezone.utc) >= end - timedelta(minutes=1):
                raise gpu.GPUError('No new platform job identity appeared after the open request')
            time.sleep(5)
        update(status='queued', job_id=owned_job, platform_state=machine['state'])
        stamp = end.isoformat(timespec='seconds')
        timer = hub.call('devmachine', 'auto-shutdown', hub.id, '--at', stamp)
        if timer.get('id') != hub.id or timer.get('closeTime') != int(end.timestamp()) * 1000:
            raise gpu.GPUError('Platform did not acknowledge the shutdown deadline')
        while True:
            machine = hub.machine()
            if machine.get('jobId') != owned_job:
                raise gpu.GPUError('Platform job changed; refusing takeover')
            update(platform_state=machine['state'], timer_visible=gpu.deadline_visible(machine, end))
            if machine['state'] == 'Running':
                if not gpu.deadline_visible(machine, end):
                    raise gpu.GPUError('Running machine did not confirm the shutdown timer')
                if (end - datetime.now(timezone.utc)).total_seconds() < min_remaining_minutes*60:
                    raise gpu.GPUError(f'Less than {min_remaining_minutes} minutes remain; training was not launched')
                return machine, end
            if machine['state'] not in gpu.BOOTING:
                raise gpu.GPUError(f"Queue ended with platform state {machine['state']}")
            if datetime.now(timezone.utc) >= end - timedelta(minutes=min_remaining_minutes):
                raise gpu.GPUError('Remaining window cannot fit the task; canceling this queued job')
            time.sleep(30)
    except BaseException:
        current = hub.machine()
        if owned_job and current.get('jobId') == owned_job:
            closed = gpu.close(hub, current, 180)
            update(platform_state=closed['state'])
        raise


def run(args, hub, package, available):
    if RECEIPT.exists() or RESULT.exists():
        raise gpu.GPUError('An existing queue receipt or scale_003 result requires inspection; no automatic re-submission')
    lock = RECEIPT.with_suffix('.lock')
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    state = dict(status='prepared', pid=os.getpid(), machine_id=hub.id, submitted_at_utc=now(),
                 bundle_sha256=package['sha256'], rate_estimate=args.rate, available_cash_at_submit=available,
                 controller_sha256=digest(Path(__file__)), gpu_controller_sha256=digest(ROOT/'scripts/gpu.py'),
                 max_minutes=args.minutes, max_cost_estimate=budget_limit(args.minutes, args.rate, args.budget, available),
                 billing_verified=False, events=[])

    def update(**values):
        changed = {k: v for k, v in values.items() if state.get(k) != v}
        state.update(values, checked_at_utc=now())
        if changed:
            state['events'].append(dict(at_utc=now(), **changed))
            gpu.report(changed)
        save(RECEIPT, state)

    machine = None
    try:
        machine, deadline = queue(hub, args.minutes, update)
        update(status='connecting')
        # Booting SSH can lag platform Running; retry read-only readiness, never
        # replay deployment or launch after an ambiguous transport error.
        ready_until = min(deadline - timedelta(minutes=5), datetime.now(timezone.utc) + timedelta(minutes=3))
        while True:
            try:
                telemetry = hub.gpu(machine)
                break
            except gpu.GPUError:
                if datetime.now(timezone.utc) >= ready_until:
                    raise
                time.sleep(10)
                machine = owned_machine(hub, state['job_id'])
        if telemetry['compute_processes'] or len(telemetry['gpus']) != 1 or '4090' not in telemetry['gpus'][0]['name']:
            raise gpu.GPUError('Expected a free single 4090; no workload launched')
        token = os.urandom(8).hex()
        config = dict(root=f'/new/scale_003_{token}', bundle=f'/new/scale24_{token}.tar',
                      sha256=package['sha256'], files=package['file_hashes'],
                      ready=json.loads(PREFLIGHT.read_text(encoding='utf-8'))['inputs']['ready'],
                      weight=json.loads(PREFLIGHT.read_text(encoding='utf-8'))['inputs']['weight'],
                      sources=['/new/source_s3/source_a_kit/data/source_v1', '/new/source_u_481d59f6_tmp/source_a_kit/data/source_v1', '/new/source_a_kit/data/source_v1'],
                      references=['/new/source_s3/source_a_kit/runs/a_002/train', '/new/source_a_kit/runs/a_002/train'],
                      pythons=['/new/source_a_kit/.venv-train/bin/python', '/new/source_s3/source_a_kit/.venv-train/bin/python'])
        config['worker'] = base64.b64encode(REMOTE.encode()).decode()
        update(remote_root=config['root'], status='deploying')
        copy(hub, machine, args.bundle, config['bundle'])
        update(deployment=remote(hub, machine, config, 'prepare', 120))
        for script in ('gpu_queue.py', 'gpu.py'):
            copy(hub, machine, ROOT/'scripts'/script, config['root'] + '/job/' + script)
        update(status='launching')
        update(worker=remote(hub, machine, config, 'launch'), status='training')
        while True:
            machine = owned_machine(hub, state['job_id'])
            if machine['state'] != 'Running':
                raise gpu.GPUError(f"Training interrupted by platform state {machine['state']}; inspect remote checkpoints")
            progress = remote(hub, machine, config, 'poll')
            update(training=progress)
            if not progress['worker_alive'] and progress['status'] in ('starting', 'running'):
                raise gpu.GPUError('Remote worker disappeared; no automatic re-training')
            if progress['status'] in ('finished', 'failed'):
                break
            time.sleep(30)
        update(remote_verification=remote(hub, machine, config, 'verify', 180), status='downloading')
        copy(hub, machine, config['root'] + '/runs/scale_003', RESULT.parent, download=True,
             timeout=max(1, min(600, int((deadline - datetime.now(timezone.utc)).total_seconds() - 60))))
        manifest = json.loads((RESULT/'training_manifest.json').read_text(encoding='utf-8'))
        if digest(RESULT/'training_manifest.json') != state['remote_verification']['manifest_sha256']:
            raise gpu.GPUError('Downloaded training manifest differs from the verified remote manifest')
        verify_inventory(RESULT, manifest)
        # Job logs are outside the sealed training inventory and retain their
        # cloud paths in this receipt. Download before shutting the machine down.
        copy(hub, machine, config['root'] + '/job', RESULT, download=True)
        update(status='download_verified', local_result=str(RESULT))
        final = gpu.close(hub, owned_machine(hub, state['job_id']), 180)
        update(status='training_complete', platform_state=final['state'], local_evaluation='pending')
    except BaseException as exc:
        update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if machine is not None:
            try:
                current = owned_machine(hub, state['job_id'])
                final = gpu.close(hub, current, 180)
                update(platform_state=final['state'])
            except gpu.GPUError as shutdown_error:
                update(shutdown_error=str(shutdown_error), platform_deadline_remains_active=True)
        raise
    finally:
        lock.unlink()


def owned_machine(hub, job):
    machine = hub.machine()
    if machine.get('jobId') != job:
        raise gpu.GPUError('Machine job changed; refusing to act on another job')
    return machine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='submit once, wait, train, retrieve and stop; otherwise read-only checks')
    parser.add_argument('--minutes', type=int, default=300)
    parser.add_argument('--rate', type=float, default=1.63)
    parser.add_argument('--budget', type=float, default=8.60)
    args = parser.parse_args()
    try:
        package = check_bundle(args.bundle.resolve())
        hub = gpu.BitaHub(json.loads(gpu.CONFIG.read_text(encoding='utf-8')))
        available = available_cash(hub.call('balance'))
        maximum = budget_limit(args.minutes, args.rate, args.budget, available)
        gpu.report(dict(check='passed', machine=gpu.summary(hub.machine()), available_cash=available,
                        maximum_cost_estimate=maximum, cloud_mutations=args.run))
        if args.run:
            run(args, hub, package, available)
        return 0
    except (gpu.GPUError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'GPU queue: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
