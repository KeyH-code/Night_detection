"""Prepare and, when explicitly authorized, queue the two original-A repeats on BitaHub."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

import gpu
from gpu_queue import available_cash, budget_limit, copy, digest, now, queue, save, verify_inventory
from seed_train import COCO, ORIGINAL_FILES, READY, ROOT, SEEDS

BUNDLE_FILES = (*ORIGINAL_FILES, *(f'scripts/{n}.py' for n in ('seed_train','train_source','source_contract',
    'source_trainer','source_resources','source_sampling','source_policy','detection_metrics','evaluate','train_monitor')),
    'environment/requirements-linux.lock','specs/s3/replica.md')
DEFAULT_SOURCE = '/new/source_s3/source_a_kit/data/source_v1'
DEFAULT_WEIGHTS = '/new/source_a_kit/weights/yolo11m.pt'
DEFAULT_VENV = '/new/source_a_kit/.venv-train/bin/python'


def persist_path(text):
    if (not text.startswith('/new/') or not re.fullmatch(r'/new/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+',text)
            or any(p in ('.','..') for p in text.split('/')[2:])):
        raise gpu.GPUError('Expected an explicit safe path on /new')
    return text


def make_bundle(path):
    if path.resolve().is_relative_to(ROOT) or path.exists():
        raise gpu.GPUError('Use a new bundle path outside the repository')
    entries = {n:digest(ROOT/n) for n in BUNDLE_FILES}
    if any(entries[n]!=v for n,v in ORIGINAL_FILES.items()):
        raise gpu.GPUError('Original configuration or effective args do not match a_001')
    with tarfile.open(path,'x') as ar:
        for n in sorted(entries):
            p=ROOT/n
            if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT):
                raise gpu.GPUError('Unexpected bundle input: '+n)
            ar.add(p,arcname=n,recursive=False)
    return verify_bundle(path,entries)


def verify_bundle(path, expected=None):
    with tarfile.open(path,'r:') as ar:
        members=ar.getmembers()
        if len(members)!=len(BUNDLE_FILES) or {m.name for m in members}!=set(BUNDLE_FILES):
            raise gpu.GPUError('Deployment package whitelist mismatch')
        entries={}
        for member in members:
            if not member.isfile() or member.size>5_000_000:
                raise gpu.GPUError('Unsafe bundle member '+member.name)
            v=hashlib.sha256(ar.extractfile(member).read()).hexdigest()
            if v!=digest(ROOT/member.name) or (expected is not None and v!=expected[member.name]):
                raise gpu.GPUError('Changed bundle member '+member.name)
            entries[member.name]=v
    if any(entries[n]!=v for n,v in ORIGINAL_FILES.items()):
        raise gpu.GPUError('Sealed original inputs differ')
    return {'sha256':digest(path),'files':entries,'bytes':path.stat().st_size}


def guarded_lifecycle(acquire, execute, close):
    machine, deadline=acquire()
    try:
        result=execute(machine,deadline)
    except BaseException as original:
        try:
            close(machine)
        except Exception as e:
            original.add_note('Automatic shutdown could not be confirmed; keep the platform deadline: '+str(e))
        raise
    close(machine)
    return result


REMOTE = r'''
import base64,hashlib,json,os,pathlib,subprocess,sys,tarfile,time
c=json.loads(base64.b64decode(sys.argv[1])); action=sys.argv[2]
root=pathlib.Path(c['root']); job=root/'job'; out=root/'replica'; bundle=pathlib.Path(c['upload'])
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(d),encoding='utf-8'); t.replace(p)
def emit(d): print(json.dumps(d),flush=True)
def stat(pid): return pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
if action=='prepare':
    if root.exists() or sha(bundle)!=c['package']['sha256']: raise RuntimeError('Unexpected deploy path or upload SHA')
    source=pathlib.Path(c['source']); weights=pathlib.Path(c['weights']); py=pathlib.Path(c['python'])
    for p in (source,weights):
        if not p.exists() or not p.resolve().is_relative_to('/new'): raise RuntimeError('Persistent input unavailable')
    # A venv's executable may legitimately symlink to the system interpreter.
    if not py.is_file() or not py.parent.resolve().is_relative_to('/new') or not (py.parent.parent/'pyvenv.cfg').is_file():
        raise RuntimeError('Persistent Python virtual environment unavailable')
    if sha(source/'READY.json')!=c['ready'] or sha(weights)!=c['weights_sha']:
        raise RuntimeError('Frozen source or official COCO weight SHA differs')
    root.mkdir()
    with tarfile.open(bundle,'r:') as ar:
        members=ar.getmembers(); expected=c['package']['files']
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected): raise RuntimeError('Package whitelist differs')
        for m in members:
            if not m.isfile() or hashlib.sha256(ar.extractfile(m).read()).hexdigest()!=expected[m.name]:
                raise RuntimeError('Unsafe package member hash')
        ar.extractall(root, filter='data')
    bundle.unlink()
    job.mkdir()
    command=[str(py),'-B','-u',str(root/'scripts/seed_train.py'),'--source',str(source),'--weights',str(weights),
             '--scratch','/tmp','--output',str(out),'--deadline',str(c['deadline']),
             '--min-seed-minutes',str(c['seed_minutes']),'--reserve-minutes',str(c['reserve_minutes'])]
    write(job/'command.json',command)
    emit({'prepared':True,'output':str(out)})
elif action=='launch':
    marker=job/'launched.json'; script=job/'worker.py'
    with marker.open('x',encoding='utf-8') as receipt: receipt.write('{}')
    script.write_bytes(base64.b64decode(c['worker']))
    with (job/'worker.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(['python3','-u',str(script),sys.argv[1],'worker'],cwd=root,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(marker,{'pid':p.pid,'start':stat(p.pid)[19]}); emit({'pid':p.pid})
elif action=='worker':
    p=job/'worker_status.json'; write(p,{'status':'running'})
    try:
        rc=subprocess.run(read(job/'command.json'),cwd=root).returncode
        write(p,{'status':'finished','exit':rc})
    except Exception as e:
        write(p,{'status':'failed','error':type(e).__name__}); raise
elif action=='poll':
    w=read(job/'launched.json'); p=job/'worker_status.json'
    s=read(p) if p.is_file() else {'status':'starting'}
    try:
        live=stat(w['pid']); s['alive']=live[19]==w['start'] and live[0]!='Z'
    except FileNotFoundError: s['alive']=False
    if (out/'status.json').is_file():
        r=read(out/'status.json'); s['seed']=r.get('active_seed'); s['completed']=list(r['runs'])
    emit(s)
elif action=='verify':
    m=read(out/'manifest.json'); s=read(job/'worker_status.json')
    if s['status']!='finished' or m['status'] not in ('complete','partial','failed') or m['final_test_executed']:
        raise RuntimeError('Worker did not produce a final safe manifest')
    if (s['exit']==0)!=(m['status']=='complete'): raise RuntimeError('Exit and manifest status disagree')
    for n,r in m['files'].items():
        q=out/n
        if not q.resolve().is_relative_to(out) or q.is_symlink() or q.stat().st_size!=r['bytes'] or sha(q)!=r['sha256']:
            raise RuntimeError('Run evidence differs: '+n)
    if m['status']=='complete' and set(m['runs'])!={'29','43'}: raise RuntimeError('Missing required seed')
    emit({'status':m['status'],'sha256':sha(out/'manifest.json'),'completed':list(m['runs'])})
else: raise ValueError(action)
'''


def remote(hub, machine, config, action, timeout=90):
    info=hub.connection(machine)
    payload=base64.b64encode(json.dumps(config).encode()).decode()
    encoded=base64.b64encode(REMOTE.encode()).decode()
    command=shlex.join(['python3','-c',f'import base64;exec(base64.b64decode({encoded!r}))',payload,action])
    response=gpu.invoke(['ssh',*info['options'],'-p',str(info['port']),f"{info['user']}@{info['host']}",command],timeout)
    if response.returncode:
        raise gpu.GPUError(f'Remote {action} failed: {response.stderr[-1500:]}')
    return json.loads(response.stdout)


def owned(hub, machine):
    current=hub.machine()
    if machine.get('jobId')!=current.get('jobId'):
        raise gpu.GPUError('Platform job changed; no action on another job')
    return current


def adoption_state(receipt, lock, target, job_id, machine_id, bundle_hash):
    if lock.exists():
        raise gpu.GPUError('The old controller lock still exists; verify it has stopped and hand off first')
    if target.exists() or not receipt.is_file():
        raise gpu.GPUError('Adoption requires an existing receipt and no local result')
    state=json.loads(receipt.read_bytes())
    if state.get('status') not in ('queued','adopted_queue') or any(
            k in state for k in ('remote_root','worker','deploy','remote_verification')):
        raise gpu.GPUError('Do not adopt a job with deployment or terminal evidence')
    if (state.get('job_id'),state.get('machine_id'),state.get('bundle_sha256'),state.get('seeds')) != (
            job_id,machine_id,bundle_hash,list(SEEDS)):
        raise gpu.GPUError('Adoption must match the recorded job, machine, bundle and seeds')
    return state


def set_adopted_timer(hub, job_id, deadline):
    current=owned(hub,{'jobId':job_id})
    if current['state'] not in gpu.STOPPABLE:
        raise gpu.GPUError('Existing job is no longer active; no timer changed')
    stamp=deadline.isoformat(timespec='seconds')
    answer=hub.call('devmachine','auto-shutdown',hub.id,'--at',stamp)
    if answer.get('id')!=hub.id or answer.get('closeTime')!=int(deadline.timestamp())*1000:
        raise gpu.GPUError('Platform did not acknowledge the new deadline')
    return owned(hub,{'jobId':job_id})


def wait_adopted(hub, job_id, cutoff, allocated_minutes, update):
    """Wait for an existing platform job. This path never submits an open request."""
    machine=owned(hub,{'jobId':job_id})
    if machine['state'] in gpu.BOOTING and datetime.now(timezone.utc)<cutoff:
        machine=set_adopted_timer(hub,job_id,cutoff)
        update(queue_timer_acknowledged=True,shutdown_at_utc=cutoff.isoformat())
    while True:
        update(platform_state=machine['state'])
        if machine['state']=='Stopped':
            raise gpu.GPUError('Existing job is Stopped; refusing to create another job')
        if machine['state']=='Running':
            try:
                latest=machine.get('latestJob') or {}
                stamp=latest.get('startedTime')
                if latest.get('id')!=job_id or not isinstance(stamp,int) or stamp<=0:
                    raise gpu.GPUError('Platform allocation timestamp not verified; training was not launched')
                allocated=datetime.fromtimestamp(stamp/1000,tz=timezone.utc)
                if allocated>=cutoff:
                    raise gpu.GPUError('GPU allocated at or after queue cutoff; training was not launched')
                deadline=allocated.replace(microsecond=0)+timedelta(minutes=allocated_minutes)
                if datetime.now(timezone.utc)>=deadline:
                    raise gpu.GPUError('The original allocated GPU window has already elapsed')
                machine=set_adopted_timer(hub,job_id,deadline)
                if machine['state']!='Running' or not gpu.deadline_visible(machine,deadline):
                    raise gpu.GPUError('Running job did not confirm its allocated shutdown deadline')
                update(gpu_allocated_at_utc=allocated.isoformat(),shutdown_at_utc=deadline.isoformat(),timer_visible=True)
                return machine,deadline
            except Exception:
                current=owned(hub,{'jobId':job_id})
                if current['state']=='Running':
                    closed=gpu.close(hub,current,180)
                    update(platform_state=closed['state'])
                raise
        if machine['state'] not in gpu.BOOTING:
            raise gpu.GPUError(f'Existing queue ended with state {machine["state"]}')
        left=(cutoff-datetime.now(timezone.utc)).total_seconds()
        if left<=0:
            closed=gpu.close(hub,machine,180)
            update(status='queue_cutoff',platform_state=closed['state'])
            raise gpu.GPUError('Queue cutoff reached without verified GPU allocation; existing job canceled')
        time.sleep(min(10,left))
        machine=owned(hub,{'jobId':job_id})


def execute_job(a, hub, package, target, budget, *, adoption=None):
    receipt=target.with_suffix('.queue.json'); lock=target.with_suffix('.queue.lock')
    if adoption:
        job_id,cutoff,allocated_minutes=adoption
        state=adoption_state(receipt,lock,target,job_id,hub.id,package['sha256'])
    else:
        if receipt.exists() or target.exists():
            raise gpu.GPUError('Existing queue receipt or result; investigate without resubmitting')
        state={'status':'prepared','started':now(),'machine_id':hub.id,'bundle_sha256':package['sha256'],
               'seeds':list(SEEDS),'minutes':a.minutes,'cost_upper_estimate_cny':budget,'billing_verified':False,
               'rate_cny_per_hour':a.rate,'seed_minutes':a.seed_minutes,'reserve_minutes':a.reserve_minutes,
               'controller_sha256':digest(Path(__file__)),'events':[]}
    with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))

    def update(**values):
        changed={k:v for k,v in values.items() if state.get(k)!=v}
        if changed:
            state.update(changed); state['events'].append({'at_utc':now(),**changed}); save(receipt,state); gpu.report(changed)

    def execute(machine,deadline):
        boot_until=min(deadline-timedelta(minutes=4),datetime.now(timezone.utc)+timedelta(minutes=3))
        while True:
            current=owned(hub,machine)
            try:
                dev=hub.gpu(current); break
            except gpu.GPUError:
                if datetime.now(timezone.utc)>=boot_until: raise
                time.sleep(8)
        if dev['compute_processes'] or len(dev['gpus'])!=1 or '4090' not in dev['gpus'][0]['name']:
            raise gpu.GPUError('Bound instance must have one idle RTX 4090')
        if (deadline-datetime.now(timezone.utc)).total_seconds()<60*(2*a.seed_minutes+a.reserve_minutes):
            raise gpu.GPUError('Queue consumed the two-seed budget; no training started')
        token=os.urandom(8).hex(); root='/new/a_replica_'+token
        cfg={'root':root,'upload':root+'.tar','package':package,'source':a.source,'python':a.python,'weights':a.weights,
             'ready':READY,'weights_sha':COCO,'deadline':deadline.timestamp(),'seed_minutes':a.seed_minutes,
             'reserve_minutes':a.reserve_minutes,'worker':base64.b64encode(REMOTE.encode()).decode()}
        update(status='uploading',remote_root=root)
        copy(hub,current,a.bundle,cfg['upload'],timeout=120)
        update(deploy=remote(hub,owned(hub,machine),cfg,'prepare'),status='starting_worker')
        update(worker=remote(hub,owned(hub,machine),cfg,'launch'),status='training')
        while True:
            current=owned(hub,machine)
            if current['state']!='Running': raise gpu.GPUError('Owned machine stopped mid-run; inspect durable project path')
            progress=remote(hub,current,cfg,'poll')
            update(worker_progress=progress)
            if progress['status'] in ('finished','failed') and not progress['alive']: break
            if not progress['alive']: raise gpu.GPUError('Worker vanished; no automatic retraining')
            time.sleep(25)
        v=remote(hub,owned(hub,machine),cfg,'verify',120)
        update(status='downloading',remote_verification=v)
        target.mkdir()
        transfer=max(1,min(600,int((deadline-datetime.now(timezone.utc)).total_seconds())-60))
        copy(hub,owned(hub,machine),root+'/replica',target,download=True,timeout=transfer)
        results=target/'replica'; m=json.loads((results/'manifest.json').read_bytes())
        if digest(results/'manifest.json')!=v['sha256'] or m['status']!=v['status']:
            raise gpu.GPUError('Downloaded manifest differs from verified remote identity')
        verify_inventory(results,m)
        copy(hub,owned(hub,machine),root+'/job',target,download=True,timeout=90)
        update(status='download_verified',local_result=str(results))
        return v['status']

    def close(machine):
        try:
            current=hub.machine()
            if current['state']!='Stopped': current=gpu.close(hub,owned(hub,machine),180)
            update(platform_state=current['state'])
        except Exception as e:
            update(shutdown_unconfirmed=str(e),platform_fallback='Configured deadline; inspect platform state')
            raise

    try:
        if adoption:
            update(status='adopted_queue',queue_until_utc=cutoff.isoformat(),allocated_minutes=allocated_minutes,
                   allocated_cost_estimate_cny=budget,adopted_controller_sha256=digest(Path(__file__)))
            acquire=lambda:wait_adopted(hub,job_id,cutoff,allocated_minutes,update)
        else:
            save(receipt,state)
            acquire=lambda:queue(hub,a.minutes,update,min_remaining_minutes=2*a.seed_minutes+a.reserve_minutes+3)
        outcome=guarded_lifecycle(acquire,execute,close)
        update(status='training_complete' if outcome=='complete' else 'partial_or_failed_training',local_evaluation='pending')
        if outcome!='complete': raise gpu.GPUError('The downloaded result does not contain two completed seeds')
    except BaseException as exc:
        update(error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true',help='create exact local bundle; no platform request')
    mode.add_argument('--status',action='store_true',help='verify bundle and query read-only platform state')
    mode.add_argument('--run',action='store_true',help='one billed job, training, download and automatic stop')
    mode.add_argument('--adopt-job',help='handoff an existing queued job without submitting another open request')
    p.add_argument('--queue-until',help='explicit ISO 8601 cutoff with time zone; adoption only')
    p.add_argument('--minutes',type=int)
    p.add_argument('--rate',type=float)
    p.add_argument('--budget',type=float)
    p.add_argument('--seed-minutes',type=int,default=75)
    p.add_argument('--reserve-minutes',type=int,default=12)
    p.add_argument('--result',default='replica_001')
    p.add_argument('--source',default=DEFAULT_SOURCE)
    p.add_argument('--weights',default=DEFAULT_WEIGHTS)
    p.add_argument('--python',default=DEFAULT_VENV)
    a=p.parse_args(); a.bundle=a.bundle.resolve()
    try:
        if a.adopt_job:
            if not a.queue_until or any(v is not None for v in (a.minutes,a.rate,a.budget)):
                p.error('Adoption requires --queue-until and uses the original receipt budget, rate and training windows')
            cutoff=datetime.fromisoformat(a.queue_until)
            if cutoff.utcoffset() is None or cutoff.microsecond:
                p.error('--queue-until needs a time zone and whole seconds')
            cutoff=cutoff.astimezone(timezone.utc)
        elif a.queue_until:
            p.error('--queue-until requires --adopt-job')
        if a.run:
            if any(x is None for x in (a.minutes,a.rate,a.budget)) or not 65<=a.seed_minutes<=150 or not 8<=a.reserve_minutes<=30:
                p.error('--run requires explicit authorized --minutes, --rate and --budget plus safe windows')
            if a.minutes < 2*a.seed_minutes+a.reserve_minutes+3:
                p.error('Time window must allow both seed and transfer allocations')
        for key in ('source','weights','python'): persist_path(getattr(a,key))
        if not re.fullmatch(r'replica_\d{3}',a.result): p.error('Use an unused replica_NNN identifier')
        package=make_bundle(a.bundle) if a.prepare else verify_bundle(a.bundle)
        if a.run or a.status or a.adopt_job:
            hub=gpu.BitaHub(json.loads(gpu.CONFIG.read_text(encoding='utf-8')))
            available=available_cash(hub.call('balance'))
            gpu.report({'machine':gpu.summary(hub.machine()),'available_cash':available,'will_mutate':bool(a.run or a.adopt_job)})
            if a.run:
                estimate=budget_limit(a.minutes,a.rate,a.budget,available)
                execute_job(a,hub,package,available_target(a.result),estimate)
            if a.adopt_job:
                target=available_target(a.result)
                state=adoption_state(target.with_suffix('.queue.json'),target.with_suffix('.queue.lock'),
                                     target,a.adopt_job,hub.id,package['sha256'])
                a.seed_minutes=state['seed_minutes']; a.reserve_minutes=state['reserve_minutes']
                allocated=2*a.seed_minutes+a.reserve_minutes+3
                estimate=budget_limit(allocated,state['rate_cny_per_hour'],state['cost_upper_estimate_cny'],available)
                execute_job(a,hub,package,target,estimate,adoption=(a.adopt_job,cutoff,allocated))
        else:
            gpu.report({'package_sha256':package['sha256'],'bytes':package['bytes'],'files':len(package['files']),
                        'cloud_mutations':False})
    except (gpu.GPUError,OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        p.exit(1,f'Source A queue: {exc}\n')


def available_target(name):
    path=ROOT/'runs'/name
    if path.resolve()!=path:
        raise gpu.GPUError('Unexpected result path resolution')
    return path


if __name__=='__main__': main()
