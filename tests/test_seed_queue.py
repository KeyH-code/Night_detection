"""Source A repeat recipe and fail-closed provisioning contract."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_repeat_uses_original_a_and_allows_only_the_missing_training_seeds():
    from seed_train import replica_config
    from source_contract import ContractError

    config, baseline = replica_config(29)
    assert config['experiment']=='source_a' and 'monitor' not in config
    assert config['training']['seed']==29
    assert config['training']['patience']==20 and config['training']['batch']==16 and config['training']['epochs']==100
    assert baseline['seed']==17 and baseline['lr0']==.01
    with pytest.raises(ContractError,match='29 or 43'):
        replica_config(17)


def test_effective_training_rejects_any_non_path_non_seed_recipe_mutation():
    from seed_train import replica_config, verify_effective
    from source_contract import ContractError

    _, baseline=replica_config(43)
    verify_effective(dict(baseline,seed=43,data='/tmp/new',device=0),baseline,43)
    with pytest.raises(ContractError,match='batch'):
        verify_effective(dict(baseline,seed=43,batch=8),baseline,43)


def test_remote_paths_reject_escape_and_non_persistent_inputs():
    from seed_queue import persist_path
    from gpu import GPUError

    assert persist_path('/new/source_a_kit/data/source_v1')=='/new/source_a_kit/data/source_v1'
    for p in ('/tmp/data','/new/../escape','/new/sh;echo','relative','/new'):
        with pytest.raises(GPUError): persist_path(p)


def test_no_stage_deploy_or_close_for_a_new_unowned_job():
    from seed_queue import guarded_lifecycle
    from gpu import GPUError

    def reject(**_):
        raise GPUError('no owned allocation')
    hit=[]
    def deploy(*a): hit.append('deploy')
    def close(*a): hit.append('close')
    with pytest.raises(GPUError,match='no owned allocation'):
        guarded_lifecycle(reject,deploy,close)
    assert hit==[]


def test_once_owned_the_workflow_closes_after_download_or_failure():
    from seed_queue import guarded_lifecycle
    from gpu import GPUError

    for failed in (False,True):
        events=[]
        def acquire(): events.append('queue'); return ('job',123)
        def execute(job,deadline):
            assert (job,deadline)==('job',123)
            events.append('deploy/train/verify/download')
            if failed: raise GPUError('transfer failure')
            return 'sealed'
        def close(job): assert job=='job'; events.append('stop')
        if failed:
            with pytest.raises(GPUError,match='transfer failure'): guarded_lifecycle(acquire,execute,close)
        else:
            assert guarded_lifecycle(acquire,execute,close)=='sealed'
        assert events==['queue','deploy/train/verify/download','stop']


def test_bundle_is_local_and_contains_only_the_declared_recipe_and_code(tmp_path):
    import tarfile
    from seed_queue import BUNDLE_FILES, REMOTE, make_bundle, verify_bundle

    p=tmp_path/'approved.tar'
    assert make_bundle(p)==verify_bundle(p)
    with tarfile.open(p) as ar:
        assert {x.name for x in ar.getmembers()}==set(BUNDLE_FILES)
        assert all(x.isfile() for x in ar.getmembers())
    compile(REMOTE,'<seed-remote>','exec')


def test_run_requires_explicit_billing_authorization_before_touching_package(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve().parents[1]/'scripts/seed_queue.py'),
                           '--bundle',str(tmp_path/'missing.tar'),'--run'],capture_output=True,text=True)
    assert result.returncode==2
    assert 'explicit authorized --minutes, --rate and --budget' in result.stderr


def test_controller_retrieves_verified_seed_runs_before_stopping(tmp_path, monkeypatch):
    from datetime import datetime,timedelta,timezone
    import hashlib
    import json
    from types import SimpleNamespace
    import seed_queue as sq

    events=[]
    job={'jobId':'owned','state':'Running'}
    hub=SimpleNamespace(id='bound',machine=lambda:job,gpu=lambda m:{'gpus':[{'name':'RTX 4090'}],'compute_processes':[]})
    opts=SimpleNamespace(minutes=200,rate=1.6,seed_minutes=75,reserve_minutes=12,source='/new/data',python='/new/venv/bin/python',
                         weights='/new/weights.pt',bundle=tmp_path/'deploy.tar')
    target=tmp_path/'replica_001'
    blob=json.dumps({'status':'complete','runs':{'29':{},'43':{}},'files':{}}).encode()
    sha=hashlib.sha256(blob).hexdigest()

    def acquire(h,minutes,update,*,min_remaining_minutes):
        assert h is hub and minutes==200 and min_remaining_minutes==165
        events.append('owned queue'); update(job_id='owned')
        return job,datetime.now(timezone.utc)+timedelta(minutes=195)

    def remote(h,m,c,action,t=90):
        assert m==job and c['root'].startswith('/new/a_replica_')
        events.append(action)
        return {'prepare':{'prepared':True},'launch':{'pid':60},'poll':{'status':'finished','alive':False},
                'verify':{'status':'complete','sha256':sha,'completed':['29','43']}}[action]

    def scp(h,m,source,dest,download=False,timeout=90):
        events.append('scp replica' if download and str(source).endswith('/replica') else 'scp job' if download else 'upload')
        if download and str(source).endswith('/replica'):
            p=dest/'replica'; p.mkdir(); (p/'manifest.json').write_bytes(blob)

    def close(h,m,timeout):
        events.append('stop'); assert (target/'replica/manifest.json').is_file(); return {'state':'Stopped'}

    for name,func in (('queue',acquire),('remote',remote),('copy',scp)):
        monkeypatch.setattr(sq,name,func)
    monkeypatch.setattr(sq.gpu,'close',close)
    sq.execute_job(opts,hub,{'sha256':'x'},target,5.33)
    assert events==['owned queue','upload','prepare','launch','poll','verify','scp replica','scp job','stop']
    status=json.loads(target.with_suffix('.queue.json').read_bytes())
    assert status['status']=='training_complete' and status['platform_state']=='Stopped'

    with pytest.raises(sq.gpu.GPUError,match='Existing queue'):
        sq.execute_job(opts,hub,{'sha256':'x'},target,5.33)


def test_adopted_queue_uses_allocated_time_not_submission_or_observation(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    import seed_queue as sq

    cutoff=datetime(2026,11,20,16,tzinfo=timezone.utc)
    allocated=cutoff-timedelta(seconds=35)
    observed=cutoff-timedelta(seconds=10)
    state={'time':observed-timedelta(seconds=15),'machine':'Queuing','deadline':None}
    calls=[]

    class Clock(datetime):
        @classmethod
        def now(cls,tz=None): return state['time']

    def machine():
        return {'state':state['machine'],'jobId':'owned','latestJob':{
            'id':'owned','startedTime':int(allocated.timestamp()*1000) if state['machine']=='Running' else None},
            'autoCloseTime':state['deadline']}

    def call(*args):
        assert args[:3]==('devmachine','auto-shutdown','bound')
        value=datetime.fromisoformat(args[-1]); calls.append(value)
        state['deadline']=int(value.timestamp()*1000)
        return {'id':'bound','closeTime':state['deadline']}

    def advance(_): state.update(time=observed,machine='Running')
    monkeypatch.setattr(sq,'datetime',Clock)
    monkeypatch.setattr(sq.time,'sleep',advance)
    h=SimpleNamespace(id='bound',machine=machine,call=call)
    running,deadline=sq.wait_adopted(h,'owned',cutoff,168,lambda **_:None)
    assert running['state']=='Running' and deadline==allocated+timedelta(minutes=168)
    assert calls==[cutoff,deadline]


def test_adopting_an_expired_queue_cancels_only_the_owned_job(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import seed_queue as sq

    cutoff=datetime(2026,11,20,16,tzinfo=timezone.utc)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None): return cutoff
    monkeypatch.setattr(sq,'datetime',Clock)
    m={'state':'Queuing','jobId':'owned'}
    events=[]
    h=SimpleNamespace(machine=lambda:m)
    monkeypatch.setattr(sq.gpu,'close',lambda hub,machine,seconds: events.append(machine['jobId']) or {'state':'Stopped'})
    with pytest.raises(sq.gpu.GPUError,match='Queue cutoff'):
        sq.wait_adopted(h,'owned',cutoff,168,lambda **_:None)
    m['jobId']='unowned'
    with pytest.raises(sq.gpu.GPUError,match='changed'):
        sq.wait_adopted(h,'owned',cutoff,168,lambda **_:None)
    assert events==['owned']


def test_adoption_refuses_deployment_marker_or_live_lock(tmp_path):
    import json
    import seed_queue as sq

    target=tmp_path/'replica_002'; receipt=target.with_suffix('.queue.json'); lock=target.with_suffix('.queue.lock')
    d={'status':'queued','job_id':'owned','machine_id':'bound','bundle_sha256':'same','seeds':[29,43]}
    receipt.write_text(json.dumps(d),encoding='utf-8'); lock.write_text('123',encoding='utf-8')
    with pytest.raises(sq.gpu.GPUError,match='controller lock'):
        sq.adoption_state(receipt,lock,target,'owned','bound','same')
    lock.unlink(); d['remote_root']='/new/already-deployed'; receipt.write_text(json.dumps(d),encoding='utf-8')
    with pytest.raises(sq.gpu.GPUError,match='deployment'):
        sq.adoption_state(receipt,lock,target,'owned','bound','same')
