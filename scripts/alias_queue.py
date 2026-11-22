"""Run one source_alias training on the bound BitaHub machine and release it before local evaluation."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import tarfile
import tempfile
import time

import gpu
from gpu_queue import available_cash, budget_limit, copy, digest, now, queue, save, verify_inventory
from seed_queue import DEFAULT_SOURCE, DEFAULT_WEIGHTS, DEFAULT_VENV, owned
from source_alias import pack
from seed_train import READY, COCO, ROOT


REMOTE=r'''
import hashlib,json,os,sys,pathlib,tarfile,subprocess
from datetime import datetime,timezone
root=pathlib.Path(sys.argv[1]);action=sys.argv[2];conf=json.loads(sys.argv[3]);out=root/'run';job=root/'job'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');t.replace(p)
if root.parent!=pathlib.Path('/new') or root.name!=conf['token']:raise ValueError('Unexpected remote path')
if action=='deploy':
 ar=root.with_suffix('.tar');assert not root.exists() and sha(ar)==conf['bundle'];root.mkdir();job.mkdir()
 with tarfile.open(ar,'r:') as t:
  assert sorted(t.getnames())==sorted(conf['files'])
  for m in t:
   if not m.isfile() or m.name.startswith('/') or '..' in pathlib.PurePosixPath(m.name).parts or hashlib.file_digest(t.extractfile(m),'sha256').hexdigest()!=conf['files'][m.name]:raise ValueError('Invalid package')
  t.extractall(root,filter='data')
 ar.unlink();py=pathlib.Path(conf['python']);source=pathlib.Path(conf['source']);weight=pathlib.Path(conf['weights'])
 assert py.is_file() and (py.parent.parent/'pyvenv.cfg').is_file() and sha(source/'READY.json')==conf['ready'] and sha(weight)==conf['coco']
 cmd=[str(py),'-B','-u',str(root/'scripts/source_alias.py'),'--source',str(source),'--weights',str(weight),'--scratch','/tmp','--output',str(out),'--seed',str(conf['seed']),'--deadline',str(conf['deadline'])];write(job/'command.json',cmd)
 if conf.get('reuse'):
  cmd.extend(['--preflight-run',conf['reuse']['remote_run'],'--expected-preflight-sha',conf['reuse']['estimate_sha256']]);write(job/'command.json',cmd)
 code="import json,subprocess,pathlib,sys,time; j=pathlib.Path(sys.argv[1]); rc=subprocess.call(json.loads((j/'command.json').read_text()),cwd=j.parent); p=j/'exit.tmp'; p.write_text(json.dumps({'code':rc,'finished_unix':time.time()})); p.replace(j/'exit.json')"
 with (job/'launched').open('x') as f, (job/'training.log').open('wb') as stream:
  p=subprocess.Popen(['python3','-u','-c',code,str(job)],cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);f.write(str(p.pid))
 value={'pid':p.pid,'command_persisted':True}
elif action=='observe':
 value=read(out/'status.json') if (out/'status.json').is_file() else {'status':'setup'}
 for stage in ('work','probe'):
  p=out/stage/'results.csv'
  if p.is_file():value.update(stage=stage,completed_csv_epochs=max(0,len(p.read_bytes().splitlines())-1));break
 if (job/'exit.json').is_file():value.update(exit=read(job/'exit.json')['code'])
elif action=='verify':
 m=read(out/'manifest.json');assert read(job/'exit.json')['code']==0 and m['status']=='complete' and m['seed']==conf['seed'] and not m['final_test_executed']
 assert m['run']['status']=='complete' and m['run']['seed']==conf['seed']
 for n,v in m['files'].items():
  p=out/n;assert p.resolve().is_relative_to(out) and not p.is_symlink() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
 value={'sha256':sha(out/'manifest.json'),'files':len(m['files']),'run':m['run'],'log_sha256':sha(job/'training.log')}
elif action=='failure':
 # Keep the full durable work in place; only collect bounded evidence, never frozen images or an unfinished run tree.
 dst=root/'failure';dst.mkdir(exist_ok=True)
 import shutil
 for n in ('status.json','initialization.json','preflight/reuse.json','preflight/workers.json','preflight/estimate.json','preflight/train_monitor.json','preflight/train_metrics.csv','preflight/results.csv','probe/results.csv','work/results.csv'):
  p=out/n
  if p.is_file():q=dst/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 p=job/'training.log';value={'exit':read(job/'exit.json'),'log_sha256':sha(p),'last_lines':p.read_bytes()[-3500:].decode('utf-8','replace').splitlines()[-8:]}
else:raise ValueError(action)
print(json.dumps(value),flush=True)
'''


def remote(hub,m,c,action,timeout=120):
    ssh=hub.connection(m);enc=base64.b64encode(REMOTE.encode()).decode();cmd=shlex.join(['python3','-c',f'import base64;exec(base64.b64decode({enc!r}))',c['root'],action,json.dumps(c,separators=(',',':'))]);resp=gpu.invoke(['ssh',*ssh['options'],'-p',str(ssh['port']),f"{ssh['user']}@{ssh['host']}",cmd],timeout)
    if resp.returncode:raise gpu.GPUError(f'Alias remote {action} failed (exit {resp.returncode}): {resp.stderr[-650:]}')
    return json.loads(resp.stdout)


def pricing(hub,minutes=180):
    m=hub.machine();p=m.get('powerConfDto') or {};cfg=hub.config
    if p.get('id')!=cfg['power_conf_id'] or m.get('powerConfId')!=cfg['power_conf_id'] or p.get('cardNum')!=1 or '4090' not in str([p.get(k) for k in ('name','displayName','cardName','cardDisplayName','cardRealName')]):raise gpu.GPUError('Live GPU specification differs from bound 4090')
    if m.get('ssdPower') not in (0,0.,None) or m.get('diskPower') not in (0,0.,None):raise gpu.GPUError('Unexpected extra storage rate')
    if minutes not in (180,240):raise gpu.GPUError('Unregistered time window')
    rate=float(p['price']);cash=available_cash(hub.call('balance'));est=budget_limit(minutes,rate,min(cash,minutes/60*1.68+1e-8),cash)
    return m,{'rate_cny_per_hour':rate,'available_cash_cny':cash,'estimate_ceiling_cny':est,'max_minutes':minutes,'machine':hub.id,'spec_id':p['id']}


def prior_measurement(name):
    if name!='alias_001':raise gpu.GPUError('Only the measured first job is registered for reuse')
    base=ROOT/'runs'/name;r=read(base.with_suffix('.queue.json'));pre=base/'failure/preflight';e=read(pre/'estimate.json');obs=e['observation'];m=read(pre/'train_monitor.json');s=read(base/'failure/status.json')
    if not (r['seed']==29 and r['platform_state']=='Stopped' and r['worker_progress']['status']=='window_insufficient' and s.get('no_formal_training') is True and
            digest(base/'job/training.log')==r['failure']['log_sha256'] and obs==s['preflight']==r['worker_progress']['preflight'] and obs['epochs']==2 and
            digest(pre/'train_metrics.csv')==obs['train_metrics_sha256'] and digest(pre/'train_monitor.json')==obs['timing_sha256'] and m['epochs_in_segment']==2 and
            m['source_identity']['ready_sha256']==READY and re.fullmatch(r'/new/alias_[a-f0-9]{16}',r['remote_root'])):raise gpu.GPUError('Preflight evidence is not bound to the closed, no-training run')
    required=e['conservative_required_seconds_at_100_epochs_with_transfer']
    if not math.isfinite(required) or not 180*60<required<225*60 or required!=e['real_preflight_seconds']*50+720:raise gpu.GPUError('Measured preflight cost incompatible with the replacement window')
    return {'remote_run':r['remote_root']+'/run','estimate_sha256':digest(pre/'estimate.json'),'source_receipt_sha256':digest(base.with_suffix('.queue.json')),'required_seconds':required}


def check_qualification(name):
    if not name or not re.fullmatch(r'alias_\d{3}',name):raise gpu.GPUError('Require explicit qualifying seed 29 run')
    base=ROOT/'runs'/name;d=base/'dev';s=read(d/'manifest.json');v=read(d/'comparison.json');r=read(base.with_suffix('.queue.json'));weight=base/'run/29/weights/best.pt'
    if not (v.get('five_pass') is True and v['seed']==29 and s['seed']==29 and s['status']=='complete' and s['source_training_seal']==digest(base/'run/manifest.json')==r['verified_remote']['sha256'] and r['formal_seed']==29 and r['platform_state']=='Stopped' and s['checkpoint']['sha256']==digest(weight) and not s['final_test_executed']):raise gpu.GPUError('Seed 29 did not qualify on sealed source_dev')
    verify_inventory(d,s)


def close_owned(hub,machine,update):
    """Retry only an explicitly failed *read* before shutdown; never replay an ambiguous close request."""
    for attempt in range(4):
        current=owned(hub,machine)
        if current['state']=='Stopped':return current
        try:return gpu.close(hub,current,180)
        except gpu.GPUError as exc:
            if str(exc)!='SSH/GPU probe failed (exit 255); GPU usage is unknown' or attempt==3:raise
            update(shutdown_probe_retry=attempt+1,shutdown_probe_error=str(exc));time.sleep(4*(attempt+1))


def run(seed,name,scratch,reuse=None,qualification=None):
    if not re.fullmatch(r'alias_\d{3}',name) or seed not in (29,17,43) or not scratch.is_dir() or scratch.is_relative_to(ROOT):raise ValueError('Invalid run, seed or scratch')
    out=ROOT/'runs'/name;receipt=out.with_suffix('.queue.json');lock=out.with_suffix('.queue.lock')
    if out.exists() or receipt.exists():raise gpu.GPUError('Run or receipt exists; no duplicate submission')
    if seed!=29:check_qualification(qualification)
    prior=prior_measurement(reuse) if reuse else None;minutes=240 if prior else 180
    hub=gpu.BitaHub(read(gpu.CONFIG));m,price=pricing(hub,minutes)
    if m['state']!='Stopped':raise gpu.GPUError('Refusing to take over active instance')
    with tempfile.TemporaryDirectory(dir=scratch,prefix='alias-bundle-') as d:
        bundle=Path(d)/'alias.tar';pack(bundle)
        with tarfile.open(bundle,'r:') as ar:files={f:digest(ROOT/f) for f in ar.getnames()}
        state={'status':'prepared','seed':seed,'experiment':'coco_person_alias','budget':price,'machine_id':hub.id,'pid':os.getpid(),'bundle_sha256':digest(bundle),'controller_sha256':digest(Path(__file__)),'reused_preflight':prior,'qualification':qualification,'events':[],'billing_verified':False};machine=None
        def update(**kw):
            delta={k:v for k,v in kw.items() if state.get(k)!=v}
            if delta:
                state.update(delta);state['events'].append({'at_utc':now(),**delta});save(receipt,state);gpu.report(delta)
        with lock.open('x') as f:f.write(str(os.getpid()))
        try:
            save(receipt,state);gpu.report(price);machine,end=queue(hub,minutes,update,min_remaining_minutes=math.ceil(prior['required_seconds']/60)+10 if prior else 100)
            def current():return owned(hub,machine)
            def observe(action):
                for attempt in range(5):
                    try:return remote(hub,current(),c,action)
                    except (gpu.GPUError,OSError,ValueError):
                        if attempt==4 or time.time()+40>=end.timestamp():raise
                        time.sleep(8)
            for n in range(15):
                try:tele=hub.gpu(current());break
                except gpu.GPUError:
                    if n==14:raise
                    time.sleep(6)
            if tele['compute_processes'] or len(tele['gpus'])!=1 or '4090' not in tele['gpus'][0]['name']:raise gpu.GPUError('GPU busy or wrong device')
            token='alias_'+os.urandom(8).hex();root='/new/'+token
            c={'root':root,'token':token,'bundle':state['bundle_sha256'],'files':files,'seed':seed,'ready':READY,'coco':COCO,'python':DEFAULT_VENV,'source':DEFAULT_SOURCE,'weights':DEFAULT_WEIGHTS,'deadline':end.timestamp(),'reuse':prior}
            update(status='deploying',remote_root=root);copy(hub,current(),bundle,root+'.tar',timeout=120);started=remote(hub,current(),c,'deploy');update(status='worker_started',worker=started)
            while True:
                if time.time()>=end.timestamp():raise gpu.GPUError('End of platform window; persistent remote workspace retained')
                info=observe('observe');update(worker_progress=info)
                if 'exit' in info:break
                time.sleep(25)
            out.mkdir()
            if info['exit']!=0:
                failure=observe('failure');update(status='worker_failed',failure=failure);copy(hub,current(),root+'/failure',out,download=True);copy(hub,current(),root+'/job',out,download=True);raise gpu.GPUError('Worker ended with nonzero exit; failure evidence collected')
            verified=observe('verify');update(status='downloading',verified_remote=verified)
            for item in ('run','job'):
                copy(hub,current(),root+'/'+item,out,download=True,timeout=max(1,min(600,int(end.timestamp()-time.time()-30))))
            loc=out/'run';manifest=read(loc/'manifest.json');verify_inventory(loc,manifest)
            if digest(loc/'manifest.json')!=verified['sha256'] or digest(out/'job/training.log')!=verified['log_sha256']:raise gpu.GPUError('Local result/log seal differs from remote')
            update(status='download_verified',formal_seed=seed,local_dev='pending')
        except BaseException as exc:
            update(error=f'{type(exc).__name__}: {exc}');raise
        finally:
            try:
                if machine is not None:
                    cur=close_owned(hub,machine,update)
                    update(platform_state=cur['state'],shutdown_readback_utc=now())
                    update(available_cash_after_stop=available_cash(hub.call('balance')))
            finally:lock.unlink(missing_ok=True)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);p.add_argument('--result',required=True);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--reuse-preflight');p.add_argument('--qualification');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:run(a.seed,a.result,a.scratch.resolve(),a.reuse_preflight,a.qualification)
    else:
        prior=prior_measurement(a.reuse_preflight) if a.reuse_preflight else None
        gpu.report({'pricing':pricing(gpu.BitaHub(read(gpu.CONFIG)),240 if prior else 180)[1],'preflight':prior,'mutates':False})
