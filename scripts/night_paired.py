"""S7: matched 340-image training and development replay on the sealed supplement."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import night_accept as ac
import night_adapt as adapt
import night_expand as exp
import night_material as mat
import night_supplement as blind
import night_train as tr
from night_contract import ROOT, _no_reparse, read_sealed_bytes, read_sealed_json, verify_dev
from source_contract import canonical_hash, check_equal, read_json, sha256, write_json

RUN=ROOT/'runs'; PAIR=RUN/'supp_002'; ARMS={'targeted':'expand_013','random':'expand_014'}
MAT='686497991c7290276d45be3f45db647ed5909a379bb7e2b48ac380780dafbbc3'
SPEC=ROOT/'specs/s7/paired.md'; SEED=43


def order(tag,value):return blind.hashed(f'paired/43/{tag}/{value}')


def schedules(original,selected,groups):
    by=defaultdict(list)
    for i,(_,n) in enumerate(original):by[n[0]].append(i)
    check_equal('original 300 exposure budget',(300,2000,{6,7}),(len(by),len(original),set(map(len,by.values()))))
    five=set(sorted(by,key=lambda k:(order('five',k),k))[:40]);vacant=sorted(i for k,idx in by.items()
        for i in sorted(idx,key=lambda i:(order('replace',i),i))[:len(idx)-(5 if k in five else 6)])
    t,r=(selected[a] for a in ARMS); common=set(t)&set(r);aligned={k:k for k in common};f=lambda k:blind.fold(groups[k])
    for j in range(5):
        sides=[sorted([k for k in a if f(k)==j and k not in common],key=lambda k:(order('match',k),k)) for a in (t,r)]
        aligned.update(zip(*sides,strict=True))
    check_equal('supplement scope',(240,40,40,4,True),(len(vacant),len(set(t)),len(set(r)),len(common),set(by).isdisjoint(t+r)))
    keys=[k for j in range(6) for k in sorted(t,key=lambda k:(order('cycle'+str(j),k),k))];result={}
    for arm,mapped in (('targeted',keys),('random',[aligned[k] for k in keys])):
        replacements=dict(zip(vacant,mapped,strict=True));plan=[(d,(replacements.get(i,n[0]),*n[1:])) for i,(d,n) in enumerate(original)]
        count=Counter(n[0] for _,n in plan);check_equal('all actual night exposures',{**{k:5 if k in five else 6 for k in by},**{k:6 for k in selected[arm]}},dict(count));result[arm]=plan
    check_equal('304 shared images slot-aligned',1784,sum(a==b for a,b in zip(result['targeted'],result['random'],strict=True)))
    return result


def load_inputs():
    base=RUN/'expand_010';bs=mat.parent('expand_010');get=lambda n:blind.sealed(base,bs,n)
    state=get('run.json');orig=get('43/inputs.json');train=get('43/train.json');sel=get('selection.json');_,_,pair,recipe=exp.configuration(state['config_sha256'])
    selected,novel=mat.scope();inv=blind.inventory();ms=read_sealed_json(mat.OUT,'manifest.json',MAT,[]);opened=blind.sealed(mat.OUT,ms,'openings.json')
    check_equal('parental input identities',(mat.LOCK,state['a_sha256'],state['floor']['teacher_sha256']),
                (opened['lock'],sha256(tr.CHECKPOINT),sha256(RUN/'expand_003/43/last.pt')))
    source=state['source_train'];sdata=read_sealed_json(tr.SOURCE,source['path'],source['sha256'],[]);day={r['key']:r for r in sdata['records']}
    check_equal('only day training inventory',('source_train',4000,{'source_train'}),(sdata['purpose'],len(day),{v['purpose'] for v in day.values()}))
    draw=read_sealed_json(RUN/'pool_001','random.json',recipe['random_sha256'],[]);d100=[r['key'] for r in draw['arms']['43']['records']]
    original=exp.expanded_plan(adapt._schedule(SEED,list(day),d100,recipe),sel['slots'],SEED,recipe['schedule_salt'])
    check_equal('verified original scheduling',(train['augment_schedule_sha256'],train['microbatches'],orig['source_keys']),
                (canonical_hash(original),[[d[0],n[0]] for d,n in original],[d[0] for d,n in original]))
    night={}
    for r,image,label in zip(orig['records'],orig['source_reads'][::2],orig['source_reads'][1::2],strict=True):
        k=r['key'];im=inv[k];check_equal('base frozen night payload',(im['image'],mat.label_path(im),im['image_sha256'],im['group_id'],'night_pool'),(image,label,r['image_sha256'],r['group_id'],r['purpose']))
        night[k]=dict(r,image=image,label=label)
    check_equal('300 base slots',sel['slots'],list(night));check_equal('sealed openings',set(novel),{r['key'] for r in opened['receipts']})
    for r in opened['receipts']:
        im=novel[r['key']];check_equal('selected image and label identity',(im['image'],im['image_sha256'],im['group_id'],mat.label_path(im)),tuple(r[k] for k in ('image','image_sha256','group_id','label')))
        night[r['key']]=dict(r,purpose='night_pool')
    plans=schedules(original,selected,{k:x['group_id'] for k,x in night.items()});proto=tr.load_protocol();dev_groups=set()
    for purpose,root in (('source_dev',tr.SOURCE),('night_dev',tr.NIGHT)):
        entry=state[purpose];ds=read_sealed_json(root,entry['path'],entry['sha256'],[]);dev_groups.update(r['group_id'] for r in ds['records'])
    reserved=RUN/'reserve_001';groups=blind.sealed(reserved,read_sealed_json(reserved,'manifest.json',blind.RESERVE,[]),'roles.json')['review_groups']
    for arm,plan in plans.items():
        used={night[n[0]]['group_id'] for _,n in plan};days={day[d[0]]['group_id'] for d,_ in plan};check_equal('no group overlap',(340,set(),set()),(len(used),used&set(groups),(used|days)&dev_groups))
        check_equal('matched day keys',orig['source_keys'],[d[0] for d,n in plan]);check_equal('locked 340 keys',set(sel['slots']+selected[arm]),{n[0] for _,n in plan})
    return state,recipe,pair,proto,selected,plans,day,night


def lock(ctx):
    state,recipe,_,_,sel,plans,_,_=ctx
    return dict(spec_sha256=sha256(SPEC),material=MAT,selection=mat.LOCK,reserve=blind.RESERVE,control=mat.PARENTS['expand_010'],seed=SEED,
        initializer=state['a_sha256'],teacher=state['floor']['teacher_sha256'],recipe=recipe,floor=state['floor'],budget_minutes_per_arm=35,
        arms={a:dict(run=ARMS[a],added=sel[a],plan_sha256=canonical_hash(p),day_and_augmentation_sha256=canonical_hash([(d,n[1:]) for d,n in p]),
                    exposures=dict(Counter(n[0] for _,n in p))) for a,p in plans.items()})


def stage(box,root,records,purpose,cv2):
    dest=box/purpose;prepared=[];reads=[]
    for r in records:
        k=r['key'];check_equal('only explicitly approved purpose',purpose,r['purpose']);prefix='images/'+purpose+'/'
        blind.require(r['image'].startswith(prefix) and r['label'].startswith(('hidden_ground_truth/' if purpose=='night_pool' else 'labels/')+purpose+'/'),'outside training split')
        s=dict(r)
        for name,extension in (('image','.jpg'),('label','.txt')):
            raw=read_sealed_bytes(root,r[name],r[name+'_sha256'],reads);s[name]=f'{name}s/{k.replace(":","_")}{extension}';p=dest/s[name];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            if name=='image' and 'width' not in s:
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);s['height'],s['width']=im.shape[:2]
            if name=='label' and 'classes' not in s:s['classes']=sorted({x[0] for x in tr._labels(raw)})
        prepared.append(s)
    check_equal('only ordered grant was opened',[r[f] for r in records for f in ('image','label')],reads)
    return tr.training_rows(dest,prepared,purpose),dict(records=prepared,source_reads=reads)


def preflight(torch,cv2,ctx,payload,arm):
    from ultralytics.nn.tasks import load_checkpoint
    from types import SimpleNamespace
    from ultralytics.utils.loss import v8DetectionLoss
    from night_floor import FloorLoss,reference_model,native_id
    state,recipe,_,_,sel,plans,*_=ctx;p=plans[arm];added=set(sel[arm]);idx=[];seen=set()
    for i,(_,n) in enumerate(p):
        if i<64 or (n[0] in added and n[0] not in seen):idx.append(i);seen.add(n[0])
    check_equal('all 40 new images in preflight',added,added&seen)
    model,_=load_checkpoint(str(tr.CHECKPOINT),device='cuda');model.train().requires_grad_(False);model.args=SimpleNamespace(**dict(model.args))
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
    teacher=reference_model(state['floor']['teacher_sha256']);native=v8DetectionLoss(model);check_equal('exact native floor pipeline',state['floor']['native_code_sha256'],native_id(model.model[-1],native))
    snaps=[[v.detach().cpu().clone() for v in m.state_dict().values()] for m in (model,teacher)];hook=FloorLoss(model.model[-1],teacher,[p[i][1][0] for i in idx])
    with torch.no_grad():
        for i in idx:
            batch=tr._batch(torch,cv2,payload,p[i],recipe['size'])
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model(batch['img']);_,terms,_=native.get_assigned_targets_and_loss(pred,batch);extra=hook.extra(pred,batch,audit=True)
            blind.require(bool(terms.isfinite().all() and extra.isfinite() and not extra.requires_grad and not terms.requires_grad),'Invalid no-update probe')
    blind.require(all(torch.equal(v.cpu(),s) for m,ss in zip((model,teacher),snaps) for v,s in zip(m.state_dict().values(),ss,strict=True)),'Probe changed model state')
    stats=hook.summary();blind.require(stats['gradient_checked_batches']>0 and stats['active_gt']>0,'No actual isolated floor gradient')
    return dict(updates=0,new_images=40,weights_unchanged=True,microbatch_indices=idx,**stats)


def codes(out):
    dest=out/'code';dest.mkdir();files={Path(v.__file__).resolve() for v in list(sys.modules.values()) if getattr(v,'__file__',None) and str(v.__file__).endswith('.py') and Path(v.__file__).resolve().parent==ROOT/'scripts'}
    for f in files:shutil.copyfile(f,dest/f.name)


def local_device():
    from seed_eval import probe_gpu
    from source_contract import ContractError
    for attempt in range(7):
        try:return probe_gpu()
        except ContractError as exc:
            if not str(exc).startswith('GPU busy or insufficient free memory') or attempt==6:raise
            print('Waiting for original local GPU admission:',exc,flush=True);time.sleep(20)


def run_arm(arm,scratch,ctx):
    import cv2,torch
    from night_floor import FloorLoss,reference_model
    from night_focus import fit_native
    state,recipe,pair,proto,_,plans,day,night=ctx;p=plans[arm];out=RUN/ARMS[arm];statefile=out/'run.json';phase=out/'43'
    check_equal('locked joint plan',lock(ctx),read_json(PAIR/'protocol.json'));check_equal('pinned train environment',state['versions'],{n:version(n) for n in tr.LIBS})
    gpu=local_device();started=time.monotonic();until=started+35*60;out.mkdir();phase.mkdir();torch.set_num_threads(4);cv2.setNumThreads(1)
    actual=dict(status='incomplete',arm=arm,seed=43,cloud_used=False,final_test_executed=False,initial_gpu=gpu,protocol_sha256=sha256(PAIR/'protocol.json'),versions=state['versions'],python=sys.version)
    write_json(statefile,actual)
    try:
        with tempfile.TemporaryDirectory(dir=scratch,prefix='paired-arm-') as tmp:
            box=Path(tmp);days=[day[d[0]] for d,_ in p];nights=[night[k] for k in dict.fromkeys(n[0] for _,n in p)]
            dp,dr=stage(box,tr.SOURCE,days,'source_train',cv2);npay,nr=stage(box,tr.NIGHT,nights,'night_pool',cv2)
            check_equal('disjoint verified fit payloads',(2000,340,set()),(len(dp),len(npay),set(dp)&set(npay)))
            write_json(phase/'inputs.json',dict(night=nr,source_keys=list(dp),source_manifest=state['source_train'],source_reads_sha256=canonical_hash(dr['source_reads']),schedule_sha256=canonical_hash(p)))
            probe=preflight(torch,cv2,ctx,dp|npay,arm);write_json(phase/'preflight.json',probe);print(arm,'preflight passed',probe['calls'],'batches',probe['gradient_checked_batches'],'gradient checks',flush=True)
            torch.cuda.empty_cache();hooks=[]
            def setup(model,criterion):
                h=FloorLoss(model.model[-1],reference_model(state['floor']['teacher_sha256']),[n[0] for _,n in p]);h.attach(criterion);hooks.append(h)
            with tr.telemetry() as usage:
                w,t=fit_native(torch,cv2,{'day':dp,'night':npay},p,SEED,recipe,{'updates':200,'model_objective':'floor'},until,phase,model_criterion_setup=setup)
            sig=hooks.pop().summary();check_equal('actual objective applied',2000,sig['calls']);blind.require(sig['active_gt']>0,'No active floor supervision')
            write_json(phase/'train.json',dict(**t,floor_signal=sig,gpu=usage));del dp,npay;torch.cuda.empty_cache();torch.use_deterministic_algorithms(False)
            dv=tr.verify_subset(tr.SOURCE,proto['data']['source_ready_sha256'],'source_dev');nt=verify_dev(tr.NIGHT,proto['data']['night_ready_sha256'])
            check_equal('official development identities',(state['source_dev'],state['night_dev']),(dv['manifest'],nt['manifest']))
            policy,ref,_,d_pairs,a_day=adapt._reference(pair,proto,dv,nt,[43]);_,a=adapt._evaluate(torch,w,phase/'dev',box/'eval',nt,dv,policy,ref,d_pairs[43],a_day,until)
            actual.update(status='complete',source_dev=dv['manifest'],night_dev=nt['manifest']);print(arm,'night',a['classes_met'],'day protection',a['day_protection']['met'],flush=True)
    except BaseException as e:
        actual['error']=repr(e);raise
    finally:
        actual['seconds']=time.monotonic()-started;write_json(statefile,actual);codes(out);write_json(out/'manifest.json',dict(status=actual['status'],files=tr._receipt(out)))


def execute(scratch):
    _no_reparse(scratch);blind.require(scratch==scratch.resolve() and not scratch.is_relative_to(ROOT),'External scratch required');_no_reparse(RUN)
    ctx=load_inputs();protocol=lock(ctx)
    if not PAIR.exists():PAIR.mkdir();write_json(PAIR/'protocol.json',protocol)
    else:check_equal('already registered pair',protocol,read_json(PAIR/'protocol.json'))
    with tempfile.TemporaryDirectory(dir=scratch,prefix='paired-env-') as tmp:
        for k in ('YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','TEMP','TMP','TMPDIR','XDG_CONFIG_HOME'):
            pp=Path(tmp)/k.lower();pp.mkdir();os.environ[k]=str(pp)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for arm,name in ARMS.items():
            if (RUN/name).exists():
                state=read_json(RUN/name/'run.json');check_equal('only skip completed sealed same arm',('complete',arm,sha256(PAIR/'protocol.json')),(state['status'],state['arm'],state['protocol_sha256']));continue
            run_arm(arm,scratch,ctx)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scratch',type=Path,required=True);args=parser.parse_args();execute(args.scratch.absolute())
