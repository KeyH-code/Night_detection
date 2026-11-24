"""Fixed-model readout ranking at the truck low-false-positive frontier."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
import math
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

import detection_metrics as dm
from source_contract import ContractError, NAMES, canonical_hash, check_equal, sha256, write_json

BASE_SHA='24975bacea26e9349d27886c624c97a3131176095af42950cf02275a1b68a2d2'
SHIFT_SHA='d5d625d4f743880a21ea0acc546a8873e28c6d83a3aec602f99f6b5ba6ef7dfa'
BUCKETS=('classification','classification_localization','localization','background_candidate')
CID=3


def pair_loss(positive,negative):
    return .10*F.softplus(math.log(2)+(negative-positive)).mean()


@dataclass(frozen=True)
class Box:
    level: int
    anchor: int
    score: float  # raw score before the correction, in logit units
    kind: str
    group: str
    gt: tuple | None
    feat: torch.Tensor


class Ledger:
    def __init__(self):
        self.history=deque(maxlen=32)

    def push(self,current):
        if len({x.anchor for x in current})!=len(current):
            raise ContractError('Duplicate native candidate in one exposure')
        result=[]
        for old in self.history:
            for x in current:
                for y in old:
                    if x.group==y.group or (x.kind=='tp')==(y.kind=='tp'): continue
                    p,n=(x,y) if x.kind=='tp' else (y,x)
                    if n.score>=p.score: result.append((p,n,x.kind=='tp'))
        self.history.append(tuple(current))
        return result


def failures(s):
    return [k for k,yes in (('tp_support',s['positive_gt']>=12),('fp_groups',s['negative_groups']>=8),
                           *((b,s['groups_by_bucket'][b]>=2) for b in BUCKETS[:2])) if not yes]


def inputs():
    import night_expand as exp
    import night_train as train
    from night_floor import balanced_inputs
    from night_contract import ROOT, _no_reparse, read_sealed_json

    _,recipe,identity,night,plan,ph,_=balanced_inputs()
    read_sealed_json(ROOT/'runs/truck_shift','manifest.json',SHIFT_SHA,[])
    parent=ROOT/'runs/expand_010'; seal=read_sealed_json(parent,'manifest.json',BASE_SHA,[])
    run,actual,rec=(exp.adapt._json(parent,seal,f) for f in ('run.json','43/train.json','43/inputs.json'))
    check_equal('prespecified parent and complete original training schedule',
                ('sealed','complete','e2',43,True,False,False,200,0,ph,[[d[0],n[0]] for d,n in plan],list(night),[d[0] for d,_ in plan]),
                (seal['status'],run['status'],run['arm'],run['seed'],bool(run.get('floor')),run['cloud_used'],run['final_test_executed'],
                 actual['updates'],actual['skipped_updates'],actual['augment_schedule_sha256'],actual['microbatches'],
                 [r['key'] for r in rec['records']],rec['source_keys']))
    w=parent/'43/last.pt'; _no_reparse(w)
    check_equal('exact fixed last weight',(actual['last_sha256'],actual['last_sha256']),
                (seal['files']['43/last.pt']['sha256'],sha256(w)))
    entry=identity['source_train']; doc=read_sealed_json(train.SOURCE,entry['path'],entry['sha256'],[])
    check_equal('only source-training metadata',('source_train',4000),(doc['purpose'],len(doc['records'])))
    day={r['key']:r for r in doc['records']}
    return w,run,recipe,night,day,plan,ph


@torch.no_grad()
def native_row(raw, logits, gt, cls):
    import torchvision
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import xywh2xyxy

    check_equal('original six-class 640 output',(1,10,8400),tuple(raw.shape))
    (out,),(idx,)=non_max_suppression(raw.clone(),.001,.7,nc=6,multi_label=True,max_det=300,
                                    max_nms=30000,return_idxs=True)
    idx=idx.reshape(-1).long(); pc=out[:,5].long()
    check_equal('native score, geometry and anchor round trip',True,
                torch.equal(out[:,:4],xywh2xyxy(raw[0,:4].T)[idx].to(out.dtype)) and
                torch.equal(out[:,4],logits[pc,idx].sigmoid().to(out.dtype)))
    gc=cls<5; keep=pc<5
    iou=box_iou(gt[gc].float(),out[keep,:4].float()).cpu().numpy()
    row={'gt':{'cls':cls[gc].tolist()},'pred':{'cls':pc[keep].tolist(),'conf':out[keep,4].tolist()},
         'iou_edges':[(int(i),int(j),float(iou[i,j])) for i,j in zip(*iou.nonzero(),strict=True)]}
    return row,idx[keep].tolist(),gc.nonzero().flatten().tolist()


def training_boundary(model, payload, until):
    import cv2
    from ultralytics.utils.ops import xywh2xyxy
    import night_train as train
    import night_accept as ac

    check_equal('fixed 300 training images',300,len(payload))
    rows=[]
    with torch.inference_mode():
        for path,w,h,labels in payload.values():
            if time.monotonic()>=until: raise ContractError('Boundary collection exceeded local budget')
            im=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
            check_equal('night boundary image dimensions',(h,w),None if im is None else im.shape[:2])
            x,targets=train.fit_image(im,labels,False,1.,640)
            raw,pred=model(torch.from_numpy(x).cuda().half().div_(255)[None])
            cls=torch.tensor([r[0] for r in labels],device='cuda',dtype=torch.long)
            gt=xywh2xyxy(torch.as_tensor(targets,device='cuda')*640.)
            row,_,_=native_row(raw,pred['scores'][0],gt,cls); rows.append(row)
    cv=dm.work_curve(rows,CID); work=ac._choose(cv,300,6)
    return {**work,'images':len(rows),'complete_training_gt':int(cv['gt'])}


class Frontier:
    def __init__(self,model,groups,threshold):
        from night_truck import score_residual

        self.model=model; self.groups=groups; self.score_residual=score_residual
        self.layers=[tower[-1] for tower in model.model[-1].cv3]
        check_equal('three ordinary final truck classifiers',[(6,(1,1),1,True)]*3,
                    [(x.out_channels,x.kernel_size,x.groups,x.bias is not None) for x in self.layers])
        device=next(model.parameters()).device
        self.delta=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(l.in_channels,device=device)) for l in self.layers])
        z=math.log(threshold/(1-threshold))
        self.low,self.high=[1/(1+math.exp(-(z+k))) for k in (-math.log(2),math.log(2))]
        self.features=[]
        self.hooks=[l.register_forward_pre_hook(lambda _,args:self.features.append(args[0])) for l in self.layers]
        self.ledger=Ledger(); self.calls=0; self.counts=Counter(); self.gt=set(); self.fp={k:set() for k in BUCKETS}

    def close(self):
        for h in self.hooks: h.remove()

    def candidates(self,batch,key,*,audit=False):
        from ultralytics.utils.ops import xywh2xyxy

        self.features.clear()
        raw,pred=self.model(batch['img'].half()); fs=tuple(self.features); self.features.clear()
        lengths=[f.shape[-2]*f.shape[-1] for f in fs]
        check_equal('ordered frozen paired 640 features',([6400,1600,400],(2,6,8400)),(lengths,tuple(pred['scores'].shape)))
        if audit:
            for f,l,d,s in zip(fs,self.layers,self.delta,pred['scores'].split(lengths,dim=-1),strict=True):
                zero=torch.cat((d,d.new_zeros(1)))
                check_equal('zero residual reproduces each native classifier',True,
                            torch.equal(F.conv2d(f,l.weight,l.bias).flatten(2),s) and
                            torch.equal(self.score_residual(s,f.flatten(2),zero),s))
            check_equal('zero residue retains exact raw decode',True,torch.equal(raw,self.model.model[-1]._inference(pred)))
        mask=batch['batch_idx']==1
        row,anchors,gtids=native_row(raw[1:],pred['scores'][1],xywh2xyxy(batch['bboxes'][mask]*640),batch['cls'].reshape(-1)[mask].long())
        _,pairs,keep=dm.image_point(row,CID,self.low)
        true={int(i):int(j) for j,i in pairs}; pp=[]; neg=defaultdict(list)
        for j in keep.tolist():
            score=row['pred']['conf'][j]
            if score>self.high:continue
            index=anchors[j]; level=0 if index<6400 else 1 if index<8000 else 2
            x=fs[level][1].flatten(1)[:,index-(0,6400,8000)[level]].detach().float()
            z=float(pred['scores'][1,CID,index]); group=self.groups[key]
            if j in true:
                pp.append((score,index,Box(level,index,z,'tp',group,(key,gtids[true[j]]),x)))
            else:
                kind=dm.error_type(row,j,pairs)['type']
                if kind in BUCKETS:
                    neg[kind].append((score,index,Box(level,index,z,kind,group,None,x)))
        return [x[2] for x in sorted(pp,key=lambda x:x[:2])[:3]] + [min(v,key=lambda x:(-x[0],x[1]))[2] for v in neg.values()]

    def objective(self,batch,key,*,audit=False):
        self.calls+=1
        with torch.no_grad():
            boxes=self.candidates(batch,key,audit=audit)
        self.counts['selected_positive']+=sum(x.kind=='tp' for x in boxes)
        for k in BUCKETS:self.counts['selected_'+k]+=sum(x.kind==k for x in boxes)
        pairs=self.ledger.push(boxes)
        if not pairs:return None
        banks=defaultdict(list); self.counts['paired_batches']+=1
        for p,n,hist_fp in pairs:
            self.gt.add(p.gt); self.fp[n.kind].add(n.group); self.counts['pairs_'+n.kind]+=1
            banks[n.kind].append((p,n,hist_fp))
        def logits(side):
            scores=torch.tensor([x.score for x in side],device=side[0].feat.device)
            if audit: scores.requires_grad_()
            updates=torch.stack([torch.dot(x.feat,self.delta[x.level]) for x in side])
            return scores,scores+updates
        losses=[]
        for items in banks.values():
            p,n,h=zip(*items,strict=True); src,pos=logits(p); rival,neg=logits(n)
            loss=pair_loss(pos,neg)
            if audit:
                gp,gn=torch.autograd.grad(loss,(src,rival),retain_graph=True)
                hs=(neg[0],n[0].level) if h[0] else (pos[0],p[0].level)
                history,=torch.autograd.grad(hs[0],self.delta[hs[1]],retain_graph=True)
                check_equal('both logit directions, nonzero historical residual and shift invariance',True,
                            bool((gp<0).all() and (gn>0).all() and (history!=0).any()) and torch.equal(gp,-gn) and
                            torch.allclose(loss,pair_loss(pos+1.,neg+1.),atol=1e-7,rtol=1e-6))
            losses.append(loss)
        value=torch.stack(losses).mean()
        if audit: self.counts['gradient_checked_batches']+=1
        return value.detach() if audit else value

    def summary(self):
        return {'calls':self.calls,'positive_gt':len(self.gt),'negative_groups':len(set().union(*self.fp.values())),
                'groups_by_bucket':{k:len(v) for k,v in self.fp.items()},'counts':dict(self.counts),'window':[self.low,self.high]}


def probe(out,scratch):
    import shutil
    from importlib.metadata import version
    import cv2
    from ultralytics.nn.tasks import load_checkpoint
    import night_train as train
    from night_tal import INDICES
    from seed_eval import probe_gpu

    start=time.monotonic(); gpu=probe_gpu(); weight,run,recipe,night,day,plan,plan_sha=inputs()
    check_equal('original library versions',run['versions'],{k:version(k) for k in train.LIBS})
    sample=[plan[i] for i in INDICES]
    src=[day[d[0]] for d,_ in sample]
    if any(r['purpose']!='source_train' or not r['image'].startswith('images/source_train/') or
           not r['label'].startswith('labels/source_train/') for r in src):raise ContractError('Forbidden training source record')
    with tempfile.TemporaryDirectory(dir=scratch,prefix='frontier-probe-') as td:
        box=Path(td); payload={}; reads={}
        for name,root,rs in (('source_train',train.SOURCE,src),('night_pool',train.NIGHT,list(night.values()))):
            staged,opened=train._stage(root,box/name,rs)
            check_equal('exact original training records only',[r[k] for r in rs for k in ('image','label')],opened)
            payload[name]=train.training_rows(box/name,staged,name)
            reads[name]={'images':len(rs),'files':len(opened),'paths_sha256':canonical_hash(opened)}
        torch.set_num_threads(4); cv2.setNumThreads(1)
        model,_=load_checkpoint(str(weight),device='cuda'); model.eval().half().requires_grad_(False)
        check_equal('frozen class order',NAMES,list(model.names.values()))
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        hook=None; used=[]
        with train.telemetry() as monitor:
            threshold=training_boundary(model,payload['night_pool'],start+600)
            if threshold['empty_output'] or not 0<threshold['threshold']<1:
                # This failure is a scientific gate, not an exception or permission to adjust the band.
                stats={'calls':0,'positive_gt':0,'negative_groups':0,'groups_by_bucket':{k:0 for k in BUCKETS},'counts':{},'window':None}
                bad=['empty_training_point']
            else:
                hook=Frontier(model,{k:r['group_id'] for k,r in night.items()},threshold['threshold'])
                data=payload['source_train']|payload['night_pool']
                try:
                    for i,m in zip(INDICES,sample,strict=True):
                        if time.monotonic()-start>=600:raise ContractError('Ten-minute frontier probe cap')
                        b=train._batch(torch,cv2,data,m,recipe['size'])
                        result=hook.objective(b,m[1][0],audit=True)
                        if result is not None and not bool(result.isfinite()):raise ContractError('Nonfinite probe loss')
                        used.append(i)
                        if len(used)%50==0:print(f'Truck frontier no-update probe {len(used)}/200',flush=True)
                finally:hook.close()
                stats=hook.summary();bad=failures(stats)
                check_equal('no residual updates',0,sum(int(d.count_nonzero()) for d in hook.delta))
                check_equal('one direct audit per paired batch',stats['counts'].get('paired_batches',0),stats['counts'].get('gradient_checked_batches',0))
        check_equal('frozen model and no model gradients',True,all(torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())
                    and all(x.grad is None for x in model.parameters()))
        v={**stats,'point':threshold,'failures':bad,'met':not bad,'indices':used,'updates':0,'weights_unchanged':True,
           'base_manifest_sha256':BASE_SHA,'base_checkpoint_sha256':sha256(weight),'shift_sha256':SHIFT_SHA,
           'plan_sha256':plan_sha,'sample_plan_sha256':canonical_hash(sample),'expand_sha256':run['config_sha256'],
           'reads':reads,'development_read':False,'holdout_read':False,'cloud_used':False,'versions':run['versions'],
           'gpu':monitor,'initial_gpu':gpu,'seconds':time.monotonic()-start}
        e=box/'evidence';e.mkdir();shutil.copyfile(__file__,e/'night_frontier.py')
        write_json(e/'metrics.json',v);write_json(e/'manifest.json',{'status':'sealed','files':train._receipt(e)})
        e.rename(out)
    return verify_probe(out,sha256(out/'manifest.json'))


def verify_probe(path,digest):
    from night_contract import read_sealed_bytes,read_sealed_json
    from night_tal import INDICES

    m=read_sealed_json(path,'manifest.json',digest,[]); files=m['files']
    check_equal('exact frontier evidence',('sealed',{'metrics.json','night_frontier.py'}),(m['status'],set(files)))
    raw={k:read_sealed_bytes(path,k,v['sha256'],[]) for k,v in files.items()}
    check_equal('sealed lengths',{k:x['bytes'] for k,x in files.items()},{k:len(v) for k,v in raw.items()})
    v=json.loads(raw['metrics.json']); empty=v['point']['empty_output'] or not 0<v['point']['threshold']<1
    bad=['empty_training_point'] if empty else failures(v)
    check_equal('verified original training-only scope and gate',
                (v['point']['images'],v['point']['fp_budget'],v['indices'],v['calls'],v['updates'],v['weights_unchanged'],
                 v['base_manifest_sha256'],v['shift_sha256'],v['development_read'],v['holdout_read'],v['cloud_used'],v['failures'],v['met']),
                (300,6,[] if empty else list(INDICES),0 if empty else 200,0,True,BASE_SHA,SHIFT_SHA,False,False,False,bad,not bad))
    check_equal('one audit for every actual pair-producing batch',v['counts'].get('paired_batches',0),v['counts'].get('gradient_checked_batches',0))
    return {**v,'manifest_sha256':digest,'code_sha256':files['night_frontier.py']['sha256']}


def main():
    import argparse,re
    from night_contract import ROOT,_no_reparse

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('probe','verify-probe'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path);p.add_argument('--manifest-sha256')
    a=p.parse_args();out=a.output.absolute();existing=a.mode=='verify-probe'
    if out.parent!=ROOT/'runs' or not re.fullmatch(r'frontier_\d{3}',out.name) or out.exists()!=existing or out.is_symlink():
        p.error('Expected a fresh or sealed runs/frontier_NNN')
    _no_reparse(out if existing else out.parent)
    if existing:
        if not a.manifest_sha256 or a.scratch:p.error('verify requires seal and no scratch')
        result=verify_probe(out,a.manifest_sha256)
    else:
        if not a.scratch or a.manifest_sha256:p.error('probe requires external scratch and no seal')
        scratch=a.scratch.absolute();_no_reparse(scratch)
        if not scratch.is_dir() or scratch.resolve()!=scratch or scratch.is_relative_to(ROOT) or scratch.drive!=out.drive:
            p.error('Expected external scratch on the same drive')
        result=probe(out,scratch)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
