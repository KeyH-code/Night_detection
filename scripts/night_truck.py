"""Calibrate only the truck score row of the sealed positive-floor detector."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import tempfile
import time

import torch
import torch.nn.functional as F

import detection_metrics as dm
from source_contract import ContractError, NAMES, canonical_hash, check_equal, sha256, write_json

CID = 3
PARENTS = {'base':('expand_010','24975bacea26e9349d27886c624c97a3131176095af42950cf02275a1b68a2d2'),
           'reference':('expand_006','26ae15193a002a9e63238fa93e13bd8ab8cbcf92c8c9f78344bb824eb95ffc84')}
PAIR_SHA = 'e83165b54fba9b5da0c77a434d55262616dd58fe2efef8be7b31733f51db9413'


def score_residual(logits, feats, delta):
    adjust = (feats.detach().float() * delta[:-1][None,:,None]).sum(1) + delta[-1]
    return torch.cat((logits[:,:CID],(logits[:,CID].float()+adjust).to(logits.dtype)[:,None],logits[:,CID+1:]),dim=1)


def failures(stats):
    return [k for k,passed in (('active_support',stats['active_gt']>=12),('near_support',stats['near_gt']>=6)) if not passed]


@torch.no_grad()
def select_pairs(reference, reference_logits, base, gt):
    import torchvision
    import numpy as np
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import xywh2xyxy

    check_equal('same signed six-class anchor grid',tuple(reference.shape),tuple(base.shape))
    (out,),(idx,)=non_max_suppression(reference.clone(),.001,.7,nc=6,multi_label=True,max_det=300,
                                    max_nms=30000,return_idxs=True)
    idx=idx.reshape(-1).long(); classes=out[:,5].long()
    check_equal('exact NMS geometry and score/index round trip',True,
                torch.equal(out[:,:4],xywh2xyxy(reference[0,:4].T)[idx].to(out.dtype)) and
                torch.equal(out[:,4],reference_logits[classes,idx].sigmoid().to(out.dtype)))
    truck=classes==CID; anchors=idx[truck]
    overlaps=box_iou(gt.float(),out[truck,:4].float())
    match=dm.native_pairs(np.full(len(anchors),CID),np.full(len(gt),CID),overlaps.cpu().numpy())
    targets,pred=torch.as_tensor(match,device=gt.device).T; anchors=anchors[pred]
    aligned=box_iou(gt.float(),xywh2xyxy(base[0,:4].T).float())
    same=aligned[targets,anchors]>=.5
    selected=anchors[same]
    return selected,targets[same],reference_logits[CID,selected].detach()


def approved():
    import night_expand as exp
    import night_train as train
    from night_contract import ROOT, _no_reparse, read_sealed_json
    from night_floor import balanced_inputs

    _,recipe,identity,night,plan,plan_sha,_=balanced_inputs()
    read_sealed_json(ROOT/'runs/truck_pair','manifest.json',PAIR_SHA,[])
    data={}; same=[]; versions=None; previous=None
    for key,(name,digest) in PARENTS.items():
        path=ROOT/'runs'/name; seal=read_sealed_json(path,'manifest.json',digest,[])
        state,choice,inputs,t=(exp.adapt._json(path,seal,k) for k in ('run.json','selection.json','43/inputs.json','43/train.json'))
        check_equal('sealed fixed last of correct treatment',('sealed','complete','e2',43,False,False,True,200,0),
                    (seal['status'],state['status'],state['arm'],state['seed'],state['cloud_used'],state['final_test_executed'],
                     bool(state.get('floor' if key=='base' else 'margin')),t['updates'],t['skipped_updates']))
        check_equal('original augmented slots, source and ordered night whitelist',
                    (plan_sha,[[d[0],n[0]] for d,n in plan],list(night),[d[0] for d,_ in plan]),
                    (t['augment_schedule_sha256'],t['microbatches'],[r['key'] for r in inputs['records']],inputs['source_keys']))
        same.append((choice,inputs,state['versions'],state['config_sha256'],state['a_sha256']))
        weight=path/'43/last.pt'; _no_reparse(weight)
        check_equal('sealed exact EMA weights',(t['last_sha256'],t['last_sha256']), (seal['files']['43/last.pt']['sha256'],sha256(weight)))
        data[key]={'path':weight,'last_sha256':t['last_sha256'],'manifest_sha256':digest}
        versions=state['versions']; previous=state
    check_equal('unchanged control training identity and only sealed prespecified parents',same[0],same[1])
    entry=identity['source_train']; doc=read_sealed_json(train.SOURCE,entry['path'],entry['sha256'],[])
    check_equal('authorized train metadata only',('source_train',4000),(doc['purpose'],len(doc['records'])))
    day={r['key']:r for r in doc['records']}
    return data,previous,versions,recipe,identity,plan,plan_sha,day,night


def stage(box,plan,day,night):
    import night_train as train

    src=[day[d[0]] for d,_ in plan]; dst=[night[k] for k in dict.fromkeys(n[0] for _,n in plan)]
    if any(r['purpose']!='source_train' or not r['image'].startswith('images/source_train/') or
           not r['label'].startswith('labels/source_train/') for r in src):
        raise ContractError('Source payload outside frozen training scope')
    data={}; hashes={}
    for name,root,records in (('source_train',train.SOURCE,src),('night_pool',train.NIGHT,dst)):
        staged,reads=train._stage(root,box/name,records)
        check_equal('authorized payloads only',[r[k] for r in records for k in ('image','label')],reads)
        data.update(train.training_rows(box/name,staged,name))
        hashes[name]={'images':len(records),'files':len(reads),'paths_sha256':canonical_hash(reads)}
    return data,hashes


class Calibration:
    def __init__(self, parents):
        from ultralytics.nn.tasks import load_checkpoint

        self.base, self.ref = [load_checkpoint(str(parents[k]['path']),device='cuda')[0].eval().half().requires_grad_(False)
                               for k in ('base','reference')]
        head=self.base.model[-1]; ref=self.ref.model[-1]
        check_equal('original class and feature grid',(NAMES,6,False,[8.,16.,32.]),
                    (list(self.base.names.values()),head.nc,head.end2end,head.stride.tolist()))
        check_equal('same ordered six-class model layout',(type(head),list(self.base.names.values()),head.stride.tolist()),
                    (type(ref),list(self.ref.names.values()),ref.stride.tolist()))
        self.layers=[seq[-1] for seq in head.cv3]
        check_equal('three ordinary 1x1 conv classifier rows',[(6,(1,1),1,True)]*3,
                    [(c.out_channels,c.kernel_size,c.groups,c.bias is not None) for c in self.layers])
        self.features=[]
        self.hooks=[layer.register_forward_pre_hook(lambda _,args:self.features.append(args[0])) for layer in self.layers]
        self.delta=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(c.in_channels+1,device='cuda')) for c in self.layers])
        self.sets={k:set() for k in ('eligible','active','near')}; self.exposures=Counter(); self.scales=Counter(); self.count=0

    def close(self):
        for h in self.hooks: h.remove()

    def objective(self,batch,key,*,audit=False):
        from ultralytics.utils.ops import xywh2xyxy

        self.count+=1
        self.features.clear()
        with torch.no_grad():
            x=batch['img'].half()
            check_equal('original ordered paired training shape',(2,3,640,640),tuple(x.shape))
            raw,pred=self.base(x); fs=tuple(self.features); self.features.clear()
            check_equal('native multi-level grid layout',([(80,80),(40,40),(20,20)],(2,6,8400)),
                        ([tuple(f.shape[-2:]) for f in fs],tuple(pred['scores'].shape)))
            scores=pred['scores']; parts=scores.split([f.shape[-2]*f.shape[-1] for f in fs],dim=-1)
            if audit:
                for layer,f,part,d in zip(self.layers,fs,parts,self.delta,strict=True):
                    check_equal('native row and zero adjustment identity',True,
                                torch.equal(F.conv2d(f,layer.weight,layer.bias).flatten(2),part) and
                                torch.equal(score_residual(part,f.flatten(2),d),part))
                decoded=self.base.model[-1]._inference(pred)
                check_equal('native full raw boxes and score order exactly replay',True,torch.equal(raw,decoded))
            night=batch['batch_idx']==1
            idx=(batch['cls'].reshape(-1)[night]==CID).nonzero().flatten()
            if not idx.numel(): return None
            gt=xywh2xyxy(batch['bboxes'][night][idx]*640)
            other,otr=self.ref(x[1:]);
            check_equal('reference order of grid levels',[tuple(f.shape[-2:]) for f in fs], [tuple(f.shape[-2:]) for f in otr['feats']])
            a,g,reference=select_pairs(other,otr['scores'][0],raw[1:],gt)
            if not a.numel(): return None
            for target,gap in zip(idx[g].tolist(),(reference.float()-scores[1,CID,a].float()).tolist(),strict=True):
                identity=key,target
                for k,ok in (('eligible',True),('active',gap>0),('near',0<gap<=math.log(2))):
                    if ok: self.sets[k].add(identity); self.exposures[k]+=1
        adjusted=[]; observers=[]; start=0
        for l,(f,part,d) in enumerate(zip(fs,parts,self.delta,strict=True)):
            n=part.shape[-1]; pick=(a>=start)&(a<start+n); j=a[pick]-start
            if j.numel():
                self.scales[l]+=j.numel()
                pairs=part[:,:,j].detach().requires_grad_(audit)
                row=score_residual(pairs,f.flatten(2)[:,:,j],d)
                adjusted.append((reference[pick].float()-row[1,CID].float()).relu())
                if audit:
                    observers.append((pairs,reference[pick].float()>part[1,CID,j].float()))
            start+=n
        loss=.10*torch.cat(adjusted).mean()
        if audit:
            scores_to_check, masks=zip(*observers,strict=True)
            grads=torch.autograd.grad(loss,tuple(scores_to_check)+tuple(self.delta),allow_unused=True)
            for g,audit_mask in zip(grads[:len(masks)],masks,strict=True):
                wanted=torch.zeros_like(g); wanted[1,CID,audit_mask]=-1
                check_equal('only lower truck night score has direct gradient',True,torch.equal(g.sign(),wanted))
            check_equal('parent, geometry and mentor are detached', (False,False,False),
                        (raw.requires_grad,pred['boxes'].requires_grad,reference.requires_grad))
            if not all(g is None or bool(g.isfinite().all()) for g in grads): raise ContractError('Nonfinite residual gradient')
            self.exposures['gradient_checked_batches']+=1
        return loss.detach() if audit else loss

    def summary(self):
        return {'calls':self.count,**{k+'_gt':len(v) for k,v in self.sets.items()},'exposures':dict(self.exposures),
                'scales':{str(k):self.scales[k] for k in range(3)}}


def seal_probe(out, scratch):
    import shutil
    from importlib.metadata import version
    import cv2
    import night_train as train
    from night_tal import INDICES
    from seed_eval import probe_gpu

    started=time.monotonic(); gpu=probe_gpu()
    parents,st,versions,recipe,_,plan,plan_sha,day,night=approved()
    check_equal('original installed package versions',versions,{k:version(k) for k in train.LIBS})
    sample=[plan[i] for i in INDICES]
    with tempfile.TemporaryDirectory(dir=scratch,prefix='truck-probe-') as td:
        box=Path(td); payload,reads=stage(box,sample,day,night)
        torch.set_num_threads(4); cv2.setNumThreads(1)
        cal=Calibration(parents)
        origin=[{k:v.detach().cpu().clone() for k,v in a.state_dict().items()} for a in (cal.base,cal.ref)]
        try:
            with train.telemetry() as monitor:
                for j,m in enumerate(sample,1):
                    if time.monotonic()-started>=600: raise ContractError('Ten-minute truck probe cap')
                    b=train._batch(torch,cv2,payload,m,recipe['size'])
                    cal.objective(b,m[1][0],audit=True)
                    if j%50==0: print(f'Truck residual probe: {j}/200',flush=True)
        finally: cal.close()
        kept=all(torch.equal(v.detach().cpu(),s[k]) for a,s in zip((cal.base,cal.ref),origin,strict=True) for k,v in a.state_dict().items())
        check_equal('no parent or residual weight changes',(True,0),(kept,sum(int(d.count_nonzero()) for d in cal.delta)))
        result=cal.summary(); bad=failures(result)
        rec={**result,'failures':bad,'met':not bad,'plan_sha256':plan_sha,'sample_plan_sha256':canonical_hash(sample),
             'expand_sha256':st['config_sha256'],'parents':{k:{x:y for x,y in v.items() if x!='path'} for k,v in parents.items()},
             'pair_sha256':PAIR_SHA,'indices':list(INDICES),'reads':reads,'updates':0,'weights_unchanged':kept,
             'development_read':False,'holdout_read':False,'cloud_used':False,'versions':versions,
             'initial_gpu':gpu,'gpu':monitor,'seconds':time.monotonic()-started}
        evidence=box/'evidence';evidence.mkdir();shutil.copyfile(__file__,evidence/'night_truck.py')
        write_json(evidence/'metrics.json',rec)
        write_json(evidence/'manifest.json',{'status':'sealed','files':train._receipt(evidence)})
        evidence.rename(out)
    return verify_probe(out,sha256(out/'manifest.json'))


def verify_probe(path, digest):
    from night_contract import read_sealed_bytes,read_sealed_json
    from night_tal import INDICES

    s=read_sealed_json(path,'manifest.json',digest,[])
    check_equal('exact truck probe evidence',('sealed',{'metrics.json','night_truck.py'}),(s['status'],set(s['files'])))
    raw={k:read_sealed_bytes(path,k,v['sha256'],[]) for k,v in s['files'].items()}
    check_equal('sealed file lengths',{k:x['bytes'] for k,x in s['files'].items()},{k:len(b) for k,b in raw.items()})
    v=json.loads(raw['metrics.json']); bad=failures(v)
    check_equal('verified isolated no-update support gate',(list(INDICES),200,0,True,False,False,False,PAIR_SHA,bad,not bad),
                (v['indices'],v['calls'],v['updates'],v['weights_unchanged'],v['development_read'],v['holdout_read'],v['cloud_used'],
                 v['pair_sha256'],v['failures'],v['met']))
    if v['met'] and not v['exposures'].get('gradient_checked_batches'): raise ContractError('Missing isolated gradient proof')
    return {**v,'manifest_sha256':digest,'code_sha256':s['files']['night_truck.py']['sha256']}


def main():
    import argparse,re
    from night_contract import ROOT,_no_reparse

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('probe','verify-probe'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path);p.add_argument('--manifest-sha256')
    a=p.parse_args();out=a.output.absolute();existing=a.mode=='verify-probe'
    if out.parent!=ROOT/'runs' or not re.fullmatch(r'truck_\d{3}',out.name) or out.exists()!=existing or out.is_symlink():
        p.error('Expected a fresh or sealed runs/truck_NNN')
    _no_reparse(out if existing else out.parent)
    if existing:
        if not a.manifest_sha256 or a.scratch:p.error('verify requires seal and no scratch')
        result=verify_probe(out,a.manifest_sha256)
    else:
        if not a.scratch or a.manifest_sha256:p.error('probe requires external scratch and no seal')
        scratch=a.scratch.absolute();_no_reparse(scratch)
        if not scratch.is_dir() or scratch.resolve()!=scratch or scratch.is_relative_to(ROOT) or scratch.drive!=out.drive:
            p.error('Expected external scratch on same drive')
        result=seal_probe(out,scratch)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
