"""One-sided score preservation on same-anchor night tram true positives."""
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch

import detection_metrics as dm
from source_contract import ContractError, NAMES, check_equal

CLASS_ID = 4
FLOOR = .001
FLOW_SHA = '6e2e4468de86374c19ebd534ed0d30d42896c8709e55c3737e7ceb75e3e6b27b'


@torch.no_grad()
def observations(reference, reference_logits, current, gt):
    import torchvision  # Same NMS provider as validation.
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.ops import xywh2xyxy

    n = reference_logits.shape[-1]
    check_equal('same native six-class grid',(1,10,n),tuple(reference.shape))
    check_equal('identical current grid',(1,10,n),tuple(current.shape))
    (out,),(ids,) = non_max_suppression(reference.clone(),FLOOR,.7,nc=6,multi_label=True,max_det=300,
                                        max_nms=30000,return_idxs=True)
    ids = ids.reshape(-1).long(); cls = out[:,5].long()
    decoded = xywh2xyxy(reference[0,:4].T)
    check_equal('exact native retained anchor and score identity',True,
                torch.equal(out[:,:4],decoded[ids].to(out.dtype)) and
                torch.equal(out[:,4],reference[0,4+cls,ids].to(out.dtype)) and
                torch.equal(out[:,4],reference_logits[cls,ids].sigmoid().to(out.dtype)))
    tram = cls == CLASS_ID; refs = ids[tram]
    ref_iou = box_iou(gt.float(),out[tram,:4].float())
    pairs = dm.native_pairs(np.full(len(refs),CLASS_ID),np.full(len(gt),CLASS_ID),ref_iou.cpu().numpy())
    g,p = torch.as_tensor(pairs,device=gt.device).T
    anchors = refs[p]
    current_iou = box_iou(gt.float(),xywh2xyxy(current[0,:4].T).float())
    same = current_iou[g,anchors] >= .5
    g,anchors = g[same],anchors[same]
    covered = ((current_iou[g] >= .5) & (current[0,4+CLASS_ID] > FLOOR)).any(dim=1)
    return anchors, reference_logits[CLASS_ID,anchors].detach(), g, ~covered, reference[0,4+CLASS_ID,anchors]


def lower_bound_loss(logits, anchors, reference):
    if not anchors.numel():
        return logits.new_zeros((),dtype=torch.float32)
    return .10 * (reference.detach().float() - logits[1,CLASS_ID,anchors].float()).relu().mean()


def gate(stats):
    checks = {'support':stats['eligible_gt']>=12, 'floor_signal':stats['below_floor_gt']>=4}
    return [k for k,v in checks.items() if not v]


class FloorLoss:
    def __init__(self, head, reference, night_keys):
        check_equal('stock detector layout',(6,False,False),(head.nc,head.end2end,head.xyxy))
        check_equal('same anchor grid and head class configuration',
                    (type(head),6,False,False,head.stride.tolist()),
                    (type(reference.model[-1]),reference.model[-1].nc,reference.model[-1].end2end,
                     reference.model[-1].xyxy,reference.model[-1].stride.tolist()))
        self.head, self.reference, self.night_keys = head, reference, tuple(night_keys)
        self.calls = self.grad_checks = 0
        self.eligible, self.signal, self.active = set(), set(), set()
        self.exposures = Counter()
        self.bins = Counter()

    def attach(self, criterion):
        native=criterion.get_assigned_targets_and_loss
        def loss(pred,batch):
            points, terms, items = native(pred,batch)
            return points,torch.stack((terms[0],terms[1]+self.extra(pred,batch),terms[2])),items
        criterion.get_assigned_targets_and_loss=loss

    def extra(self, preds, batch, *, audit=False):
        from ultralytics.utils.ops import xywh2xyxy

        logits = preds['scores']; boxes = preds['boxes']
        if self.calls >= len(self.night_keys):
            raise ContractError('Score-floor call outside original schedule')
        key = self.night_keys[self.calls]
        self.calls += 1
        check_equal('paired 640 train batch',(2,6,(2,3,640,640)),(logits.shape[0],logits.shape[1],tuple(batch['img'].shape)))
        night = batch['batch_idx']==1
        matched_labels = (batch['cls'].reshape(-1)[night] == CLASS_ID).nonzero().flatten()
        if not matched_labels.numel():
            return logits.new_zeros((),dtype=torch.float32)
        with torch.no_grad():
            gt = xywh2xyxy(batch['bboxes'][night][matched_labels]*640)
            # Keep reference native FP16 inference separate from the student's training autocast.
            with torch.autocast(logits.device.type,enabled=False):
                reference, features = self.reference(batch['img'][1:].half())
            check_equal('same ordered grid in every level', [tuple(x.shape[-2:]) for x in preds['feats']],
                        [tuple(x.shape[-2:]) for x in features['feats']])
            current = self.head._inference({'boxes':boxes[1:].detach(),'scores':logits[1:].detach(),
                                            'feats':[v[1:].detach() for v in preds['feats']]})
            anchors, targets, ids, below, scores = observations(reference,features['scores'][0],current,gt)
            positive = logits[1,CLASS_ID,anchors] < targets
            for i, lost, act, prob in zip(matched_labels[ids].tolist(),below.tolist(),positive.tolist(),scores.tolist(),strict=True):
                gh = key,i
                self.eligible.add(gh)
                self.exposures['eligible']+=1
                if lost:
                    self.signal.add(gh); self.exposures['below_floor']+=1
                if act:
                    self.active.add(gh); self.exposures['active']+=1
                self.bins['(.001,.01]' if prob<=.01 else '(.01,.05]' if prob<=.05 else '>.05']+=1
        if not audit:
            return lower_bound_loss(logits,anchors,targets)
        if not anchors.numel():
            return logits.new_zeros((),dtype=torch.float32)
        with torch.enable_grad():
            z=logits.detach().requires_grad_(); d=boxes.detach().requires_grad_()
            ref=targets.detach().requires_grad_()
            extra=lower_bound_loss(z,anchors,ref)
            grad, geom, teacher = torch.autograd.grad(extra,(z,d,ref),allow_unused=True)
            expected=torch.zeros_like(grad); expected[1,CLASS_ID,anchors[positive]]=-1
            check_equal('only intended night score gradient; no geometry or teacher gradient',True,
                        geom is None and teacher is None and torch.equal(grad.sign(),expected))
            self.grad_checks+=1
        return extra.detach()

    def summary(self):
        return {'calls':self.calls,'gradient_checked_batches':self.grad_checks,'eligible_gt':len(self.eligible),
                'below_floor_gt':len(self.signal),'active_gt':len(self.active),'exposures':dict(self.exposures),
                'reference_score_bands':dict(self.bins)}


def balanced_inputs():
    import night_expand as exp
    import night_train as train
    from night_contract import ROOT, read_sealed_json
    from night_tal import BALANCED_SHA
    from source_contract import canonical_hash, sha256

    parent = ROOT/'runs/expand_003'
    seal = read_sealed_json(parent,'manifest.json',BALANCED_SHA,[])
    run, chosen, inputs, actual = (exp.adapt._json(parent,seal,f) for f in ('run.json','selection.json','43/inputs.json','43/train.json'))
    check_equal('frozen untreated balanced seed 43',('sealed','complete','e2',43,False,False,200,0,(None,)*5),
                (seal['status'],run['status'],run['arm'],run['seed'],run['cloud_used'],run['final_test_executed'],actual['updates'],
                 actual['skipped_updates'],tuple(run.get(k) for k in ('e2_mix','margin','crop','ring','pairloss'))))
    digest=run['config_sha256']; _,cfg,pair,recipe=exp.configuration(digest)
    _,identity,pool,inv,_,d300,base,_=exp.context(cfg,pair,recipe,'e2')
    slots=exp.align_slots(d300,chosen['keys']); plan=exp.expanded_plan(base,slots,43,recipe['schedule_salt']); plan_sha=canonical_hash(plan)
    check_equal('exact approved full order and image-only selection',
                (slots,plan_sha,plan_sha,[[d[0],n[0]] for d,n in plan],[inv[k] for k in slots],pool,False,slots),
                (chosen['slots'],chosen['schedule_sha256'],actual['augment_schedule_sha256'],actual['microbatches'],
                 chosen['records'],chosen['pool_sha256'],chosen['labels_read_in_selection'],[r['key'] for r in inputs['records']]))
    check_equal('locked original input and initializer hashes',(identity['A_sha256'],identity['A_sha256'],seal['files']['code/night_train.py']['sha256']),
                (run['a_sha256'],sha256(train.CHECKPOINT),sha256(ROOT/'scripts/night_train.py')))
    night={}
    for r,image,label in zip(inputs['records'],inputs['source_reads'][::2],inputs['source_reads'][1::2],strict=True):
        im=inv[r['key']]
        check_equal('exact selected payloads',('night_pool',im['image'],im['image_sha256'],im['group_id'],
                    'hidden_ground_truth/night_pool/'+PurePosixPath(im['image']).with_suffix('.txt').name),
                    (r['purpose'],image,r['image_sha256'],r['group_id'],label))
        night[r['key']]=dict(r,image=image,label=label)
    return run,recipe,identity,night,plan,plan_sha,seal['files']['43/last.pt']['sha256']


def reference_model(expected_sha):
    from ultralytics.nn.tasks import load_checkpoint
    from night_contract import ROOT, _no_reparse
    from source_contract import sha256

    p=ROOT/'runs/expand_003/43/last.pt'; _no_reparse(p)
    check_equal('exact fixed balanced EMA',expected_sha,sha256(p))
    model, receipt=load_checkpoint(str(p),device='cuda',fuse=True)
    check_equal('reference fixed last and original class order',(200,NAMES),(receipt['updates'],list(model.names.values())))
    return model.half().eval().requires_grad_(False)


def native_id(head, loss):
    import inspect
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import box_iou
    from source_contract import sha256

    return {Path(p).name:sha256(Path(p)) for p in map(inspect.getfile,(type(head),nms,box_iou,type(loss)))}


def preflight(scratch,out):
    import os, random, shutil, tempfile, time
    from importlib.metadata import version
    from types import SimpleNamespace
    import cv2
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.loss import v8DetectionLoss
    import night_train as train
    from night_contract import ROOT, read_sealed_json
    from night_tal import BALANCED_SHA, INDICES
    from seed_eval import probe_gpu
    from source_contract import canonical_hash, sha256, write_json

    start=time.monotonic(); gpu=probe_gpu()
    state,recipe,identity,night,plan,plan_sha,ref_sha=balanced_inputs()
    flow=read_sealed_json(ROOT/'runs/tram_flow','manifest.json',FLOW_SHA,[])
    check_equal('diagnostic registered', 'sealed', flow['status'])  # Do not inspect the GT rows in the diagnostic.
    versions={n:version(n) for n in train.LIBS}
    check_equal('exact environment',state['versions'],versions)
    src=identity['source_train']; doc=read_sealed_json(train.SOURCE,src['path'],src['sha256'],[])
    check_equal('only day train metadata',('source_train',4000),(doc['purpose'],len(doc['records'])))
    day={r['key']:r for r in doc['records']}; sample=[plan[i] for i in INDICES]
    days=[day[d[0]] for d,_ in sample]; nights=[night[k] for k in dict.fromkeys(n[0] for _,n in sample)]
    if any(r['purpose']!='source_train' or not r['image'].startswith('images/source_train/') or
           not r['label'].startswith('labels/source_train/') for r in days):
        raise ContractError('Day payload outside training scope')
    with tempfile.TemporaryDirectory(dir=scratch,prefix='floor-') as td:
        box=Path(td); data={}; reads={}
        for name,r,rs in (('source_train',train.SOURCE,days),('night_pool',train.NIGHT,nights)):
            staged,seen=train._stage(r,box/name,rs)
            check_equal('exact sampled payloads',[x[k] for x in rs for k in ('image','label')],seen)
            data.update(train.training_rows(box/name,staged,name))
            reads[name]={'images':len(rs),'files':len(seen),'paths_sha256':canonical_hash(seen)}
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; torch.set_num_threads(4); cv2.setNumThreads(1)
        torch.manual_seed(43); random.seed(43); np.random.seed(43)
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        model,_=load_checkpoint(str(train.CHECKPOINT),device='cuda'); model.train().requires_grad_(False)
        model.args=SimpleNamespace(**dict(model.args))
        check_equal('student class order',NAMES,list(model.names.values()))
        for layer in model.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm): layer.eval()
        teacher=reference_model(ref_sha)
        snapshots=[{k:v.detach().cpu().clone() for k,v in a.state_dict().items()} for a in (model,teacher)]
        hook=FloorLoss(model.model[-1],teacher,[n[0] for _,n in sample]); native=v8DetectionLoss(model); processed=[]
        with torch.no_grad(),train.telemetry() as monitor:
            for i,m in zip(INDICES,sample,strict=True):
                if time.monotonic()-start>=600: raise ContractError('Ten-minute score-floor probe cap')
                batch=train._batch(torch,cv2,data,m,recipe['size'])
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=model(batch['img']); _,loss,_=native.get_assigned_targets_and_loss(pred,batch)
                    extra=hook.extra(pred,batch,audit=True)
                check_equal('forward terms finite and detached',True,bool(loss.isfinite().all() & extra.isfinite()) and
                            not loss.requires_grad and not extra.requires_grad)
                processed.append(i)
                if len(processed)%50==0: print(f'Tram score-floor no-update probe: {len(processed)}/200',flush=True)
        for a,s in zip((model,teacher),snapshots,strict=True):
            check_equal('both frozen states exactly unchanged',True,all(torch.equal(v.detach().cpu(),s[k]) for k,v in a.state_dict().items()))
        stats=hook.summary(); failed=gate(stats)
        rec={**stats,'failures':failed,'met':not failed,'updates':0,'weights_unchanged':True,'indices':processed,
             'development_read':False,'holdout_read':False,'cloud_used':False,'reads':reads,'plan_sha256':plan_sha,
             'sample_plan_sha256':canonical_hash(sample),'expand_sha256':state['config_sha256'],'A_sha256':state['a_sha256'],
             'balanced_sha256':BALANCED_SHA,'teacher_sha256':ref_sha,'flow_sha256':FLOW_SHA,'versions':versions,
             'native_code_sha256':native_id(model.model[-1],native),'matcher_sha256':sha256(Path(dm.__file__)),
             'gpu':monitor,'initial_gpu':gpu,'seconds':time.monotonic()-start}
        evidence=box/'evidence'; evidence.mkdir(); shutil.copyfile(__file__,evidence/'night_floor.py')
        write_json(evidence/'metrics.json',rec)
        write_json(evidence/'manifest.json',{'status':'sealed','files':train._receipt(evidence)})
        evidence.rename(out)
    return verify(out,sha256(out/'manifest.json'))


def verify(out,sha):
    import json
    from night_contract import read_sealed_bytes,read_sealed_json
    from night_tal import BALANCED_SHA,INDICES

    seal=read_sealed_json(out,'manifest.json',sha,[]); files=seal['files']
    check_equal('exact floor files',('sealed',{'metrics.json','night_floor.py'}),(seal['status'],set(files)))
    b={k:read_sealed_bytes(out,k,x['sha256'],[]) for k,x in files.items()}
    check_equal('exact lengths',{k:x['bytes'] for k,x in files.items()},{k:len(x) for k,x in b.items()})
    v=json.loads(b['metrics.json']); bad=gate(v)
    check_equal('fixed schedule, official anchors and no development', (list(INDICES),200,0,True,BALANCED_SHA,FLOW_SHA,False,False,False),
                (v['indices'],v['calls'],v['updates'],v['weights_unchanged'],v['balanced_sha256'],v['flow_sha256'],
                 v['development_read'],v['holdout_read'],v['cloud_used']))
    check_equal('recomputed floor decision',(bad,not bad),(v['failures'],v['met']))
    if v['active_gt'] and not v['gradient_checked_batches']:
        raise ContractError('Missing frozen-target direct gradient audit')
    return {**v,'manifest_sha256':sha,'code_sha256':files['night_floor.py']['sha256']}


def main():
    import argparse, json, re
    from night_contract import ROOT, _no_reparse

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('probe','verify')); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path); p.add_argument('--manifest-sha256'); a=p.parse_args()
    out=a.output.absolute(); existing=a.mode=='verify'
    if out.parent!=ROOT/'runs' or not re.fullmatch(r'floor_\d{3}',out.name) or out.exists()!=existing or out.is_symlink():
        p.error('Expected a fresh or sealed runs/floor_NNN')
    _no_reparse(out if existing else out.parent)
    if existing:
        if not a.manifest_sha256 or a.scratch: p.error('verify requires seal and no scratch')
        result=verify(out,a.manifest_sha256)
    else:
        if not a.scratch or a.manifest_sha256: p.error('probe requires external scratch and no seal')
        scratch=a.scratch.absolute(); _no_reparse(scratch)
        if not scratch.is_dir() or scratch.resolve()!=scratch or scratch.is_relative_to(ROOT) or scratch.drive!=out.drive:
            p.error('Expected external scratch on the same drive')
        result=preflight(scratch,out)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
