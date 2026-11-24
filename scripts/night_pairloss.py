"""Training-only same-class ranking across recent, disjoint night risk groups."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
import torch.nn.functional as F

import detection_metrics as dm
from source_contract import ContractError, NAMES, check_equal

BUCKETS = ('classification', 'classification_localization', 'localization', 'background_candidate')
CLASSES = (3, 4)


@dataclass(frozen=True)
class Candidate:
    cls: int
    anchor: int
    kind: str
    score: float
    logit: float
    gt: tuple | None


@torch.no_grad()
def candidates(raw, logits, gt, gt_cls, key):
    import torchvision  # Match the native validation NMS backend.
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.ops import xywh2xyxy

    check_equal('native six-class raw layout', (1, 10, logits.shape[-1]), tuple(raw.shape))
    output, kept = nms.non_max_suppression(raw.clone(), .001, .7, nc=6, multi_label=True,
                                            max_det=300, max_nms=30000, return_idxs=True)
    pred = output[0]; anchors = kept[0].reshape(-1).long(); cls = pred[:, 5].long()
    check_equal('native NMS indices cover every emitted row', len(pred), anchors.numel())
    if not len(pred):
        return []
    check_equal('NMS class scores and anchor geometry round-trip', True,
                torch.equal(pred[:, 4], raw[0, 4+cls, anchors].to(pred.dtype)) and
                torch.equal(pred[:, :4], xywh2xyxy(raw[0, :4, anchors].T).to(pred.dtype)) and
                torch.equal(pred[:, 4], logits[cls, anchors].sigmoid().to(pred.dtype)))
    mask = cls < 5
    pred, anchors, cls = pred[mask], anchors[mask], cls[mask]
    keep_gt = (gt_cls < 5).nonzero().flatten()
    p_cls = cls.tolist(); g_cls = gt_cls[keep_gt].tolist(); ids = anchors.tolist()
    value = pred[:, 4].tolist(); live = logits[cls, anchors].tolist()
    iou = box_iou(gt[keep_gt].float(), pred[:, :4].float()).cpu().numpy()
    matrix = {'gt': {'cls': g_cls}, 'pred': {'cls': p_cls, 'conf': value},
              'iou_edges': [(int(i), int(j), float(iou[i,j])) for i,j in zip(*iou.nonzero(), strict=True)]}
    g_ids = keep_gt.tolist(); result = []
    for c in CLASSES:
        _, pairs, chosen = dm.image_point(matrix, c, 0.)
        matched = {int(j): g_ids[int(i)] for i,j in pairs}
        positives = []; negatives = []
        for j in chosen.tolist():
            if j in matched:
                positives.append(Candidate(c, ids[j], 'tp', value[j], live[j], (key, matched[j])))
            else:
                k = dm.error_type(matrix, j, pairs)['type']
                if k in BUCKETS:
                    negatives.append(Candidate(c, ids[j], k, value[j], live[j], None))
        result.extend(sorted(positives, key=lambda r: (r.score, r.anchor))[:3])
        result.extend(next(r for r in sorted(negatives, key=lambda r: (-r.score, r.anchor)) if r.kind == k)
                      for k in BUCKETS if any(r.kind == k for r in negatives))
    return result


class Queue:
    def __init__(self):
        self.history = deque(maxlen=32)

    def push(self, scores, group, current):
        if not group:
            raise ContractError('Ranking requires a sealed night risk group')
        history = [(g,r) for g,rows in self.history if g != group for r in rows]
        classes = []; records = []
        for c in CLASSES:
            past = [(g,r) for g,r in history if r.cls == c]
            now = [r for r in current if r.cls == c]
            refs = sorted(((g,r) for g,r in past if r.kind == 'tp'), key=lambda a: (a[1].score,a[1].anchor))[:3]
            pos_buckets = []; neg_buckets = []
            for kind in BUCKETS:
                negatives = [(g,r) for g,r in past if r.kind == kind]
                if negatives:
                    ng, neg = min(negatives, key=lambda a: (-a[1].score,a[1].anchor))
                    pos = [r for r in now if r.kind == 'tp']
                    if pos:
                        z = scores[1,c,[r.anchor for r in pos]].float()
                        pos_buckets.append(F.softplus(1 + neg.logit - z).mean())
                        records.extend((p,neg,ng,True) for p in pos)
                for neg in (r for r in now if r.kind == kind):
                    if refs:
                        t = scores.new_tensor([p.logit for _,p in refs], dtype=torch.float32)
                        neg_buckets.append(F.softplus(1 + scores[1,c,neg.anchor].float() - t).mean())
                        records.extend((p,neg,group,False) for _,p in refs)
            directions = [torch.stack(d).mean() for d in (pos_buckets,neg_buckets) if d]
            if directions:
                classes.append(torch.stack(directions).mean())
        self.history.append((group, tuple(current)))
        return (torch.stack(classes).mean() * .10 if classes else scores.new_zeros((),dtype=torch.float32)), records


class PairLoss:
    def __init__(self, head, rows):
        check_equal('stock non-end-to-end six class detector', (6, False, False), (head.nc, head.end2end, head.xyxy))
        self.head, self.rows = head, tuple(rows)
        self.queue = Queue()
        self.calls = self.active = self.grad_checks = 0
        self.stats = {c:{'gt':set(),'inverted':set(),'groups':{k:set() for k in BUCKETS},'pairs':Counter()} for c in CLASSES}

    def attach(self, criterion):
        native = criterion.get_assigned_targets_and_loss
        def loss(preds, batch):
            assignment, terms, items = native(preds, batch)
            extra = self.extra(preds, batch)
            return assignment, torch.stack((terms[0], terms[1]+extra, terms[2])), items
        criterion.get_assigned_targets_and_loss = loss

    def extra(self, preds, batch, *, audit=False):
        from ultralytics.utils.ops import xywh2xyxy

        if self.calls >= len(self.rows):
            raise ContractError('Extra ranking call outside sealed schedule')
        key, group = self.rows[self.calls]
        scores, boxes = preds['scores'], preds['boxes']
        check_equal('original paired 640 training batch', (2, 6, (2,3,640,640)),
                    (scores.shape[0], scores.shape[1], tuple(batch['img'].shape)))
        with torch.no_grad():
            raw = self.head._inference({k: preds[k][1:].detach() for k in ('boxes','scores')} |
                                       {'feats':[v[1:].detach() for v in preds['feats']]})
            night = batch['batch_idx']==1
            gt_cls = batch['cls'].reshape(-1)[night].long()
            gt = xywh2xyxy(batch['bboxes'][night]*640)
            found = candidates(raw, scores[1], gt, gt_cls, key)
        if audit:
            with torch.enable_grad():
                z = scores.detach().requires_grad_(); d = boxes.detach().requires_grad_()
                extra, records = self.queue.push(z, group, found)
                if records:
                    g, geom = torch.autograd.grad(extra,(z,d),allow_unused=True)
                    target = torch.zeros_like(g,dtype=torch.int8)
                    for pos, neg, _, is_pos in records:
                        r = pos if is_pos else neg
                        target[1,r.cls,r.anchor] = -1 if is_pos else 1
                    check_equal('auxiliary gradient has only intended signs and no geometry path', True,
                                geom is None and torch.equal(g.sign(),target.to(g.dtype)))
                    self.grad_checks += 1
                extra = extra.detach()
        else:
            extra, records = self.queue.push(scores, group, found)
        self.calls += 1
        self.active += bool(records)
        for p,n,group,_ in records:
            s = self.stats[p.cls]
            s['gt'].add(p.gt); s['groups'][n.kind].add(group); s['pairs'][n.kind] += 1
            if n.score > p.score:
                s['inverted'].add(p.gt)
        return extra

    def summary(self):
        return {'calls':self.calls, 'active_batches':self.active, 'gradient_checked_batches':self.grad_checks,
                'classes': {NAMES[c]:{'positive_gt':len(s['gt']), 'inverted_gt':len(s['inverted']),
                                      'negative_groups':len(set().union(*s['groups'].values())),
                                      'groups_by_bucket':{k:len(v) for k,v in s['groups'].items()},
                                      'pairs_by_bucket':{k:s['pairs'][k] for k in BUCKETS}} for c,s in self.stats.items()}}


def failures(stats):
    check_equal('both preregistered tail classes', {'truck','tram'}, set(stats['classes']))
    check_equal('one detached gradient audit per active sampled batch', (200, stats['active_batches']),
                (stats['calls'],stats['gradient_checked_batches']))
    result = {}
    for c, q in stats['classes'].items():
        flags = {'positive_gt':q['positive_gt']>=12,'negative_groups':q['negative_groups']>=12,
                 'classification':q['groups_by_bucket']['classification']>=4,
                 'classification_localization':q['groups_by_bucket']['classification_localization']>=4,
                 'inversions':10*q['inverted_gt']>=q['positive_gt']}
        result[c] = [k for k,ok in flags.items() if not ok]
    return result


def preflight(scratch, out):
    import inspect
    import os
    import random
    import shutil
    import tempfile
    import time
    from importlib.metadata import version
    from types import SimpleNamespace

    import cv2
    import numpy as np
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils import nms
    from ultralytics.utils.loss import v8DetectionLoss
    import night_expand as exp
    import night_train as train
    from night_contract import ROOT, read_sealed_json
    from night_tal import BALANCED_SHA, INDICES
    from seed_eval import probe_gpu
    from source_contract import canonical_hash, sha256, write_json

    start = time.monotonic(); gpu = probe_gpu()
    parent = ROOT/'runs/expand_003'
    seal = read_sealed_json(parent, 'manifest.json', BALANCED_SHA, [])
    state, chosen, inputs, actual = (exp.adapt._json(parent, seal, f) for f in
                                      ('run.json','selection.json','43/inputs.json','43/train.json'))
    check_equal('original untreated E2 train identity', ('e2',43,'complete',None,None,None,None,False),
                (state['arm'],state['seed'],state['status'],state.get('ring'),state.get('margin'),state.get('crop'),
                 state.get('e2_mix'),state['final_test_executed']))
    digest = state['config_sha256']; _, cfg, pair, recipe = exp.configuration(digest)
    _, identity, pool, inv, _, d300, base, _ = exp.context(cfg, pair, recipe, 'e2')
    slots = exp.align_slots(d300,chosen['keys']); plan = exp.expanded_plan(base,slots,43,recipe['schedule_salt'])
    plan_sha = canonical_hash(plan)
    check_equal('original plan, actual order and image-only selection',
                (slots,plan_sha,plan_sha,[[d[0],n[0]] for d,n in plan],[inv[k] for k in slots],pool,False),
                (chosen['slots'],chosen['schedule_sha256'],actual['augment_schedule_sha256'],actual['microbatches'],
                 chosen['records'],chosen['pool_sha256'],chosen['labels_read_in_selection']))
    check_equal('frozen balanced initializer and trainer', (identity['A_sha256'], identity['A_sha256'], seal['files']['code/night_train.py']['sha256']),
                (state['a_sha256'], sha256(train.CHECKPOINT), sha256(ROOT/'scripts/night_train.py')))
    versions = {k:version(k) for k in train.LIBS}
    check_equal('same package versions as original', state['versions'], versions)
    check_equal('exact original label slots', slots, [r['key'] for r in inputs['records']])
    night = {}
    for r, image, label in zip(inputs['records'], inputs['source_reads'][::2], inputs['source_reads'][1::2], strict=True):
        info = inv[r['key']]
        check_equal('sealed night payload purpose and identity',('night_pool',info['image'],info['image_sha256'],info['group_id'],
                    'hidden_ground_truth/night_pool/'+PurePosixPath(info['image']).with_suffix('.txt').name),
                    (r['purpose'],image,r['image_sha256'],r['group_id'],label))
        night[r['key']] = dict(r,image=image,label=label)
    src=identity['source_train']; doc=read_sealed_json(train.SOURCE,src['path'],src['sha256'],[])
    check_equal('only source training metadata', ('source_train',4000), (doc['purpose'],len(doc['records'])))
    day = {r['key']:r for r in doc['records']}; sample=[plan[i] for i in INDICES]
    day_rows=[day[d[0]] for d,_ in sample]
    if any(r['purpose']!='source_train' or not r['image'].startswith('images/source_train/') or
           not r['label'].startswith('labels/source_train/') for r in day_rows):
        raise ContractError('Forbidden source payload')
    night_rows=[night[k] for k in dict.fromkeys(n[0] for _,n in sample)]
    with tempfile.TemporaryDirectory(dir=scratch,prefix='pairloss-') as td:
        box=Path(td); payload={}; reads={}
        for kind,root,rows in (('source_train',train.SOURCE,day_rows),('night_pool',train.NIGHT,night_rows)):
            staged,paths=train._stage(root,box/kind,rows)
            check_equal('exact authorized train payloads',[r[k] for r in rows for k in ('image','label')],paths)
            payload.update(train.training_rows(box/kind,staged,kind))
            reads[kind]={'images':len(rows),'files':len(paths),'paths_sha256':canonical_hash(paths)}
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.set_num_threads(4); cv2.setNumThreads(1)
        torch.manual_seed(43); random.seed(43); np.random.seed(43)
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        model,_=load_checkpoint(str(train.CHECKPOINT),device='cuda')
        model.train().requires_grad_(False); model.args=SimpleNamespace(**dict(model.args))
        check_equal('original classes', NAMES,list(model.names.values()))
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm): layer.eval()
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        hook=PairLoss(model.model[-1],[(n[0],night[n[0]]['group_id']) for _,n in sample]); native=v8DetectionLoss(model)
        used=[]
        with torch.no_grad(),train.telemetry() as monitor:
            for i,micro in zip(INDICES,sample,strict=True):
                if time.monotonic()-start>=600: raise ContractError('Ten-minute ranking probe budget exhausted')
                batch=train._batch(torch,cv2,payload,micro,recipe['size'])
                with torch.autocast('cuda',dtype=torch.float16):
                    preds=model(batch['img'])
                    _,loss,_=native.get_assigned_targets_and_loss(preds,batch)
                    extra=hook.extra(preds,batch,audit=True)
                check_equal('finite native and auxiliary terms without gradient to model',True,
                            bool(loss.isfinite().all() & extra.isfinite()) and not loss.requires_grad and not extra.requires_grad)
                used.append(i)
                if len(used)%50==0: print(f'Pair ranking no-update probe: {len(used)}/200',flush=True)
        check_equal('frozen state exactly untouched',True,all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in model.state_dict().items()))
        stats=hook.summary(); failed=failures(stats)
        receipt={**stats,'failures':failed,'met':not any(failed.values()),'indices':used,
                 'plan_sha256':plan_sha,'sample_plan_sha256':canonical_hash(sample),'balanced_sha256':BALANCED_SHA,
                 'expand_sha256':digest,'A_sha256':state['a_sha256'],'reads':reads,'updates':0,'weights_unchanged':True,
                 'development_read':False,'holdout_read':False,'cloud_used':False,'versions':versions,
                 'native_code_sha256':{Path(f).name:sha256(Path(f)) for f in map(inspect.getfile,(type(hook.head),nms,v8DetectionLoss))},
                 'active_matcher_sha256':sha256(Path(dm.__file__)),
                 'initial_gpu':gpu,'gpu':monitor,'seconds':time.monotonic()-start}
        evidence=box/'evidence'; evidence.mkdir(); shutil.copyfile(__file__,evidence/'night_pairloss.py')
        write_json(evidence/'metrics.json',receipt)
        write_json(evidence/'manifest.json',{'status':'sealed','files':train._receipt(evidence)})
        evidence.rename(out)
    return verify(out,sha256(out/'manifest.json'))


def verify(out, digest):
    import json
    from night_contract import read_sealed_bytes, read_sealed_json
    from night_tal import BALANCED_SHA, INDICES
    from source_contract import sha256

    seal=read_sealed_json(out,'manifest.json',digest,[])
    check_equal('sealed ranking evidence',('sealed',{'metrics.json','night_pairloss.py'}),(seal['status'],set(seal['files'])))
    contents={}
    for name,ent in seal['files'].items():
        b=read_sealed_bytes(out,name,ent['sha256'],[])
        check_equal('sealed file length',ent['bytes'],len(b)); contents[name]=b
    v=json.loads(contents['metrics.json']); failed=failures(v)
    check_equal('original schedule and no development/updates', (list(INDICES),0,True,False,False,False,BALANCED_SHA),
                (v['indices'],v['updates'],v['weights_unchanged'],v['development_read'],v['holdout_read'],v['cloud_used'],v['balanced_sha256']))
    check_equal('recomputed decision', (failed, not any(failed.values())), (v['failures'],v['met']))
    return {**v,'manifest_sha256':sha256(out/'manifest.json'),'code_sha256':seal['files']['night_pairloss.py']['sha256']}


def main():
    import argparse
    import json
    import re
    from night_contract import ROOT, _no_reparse

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('probe','verify')); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path); p.add_argument('--manifest-sha256')
    a=p.parse_args(); out=a.output.absolute(); existing=a.mode=='verify'
    if out.parent!=ROOT/'runs' or not re.fullmatch(r'pairloss_\d{3}',out.name) or out.exists()!=existing or out.is_symlink():
        p.error('Expected a fresh or sealed runs/pairloss_NNN')
    _no_reparse(out if existing else out.parent)
    if existing:
        if not a.manifest_sha256 or a.scratch: p.error('verify requires a seal and no scratch')
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
