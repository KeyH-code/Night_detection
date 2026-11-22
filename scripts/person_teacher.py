"""Measure train-only supervision available from a higher-resolution copy of seed 29."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

import detection_metrics as dm
import person_fusion as pf
import seed_eval as se
from evaluate import original_gt, verify_subset
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json


def representative(output, gt):
    # All classes participate in resolving ambiguous boxes; no candidate votes for two GT.
    scores, boxes = output
    overlap = dm.box_iou([b for _,b in gt],boxes)
    owner = overlap.argmax(axis=0) if len(gt) else []
    result = {}
    for g,(c,_) in enumerate(gt):
        if c!=0:continue
        ids = np.flatnonzero((owner==g)&(overlap[g]>=.5)&(scores>pf.FLOOR))
        if len(ids):
            i=int(ids[np.argmax(scores[ids])]);result[g]={'score':float(scores[i]),'iou':float(overlap[g,i]),'box':boxes[i].tolist()}
        else:result[g]=None
    return result


def qualifies(a,b):
    return bool(a and b and a['score']<.35 and b['score']>=.50 and b['score']-a['score']>=.20)


def run(out,scratch):
    if out.exists() or out.parent!=pf.ROOT/'runs' or not re.fullmatch(r'teacher_\d{3}',out.name) or scratch.is_relative_to(pf.ROOT):
        raise ContractError('Choose a new teacher_NNN and external scratch')
    gpu=se.probe_gpu();train=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train')
    dev=verify_subset(pf.SOURCE,pf.READY_SHA,'source_dev')
    groups={r['group_id'] for r in train['records']}
    if groups&{r['group_id'] for r in dev['records']}:raise ContractError('train/dev overlap')
    check_equal('training count',4000,len(train['records']));check_equal('replica seal',se.REPLICA_SHA,sha256(se.REPLICA/'manifest.json'))
    weight=se.REPLICA/'29/weights/best.pt';digest=read_json(se.REPLICA/'manifest.json')['runs']['29']['best_sha256']
    check_equal('seed 29 best',digest,sha256(weight));exclude=pf.ROOT/'configs/rerank.json';skip={x['key'] for x in read_json(exclude)['candidates']}
    ordered=sorted((r for r in train['records'] if r['key'] not in skip),key=lambda r:hashlib.sha256(('teacher29:'+r['key']).encode()).digest())
    picked=[];seen=set()
    for r in ordered:
        if r['group_id'] not in seen:
            seen.add(r['group_id']);picked.append(r)
        if len(picked)==512:break
    check_equal('sample size',512,len(picked)); keys={r['key'] for r in picked}
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        for k in ('YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','XDG_CONFIG_HOME'):os.environ[k]=tmp
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import cv2
        import torch
        import torchvision
        import ultralytics
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops
        versions=[torch.__version__,torchvision.__version__,ultralytics.__version__]
        check_equal('locked versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions)
        torch.set_num_threads(4);model,_=load_checkpoint(str(weight),device='cuda:0',fuse=True);model.half().eval()
        check_equal('model classes',NAMES,list(model.names.values()))
        def forward(img,side):
            padded=LetterBox((side,side),auto=True,stride=32)(image=img)
            x=torch.from_numpy(np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))).to('cuda').half()[None]/255
            with torch.inference_mode():pred=model(x)[0][0]
            if pred.shape[0]!=10 or not pred.isfinite().all():raise ContractError('invalid decoder')
            p=pred[4];idx=p>pf.FLOOR;box=ops.xywh2xyxy(pred[:4,idx].T.float())
            box=ops.scale_boxes(x.shape[-2:],box,img.shape[:2]);return p[idx].float().cpu().numpy(),box.cpu().numpy()
        def summary(rows):
            positives=[(r,t) for r in rows for t in r['targets']]; eligible=[(r,t) for r,t in positives if t['eligible']]
            return {'images':len(rows),'groups':len({r['group_id'] for r in rows}),'pedestrian_gt':len(positives),
                    'eligible_gt':len(eligible),'eligible_groups':len({r['group_id'] for r,t in eligible}),
                    'reverse_gt':sum(t['reverse'] for r,t in positives),'no_student_box':sum(t['student'] is None for r,t in positives),
                    'eligible_small_thin':sum(t['small_thin'] for r,t in eligible)}
        rows=[];stages=[]
        for batch in (picked,[r for r in ordered if r['key'] not in keys]):
            for r in batch:
                gt=original_gt(r,train['root']);targets=[]
                if any(c==0 for c,_ in gt):
                    img=cv2.imdecode(np.fromfile(train['root']/r['image'],np.uint8),cv2.IMREAD_COLOR)
                    check_equal('image shape',(r['height'],r['width']),img.shape[:2]);a,b=[representative(forward(img,z),gt) for z in (640,960)]
                    tags=dm.slice_tags(dict(r,gt={'original_boxes':[box for _,box in gt],'cls':[c for c,_ in gt]}))
                    for g in a:targets.append({'gt':g,'small_thin':bool({'area_S','width_le8'}&set(tags[g])),
                                    'student':a[g],'teacher':b[g],'eligible':qualifies(a[g],b[g]),'reverse':qualifies(b[g],a[g])})
                rows.append({'key':r['key'],'group_id':r['group_id'],'targets':targets})
            sums=summary(rows);stages.append(sums); print('TEACHER_SIGNAL',sums,flush=True)
            if sums['eligible_gt']>=25 and sums['eligible_groups']>=20:break
        check_equal('weight after',digest,sha256(weight));stage=Path(tmp)/'result';stage.mkdir()
        dm.write_rows(stage/'targets.jsonl.gz',rows);write_json(stage/'signal.json',{'stages':stages,'usable':sums['eligible_gt']>=25 and sums['eligible_groups']>=20,
                    'excluded_keys':sorted(skip),'method':'train best-GT-owned dense pedestrian score above .001; IoU>=.5; before NMS; positives only'})
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_train','checkpoint':{'path':weight.relative_to(pf.ROOT).as_posix(),'sha256':digest},
            'rules':sha256(pf.ROOT/'RULES.md'),'ready':pf.READY_SHA,'train_manifest':train['manifest'],'exclusions_sha256':sha256(exclude),
            'gpu':gpu,'versions':versions,'training_performed':False,'final_test_executed':False,'cloud_cny':0,'code_sha256':sha256(Path(__file__)),
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in stage.iterdir()}});shutil.copytree(stage,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
