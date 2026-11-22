"""Fuse only tricycle detections against a sealed, same-checkpoint development anchor."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from shutil import copyfile
import sys
import tempfile

import numpy as np

import detection_metrics as dm
import person_fusion as pf
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json

CLASS = NAMES.index('tricycle')
LOCKED = pf.ROOT/'runs/scale_004'
LOCKED_SHA = '453ee4c77ea4b23fa65bbe09d3dc88728f94002ada68075545a48e0773f67853'
PROFILES = {'low448':((448,512,640,800),'consensus'), 'low416':((416,448,512,640,800),'consensus'),
            'low416-or':((416,448,512,640,800),'or')}


def _bounded_clusters(boxes,scores,votes,num_views):
    sums, means = np.zeros_like(boxes), np.zeros_like(boxes)
    used = np.zeros((len(boxes),num_views),dtype=bool)
    masses = np.zeros_like(scores)
    members = []
    for i in np.argsort(-scores,kind='stable'):
        overlaps=pf._iou(boxes[i],means[:len(members)])
        overlaps[used[:len(members),votes[i]]]=-1
        j=int(np.argmax(overlaps)) if len(members) else -1
        if j<0 or overlaps[j]<pf.CLUSTER_IOU:
            j=len(members)
            members.append([])
        masses[j]+=scores[i]
        sums[j]+=boxes[i]*scores[i]
        means[j]=sums[j]/masses[j]
        used[j,votes[i]]=True
        members[j].append(i)
    bounded=np.asarray([1.-np.prod(1.-scores[group]) for group in members],dtype=np.float32)
    return means[:len(members)], bounded


def predict(views, mode='consensus'):
    """Fuse box/score/class predictions without labels; output only tricycles."""
    if not views or mode not in ('consensus','max','or'):
        raise ValueError('Expected views with consensus, max or bounded union scoring')
    candidates = [(box, score, i) for i,pred in enumerate(views) for c,score,box in pf._detections(pred)
                  if c==CLASS and score>pf.FLOOR]
    if not candidates:
        return []
    boxes, scores, votes = zip(*candidates, strict=True)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if mode=='consensus':
        boxes, scores = pf._clusters(boxes,scores,votes,len(views))
    if mode=='or':
        boxes, scores = _bounded_clusters(boxes,scores,votes,len(views))
    active = np.argsort(-scores,kind='stable')
    active = active[scores[active]>pf.FLOOR]
    output = []
    while len(active):
        i,rest=active[0],active[1:]
        output.append((CLASS,float(scores[i]),boxes[i].tolist()))
        active=rest[pf._iou(boxes[i],boxes[rest])<=pf.NMS_IOU]
    return output


def replace_tricycle(row, detections):
    """Preserve the other five classes' outputs and original matching edges."""
    anchor=row['pred']
    retained={index:i for i,index in enumerate(j for j,c in enumerate(anchor['cls']) if c!=CLASS)}
    protected=len(retained)
    capacity=pf.MAX_DET-protected
    if capacity<0 or any(d[0]!=CLASS for d in detections):
        raise ContractError('Tricycle replacement cannot override the five-class output budget')
    pred={k:[anchor[k][i] for i in retained] for k in pf.FIELDS}
    for d in detections[:capacity]:
        for k,value in zip(pf.FIELDS,d,strict=True):
            pred[k].append(value)
    matrix=dm.box_iou(row['gt']['original_boxes'],pred['original_boxes'][protected:])
    gt_idx,pred_idx=np.nonzero(matrix)
    edges=[[g,retained[p],v] for g,p,v in row['iou_edges'] if p in retained]
    edges.extend([int(g),protected+int(p),float(matrix[g,p])] for g,p in zip(gt_idx,pred_idx,strict=True))
    result=dict(row,pred=dict(pred,ids=list(range(len(pred['cls']))),boxes=pred['original_boxes']),iou_edges=edges)
    tp,pairs=dm.ap_flags(result)
    result.update(ap_tp=tp.tolist(),ap_pairs=pairs)
    return result,max(0,len(detections)-capacity)


def merge_dev(anchor, views, mode):
    """Only identity-aligned white-listed dev predictions can enter evaluation."""
    keys=set(pf._dev_index(anchor))
    groups=[pf._dev_index(v) for v in views]
    for g in groups:
        check_equal('tricycle dev coverage',keys,set(g))
    result, caps=[],[]
    for a in anchor:
        key=a['key']
        aligned=[g[key] for g in groups]
        for r in aligned:
            check_equal('tricycle frozen data identity '+key,pf._identity(a),pf._identity(r))
        additions=predict(tuple(r['pred'] for r in aligned),mode)
        row, dropped=replace_tricycle(a,additions)
        result.append(row)
        if dropped:
            caps.append({'key':key,'tricycle_dropped':dropped})
    return result,caps


def lock():
    check_equal('locked five-class manifest',LOCKED_SHA,sha256(LOCKED/'manifest.json'))
    m=read_json(LOCKED/'manifest.json')
    for name in ('source_dev/objects.jsonl.gz','comparison.json'):
        check_equal('locked '+name,m['files'][name]['sha256'],sha256(LOCKED/name))
    base=dm.read_rows(LOCKED/'source_dev/objects.jsonl.gz')
    check_equal('locked count',500,len(pf._dev_index(base)))
    return m,base


def run(profile, scratch, output):
    from evaluate import stage_subset, verify_subset
    from source_policy import (SelectionPolicy, load_rules, paired_effects, recall_intervals, reference_at, summarize)

    scratch,output=Path(scratch).resolve(),Path(output).resolve()
    if scratch.is_relative_to(pf.ROOT) or (output.is_relative_to(pf.ROOT) and not output.is_relative_to(pf.ROOT/'runs')):
        raise ContractError('Scratch must be external; in-project evidence must stay in runs/')
    if output.exists():
        raise ContractError('Will not overwrite an existing tricycle run')
    sizes,score_mode=PROFILES[profile]
    m,base=lock()
    checkpoint=pf.ROOT/m['checkpoint']['path']
    expected=m['checkpoint']['sha256']
    check_equal('locked model',expected,sha256(checkpoint))
    contract=verify_subset(pf.SOURCE,m['ready'],'source_dev')
    rules=contract['rules']=load_rules(pf.ROOT/'RULES.md')
    check_equal('locked rules',m['rules'],rules['sha256'])
    check_equal('locked dev split',m['dev_manifest'],contract['manifest'])
    views,items=[],[]
    with tempfile.TemporaryDirectory(prefix='tricycle-',dir=scratch) as tmp:
        folder=Path(tmp)
        for key in ('TMP','TEMP','TMPDIR','YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','XDG_CONFIG_HOME'):
            dest=folder/key.lower(); dest.mkdir(); os.environ[key]=str(dest)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch
        import torchvision
        import ultralytics
        torch.set_num_threads(4)
        depot=stage_subset(contract,folder/'data')
        for size in sizes:
            rows,obs=pf._capture(torch,contract,depot,checkpoint,folder,size,False)
            if size==640:
                a={r['key']:[p for p in pf._detections(r['pred']) if p[0]==CLASS] for r in base}
                b={r['key']:[p for p in pf._detections(r['pred']) if p[0]==CLASS] for r in rows}
                check_equal('fresh native 640 and locked tricycle tuples',a,b)
            views.append(rows); items.append(obs)
            print(f'TRICYCLE_VIEW {size} {len(rows)} images {obs["seconds"]:.2f}s',flush=True)
        rows,caps=merge_dev(base,views,score_mode)
        check_equal('model after inference',expected,sha256(checkpoint))
        check_equal('READY after inference',m['ready'],sha256(pf.SOURCE/'READY.json'))

    br,_,_=summarize(base,rules)
    rr,rm,curves=summarize(rows,rules)
    checks,guards=SelectionPolicy(rules,reference_at(pf.ROOT)).assess(rr)
    count=flags_changed=0
    for a,b in zip(base,rows,strict=True):
        check_equal('protected box and score tuples '+a['key'],[p for p in pf._detections(a['pred']) if p[0]!=CLASS],
                    [p for p in pf._detections(b['pred']) if p[0]!=CLASS])
        old=[v for c,v in zip(a['pred']['cls'],a['ap_tp'],strict=True) if c!=CLASS]
        new=[v for c,v in zip(b['pred']['cls'],b['ap_tp'],strict=True) if c!=CLASS]
        count+=len(old); flags_changed+=sum(u!=v for u,v in zip(old,new,strict=True))
    for n in NAMES:
        if n!='tricycle':
            check_equal('protected workpoint '+n,br['workpoints'][n],rr['workpoints'][n])
    compare={'profile':profile,'baseline':br,'candidate':rr,'checks':checks,'guards':guards,
             'five_class_box_score_tuples_equal':count,'five_class_match_flags_changed':flags_changed,
             'five_class_workpoints_exact':True,'merged_cap':caps,'intervals':recall_intervals(rows,rm),
             'paired':paired_effects([(17,base,rows)],rules),
             'cloud':{'budget_hours':0,'budget_cny':0,'used_hours':0,'used_cny':0}}
    output.mkdir(parents=True,exist_ok=False)
    (output/'code').mkdir(); (output/'source_dev').mkdir()
    copyfile(__file__,output/'code/tricycle_fusion.py')
    dm.write_rows(output/'source_dev/objects.jsonl.gz',rows)
    dm.write_rows(output/'source_dev/matches.jsonl.gz',rm)
    np.savez_compressed(output/'source_dev/pr.npz',**curves)
    write_json(output/'comparison.json',compare)
    files={p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)}
           for p in output.rglob('*') if p.is_file()}
    inputs=('tricycle_fusion','person_fusion','evaluate','source_contract','source_policy','detection_metrics')
    write_json(output/'manifest.json',{'status':'complete','scope':'source_dev','profile':profile,
                                       'sizes':sizes,'score_mode':score_mode,'locked_manifest_sha256':LOCKED_SHA,
                                       'locked_run':'runs/scale_004','checkpoint':m['checkpoint'],
                                       'ready':m['ready'],'dev_manifest':m['dev_manifest'],'rules':m['rules'],
                                       'training_performed':False,'final_test_executed':False,'observations':items,
                                       'fusion':m['fusion'],'environment':{'python':sys.version,'torch':torch.__version__,
                                       'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,
                                       'gpu':torch.cuda.get_device_name(0)},
                                       'code_sha256':{f'scripts/{n}.py':sha256(pf.ROOT/f'scripts/{n}.py') for n in inputs},
                                       'files':files})
    print('TRICYCLE_RESULT',rr['metrics']['per_class']['tricycle'],rr['workpoints']['tricycle'],flush=True)
    print('TRICYCLE_PROTECT',count,'changes',flags_changed,'all gates',all(checks.values()) and all(guards.values()),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--profile',choices=PROFILES,required=True)
    cli.add_argument('--scratch',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    run(args.profile,args.scratch,args.output)
