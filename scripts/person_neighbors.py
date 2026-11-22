"""Falsify a leave-one-person-out scene prior using only sealed detector outputs."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

import person_local as pl
from person_local import ROOT,dm,pf,se
from source_endpoints import grade
from source_contract import ContractError,NAMES,check_equal,read_json,sha256,write_json
from source_policy import SelectionPolicy,reference_at
from evaluate import verify_subset


def support(boxes,scores):
    b=np.asarray(boxes,np.float64).reshape(-1,4);p=np.asarray(scores,np.float64);n=len(p)
    if n<2:return np.zeros(n)
    w=np.maximum(1e-7,b[:,2]-b[:,0]);h=np.maximum(1e-7,b[:,3]-b[:,1]);x=(b[:,0]+b[:,2])*.5
    height=np.sqrt(h[:,None]*h);dx=np.abs(x[:,None]-x);dy=b[:,3,None]-b[:,3];dh=np.log(h[:,None]/h)
    good=(dm.box_iou(b,b)<=.1)&(dx>=.3*(w[:,None]+w));np.fill_diagonal(good,False)
    evidence=p[None,:]*np.exp(-.5*((dx/(3*height))**2+(dy/(.65*height))**2+(dh/.6)**2))
    return np.where(good,evidence,0.).max(axis=1)


def rescore(r):
    v=r['pred'];ids=np.array([i for i,c in enumerate(v['cls']) if c==0],int);old=np.asarray(v['conf'],float);s=old[ids];boxes=np.asarray(v['original_boxes'],float).reshape(-1,4);q=support(boxes[ids],s)
    nw=old.copy();nw[ids]=s*(1+q)/(1+s*q);order=np.arange(len(old));order[ids]=ids[np.argsort(-nw[ids],kind='stable')];rev=np.argsort(order)
    pred={k:[v[k][i] for i in order] for k in ('boxes','cls','original_boxes')};pred.update(conf=nw[order].tolist(),ids=list(range(len(old))))
    result=dict(r,pred=pred,iou_edges=[[g,int(rev[j]),vv] for g,j,vv in r['iou_edges']]);flags,pa=dm.ap_flags(result);result.update(ap_tp=flags.tolist(),ap_pairs=pa)
    return result


def quick(rows,rule):
    aps=dm.summarize_ap(rows)[0]['per_class']['pedestrian'];point=dm.choose_point(dm.work_curve(rows,0),rule['precision'],rule['recall'])
    return {'ap50':aps['ap50'],'ap50_95':aps['ap50_95'],'point':point}


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'views_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Expected new views_NNN and external scratch')
    rules,dev,arms,paths=pl.prepare();tr=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');rec={r['key']:r for r in tr['records']};tp=ROOT/'runs/a_001_diag/train/objects.jsonl.gz';diag=read_json(tp.parents[1]/'manifest.json');check_equal('train source',('b8c5610195ddcf012b030c1c3574e018eccff2b90885de091f2ba65303c2256e',diag['files']['train/objects.jsonl.gz']['sha256'],arms[17]['sha256']),(sha256(tp),sha256(tp),diag['checkpoint_sha256']))
    with gzip.open(tp,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    check_equal('train coverage',(4000,set(rec)),(len(rows),{r['key'] for r in rows}));check_equal('split isolation',set(),{r['group_id'] for r in tr['records']}&{r['group_id'] for r in dev['records']})
    for r in rows:
        for k in ('purpose','group_id','image_sha256','label_sha256'):check_equal(k,rec[r['key']][k],r[k])
    rule=rules['classes']['pedestrian'];base=quick(rows,rule);probe=quick([rescore(r) for r in rows],rule);del rows
    passed=probe['point']['origin']=='rule' and probe['ap50']>=base['ap50'] and probe['point']['recall']>=base['point']['recall'] and (probe['ap50']>base['ap50'] or probe['point']['recall']>base['point']['recall'])
    print('NEIGHBOR_TRAIN',base,probe,passed,flush=True)
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        stage=Path(d)/'result';stage.mkdir();chosen=None;screen={};effects=[]
        if passed:
            policy=SelectionPolicy(rules,reference_at(ROOT));idx={r['key']:r for r in dev['records']}
            for s in (29,17,43):
                src=dm.read_rows(paths[s]);check_equal('complete dev',set(idx),set(pf._dev_index(src)))
                for r in src:
                    for k in ('purpose','group_id','image_sha256','label_sha256'):check_equal(k,idx[r['key']][k],r[k])
                rows=[rescore(r) for r in src];old=read_json(pl.BASE/str(s)/'metrics.json');m,ma,curve,failed,warn=grade(rows,rules,policy,old)
                keep=lambda r:[d for d in pf._detections(r['pred']) if d[0]!=0]
                for a,b in zip(src,rows,strict=True):check_equal('nonperson output',keep(a),keep(b))
                for k in NAMES[1:]:check_equal('other workpoint '+k,old['workpoints'][k],m['workpoints'][k])
                trans,g=pl.transitions(rows,dm.read_rows(pl.BASE/str(s)/'matches.jsonl.gz'),ma);effects.append(g);m.update(failed_five=failed,relative_warnings=warn,transitions=trans,paired_recall=pl.paired_interval([g]));screen[s]={'failed_five':failed,'relative_warnings':warn}
                folder=stage/'neighbors'/str(s);folder.mkdir(parents=True);write_json(folder/'metrics.json',m);dm.write_rows(folder/'objects.jsonl.gz',rows);dm.write_rows(folder/'matches.jsonl.gz',ma);np.savez_compressed(folder/'pr.npz',**curve);print('NEIGHBOR_DEV',s,m['metrics']['per_class']['pedestrian']['ap50'],m['workpoints']['pedestrian'],failed,warn,flush=True)
            if all(not v['failed_five'] for v in screen.values()) and not any(sum(k in v['relative_warnings'] for v in screen.values())>=2 for k in ('area_S','width_le8','small_thin_union','ap75')):chosen='neighbors'
        write_json(stage/'comparison.json',{'train':{'original':base,'candidate':probe,'pass':passed},'selected':chosen,'screen':screen,'paired_recall':pl.paired_interval(effects) if effects else None});write_json(stage/'manifest.json',{'status':'complete','scope':'source_train mechanism screen; source_dev only if passed','ready':pf.READY_SHA,'baseline':pl.SEAL,'train_objects':sha256(tp),'arms':arms,'rules':rules['sha256'],'train_manifest':tr['manifest'],'dev_manifest':dev['manifest'],'training_performed':False,'gpu_hours':0,'cloud_cny':0,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('person_neighbors.py','person_local.py','source_endpoints.py','person_fusion.py','detection_metrics.py','source_policy.py','evaluate.py')},'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('NEIGHBOR_SEALED',out,chosen)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
