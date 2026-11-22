"""Train an appearance residual on original-image crops using each seed's own frozen YOLO backbone."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

import person_local as pl
from person_local import ROOT,pf,dm,se
import person_consistency as pc
from source_contract import ContractError,NAMES,check_equal,read_json,sha256,write_json
from evaluate import verify_subset,stage_subset,original_gt
from source_policy import SelectionPolicy,reference_at
from source_endpoints import grade


def chip(im,b):
    import cv2
    x,y,u,v=b;w=max(2.,1.5*(u-x));h=max(2.,1.5*(v-y));left=(x+u-w)/2;top=(y+v-h)/2
    matrix=np.float32([[w/96,0,left+.5*w/96-.5],[0,h/160,top+.5*h/160-.5]])
    return cv2.warpAffine(im,matrix,(96,160),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))


def vectors(out,records,ct,weight):
    import torch,cv2
    import torch.nn.functional as F
    from ultralytics.nn.tasks import load_checkpoint
    model,_=load_checkpoint(str(weight),fuse=True,device='cuda');model.eval().half().requires_grad_(False);layers=list(model.model[:9]);check_equal('sequential YOLO backbone',[-1]*9,[m.f for m in layers]);features={};buf=[];loc=[]
    def flush():
        with torch.inference_mode():
            a=torch.from_numpy(np.ascontiguousarray(np.stack(buf)[:,:,:,::-1].transpose(0,3,1,2))).cuda().half()/255;f=[]
            for i,l in enumerate(layers):
                a=l(a)
                if i in (4,6,8):f.append(F.adaptive_avg_pool2d(a.float(),(2,1)).flatten(1))
            z=torch.cat(f,1).cpu().numpy()
        if not np.isfinite(z).all():raise ContractError('Non-finite crop features')
        for (key,j),x in zip(loc,z,strict=True):features[key][j]=x.copy()
        loc.clear();buf.clear()
    for r in records:
        k=r['key'];candidates=out[k];features[k]={};active=[(j,b) for j,(s,b) in enumerate(candidates) if s>=.02]
        if not active:continue
        im=cv2.imdecode(np.fromfile(ct['root']/r['image'],np.uint8),cv2.IMREAD_COLOR);check_equal('frame size',(r['height'],r['width']),im.shape[:2])
        for j,b in active:
            buf.append(chip(im,b));loc.append((k,j))
            if len(buf)==32:flush()
    if buf:flush()
    del model;torch.cuda.empty_cache();return features


def fit(ct,records,det,features):
    import torch
    import torch.nn.functional as F
    xx=[];ss=[];yy=[];counts={'positive':0,'wrong_class':0,'duplicate':0,'soft':0}
    for r in records:
        key=r['key'];out=det[key];row=pc.personal(r,original_gt(r,ct['root']),out);gc,_,score,iou=dm.arrays(row);pairs=np.asarray(row['ap_pairs'][0],int).reshape(-1,2);hits=set(pairs[:,1]);occupied=pairs[:,0]
        for j,x in features[key].items():
            s=float(score[j]);y=s;kind='soft'
            if j in hits:y=1.;kind='positive'
            elif (iou[occupied,j]>=.5).any():y=0.;kind='duplicate'
            elif iou[gc!=0,j].max(initial=0)>=.5 and iou[gc==0,j].max(initial=0)<.1:y=0.;kind='wrong_class'
            xx.append(x);ss.append(s);yy.append(y);counts[kind]+=1
    x=np.stack(xx);mean=x.mean(0,dtype=np.float64);sd=np.maximum(x.std(0,dtype=np.float64),.02);des=torch.tensor(np.clip((x-mean)/sd,-5,5),dtype=torch.float64);w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True);b=torch.zeros((),dtype=torch.float64,requires_grad=True);z=torch.logit(torch.tensor(ss,dtype=torch.float64).clamp(1e-5,1-1e-5));y=torch.tensor(yy,dtype=torch.float64);opt=torch.optim.LBFGS([w,b],max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(z+des@w+b,y)+.2*w.square().sum();loss.backward();return loss
    opt.step(closure);return {'mean':mean.tolist(),'scale':sd.tolist(),'w':w.detach().tolist(),'b':b.item(),'fit_counts':counts,'iterations':opt.state[w]['n_iter'],'loss':float(closure().detach()),'layers':[4,6,8],'shape':[160,96],'residual':.5,'l2':.2}


def rank(out,features,m):
    if not features:return list(out)
    ix=np.array(list(features));x=np.stack([features[j] for j in ix]);q=np.clip((x-m['mean'])/m['scale'],-5,5)@m['w']+m['b'];s=np.clip(np.array([out[j][0] for j in ix]),1e-5,1-1e-5);score=1/(1+np.exp(-(np.log(s/(1-s))+.5*np.clip(q,-2,2))));res=list(out)
    for j,p in zip(ix,score,strict=True):res[j]=(float(p),out[j][1])
    return res


def validate(ct,records,det,feats,mod,rules):
    from person_neighbors import quick
    before=[];after=[]
    for r in records:
        k=r['key'];gt=original_gt(r,ct['root']);before.append(pc.personal(r,gt,det[k]));after.append(pc.personal(r,gt,sorted(rank(det[k],feats[k],mod),key=lambda v:-v[0])))
    rule=rules['classes']['pedestrian'];a,b=[quick(x,rule) for x in (before,after)];pr=b['point'];ps=a['point'];ok=pr['origin']=='rule' and b['ap50']>=a['ap50'] and pr['recall']>=ps['recall'] and (b['ap50']>a['ap50'] or pr['recall']>ps['recall']);return {'control':a,'appearance':b,'passed':ok}


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or scratch.is_relative_to(ROOT) or not pl.re.fullmatch(r'views_\d{3}',out.name):raise ContractError('Expected fresh views_NNN and external scratch')
    check_equal('alias mean did not qualify',False,read_json(ROOT/'runs/seed_004/comparison.json')['numeric_five'])
    rules,dev,arms,paths=pl.prepare();tr=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');records=pc.samples(tr);tr=dict(tr,records=records,rules=rules);gpu=se.probe_gpu();pol=SelectionPolicy(rules,reference_at(ROOT));screen={};groups=[]
    if {r['group_id'] for r in records}&{r['group_id'] for r in dev['records']}:raise ContractError('Group leakage')
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'result';stage.mkdir()
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled');import torch,torchvision,ultralytics
        vers=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('locked versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],vers);torch.set_num_threads(4);data=stage_subset(tr,tmp/'train')
        for seed in (29,17,43):
            folder=stage/'appearance'/str(seed);folder.mkdir(parents=True);weight=ROOT/arms[seed]['path'];found,obs=pc.extract(tr,data,weight,tmp/f'capture{seed}');det={k:v[0] for k,v in found.items()};del found
            feats=vectors(det,records,tr,weight);mod=fit(tr,records[:1024],det,feats);inner=validate(tr,records[1024:],det,feats,mod,rules);write_json(folder/'fit.json',{'model':mod,'validation':inner,'checkpoint':arms[seed],'observations':obs});del feats,det;print('APPEARANCE_INTERNAL',seed,inner,flush=True);screen[seed]={'inner_pass':inner['passed']}
            if not inner['passed']:break
            baseline=dm.read_rows(paths[seed]);det={r['key']:[(s,b) for c,s,b in pf._detections(r['pred']) if c==0] for r in baseline};idx={r['key']:r for r in dev['records']};check_equal('dev keys',set(idx),set(pf._dev_index(baseline)))
            for r in baseline:
                for k in ('group_id','image_sha256','label_sha256','width','height'):check_equal(k,idx[r['key']][k],r[k])
            start=time.perf_counter();feats=vectors(det,dev['records'],dev,weight);rows=[]
            for r in baseline:
                k=r['key'];row,cap=pl.replace(r,rank(det[k],feats[k],mod),False);check_equal('cap',0,cap);rows.append(row)
            elapsed=time.perf_counter()-start;old=read_json(pl.BASE/str(seed)/'metrics.json');m,ma,c,failed,warn=grade(rows,rules,pol,old)
            for n in NAMES[1:]:check_equal('other workpoint',old['workpoints'][n],m['workpoints'][n])
            trans,g=pl.transitions(rows,dm.read_rows(pl.BASE/str(seed)/'matches.jsonl.gz'),ma);groups.append(g);screen[seed].update(failed_five=failed,warnings=warn);m.update(failed_five=failed,relative_warnings=warn,transitions=trans,paired_recall=pl.paired_interval([g]),crop_count=sum(len(v) for v in feats.values()),crop_stage_seconds=elapsed,calibration_sha256=sha256(folder/'fit.json'));write_json(folder/'metrics.json',m);dm.write_rows(folder/'objects.jsonl.gz',rows);dm.write_rows(folder/'matches.jsonl.gz',ma);np.savez_compressed(folder/'pr.npz',**c);check_equal('unchanged detector',arms[seed]['sha256'],sha256(weight));print('APPEARANCE_DEV',seed,m['metrics']['per_class']['pedestrian']['ap50'],m['workpoints']['pedestrian'],failed,warn,flush=True)
            if failed:break
        chosen='appearance' if len(screen)==3 and all(v['inner_pass'] and not v['failed_five'] for v in screen.values()) and not any(sum(t in v['warnings'] for v in screen.values())>=2 for t in ('area_S','width_le8','small_thin_union','ap75')) else None
        write_json(stage/'train.json',{'fit':[r['key'] for r in records[:1024]],'inner':[r['key'] for r in records[1024:]]});write_json(stage/'comparison.json',{'selected':chosen,'screen':screen,'paired_recall':pl.paired_interval(groups) if groups else None});write_json(stage/'manifest.json',{'status':'complete','scope':'source_train grouped fit; source_dev after internal gate','baseline':pl.SEAL,'ready':pf.READY_SHA,'rules':rules['sha256'],'train_manifest':tr['manifest'],'dev_manifest':dev['manifest'],'arms':arms,'gpu':gpu,'versions':vers,'cloud_cny':0,'calibration_trained':True,'detector_training':False,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('person_appearance.py','person_consistency.py','person_neighbors.py','source_endpoints.py','person_fusion.py','person_local.py','detection_metrics.py','source_policy.py','evaluate.py')},'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('APPEARANCE_SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
