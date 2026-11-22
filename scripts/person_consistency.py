"""Score locked four-view pedestrian boxes using train-only estimates of disagreement."""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import person_local as pl
from person_local import ROOT,pf,dm,se
from source_endpoints import grade
from evaluate import stage_subset,verify_subset,original_gt
from source_policy import SelectionPolicy,reference_at
from source_contract import ContractError,NAMES,check_equal,read_json,sha256,write_json

SPECS=pf.PROFILES['a1'][3]
FEATURES=('vote_fraction','high_minus_low','flip_minus_orig','score_span','relative_sigma','maximum_share','iou_mean','iou_min','center_x_sd','center_y_sd','log_width_sd','log_height_sd','high_minus_low_x_width','vote_x_score')


def clusters(views):
    vals=[(b,s,j) for j,v in enumerate(views) for s,b in v if s>pf.FLOOR]
    if not vals:return [],np.empty((0,len(FEATURES)))
    boxes,scores,votes=zip(*vals,strict=True);boxes=np.asarray(boxes,np.float32);scores=np.asarray(scores,np.float32);votes=np.asarray(votes,int)
    sums,means=np.zeros_like(boxes),np.zeros_like(boxes);masses=np.zeros_like(scores);used=np.zeros((len(vals),4),bool);members=[]
    for i in np.argsort(-scores,kind='stable'):
        overlaps=pf._iou(boxes[i],means[:len(members)]);overlaps[used[:len(members),votes[i]]]=-1;j=int(np.argmax(overlaps)) if len(overlaps) else -1
        if j<0 or overlaps[j]<pf.CLUSTER_IOU:j=len(members);members.append([])
        masses[j]+=scores[i];sums[j]+=boxes[i]*scores[i];means[j]=sums[j]/masses[j];used[j,votes[i]]=True;members[j].append(i)
    total=masses[:len(members)]/4;active=np.argsort(-total,kind='stable');active=active[total[active]>pf.FLOOR];outputs=[];features=[]
    while len(active):
        i,rest=active[0],active[1:];active=rest[pf._iou(means[i],means[rest])<=pf.NMS_IOU];ids=members[i];b=boxes[ids].astype(float);s=np.zeros(4);s[votes[ids]]=scores[ids]
        outputs.append((float(total[i]),means[i].tolist()));vote=len(ids)/4;hl=(s[1]+s[3]-s[0]-s[2])/2;flip=(s[2]+s[3]-s[0]-s[1])/2
        iou=dm.box_iou(b,b)[np.triu_indices(len(ids),1)] if len(ids)>1 else np.array([0.]);w=max(1,means[i,2]-means[i,0]);h=max(1,means[i,3]-means[i,1]);wh=np.maximum(b[:,2:]-b[:,:2],1);cent=(b[:,2:]+b[:,:2])/2
        features.append([vote,hl,flip,np.ptp(s),s.std()/(s.mean()+.05),s.max()/(s.sum()+1e-8),iou.mean(),iou.min(),cent[:,0].std()/w,cent[:,1].std()/h,np.log(wh[:,0]).std(),np.log(wh[:,1]).std(),hl*np.clip(np.log(w/24),-3,3),vote*total[i]])
    return outputs,np.array(features).reshape(-1,len(FEATURES))


def extract(ct,data,weight,folder):
    import torch
    folder.mkdir();views={};obs=[]
    for spec in SPECS:
        rows,info=pf._capture(torch,ct,data,weight,folder,*spec);obs.append(info)
        for r in rows:views.setdefault(r['key'],[]).append([(s,b) for c,s,b in pf._detections(r['pred']) if c==0])
        del rows;gc.collect()
    return {k:clusters(v) for k,v in views.items()},obs


def personal(r,gt,out):
    scores=[s for s,_ in out];b=[box for _,box in out];cls=[0]*len(b);gc=[c for c,_ in gt];gb=[b for _,b in gt];row={'key':r['key'],'group_id':r['group_id'],'gt':{'cls':gc,'boxes':gb},'pred':{'conf':scores,'cls':cls,'boxes':b}};tp,pairs=dm.ap_flags(row);row.update(ap_tp=tp.tolist(),ap_pairs=pairs);return row


def samples(ct):
    excluded={v['key'] for v in read_json(ROOT/'configs/rerank.json')['candidates']};groups=set();res=[]
    for r in sorted(ct['records'],key=lambda x:hashlib.sha256(('consistency:'+x['key']).encode()).digest()):
        if r['group_id'] in groups or r['key'] in excluded:continue
        res.append(r);groups.add(r['group_id'])
        if len(res)==1536:break
    check_equal('distinct training groups',1536,len(groups));return res


def train(ct,records,features):
    import torch
    import torch.nn.functional as F
    xx=[];ss=[];yy=[];counts={'positive':0,'wrong_class':0,'duplicate':0,'soft':0}
    for r in records:
        out,x=features[r['key']];gt=original_gt(r,ct['root']);row=personal(r,gt,out);g,p,s,iou=dm.arrays(row);pairs=np.array(row['ap_pairs'][0],int).reshape(-1,2);hits=set(pairs[:,1]);occupied=pairs[:,0]
        for j in np.flatnonzero(s>=.02):
            if j in hits:label=1.;kind='positive'
            elif (iou[occupied,j]>=.5).any():label=0.;kind='duplicate'
            elif iou[g!=0,j].max(initial=0)>=.5 and iou[g==0,j].max(initial=0)<.1:label=0.;kind='wrong_class'
            else:label=float(s[j]);kind='soft'
            counts[kind]+=1;xx.append(x[j]);ss.append(s[j]);yy.append(label)
    x=np.stack(xx);mean=x.mean(0);sd=np.maximum(x.std(0),.03);feat=torch.tensor(np.clip((x-mean)/sd,-5,5),dtype=torch.float64);target=torch.tensor(yy,dtype=torch.float64);s=torch.tensor(ss,dtype=torch.float64).clamp(.00001,.99999);z=torch.logit(s)
    w=torch.zeros(len(FEATURES),dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([w,bias],max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(z+feat@w+bias,target)+.02*w.square().sum();loss.backward();return loss
    opt.step(closure);return {'mean':mean.tolist(),'scale':sd.tolist(),'weights':w.detach().tolist(),'bias':float(bias.detach()),'counts':counts,'iterations':opt.state[w]['n_iter'],'loss':float(closure().detach()),'features':FEATURES}


def apply(out,x,model):
    if not out:return []
    s=np.array([v for v,b in out]);active=s>=.02;z=np.clip((x-model['mean'])/model['scale'],-5,5)@model['weights']+model['bias'];logit=np.log(np.clip(s,1e-5,1-1e-5)/np.clip(1-s,1e-5,1));news=np.where(active,1/(1+np.exp(-(logit+.5*np.clip(z,-2,2)))),s);return [(float(p),b) for p,(_,b) in zip(news,out,strict=True)]


def validate(ct,records,features,model,rules):
    vals=[[],[]];rule=rules['classes']['pedestrian']
    for r in records:
        out,x=features[r['key']];gt=original_gt(r,ct['root']);vals[0].append(personal(r,gt,out));vals[1].append(personal(r,gt,sorted(apply(out,x,model),key=lambda t:-t[0])))
    result=[]
    for rows in vals:
        ap=dm.summarize_ap(rows)[0]['per_class']['pedestrian'];point=dm.choose_point(dm.work_curve(rows,0),rule['precision'],rule['recall']);result.append({'ap50':ap['ap50'],'ap50_95':ap['ap50_95'],'workpoint':point})
    a,b=result;passed=b['workpoint']['origin']=='rule' and b['ap50']>=a['ap50'] and b['workpoint']['recall']>=a['workpoint']['recall'] and (b['ap50']>a['ap50'] or b['workpoint']['recall']>a['workpoint']['recall']);return result,passed


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'views_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Expected fresh views_NNN and external scratch')
    rules,dev,arms,paths=pl.prepare();tr=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');records=samples(tr);tr=dict(tr,records=records,rules=rules);dev['rules']=rules;gpu=se.probe_gpu();policy=SelectionPolicy(rules,reference_at(ROOT))
    if {r['group_id'] for r in records}&{r['group_id'] for r in dev['records']}:raise ContractError('training/development leakage')
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'result';stage.mkdir()
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch,torchvision,ultralytics
        versions=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions);torch.set_num_threads(4);features,obs=extract(tr,stage_subset(tr,tmp/'train'),ROOT/arms[29]['path'],tmp/'capture');model=train(tr,records[:1024],features);inner,passed=validate(tr,records[1024:],features,model,rules);del features
        write_json(stage/'fit.json',{'model':model,'internal':inner,'passed':passed,'fitting_keys':[r['key'] for r in records[:1024]],'validation_keys':[r['key'] for r in records[1024:]],'native':obs});print('CONSISTENCY_INTERNAL',inner,passed,flush=True)
        screen={};groups=[];chosen=None
        if passed:
            data=stage_subset(dev,tmp/'dev')
            for seed in (29,17,43):
                baseline=dm.read_rows(paths[seed]);features,obs=extract(dev,data,ROOT/arms[seed]['path'],tmp/f'seed-{seed}');rows=[]
                for r in baseline:
                    outp,x=features[r['key']];old=[(s,b) for c,s,b in pf._detections(r['pred']) if c==0];check_equal('expected original consensus '+r['key'],old,outp[:len(old)])
                    z,cap=pl.replace(r,apply(old,x[:len(old)],model),False);check_equal('no extra cap',0,cap);rows.append(z)
                old=read_json(pl.BASE/str(seed)/'metrics.json');m,ma,curves,fail,warn=grade(rows,rules,policy,old)
                for n in NAMES[1:]:check_equal('protected workpoint '+n,old['workpoints'][n],m['workpoints'][n])
                trans,g=pl.transitions(rows,dm.read_rows(pl.BASE/str(seed)/'matches.jsonl.gz'),ma);groups.append(g);m.update(failed_five=fail,relative_warnings=warn,transitions=trans,paired_recall=pl.paired_interval([g]),native=obs);f=stage/'consistency'/str(seed);f.mkdir(parents=True);write_json(f/'metrics.json',m);dm.write_rows(f/'objects.jsonl.gz',rows);dm.write_rows(f/'matches.jsonl.gz',ma);np.savez_compressed(f/'pr.npz',**curves);screen[seed]={'failed_five':fail,'warnings':warn};print('CONSISTENCY_DEV',seed,m['metrics']['per_class']['pedestrian']['ap50'],m['workpoints']['pedestrian'],fail,warn,flush=True)
                if fail:break
            if len(screen)==3 and all(not v['failed_five'] for v in screen.values()) and not any(sum(t in v['warnings'] for v in screen.values())>=2 for t in ('area_S','width_le8','small_thin_union','ap75')):chosen='consistency'
        write_json(stage/'comparison.json',{'selected':chosen,'internal_pass':passed,'screen':screen,'paired_recall':pl.paired_interval(groups) if groups else None});write_json(stage/'manifest.json',{'status':'complete','scope':'source_train fitting and internal validation; source_dev only after gate','ready':pf.READY_SHA,'rules':rules['sha256'],'baseline':pl.SEAL,'arms':arms,'train_manifest':tr['manifest'],'dev_manifest':dev['manifest'],'gpu':gpu,'versions':versions,'cloud_cny':0,'detector_training':False,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('person_consistency.py','person_fusion.py','source_endpoints.py','person_local.py','detection_metrics.py','source_policy.py','evaluate.py')},'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('CONSISTENCY_SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
