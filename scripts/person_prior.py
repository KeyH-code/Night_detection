"""Train-only linear transfer into the three existing SODA pedestrian output rows."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import person_local as pl
from person_local import ROOT,pf,dm,se
from source_contract import ContractError,NAMES,check_equal,read_json,sha256,write_json
from source_policy import SelectionPolicy,reference_at,summarize
from source_endpoints import grade
from seed_train import COCO
from evaluate import stage_subset,verify_subset,original_gt


def groups(train,dev):
    excluded={r['key'] for r in read_json(ROOT/'configs/rerank.json')['candidates']};used=set();records=[]
    key=lambda r:hashlib.sha256(('prior-1122:'+r['key']).encode()).hexdigest()
    for r in sorted(train['records'],key=key):
        if r['key'] in excluded or r['group_id'] in used:continue
        used.add(r['group_id']);records.append(r)
        if len(records)==2048:break
    check_equal('distinct groups',2048,len(used))
    if used & {r['group_id'] for r in dev['records']}:raise ContractError('train/dev leakage')
    return records


def collect(model,tr,records):
    import torch,torchvision,cv2
    from ultralytics.utils import ops
    head=model.model[-1];captured={};hooks=[]
    for j,t in enumerate(head.cv3):
        def save(module,a,level=j):captured[level]=a[0].detach()
        hooks.append(t[-1].register_forward_pre_hook(save))
    flat=[[] for _ in range(3)];targets=[[] for _ in range(3)];val=[]
    try:
        for seq,rec in enumerate(records):
            side=(960,1280)[seq%2];im=cv2.imdecode(np.fromfile(tr['root']/rec['image'],np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2];check_equal('shape',(rec['height'],rec['width']),(h,w))
            scale=side/max(w,h);nw,nh=round(w*scale),round(h*scale)
            # Ultralytics rect batch=1 adds half a stride before ceil, not a whole stride unconditionally.
            ow=int(np.ceil(nw/32+.5))*32;oh=int(np.ceil(nh/32+.5))*32
            l=(ow-nw)//2;t=(oh-nh)//2;im=cv2.copyMakeBorder(cv2.resize(im,(nw,nh)),t,oh-nh-t,l,ow-nw-l,cv2.BORDER_CONSTANT,value=(114,114,114))
            a=torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))).cuda().half()[None]/255
            with torch.inference_mode():raw=model(a)[0][0]
            if raw.shape[0]!=10 or not raw.isfinite().all():raise ContractError('decoder output')
            score=raw[4].float();ids=torch.where(score>=.02)[0];boxes=ops.xywh2xyxy(raw[:4,ids].T.float());keep=torchvision.ops.nms(boxes,score[ids],.7);ids=ids[keep];boxes=boxes[keep]
            boxes[:,[0,2]]=(boxes[:,[0,2]]-l)/(nw/w);boxes[:,[1,3]]=(boxes[:,[1,3]]-t)/(nh/h);b=ops.clip_boxes(boxes,(h,w)).cpu().numpy();s=score[ids].cpu().numpy();gt=original_gt(rec,tr['root']);gc=np.array([c for c,_ in gt],int);gb=[box for _,box in gt];ious=dm.box_iou(gb,b);labels=np.full(len(s),-1,int)
            match=dm.native_pairs(np.zeros(len(b),int),gc,ious,.5)
            if len(match):labels[match[:,1]]=1
            labels[(ious[gc!=0].max(0,initial=0)>=.5)&(ious[gc==0].max(0,initial=0)<.1)]=0
            offset=0;levels=np.zeros(len(ids),int);features=np.empty((len(ids),256),np.float32)
            for j in range(3):
                z=captured[j][0].reshape(256,-1);mask=(ids>=offset)&(ids<offset+z.shape[1]);use=mask.cpu().numpy();selected=z[:,ids[mask]-offset].T.float().cpu().numpy();features[use]=selected;levels[use]=j
                if seq<1536:
                    flat[j].append(selected);targets[j].append(np.column_stack([s[use],labels[use]]))
                offset+=z.shape[1]
            check_equal('dense anchor count',raw.shape[1],offset)
            if seq>=1536:val.append((rec['key'],rec['group_id'],gc,gb,b,s,levels,features))
            if (seq+1)%512==0:print('PRIOR_FEATURES',seq+1,flush=True)
    finally:
        for h in hooks:h.remove()
    return [np.concatenate(v) for v in flat],[np.concatenate(v) for v in targets],val


def fit(x,targets,old_w,old_b,prior,mode):
    import torch
    import torch.nn.functional as F
    x=x.astype(np.float64);old_w=old_w.astype(np.float64);s,hard=targets.T;z=x@old_w+old_b
    if len(x) and np.max(np.abs(1/(1+np.exp(-np.clip(z,-50,50)))-s))>.005:raise ContractError('captured logits inconsistent with checkpoint')
    matrix=prior.astype(float)[None,:] if mode=='prior' else np.vstack([prior.astype(float),np.eye(len(old_w))]);q=x@matrix.T;diff=z-z.mean();slope=(diff@(q-q.mean(0)))/max(1e-8,diff@diff)
    res=q-q.mean(0)-diff[:,None]*slope;std=np.maximum(res.std(0),.02);design=torch.from_numpy(res/std);theta=torch.zeros(len(matrix),dtype=torch.float64,requires_grad=True)
    target=torch.from_numpy(np.where(hard<0,s,hard));weight=torch.from_numpy(np.maximum(.1,4*s*(1-s))*np.where(hard<0,.75,1.));offset=torch.from_numpy(z)
    opt=torch.optim.LBFGS([theta],lr=1.,max_iter=60,line_search_fn='strong_wolfe');pen=torch.full_like(theta,.05);pen[0]=.012
    def objective():
        opt.zero_grad();loss=(F.binary_cross_entropy_with_logits(offset+design@theta,target,reduction='none')*weight).sum()/weight.sum()+(pen*theta.square()).sum();loss.backward();return loss
    opt.step(objective);loss=float(objective().detach());coef=theta.detach().numpy()*.5/std
    dw=coef@(matrix-slope[:,None]*old_w);db=float(coef@(-q.mean(0)+(z.mean()-old_b)*slope))
    np.testing.assert_allclose(x@(old_w+dw)+old_b+db,z+.5*(design.detach().numpy()@theta.detach().numpy()),rtol=0,atol=2e-9)
    return (old_w+dw).astype(np.float16).astype(np.float32),float(np.float16(old_b+db)),{'samples':len(x),'positive':int((hard==1).sum()),'confirmed_other':int((hard==0).sum()),'soft_preserved':int((hard<0).sum()),'loss':loss,'iterations':opt.state[theta]['n_iter'],'delta_norm':float(np.linalg.norm(dw)),'delta_bias':db}


def internal(val,params,rules):
    rows=[]
    for key,group,gc,gb,b,s,levels,x in val:
        score=s.copy()
        if params:
            for j,(w,bias,*_) in enumerate(params):
                use=levels==j;score[use]=1/(1+np.exp(-np.clip(x[use]@w+bias,-40,40)))
        ix=np.argsort(-score,kind='stable');r={'key':key,'group_id':group,'gt':{'cls':gc.tolist(),'boxes':gb},'pred':{'cls':[0]*len(ix),'conf':score[ix].tolist(),'boxes':b[ix].tolist()}}
        flags,_=dm.ap_flags(r);r['ap_tp']=flags.tolist();rows.append(r)
    a=dm.summarize_ap(rows)[0]['per_class']['pedestrian'];rule=rules['classes']['pedestrian'];w=dm.choose_point(dm.work_curve(rows,0),rule['precision'],rule['recall']);return {'ap50':a['ap50'],'ap50_95':a['ap50_95'],'point':w}


def fit_seed(seed,weights,tr,records,folder,rules,fixed=None):
    import torch
    from ultralytics.nn.tasks import load_checkpoint
    model,_=load_checkpoint(str(weights),device='cuda',fuse=True);model.half().eval().requires_grad_(False)
    params=[(t[-1].weight[0].detach().cpu().float().numpy().ravel(),float(t[-1].bias[0])) for t in model.model[-1].cv3]
    pre=ROOT/'weights/yolo11m.pt';check_equal('COCO pretrain',COCO,sha256(pre));coco,_=load_checkpoint(str(pre),device='cpu');check_equal('COCO row','person',coco.names[0]);pr=[z[-1].weight[0].detach().numpy().reshape(-1) for z in coco.model[-1].cv3];del coco
    x,target,val=collect(model,tr,records);del model;torch.cuda.empty_cache();baseline=internal(val,None,rules);screen={};chosen=None;solution=None
    for mode in ((fixed,) if fixed else ('prior','feature')):
        p=[fit(x[j],target[j],*params[j],pr[j],mode) for j in range(3)];score=internal(val,p,rules);screen[mode]={'fit':[v[2] for v in p],'internal':score}
        if score['point']['origin']=='rule' and score['ap50']>=baseline['ap50'] and score['point']['recall']>=baseline['point']['recall'] and (score['point']['recall']>baseline['point']['recall'] or score['ap50']>baseline['ap50']):chosen=mode;solution=p;break
    write_json(folder/'fit.json',{'seed':seed,'baseline':baseline,'screen':screen,'chosen':chosen,'inputs_sha256':sha256(weights),'pretrain':COCO,'inner_scope':'512 source_train risk groups; fixed native NMS candidate pool conf>=.02','recipe':'1/2 regularized residual folded into three fp16 rows'})
    print('PRIOR_INTERNAL',seed,chosen,baseline,screen,flush=True)
    if chosen:
        src,_=load_checkpoint(str(weights),device='cpu');state=src.state_dict();changed={}
        with torch.no_grad():
            for j,(w,bias,_) in enumerate(solution):
                layer=src.model[-1].cv3[j][-1];layer.weight[0].copy_(torch.from_numpy(w).reshape_as(layer.weight[0]));layer.bias[0]=bias
                for k,t in src.named_parameters():
                    if t is layer.weight or t is layer.bias:changed[k]=t[0].clone()
        check_equal('three person weights and biases',6,len(changed));pt=folder/'model.pt';torch.save({'model':src.half().eval(),'train_args':src.args,'epoch':-1,'scope':'source-only person row adapter'},pt)
        restored,_=load_checkpoint(str(pt),device='cpu');orig,_=load_checkpoint(str(weights),device='cpu');old=orig.state_dict()
        for k,v in restored.state_dict().items():
            if k in changed:
                check_equal('folded coefficients '+k,True,torch.equal(v[0],changed[k]));check_equal('other classes '+k,True,torch.equal(v[1:],old[k][1:]))
            else:check_equal('frozen '+k,True,torch.equal(v,old[k]))
    return chosen


def evaluate(seed,weight,folder,ct,data,scratch,policy,rules,paths):
    import torch
    v={};obs=[];native,ped=pf.PROFILES['a1'][2:];sizes,mode=se.tf.PROFILES['low416-or'];scratch.mkdir()
    for size,flip in dict.fromkeys((native,*ped,*[(z,False) for z in sizes])):
        rows,info=pf._capture(torch,ct,data,weight,scratch,size,flip);v[size,flip]=rows;obs.append(info)
    a,pc=pf.evaluate_rows(v[native],tuple(v[k] for k in ped));rows,tc=se.tf.merge_dev(a,tuple(v[z,False] for z in sizes),mode);base=dm.read_rows(paths[seed]);check_equal('GT and groups',[(r['key'],pf._identity(r)) for r in base],[(r['key'],pf._identity(r)) for r in rows])
    old=read_json(pl.BASE/str(seed)/'metrics.json');rec,ma,curves,failed,warn=grade(rows,rules,policy,old);moved,grp=pl.transitions(rows,dm.read_rows(pl.BASE/str(seed)/'matches.jsonl.gz'),ma);single,_,_=summarize(v[native],rules)
    rec.update(failed_five=failed,relative_warnings=warn,transitions=moved,paired_recall=pl.paired_interval([grp]),single640={'metrics':single['metrics'],'workpoints':single['workpoints']},observations=obs,checkpoint_sha256=sha256(weight),caps={'person':pc,'tricycle':tc});write_json(folder/'metrics.json',rec);dm.write_rows(folder/'objects.jsonl.gz',rows);dm.write_rows(folder/'matches.jsonl.gz',ma);np.savez_compressed(folder/'pr.npz',**curves)
    return failed,warn,grp


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'prior_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Expected new prior_NNN and external scratch')
    rules,dev,arms,paths=pl.prepare();dev['rules']=rules;gpu=se.probe_gpu();train=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');records=groups(train,dev);policy=SelectionPolicy(rules,reference_at(ROOT));screen={};effects=[];chosen=None
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'results';stage.mkdir()
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch,torchvision,ultralytics
        vers=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],vers);torch.set_num_threads(4);data=stage_subset(dev,tmp/'dev')
        for seed in (29,17,43):
            folder=stage/'adapter'/str(seed);folder.mkdir(parents=True);mode=fit_seed(seed,ROOT/arms[seed]['path'],train,records,folder,rules,screen[29]['internal'] if seed!=29 else None);screen[seed]={'internal':mode}
            if not mode:break
            failed,warn,grp=evaluate(seed,folder/'model.pt',folder,dev,data,tmp/f'views-{seed}',policy,rules,paths);effects.append(grp);screen[seed].update(failed_five=failed,relative_warnings=warn);print('PRIOR_DEV',seed,failed,warn,flush=True)
            if failed:break
        if len(screen)==3 and all(v['internal'] and not v.get('failed_five') for v in screen.values()) and not any(sum(k in v['relative_warnings'] for v in screen.values())>=2 for k in ('area_S','width_le8','small_thin_union','ap75')):chosen='adapter'
        write_json(stage/'comparison.json',{'selected':chosen,'screen':screen,'paired_recall_on_evaluated_seeds':pl.paired_interval(effects) if effects else None});write_json(stage/'train.json',{'train_keys':[r['key'] for r in records[:1536]],'validation_keys':[r['key'] for r in records[1536:]],'sample_order':'sha256 prior-1122:image key; one image per group','scales':[960,1280]})
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_train calibration and source_dev evaluation if internal gate passes','baseline':pl.SEAL,'selected':chosen,'ready':pf.READY_SHA,'rules':rules['sha256'],'dev_manifest':dev['manifest'],'train_manifest':train['manifest'],'arms':arms,'coco':COCO,'versions':vers,'gpu':gpu,'training_performed':True,'cloud_cny':0,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('person_prior.py','source_endpoints.py','seed_train.py','person_fusion.py','tricycle_fusion.py','person_local.py','source_policy.py','detection_metrics.py','evaluate.py')},'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('PRIOR_SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
