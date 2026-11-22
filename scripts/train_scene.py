"""Multi-scale, full-scene supervised adaptation with immutable BN and fixed-epoch EMA selection."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np

import person_local as pl
from person_local import ROOT,pf,se
from source_contract import ContractError,NAMES,read_json,check_equal,sha256,write_json,canonical_hash
from evaluate import verify_subset,original_gt,stage_subset
from source_policy import SelectionPolicy,reference_at
from person_prior import evaluate

SCALES=(640,960,1280,960)
RECIPE={'epochs':6,'microbatch':2,'accumulate':16,'updates_per_epoch':125,'scales':SCALES,'lr_start':.0008,'lr_end':.00008,'momentum':.937,'weight_decay':.00025,'clip':10.,'fixed_bn':True,'fixed_final_ema':True}


def geometry(w,h,size):
    nw,nh=round(w*size/max(w,h)),round(h*size/max(w,h));ow=int(math.ceil(nw/32+.5))*32;oh=int(math.ceil(nh/32+.5))*32
    return nw,nh,ow,oh,(ow-nw)//2,(oh-nh)//2


def transformed(box,w,h,size,flip):
    nw,nh,ow,oh,l,t=geometry(w,h,size);x,y,u,v=box
    cx=((x+u)*.5*nw/w+l)/ow;cy=((y+v)*.5*nh/h+t)/oh
    return (1-cx if flip else cx,cy,(u-x)*nw/w/ow,(v-y)*nh/h/oh)


class SceneDataset:
    def __init__(self,ct):
        self.root=ct['root'];self.records=ct['records'];self.labels=[original_gt(r,self.root) for r in self.records]
        check_equal('training only',{'source_train'},{r['purpose'] for r in self.records});check_equal('4000 unique images',4000,len({r['key'] for r in self.records}))
    def __len__(self):return len(self.records)
    def __getitem__(self,item):
        import torch,cv2
        idx,size,flip,gain=item;r=self.records[idx];im=cv2.imdecode(np.fromfile(self.root/r['image'],np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2];check_equal('image dimensions',(r['height'],r['width']),(h,w));nw,nh,ow,oh,l,t=geometry(w,h,size)
        im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR);im=cv2.LUT(im,np.rint(np.clip(np.arange(256)*gain,0,255)).astype(np.uint8));im=cv2.copyMakeBorder(im,t,oh-nh-t,l,ow-nw-l,cv2.BORDER_CONSTANT,value=(114,114,114))
        if flip:im=im[:,::-1]
        y=self.labels[idx];boxes=np.asarray([transformed(b,w,h,size,flip) for c,b in y],np.float32).reshape(-1,4)
        return idx,size,torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))),torch.tensor([c for c,b in y],dtype=torch.float32).reshape(-1,1),torch.from_numpy(boxes)


def collate(items):
    import torch
    idx,size,image,cl,box=zip(*items,strict=True);check_equal('one scale per batch',1,len(set(size)))
    return {'sequence':list(idx),'size':size[0],'img':torch.stack(image),'cls':torch.cat(cl),'bboxes':torch.cat(box),'batch_idx':torch.cat([torch.full((len(c),),float(i)) for i,c in enumerate(cl)])}


def schedule(seed,epoch):
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,20261122]));ids=rng.permutation(4000).tolist();items=[]
    for j,idx in enumerate(ids):items.append((idx,SCALES[(j//2)%4],bool(rng.random()<.5),float(rng.uniform(.85,1.15))))
    return [items[i:i+2] for i in range(0,len(items),2)]


def atomic(path,record):
    t=path.with_suffix('.tmp');write_json(t,record);t.replace(path)


def fit(folder,ds,seed,weight,spec):
    import torch
    from torch.nn.modules.batchnorm import _BatchNorm
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import ModelEMA
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    m,_=load_checkpoint(str(weight),device='cuda');m.train();args=dict(m.args);m.args=SimpleNamespace(**args);check_equal('6 class detector',NAMES,list(m.names.values()))
    for name,p in m.named_parameters():p.requires_grad_('.dfl.' not in name)
    bn={};fixed=set()
    for n,l in m.named_modules():
        if isinstance(l,_BatchNorm):
            l.eval().requires_grad_(False);fixed.update(f'{n}.{k}' for k in l.state_dict());bn[n]=l
    initial={k:v.detach().clone() for k,v in m.state_dict().items() if k in fixed};wd=[];free=[]
    for n,l in m.named_modules():
        for k,p in l.named_parameters(recurse=False):
            if p.requires_grad:(wd if isinstance(l,torch.nn.Conv2d) and k=='weight' else free).append(p)
    opt=torch.optim.SGD([{'params':wd,'weight_decay':.00025},{'params':free,'weight_decay':0.}],lr=.0008,momentum=.937);scaler=torch.amp.GradScaler('cuda',init_scale=1.);ema=ModelEMA(m);crit=v8DetectionLoss(m);hist=[];snap=folder/'resume.pt'
    folder.mkdir(parents=True,exist_ok=True)
    if snap.exists():
        state=torch.load(snap,map_location='cpu',weights_only=False);check_equal('resume recipe',spec,state['identity']);m.load_state_dict(state['model']);ema.ema.load_state_dict(state['ema']);ema.updates=state['updates'];opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);hist=state['history'];del state
    for epoch in range(len(hist),6):
        start=time.monotonic();schedule_batches=schedule(seed,epoch);expected=[i[0] for b in schedule_batches for i in b];lr=.00008+(.0008-.00008)*(1+math.cos(math.pi*epoch/5))/2
        for v in opt.param_groups:v['lr']=lr
        loader=torch.utils.data.DataLoader(ds,batch_sampler=schedule_batches,num_workers=2,collate_fn=collate,pin_memory=True,persistent_workers=False,generator=torch.Generator().manual_seed(seed*100+epoch));observed=[];sizes=Counter();sums=np.zeros(3);updates=0;opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();nonempty=0
        for step,b in enumerate(loader,1):
            observed+=b.pop('sequence');sizes[b.pop('size')]+=len(b['img']);b={k:v.cuda(non_blocking=True) for k,v in b.items()};b['img']=b['img'].float().div_(255);nonempty+=len(b['cls'])
            with torch.autocast('cuda',dtype=torch.float16):
                pred=m(b['img']);_,loss,_=crit.get_assigned_targets_and_loss(pred,b);objective=2*loss.sum()
            if not objective.isfinite():raise ContractError('Nonfinite native detection loss')
            scaler.scale(objective).backward();sums+=loss.detach().float().cpu().numpy()
            if step%16==0:
                scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(m.parameters(),10.,error_if_nonfinite=True);scale=scaler.get_scale();scaler.step(opt);scaler.update();check_equal('no skipped update',True,scaler.get_scale()>=scale);opt.zero_grad(set_to_none=True);ema.update(m);updates+=1
            if step%400==0:print('SCENE_BATCH',seed,epoch+1,step,flush=True)
        check_equal('exact epoch traversal',expected,observed);check_equal('all training sources',list(range(4000)),sorted(observed));check_equal('native scales',{640:1000,960:2000,1280:1000},dict(sizes));check_equal('optimizer steps',125,updates)
        for k,v in initial.items():
            if not torch.equal(m.state_dict()[k],v):raise ContractError('BN changed '+k)
        hist.append({'epoch':epoch+1,'lr':lr,'updates':updates,'ema_updates':ema.updates,'main_images':len(observed),'ground_truth_count':nonempty,'scale_counts':dict(sizes),'augmentation_schedule_sha256':canonical_hash(schedule_batches),'consumption_sha256':canonical_hash(observed),'native_box_cls_dfl':(sums/len(loader)).tolist(),'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated()});torch.save({'identity':spec,'history':hist,'updates':ema.updates,'model':m.state_dict(),'ema':ema.ema.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict()},snap.with_suffix('.tmp.pt'));snap.with_suffix('.tmp.pt').replace(snap);atomic(folder/'train.json',{'identity':spec,'history':hist,'status':'training'});print('SCENE_EPOCH',seed,hist[-1],flush=True)
    check_equal('total updates',750,ema.updates)
    with torch.no_grad():
        ema_state=ema.ema.state_dict()
        for k,v in initial.items():ema_state[k].copy_(v)
    model=deepcopy(ema.ema).cpu().half().eval();model.args=args;pt=folder/'last.pt';torch.save({'ema':model,'model':None,'epoch':5,'updates':750,'train_args':args,'scope':'fixed last scene adaptation'},pt.with_suffix('.tmp.pt'));pt.with_suffix('.tmp.pt').replace(pt)
    reloaded,_=load_checkpoint(str(pt),device='cpu');check_equal('published frozen BN',True,all(torch.equal(reloaded.state_dict()[k],v.cpu()) for k,v in initial.items()));check_equal('EMA reload',True,all(torch.equal(v.float(),reloaded.state_dict()[k].float()) for k,v in model.state_dict().items()));atomic(folder/'train.json',{'identity':spec,'history':hist,'status':'complete','updates':750,'checkpoint_sha256':sha256(pt),'bn_frozen':len(bn),'role':'last_fixed_no_epoch_selection'});snap.unlink();return pt


def run(out,scratch,resume):
    if out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'align_\d{3}',out.name) or scratch.is_relative_to(ROOT) or out.exists()!=resume or (out/'manifest.json').exists():raise ContractError('New align_NNN, or explicit resume of incomplete run required')
    rules,dev,arms,paths=pl.prepare();dev['rules']=rules;tr=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');policy=SelectionPolicy(rules,reference_at(ROOT));gpu=se.probe_gpu();check_equal('train/dev group overlap',set(),{r['group_id'] for r in tr['records']}&{r['group_id'] for r in dev['records']})
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';ds=SceneDataset(tr);screen={};effects=[]
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d)
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled');import torch,torchvision,ultralytics,cv2
        vers=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('locked environment',['2.14.0+cu130','0.29.0+cu130','8.4.171'],vers);torch.set_num_threads(4);cv2.setNumThreads(1);data=stage_subset(dev,tmp/'dev');out.mkdir(exist_ok=resume)
        code={n:sha256(ROOT/'scripts'/n) for n in ('train_scene.py','person_prior.py','person_local.py','source_endpoints.py','person_fusion.py','tricycle_fusion.py','evaluate.py','source_policy.py','detection_metrics.py','source_contract.py')}
        for seed in (29,17,43):
            f=out/str(seed);spec={'recipe':RECIPE,'seed':seed,'source_best':arms[seed],'train_manifest':tr['manifest'],'ready':pf.READY_SHA,'versions':vers,'code_sha256':code,'rules':rules['sha256']};weight=ROOT/arms[seed]['path'];pt=f/'last.pt';train=f/'train.json'
            if pt.exists() and train.exists() and read_json(train)['status']=='complete':
                check_equal('sealed training identity',canonical_hash(spec),canonical_hash(read_json(train)['identity']));check_equal('sealed weight',read_json(train)['checkpoint_sha256'],sha256(pt))
            else:pt=fit(f,ds,seed,weight,spec)
            torch.cuda.empty_cache();sub=f/'dev';sub.mkdir(exist_ok=resume);failed,warn,g=evaluate(seed,pt,sub,dev,data,tmp/f'eval{seed}',policy,rules,paths);effects.append(g);screen[seed]={'failed_five':failed,'warnings':warn};record=read_json(sub/'metrics.json');print('SCENE_EVAL',seed,record['metrics']['per_class']['pedestrian']['ap50'],record['workpoints']['pedestrian'],failed,warn,flush=True)
            if failed:break
        chosen=len(screen)==3 and all(not v['failed_five'] for v in screen.values()) and not any(sum(k in v['warnings'] for v in screen.values())>=2 for k in ('area_S','width_le8','small_thin_union','ap75'))
        write_json(out/'comparison.json',{'recommended':chosen,'screen':screen,'paired_recall_on_evaluated_seeds':pl.paired_interval(effects)});write_json(out/'manifest.json',{'status':'complete','scope':'source_train six supervised epochs; full source_dev each completed seed','source_train':tr['manifest'],'source_dev':dev['manifest'],'ready':pf.READY_SHA,'baseline':pl.SEAL,'source_checkpoints':arms,'recipe':RECIPE,'gpu':gpu,'versions':vers,'cloud_cny':0,'training_performed':True,'final_test_executed':False,'rules':rules['sha256'],'code_sha256':code,'files':{f.relative_to(out).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in out.rglob('*') if f.is_file()}});print('SCENE_SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.output.resolve(),a.scratch.resolve(),a.resume)
