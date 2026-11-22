"""Calibrate running BN moments using only a small, group-disjoint source_train whitelist."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import person_local as pl
from person_local import ROOT,pf,se
from person_prior import evaluate
from evaluate import stage_subset,verify_subset
from source_contract import ContractError,read_json,check_equal,sha256,write_json
from source_policy import SelectionPolicy,reference_at


def calibrate(weight,ct,records,folder):
    import cv2,torch
    from ultralytics.nn.tasks import load_checkpoint
    model,_=load_checkpoint(str(weight),device='cuda',fuse=False);model.float().eval().requires_grad_(False);mom={};hooks=[]
    layers={n:z for n,z in model.named_modules() if isinstance(z,torch.nn.BatchNorm2d)}
    if not layers:raise ContractError('Unexpected fused model')
    for n,bn in layers.items():
        def save(m,x,key=n):
            var,mean=torch.var_mean(x[0].float(),dim=(0,2,3),correction=0);second=var+mean.square()
            if key not in mom:mom[key]=[mean.double(),second.double(),1]
            else:mom[key][0].add_(mean);mom[key][1].add_(second);mom[key][2]+=1
        hooks.append(bn.register_forward_pre_hook(save))
    try:
        with torch.inference_mode():
            for j,r in enumerate(records):
                im=cv2.imdecode(np.fromfile(ct['root']/r['image'],np.uint8),cv2.IMREAD_COLOR);h,w=im.shape[:2];check_equal('shape',(r['height'],r['width']),(h,w));side=(640,960,1280,960)[j%4]
                nw,nh=round(w*side/max(w,h)),round(h*side/max(w,h));ow=int(np.ceil(nw/32+.5))*32;oh=int(np.ceil(nh/32+.5))*32;l=(ow-nw)//2;t=(oh-nh)//2
                arr=cv2.copyMakeBorder(cv2.resize(im,(nw,nh)),t,oh-nh-t,l,ow-nw-l,cv2.BORDER_CONSTANT,value=(114,114,114));img=torch.from_numpy(np.ascontiguousarray(arr[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
                if not model(img)[0].isfinite().all():raise ContractError('Invalid calibration activations')
    finally:
        for hook in hooks:hook.remove()
    changes={};keys=set()
    for n,bn in layers.items():
        obs=mom[n];check_equal('calibration images per layer',256,obs[2]);mu=obs[0]/256;var=(obs[1]/256-mu.square()).clamp_min(0);old=bn.running_mean.double();v=bn.running_var.double();d=mu-old
        changes[n]={'channels':len(mu),'median_abs_shift_in_old_sd':float((d.abs()/torch.sqrt(v+bn.eps)).median()),'median_observed_to_old_variance':float(((var+bn.eps)/(v+bn.eps)).median())}
        bn.running_mean.copy_(old+.25*d);bn.running_var.copy_(.75*v+.25*var+.1875*d.square());keys.update((n+'.running_mean',n+'.running_var'))
    model=model.cpu().half().eval();pt=folder/'model.pt';torch.save({'model':model,'train_args':model.args,'epoch':-1,'scope':'source_train BN buffers only'},pt)
    after,_=load_checkpoint(str(pt),device='cpu');before,_=load_checkpoint(str(weight),device='cpu');old=before.state_dict();state=after.state_dict();check_equal('tensor names',set(old),set(state));different=[]
    for k,v in state.items():
        if not torch.equal(v,old[k]):different.append(k)
    if not different or not set(different)<=keys:raise ContractError('Unregistered tensor changes')
    write_json(folder/'calibration.json',{'source_sha256':sha256(weight),'calibrated_sha256':sha256(pt),'bn_modules':len(layers),'images':256,'changed_buffers':different,'all_learned_parameters_exact':True,'layers':changes});return pt


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'prior_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Expected fresh prior_NNN and external scratch')
    check_equal('first adaptation unresolved',None,read_json(ROOT/'runs/prior_001/comparison.json')['selected'])
    rules,dev,arms,paths=pl.prepare();dev['rules']=rules;tr=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');gpu=se.probe_gpu();p=SelectionPolicy(rules,reference_at(ROOT));records=[];used=set()
    for r in sorted(tr['records'],key=lambda x:hashlib.sha256(('moments:'+x['key']).encode()).digest()):
        if r['group_id'] not in used:records.append(r);used.add(r['group_id'])
        if len(records)==256:break
    if used&{r['group_id'] for r in dev['records']} or len(used)!=256:raise ContractError('invalid sampling')
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'result';stage.mkdir()
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            z=tmp/k.lower();z.mkdir();os.environ[k]=str(z)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch,torchvision,ultralytics
        vers=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],vers);torch.set_num_threads(4);data=stage_subset(dev,tmp/'dev');screen={};effects=[]
        for s in (29,17,43):
            folder=stage/'moments'/str(s);folder.mkdir(parents=True);weight=calibrate(ROOT/arms[s]['path'],tr,records,folder);torch.cuda.empty_cache();failed,warn,grp=evaluate(s,weight,folder,dev,data,tmp/f'views-{s}',p,rules,paths);screen[s]={'failed_five':failed,'relative_warnings':warn};effects.append(grp);print('MOMENTS',s,screen[s],flush=True)
            if failed:break
        chosen='moments' if len(screen)==3 and all(not v['failed_five'] for v in screen.values()) and not any(sum(k in v['relative_warnings'] for v in screen.values())>=2 for k in ('area_S','width_le8','small_thin_union','ap75')) else None
        write_json(stage/'comparison.json',{'selected':chosen,'screen':screen,'paired_recall_on_evaluated_seeds':pl.paired_interval(effects)});write_json(stage/'train.json',{'samples':[r['key'] for r in records],'scales':[640,960,1280,960],'fraction_new':.25,'group_overlap_with_dev':0})
        write_json(stage/'manifest.json',{'status':'complete','scope':'unlabeled input moments on source_train; source_dev evaluation','baseline':pl.SEAL,'selected':chosen,'ready':pf.READY_SHA,'rules':rules['sha256'],'dev_manifest':dev['manifest'],'train_manifest':tr['manifest'],'arms':arms,'versions':vers,'gpu':gpu,'gradient_training':False,'cloud_cny':0,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('source_moments.py','person_prior.py','source_endpoints.py','person_fusion.py','tricycle_fusion.py','person_local.py','source_policy.py','detection_metrics.py','evaluate.py')},'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('MOMENTS_SEALED',out,chosen,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--scratch',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.output.resolve(),v.scratch.resolve())
