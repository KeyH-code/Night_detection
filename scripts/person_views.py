"""Full-frame pedestrian view fusion with horizontal sampling or a fixed exposure transform."""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

import person_local as pl
from person_local import ROOT,pf,se,dm
from source_contract import ContractError,NAMES,check_equal,read_json,sha256,write_json
from source_policy import SelectionPolicy,reference_at
from source_endpoints import grade
from evaluate import stage_subset

SPECS=pf.PROFILES['a1'][3]
LUT=np.rint(255*(np.arange(256,dtype=float)/255)**.65).astype(np.uint8)


def native(torch,ct,data,weight,cache,baseline):
    cache.mkdir();out={};obs=[]
    for side,flip in SPECS:
        rows,o=pf._capture(torch,ct,data,weight,cache,side,flip);obs.append(o)
        for r in rows:
            b=out.setdefault(r['key'],{});b[side,flip]={k:r['pred'][k] for k in pf.FIELDS}
            if not flip:b[side]=[r['input_shape'],r['ratio_pad']]
    for r in baseline:
        p=out[r['key']];actual=pf._people(tuple(p[s] for s in SPECS));expected=[d for d in pf._detections(r['pred']) if d[0]==0]
        check_equal('original four-view pedestrian replay '+r['key'],expected,actual[:len(expected)])
    return out,obs


def extra(model,im,side,geometry,recipe):
    import cv2,torch
    from ultralytics.utils import ops,nms
    h,w=im.shape[:2];(ih,iw),(gain,pad)=geometry
    wh=(round(gain[1]*w),round(gain[0]*h)); factor=1.25 if recipe=='wide' else 1.
    nw,nh=round(wh[0]*factor),wh[1]; ow=math.ceil(iw*factor/32)*32;oh=ih
    x=round((ow-nw)/2-.1); y=round((oh-nh)/2-.1)
    if recipe=='gamma':
        check_equal('identical native content placement',list(pad),[x,y]);im=cv2.LUT(im,LUT)
    content=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR);padded=cv2.copyMakeBorder(content,y,oh-nh-y,x,ow-nw-x,cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=torch.from_numpy(np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))).to('cuda').half()/255
    with torch.inference_mode():pred=model(torch.stack((tensor,tensor.flip(-1))))[0]
    if pred.shape[:2]!=(2,10) or not pred.isfinite().all() or int((pred[:,4:]>pf.FLOOR).sum((1,2)).max())>=30000:raise ContractError('invalid dense output or pre-NMS truncation')
    detections=nms.non_max_suppression(pred,pf.FLOOR,pf.NMS_IOU,nc=6,multi_label=True,max_det=30000,max_time_img=10.)
    result=[];caps=[]
    for j,d in enumerate(detections):
        caps.append(max(0,len(d)-pf.MAX_DET));d=d[:pf.MAX_DET];d=d[d[:,5]==0];b=d[:,:4].float().clone()
        if j:pf.unmirror_boxes(b,ow)
        b[:,[0,2]]=(b[:,[0,2]]-x)/(nw/w);b[:,[1,3]]=(b[:,[1,3]]-y)/(nh/h);ops.clip_boxes(b,(h,w))
        result.append({'cls':[0]*len(d),'conf':d[:,4].float().cpu().tolist(),'original_boxes':b.cpu().tolist()})
    return result,caps,(oh,ow)


def run(out,scratch):
    if out.parent!=ROOT/'runs' or out.exists() or not pl.re.fullmatch(r'views_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Expected fresh views_NNN and external scratch')
    check_equal('endpoint recipe did not qualify',None,read_json(ROOT/'runs/seed_003/comparison.json')['selected'])
    rules,ct,arms,paths=pl.prepare();ct['rules']=rules;gpu=se.probe_gpu();policy=SelectionPolicy(rules,reference_at(ROOT));recipe_results={};chosen=None
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'result';stage.mkdir()
        for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','XDG_CONFIG_HOME','TEMP','TMP','TMPDIR'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import cv2,torch,torchvision,ultralytics
        from ultralytics.nn.tasks import load_checkpoint
        versions=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions)
        torch.set_num_threads(4);data=stage_subset(ct,tmp/'data');idx={r['key']:r for r in ct['records']};native_by_seed={};native_obs={}
        for recipe in ('wide','gamma'):
            screen={};groups=[]
            for seed in (29,17,43):
                weight=ROOT/arms[seed]['path'];base=dm.read_rows(paths[seed]);check_equal('dev keys',set(idx),set(pf._dev_index(base)))
                if seed not in native_by_seed:native_by_seed[seed],native_obs[seed]=native(torch,ct,data,weight,tmp/f'native-{seed}',base)
                source=native_by_seed[seed];model,_=load_checkpoint(str(weight),device='cuda',fuse=True);model.half().eval().requires_grad_(False);check_equal('labels',NAMES,list(model.names.values()))
                rows=[];caps=[];view_caps=[];shape={};start=time.perf_counter()
                for r in base:
                    key=r['key'];rec=idx[key];p=source[key];im=cv2.imdecode(np.fromfile(ct['root']/rec['image'],np.uint8),cv2.IMREAD_COLOR);check_equal('dimensions',(rec['height'],rec['width']),im.shape[:2]);v={}
                    for size in (960,1280):
                        (v[size,False],v[size,True]),trunc,hw=extra(model,im,size,p[size],recipe);shape[size]=hw
                        if any(trunc):view_caps.append([key,size,trunc])
                    new=pf._people(tuple(p[s] for s in SPECS)+tuple(v[s] for s in SPECS));after,n=pl.replace(r,[(s,b) for c,s,b in new],False);rows.append(after)
                    keep=lambda x:[d for d in pf._detections(x['pred']) if d[0]!=0];check_equal('untouched nonpersons '+key,keep(r),keep(after))
                    if n:caps.append([key,n])
                del model;gc.collect();torch.cuda.empty_cache();check_equal('checkpoint after',arms[seed]['sha256'],sha256(weight));old=read_json(pl.BASE/str(seed)/'metrics.json');rec,ma,curves,failed,warn=grade(rows,rules,policy,old)
                for name in NAMES[1:]:check_equal('unchanged workpoint '+name,old['workpoints'][name],rec['workpoints'][name])
                moved,gr=pl.transitions(rows,dm.read_rows(pl.BASE/str(seed)/'matches.jsonl.gz'),ma);groups.append(gr)
                rec.update(failed_five=failed,relative_warnings=warn,weight=arms[seed],transitions=moved,paired_recall=pl.paired_interval([gr]),native_observations=native_obs[seed],
                         additional=dict(shapes=shape,forward_and_fusion_seconds=time.perf_counter()-start,view_cap_drops=view_caps,final_cap=caps))
                folder=stage/recipe/str(seed);folder.mkdir(parents=True);write_json(folder/'metrics.json',rec);dm.write_rows(folder/'objects.jsonl.gz',rows);dm.write_rows(folder/'matches.jsonl.gz',ma);np.savez_compressed(folder/'pr.npz',**curves)
                screen[seed]={'failed_five':failed,'warnings':warn};print('AUG_RESULT',recipe,seed,'AP50',rec['metrics']['per_class']['pedestrian']['ap50'],rec['workpoints']['pedestrian'],failed,warn,flush=True)
                if failed:break
            recipe_results[recipe]={'screen':screen,'conditional_paired_recall_on_evaluated_seeds':pl.paired_interval(groups)}
            if len(screen)==3 and not any(v['failed_five'] for v in screen.values()) and not any(sum(k in v['warnings'] for v in screen.values())>=2 for k in ('area_S','width_le8','small_thin_union','ap75')):
                chosen=recipe;break
        write_json(stage/'comparison.json',{'selected':chosen,'recipes':recipe_results});write_json(stage/'manifest.json',{'status':'complete','scope':'source_dev','baseline':pl.SEAL,'ready':pf.READY_SHA,'rules':rules['sha256'],'arms':arms,'dev_manifest':ct['manifest'],'gpu':gpu,'versions':versions,
                 'native':SPECS,'augmented':SPECS,'selected':chosen,'training_performed':False,'cloud_cny':0,'final_test_executed':False,'implementation':{n:sha256(ROOT/'scripts'/n) for n in ('person_views.py','source_endpoints.py','person_local.py','person_fusion.py','seed_eval.py','source_policy.py','detection_metrics.py','evaluate.py')},
                 'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('AUG_SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
