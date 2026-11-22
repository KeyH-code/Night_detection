"""Rescore existing source-dev pedestrian boxes with road geometry or same-detector ROI evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

import person_local as pl
from person_local import BASE, ROOT, dm, pf, se, check_equal, read_json, sha256, write_json
from source_contract import ContractError, NAMES
from source_policy import SelectionPolicy, recall_intervals, reference_at, summarize
from evaluate import verify_subset


def logit(s):
    s = np.clip(s,.00001,.99999)
    return np.log(s/(1-s))


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))


def roi(box,width,height):
    x1,y1,x2,y2 = box
    w = min(width, round(np.clip(8*(y2-y1),384,1536))); h = min(height, round(.75*w))
    x = int(np.clip(round((x1+x2-w)/2),0,width-w)); y = int(np.clip(round((y1+y2-h)/2),0,height-h))
    return x,y,x+w,y+h


def capture(scratch):
    if scratch.is_relative_to(ROOT) or not scratch.is_dir() or any(scratch.iterdir()):
        raise ContractError('Choose an empty external scratch directory')
    rules,ct,arms,paths = pl.prepare(); gpu = se.probe_gpu()
    for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','XDG_CONFIG_HOME'):
        os.environ[k] = str(scratch)
    os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
    import cv2
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import nms

    class VerifyPredictor(DetectionPredictor):
        def postprocess(self,preds,img,orig_imgs,**kwargs):
            raw = preds[0] if isinstance(preds,(tuple,list)) else preds
            if raw.shape[1]!=10 or not raw.isfinite().all(): raise ContractError('Invalid raw predictions')
            raw = raw[:,:5].contiguous()
            if int((raw[:,4]>pf.FLOOR).sum(1).max())>=30000: raise ContractError('Pre-NMS cap')
            output = nms.non_max_suppression(raw,pf.FLOOR,pf.NMS_IOU,nc=1,max_det=300,max_time_img=10.)
            if any(len(x)>=300 for x in output): raise ContractError('ROI cap')
            return self.construct_results(output,img,orig_imgs)

    versions = [torch.__version__,torchvision.__version__,ultralytics.__version__]
    check_equal('locked inference versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions)
    torch.set_num_threads(4); records = {r['key']:r for r in ct['records']}; obs = {}
    for seed in (29,17,43):
        rows = dm.read_rows(paths[seed]); check_equal('dev only',set(records),set(pf._dev_index(rows)))
        model = YOLO(str(ROOT/arms[seed]['path'])); start=time.perf_counter(); saved=[]; count=0
        check_equal('detector names',NAMES,list(model.names.values()))
        for row in rows:
            rec = records[row['key']]; im = cv2.imdecode(np.fromfile(ct['root']/rec['image'],np.uint8),cv2.IMREAD_COLOR)
            h,w = im.shape[:2]; check_equal('image identity',(rec['image_sha256'],rec['width'],rec['height']),
                                                       (row['image_sha256'],w,h))
            idx = [i for i,(c,s,b) in enumerate(pf._detections(row['pred'])) if c==0 and s>=.02]; scored={}
            for off in range(0,len(idx),4):
                ids = idx[off:off+4]; crops = [roi(row['pred']['original_boxes'][i],w,h) for i in ids]
                imgs = [np.ascontiguousarray(im[y:v,x:u]) for x,y,u,v in crops]
                batch = model.predict(imgs,predictor=VerifyPredictor,imgsz=640,rect=False,quantize=16,device=0,augment=False,
                                      conf=pf.FLOOR,iou=pf.NMS_IOU,max_det=300,verbose=False,save=False,project=str(scratch/'predict'))
                for i,(x,y,u,v),result in zip(ids,crops,batch,strict=True):
                    det=result.boxes.data.float().cpu().numpy(); boxes=det[:,:4].copy(); boxes[:,[0,2]]+=x; boxes[:,[1,3]]+=y
                    overlap=dm.box_iou([row['pred']['original_boxes'][i]],boxes)[0]
                    score=float(det[overlap>=.5,4].max(initial=pf.FLOOR));scored[str(i)]=score
            saved.append({'key':row['key'],'scores':scored});count+=len(idx)
        dm.write_rows(scratch/f'{seed}.gz',saved); del model; torch.cuda.empty_cache()
        check_equal('checkpoint',arms[seed]['sha256'],sha256(ROOT/arms[seed]['path']))
        obs[seed]={'seconds':time.perf_counter()-start,'proposals':count};print('VERIFIED_CROPS',seed,obs[seed],flush=True)
    write_json(scratch/'roi.json',{'seed_002':pl.SEAL,'ready':pf.READY_SHA,'arms':arms,'versions':versions,'gpu':gpu,
               'gpu_name':torch.cuda.get_device_name(0),'observations':obs,'files':{p.name:sha256(p) for p in scratch.glob('*.gz')}})


def geometry(row):
    pred=row['pred']; cls=np.asarray(pred['cls']); score=np.asarray(pred['conf']); boxes=np.asarray(pred['original_boxes']).reshape(-1,4)
    idx=np.flatnonzero(cls==0); a=boxes[idx]; w=np.maximum(a[:,2]-a[:,0],1); h=np.maximum(a[:,3]-a[:,1],1)
    ht=row['height']; wd=row['width']; cars=boxes[(cls==2)&(score>=.3)]
    horizon=np.median(cars[:,1]+.15*(cars[:,3]-cars[:,1]))/ht if len(cars) else .63
    horizon=float(np.clip(horizon,.50,.75)); eye=(a[:,1]+.15*h)/ht; x=(a[:,0]+.5*w)/wd-.5
    aspect=np.log(w/h/.38); size=np.log(h/ht/.09); off=(eye-horizon)/.05; ground=(eye-horizon)*ht/np.maximum(h,24)
    rider=(dm.box_iou(a,boxes[cls==1])**2*score[cls==1]).max(axis=1,initial=0.)
    feats=np.column_stack([aspect,np.abs(aspect),size,np.abs(size),off,np.abs(off),ground,np.abs(ground),
                           (eye-.63)/.05,np.abs(eye-.63)/.05,x,np.abs(x),rider,np.sqrt(rider),
                           ((a[:,0]<2)|(a[:,2]>wd-2)).astype(float)])
    return idx,np.clip(feats,-5.,5.)


def apply(row,model=None,scores=None):
    idx,x=geometry(row); s=np.asarray(row['pred']['conf'])[idx]
    if scores is None:
        bias=model['bias']; coef=model['coefficients']; residual=(x-model['mean'])/model['scale']@coef+bias
        new=sigmoid(logit(s)+.5*np.clip(residual,-3,3))
    else:
        expected={str(i) for i,v in zip(idx,s,strict=True) if v>=.02}; check_equal('verified proposal IDs', expected,set(scores))
        new=[sigmoid(.65*logit(v)+.35*logit(scores[str(i)])) if str(i) in scores else v for i,v in zip(idx,s,strict=True)]
    people=[(float(v),row['pred']['original_boxes'][i]) for i,v in zip(idx,new,strict=True)]
    return pl.replace(row,people,False)[0]


def person_metrics(rows):
    # score-only internal comparison: only person order matters; no class-averaged F1.
    ap=dm.summarize_ap(rows)[0]['per_class']['pedestrian']; rules=pl.se.load_rules(ROOT/'RULES.md')['classes']['pedestrian']
    pt=dm.choose_point(dm.work_curve(rows,0),rules['precision'],rules['recall'])
    return {'ap50':ap['ap50'],'ap50_95':ap['ap50_95'],'workpoint':pt}


def fit(target):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits
    if target.exists() or target.is_relative_to(ROOT): raise ContractError('Choose a new external calibration file')
    root=ROOT/'runs/a_001_diag'; seal=read_json(root/'manifest.json'); path=root/'train/objects.jsonl.gz'
    check_equal('training objects SHA', 'b8c5610195ddcf012b030c1c3574e018eccff2b90885de091f2ba65303c2256e',sha256(path))
    check_equal('train object provenance',(pf.READY_SHA,pf.PROFILES['a1'][1],sha256(path)),
                (seal['ready_sha256'],seal['checkpoint_sha256'],seal['files']['train/objects.jsonl.gz']['sha256']))
    ct=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train'); dev=verify_subset(pf.SOURCE,pf.READY_SHA,'source_dev')
    rec={r['key']:r for r in ct['records']}; groups=sorted({r['group_id'] for r in rec.values()},
              key=lambda g:hashlib.sha256(('17:'+str(g)).encode()).digest()); inner=set(groups[:len(groups)//5]); seen=set()
    if set(groups)&{r['group_id'] for r in dev['records']}: raise ContractError('group leakage')
    excludes=read_json(ROOT/'configs/rerank.json');skip={v['key'] for v in excludes['candidates']}
    xs=[];ss=[];ys=[];val=[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); k=r['key']; ref=rec[k]
            if k in seen: raise ContractError('Duplicate training image');
            seen.add(k)
            for n in ('purpose','group_id','image_sha256','label_sha256','width','height'):check_equal(k+n,ref[n],r[n])
            if k in skip:continue
            if r['group_id'] in inner:val.append(r); continue
            ix,x=geometry(r); gc,pc,s,iou=dm.arrays(r); true=np.asarray(r['ap_tp'],bool).reshape(-1,10)[ix,0]
            fitmask=true|(iou[gc==0][:,ix].max(axis=0,initial=0)<.3)
            xs.append(x[fitmask]);ss.extend(s[ix][fitmask]);ys.extend(true[fitmask])
    check_equal('all train keys',set(rec),seen)
    x=np.concatenate(xs); mean=x.mean(0);scale=np.maximum(x.std(0),.05)
    torch.set_num_threads(2); t=lambda a:torch.tensor(a,dtype=torch.float64)
    feat=t((x-mean)/scale); y=t(np.asarray(ys,float)); offset=t(logit(np.asarray(ss,float)))
    w=torch.zeros(feat.shape[1],dtype=torch.float64,requires_grad=True); b=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([w,b],max_iter=80,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
    def closure():
        opt.zero_grad(); loss=binary_cross_entropy_with_logits(offset+feat@w+b,y)+.001*w.square().sum()
        loss.backward();return loss
    opt.step(closure); loss=float(closure().detach()); model={'coefficients':w.detach().tolist(),'bias':b.item(),
          'mean':mean.tolist(),'scale':scale.tolist()}
    baseline=person_metrics(val); changed=person_metrics([apply(r,model) for r in val]); acc=(changed['ap50']>=baseline['ap50'] and
                  changed['workpoint']['recall']>=baseline['workpoint']['recall'] and changed['workpoint']['origin']=='rule')
    write_json(target,{'model':model,'source_train_objects_sha256':sha256(path),'train_manifest':ct['manifest'],
         'train_records':len(rec),'inner_groups':sorted(inner),'train_dev_risk_groups_disjoint':True,'excluded_keys':sorted(skip),
         'inner_images':len(val),'fit_candidates':len(y),'fit_positive':int(y.sum()),'l2':.001,'iterations':opt.state[w]['n_iter'],
         'final_loss':loss,'torch':torch.__version__,'baseline':baseline,'calibrated':changed,'passed_internal':acc})
    print('TRAIN_INNER',baseline,changed,acc,flush=True)


def evaluate(scratch,out):
    if (out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'views_\d{3}',out.name)
            or scratch.is_relative_to(ROOT)):
        raise ContractError('Choose a new views_NNN run and external calibration/ROI files')
    rules,ct,arms,paths=pl.prepare(); policy=SelectionPolicy(rules,reference_at(ROOT))
    calib=read_json(scratch/'geometry.json'); previous=ROOT/'runs/views_002/manifest.json'
    check_equal('first comparison failed',None,read_json(previous)['selected'])
    screen={}; summary={}; pergroup={}; triggers={}; selected=None; obs=None
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        stage=Path(temp); result=stage/'result'; result.mkdir()
        for mode in ('geometry','roi'):
            if mode=='geometry' and not calib['passed_internal']:continue
            if mode=='roi':
                obs=read_json(scratch/'roi/roi.json');check_equal('crop source',(pl.SEAL,pf.READY_SHA,arms),
                    (obs['seed_002'],obs['ready'],{int(k):v for k,v in obs['arms'].items()}))
            records={};groups=[];warnings={}
            for seed in se.SEEDS:
                base=dm.read_rows(paths[seed]); keys=set(pf._dev_index(base));check_equal('frozen keys',{r['key'] for r in ct['records']},keys)
                vals=None
                if mode=='roi':
                    f=scratch/'roi'/f'{seed}.gz';check_equal(f.name,obs['files'][f.name],sha256(f));cached=dm.read_rows(f)
                    vals={r['key']:r['scores'] for r in cached};check_equal('ROI keys/count',(keys,500),(set(vals),len(cached)))
                rows=[apply(row,calib['model'],None if vals is None else vals[row['key']]) for row in base]
                old=read_json(BASE/str(seed)/'metrics.json');before=dm.read_rows(BASE/str(seed)/'matches.jsonl.gz')
                rec,after,curves=summarize(rows,rules);checks,guards=policy.assess(rec)
                for a,b in zip(base,rows,strict=True):
                    other=lambda x:[v for v in pf._detections(x['pred']) if v[0]!=0]
                    check_equal('untouched nonperson boxes',other(a),other(b))
                for name in NAMES[1:]:check_equal('untouched workpoints',old['workpoints'][name],rec['workpoints'][name])
                changes,grp=pl.transitions(rows,before,after);groups.append(grp)
                tags=('area_S','width_le8','small_thin_union');prior={s['slice']:s['recall50'] for s in old['slices'] if s['class']=='pedestrian'}
                warnings[seed]=[s['slice'] for s in rec['slices'] if s['class']=='pedestrian' and s['slice'] in tags
                                and prior[s['slice']]-s['recall50']>rules['slice_drop']]
                if old['metrics']['per_class']['pedestrian']['ap75']-rec['metrics']['per_class']['pedestrian']['ap75']>rules['ap75_drop']:
                    warnings[seed].append('ap75')
                rec.pop('nms'); rec.update(checks=checks,guards=guards,intervals=recall_intervals(rows,after)['intervals'],
                       transition=changes,paired_recall=pl.paired_interval([grp]),
                       metric_scope='same seed_002 proposals; source_dev; '+mode)
                rec['diagnosis']=pl.diagnose(rows,after,curves,rules['classes']['pedestrian']);rec['diagnosis'].pop('fn_detail')
                records[seed]=rec;folder=stage/mode/str(seed);folder.mkdir(parents=True)
                dm.write_rows(folder/'objects.jsonl.gz',rows);dm.write_rows(folder/'matches.jsonl.gz',after);np.savez_compressed(folder/'pr.npz',**curves)
                print('RANK_RESULT',mode,seed,rec['metrics']['per_class']['pedestrian']['ap50'],rec['workpoints']['pedestrian'],flush=True)
            summary[mode]=se.summarize_seeds(records);screen[mode]=records;pergroup[mode]=pl.paired_interval(groups);triggers[mode]=warnings
            if not any(any(not k.startswith('tricycle/') for k in failed) for failed in summary[mode]['failed_by_seed'].values()) and not any(
                sum(tag in warnings[s] for s in se.SEEDS)>=2 for tag in (*tags,'ap75')):
                selected=mode
                for seed in se.SEEDS:shutil.move(str(stage/mode/str(seed)),result/str(seed))
                break
        write_json(result/'screen.json',screen);write_json(result/'comparison.json',{'modes':summary,'selected':selected,'paired_recall':pergroup,'warnings':triggers})
        shutil.copyfile(scratch/'geometry.json',result/'geometry.json');(result/'code').mkdir();shutil.copyfile(__file__,result/'code/person_verify.py')
        files={p.relative_to(result).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in result.rglob('*') if p.is_file()}
        write_json(result/'manifest.json',{'status':'complete','scope':'source_dev','images':500,'arms':arms,'ready':pf.READY_SHA,
                'dev_manifest':ct['manifest'],'rules':rules['sha256'],'seed_002':pl.SEAL,'views_002':sha256(previous),'selected':selected,
                'roi_evaluation':obs,'geometry_fit':'source_train only','detector_training':False,'final_test_executed':False,'cloud_cny':0,
                'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('person_verify.py','person_local.py','person_fusion.py','source_contract.py',
                  'seed_eval.py','evaluate.py','detection_metrics.py','source_policy.py')},'files':files})
        shutil.copytree(result,out)
    print('SEALED',out,selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--capture',type=Path);g.add_argument('--fit',type=Path);g.add_argument('--evaluate',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'runs/views_003')
    args=p.parse_args()
    if args.capture: capture(args.capture.resolve())
    elif args.fit: fit(args.fit.resolve())
    else: evaluate(args.evaluate.resolve(),args.output.resolve())
