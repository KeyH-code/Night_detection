"""Paired 512-image mechanism pilot on sealed training-only high-scale teacher targets."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

import person_local as pl
from person_local import ROOT, dm, pf, se
from evaluate import original_gt, stage_subset, verify_subset
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json
from source_policy import SelectionPolicy, recall_intervals, reference_at, summarize
from rider_supervision import native_labels, checkpoints_ok

SIGNAL = ROOT/'runs/teacher_001'
EPOCHS = 4


def fit(arm, weight, data, subset, scores, location):
    import torch
    import torch.nn.functional as F
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.loss import v8DetectionLoss
    torch.manual_seed(29)
    ds=YOLODataset(img_path=str(data.parent/'images/source_train'),imgsz=640,batch_size=4,augment=False,hyp=DEFAULT_CFG,
         rect=True,stride=32,pad=0.,cache=False,task='detect',data={'names':dict(enumerate(NAMES)),'nc':6})
    check_equal('common batching shape',1,len(np.unique(ds.batch_shapes,axis=0)))
    labels={r['key']:original_gt(r,subset['root']) for r in subset['records']};names=native_labels(ds,subset['records'],labels)
    model,_=load_checkpoint(str(weight),device='cuda:0');model.eval().requires_grad_(False);head=model.model[-1];head.training=True
    params=[p for tower in head.cv3 for p in (tower[-1].bias,tower[-1].weight)];param_names={n for n,p in model.named_parameters() if any(p is v for v in params)}
    initial={k:v.cpu().clone() for k,v in model.state_dict().items()};args=dict(model.args);model.args=SimpleNamespace(**args)
    for p in params:p.requires_grad_(True)
    criterion=v8DetectionLoss(model); opt=torch.optim.SGD(params,lr=.004,momentum=.937);history=[]
    for epoch in range(EPOCHS):
        lr=.004*(.1+.9*(1+math.cos(math.pi*epoch/(EPOCHS-1)))/2);opt.param_groups[0]['lr']=lr
        order=np.random.default_rng(29+epoch).permutation(len(ds)).tolist(); seen=[]; hits=[];logs=[];opt.zero_grad(set_to_none=True)
        for step,batch in enumerate(torch.utils.data.DataLoader(ds,batch_size=4,sampler=order,collate_fn=ds.collate_fn,num_workers=0),1):
            keys=[names[Path(f).name]['key'] for f in batch['im_file']]; seen.extend(keys)
            for n in ('cls','batch_idx','bboxes'):batch[n]=batch[n].cuda()
            img=batch['img'].cuda().float()/255
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model.predict(img); assigned,loss,_=criterion.get_assigned_targets_and_loss(pred,batch)
                fg,gi,gtboxes,anchors,stride=assigned
                with torch.no_grad():
                    decoded=criterion.bbox_decode(anchors,pred['boxes'].permute(0,2,1).contiguous())*stride
                    a=decoded.float(); b=gtboxes.float();inter=(torch.minimum(a[:,:,2:],b[:,:,2:])-torch.maximum(a[:,:,:2],b[:,:,:2])).clamp_min(0).prod(2)
                    ov=inter/((a[:,:,2:]-a[:,:,:2]).prod(2)+(b[:,:,2:]-b[:,:,:2]).prod(2)-inter).clamp_min(1e-7)
                logits=[]; targets=[]
                for j,key in enumerate(keys):
                    for g,soft in scores[key].items():
                        allowed=fg[j]&(gi[j]==g)&(ov[j]>=.5)
                        if not allowed.any():continue
                        high=pred['scores'][j,0][allowed].max();logits.append(high);targets.append(soft);hits.append([key,g])
                aux=F.binary_cross_entropy_with_logits(torch.stack(logits).float(),torch.tensor(targets,device='cuda')) if logits else loss[1]*0
                total=loss.sum()+(.2*aux if arm=='teacher' else 0.)
            if not total.isfinite():raise ContractError('nonfinite loss')
            (total/4).backward();logs.append([*loss.detach().float().tolist(),float(aux.detach())])
            if step%4==0:
                for p in params:p.grad[1:].zero_()
                torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
        check_equal('exact sample sequence',[names[Path(ds.im_files[i]).name]['key'] for i in order],seen)
        check_equal('one hit per eligible GT per epoch',len({tuple(t) for t in hits}),len(hits))
        if not hits: raise ContractError('No valid native foreground teacher anchors')
        history.append({'epoch':epoch+1,'lr':lr,'updates':len(order)//16,'sequence':seen,'hits':hits,'mean_box_cls_dfl_aux':np.mean(logs,0).tolist()})
        print('TRAIN_HEAD',arm,epoch+1,len(hits),history[-1]['mean_box_cls_dfl_aux'],flush=True)
    model.args=args;model.eval();packed=deepcopy(model).half().cpu();path=location/'last.pt'
    torch.save({'model':packed,'epoch':EPOCHS-1,'train_args':args,'scope':'paired teacher pilot final only'},path)
    actual,_=load_checkpoint(str(path),device='cpu');changed=checkpoints_ok(initial,actual,param_names,active_rows=1)
    write_json(location/'train.json',{'arm':arm,'history':history,'epochs':4,'updates':128,'sha256':sha256(path),'changed_fp16_values':changed,
           'trainable_rows':sorted(param_names),'other_parameters_and_bn_exact':True,'role':'fixed_last','seed':29,'input_tensor_hw':list(map(int,ds.batch_shapes[0]))})
    return path


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'teacher_\d{3}',out.name) or scratch.is_relative_to(ROOT):
        raise ContractError('Choose a new teacher_NNN output and external scratch')
    rules,dev,arms,_=pl.prepare();train=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');gpu=se.probe_gpu();manifest=read_json(SIGNAL/'manifest.json')
    check_equal('signal identity',(train['manifest'],arms[29]['sha256']),(manifest['train_manifest'],manifest['checkpoint']['sha256']))
    for n,v in manifest['files'].items():check_equal(n,v['sha256'],sha256(SIGNAL/n))
    source=dm.read_rows(SIGNAL/'targets.jsonl.gz');lookup={r['key']:r for r in train['records']}; records=[lookup[r['key']] for r in source]
    check_equal('exact pilot sample',(512,512),(len(source),len({r['group_id'] for r in records})))
    if {r['group_id'] for r in records}&{r['group_id'] for r in dev['records']}:raise ContractError('train/dev risk group collision')
    scores={r['key']:{v['gt']:v['teacher']['score'] for v in r['targets'] if v['eligible']} for r in source}
    for r in records:
        gt=original_gt(r,train['root'])
        for g in scores[r['key']]:check_equal('teacher class',0,gt[g][0])
    check_equal('pilot teacher count',46,sum(map(len,scores.values())))
    subset=dict(train,records=records);dev['rules']=rules;weight=ROOT/arms[29]['path'];policy=SelectionPolicy(rules,reference_at(ROOT))
    with tempfile.TemporaryDirectory(dir=scratch) as name:
        tmp=Path(name)
        for k in ('TEMP','TMP','TMPDIR','YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','XDG_CONFIG_HOME'):
            d=tmp/k.lower();d.mkdir();os.environ[k]=str(d)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch
        import torchvision
        import ultralytics
        versions=[torch.__version__,torchvision.__version__,ultralytics.__version__]
        check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions);torch.set_num_threads(4)
        dataset=stage_subset(subset,tmp/'train');data=stage_subset(dev,tmp/'dev');stage=tmp/'result';stage.mkdir();observ={};summary={};match_all={}
        for arm in ('ordinary','teacher'):
            loc=stage/arm;loc.mkdir();fit(arm,weight,dataset,subset,scores,loc);torch.cuda.empty_cache()
        for arm in ('ordinary','teacher'):
            loc=stage/arm;path=loc/'last.pt';views={};observ[arm]=[];directory=tmp/arm;directory.mkdir()
            sizes,mode=se.tf.PROFILES['low416-or'];native,person=pf.PROFILES['a1'][2:]
            for size,mirror in dict.fromkeys((native,*person,*[(s,False) for s in sizes])):
                rows,item=pf._capture(torch,dev,data,path,directory,size,mirror);views[size,mirror]=rows;observ[arm].append(item)
            fused,_=pf.evaluate_rows(views[native],tuple(views[v] for v in person));fused,_=se.tf.merge_dev(fused,tuple(views[s,False] for s in sizes),mode)
            summary[arm]={}
            for scope,rows in [('single640',views[native]),('locked',fused)]:
                rec,matches,curves=summarize(rows,rules);checks,guards=policy.assess(rec);rec.update(checks=checks,guards=guards,
                       intervals=recall_intervals(rows,matches)['intervals']);summary[arm][scope]=rec;match_all[arm,scope]=matches
                sub=loc/scope;sub.mkdir();dm.write_rows(sub/'objects.jsonl.gz',rows);dm.write_rows(sub/'matches.jsonl.gz',matches);np.savez_compressed(sub/'pr.npz',**curves)
                print('TRAIN_RESULT',arm,scope,rec['metrics']['per_class']['pedestrian']['ap50'],rec['workpoints']['pedestrian'],flush=True)
        effects={}
        for scope in ('single640','locked'):
            rows=dm.read_rows(stage/'teacher'/scope/'objects.jsonl.gz')
            refs={'ordinary':match_all['ordinary',scope]}
            if scope=='locked':refs['seed_002']=dm.read_rows(BASE29/'matches.jsonl.gz')
            for label,before in refs.items():
                trans,groups=pl.transitions(rows,before,match_all['teacher',scope]);effects[scope+'_vs_'+label]={'transitions':trans,'paired_recall':pl.paired_interval([groups])}
        write_json(stage/'comparison.json',{'arms':summary,'effects':effects});(stage/'code').mkdir();shutil.copyfile(__file__,stage/'code/train_teacher.py')
        write_json(stage/'manifest.json',{'status':'complete','scope':'paired source_train pilot + source_dev','observations':observ,
              'eligible_targets':46,'frozen_initial':arms[29],'signal_sha256':sha256(SIGNAL/'manifest.json'),'dev_manifest':dev['manifest'],
              'train_manifest':train['manifest'],'ready':pf.READY_SHA,'rules':rules['sha256'],'versions':versions,'initial_gpu':gpu,'cloud_cny':0,
              'final_test_executed':False,'files':{p.relative_to(stage).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in stage.rglob('*') if p.is_file()}})
        shutil.copytree(stage,out)


BASE29=pl.BASE/'29'
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
