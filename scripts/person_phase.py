"""Read the same 960 scene at two feature-grid offsets and rescore locked pedestrian boxes."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import person_local as pl
from person_local import dm, pf, se, ROOT
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json
from source_policy import summarize, SelectionPolicy, recall_intervals, reference_at

OFFSETS=((0,0),(5,13)); MODES=('aligned','shifted')


def evidence(model,img,ref):
    import torch
    import torch.nn.functional as F
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    padded=LetterBox((960,960),auto=True,stride=32)(image=img);h,w=padded.shape[:2]
    pixel=torch.from_numpy(np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1))).to('cuda')
    views=[F.pad(pixel.flip(-1) if flip else pixel,(dx,32-dx,dy,32-dy),value=114) for dx,dy in OFFSETS for flip in (False,True)]
    with torch.inference_mode(): raw=model(torch.stack(views).half()/255)[0]
    if raw.shape[:2]!=(4,10) or not raw.isfinite().all():raise ContractError('invalid dense output')
    supports=[]
    for j,z in enumerate(raw):
        dx,dy=OFFSETS[j//2];mask=z[4]>pf.FLOOR;score=z[4,mask].float().cpu().numpy();box=ops.xywh2xyxy(z[:4,mask].float().T)
        box[:,[0,2]]-=dx;box[:,[1,3]]-=dy
        if j%2:pf.unmirror_boxes(box,w)
        box=ops.scale_boxes((h,w),box,img.shape[:2]).cpu().numpy();ov=dm.box_iou(ref,box)
        supports.append(np.where(ov>=.5,score[None,:],0.).max(axis=1,initial=0))
    return np.stack(supports,axis=1)


def updated(row,support,mode):
    people=[(s,b) for c,s,b in pf._detections(row['pred']) if c==0]
    values=np.asarray(support,np.float32).reshape(-1,4);check_equal('four scores per original person',len(people),len(values))
    extra=values[:,2*mode:2*mode+2].mean(1);floor=float(np.nextafter(np.float32(pf.FLOOR),np.float32(1)))
    rescored=[(max(floor,.75*s+.25*float(v)),box) for (s,box),v in zip(people,extra,strict=True)]
    after,cap=pl.replace(row,rescored,False);check_equal('same prediction count, no cap',(len(row['pred']['cls']),0),(len(after['pred']['cls']),cap))
    check_equal('untouched other outputs',[d for d in pf._detections(row['pred']) if d[0]!=0],[d for d in pf._detections(after['pred']) if d[0]!=0])
    check_equal('same person boxes including multiplicity',Counter(tuple(b) for _,b in people),Counter(tuple(b) for c,s,b in pf._detections(after['pred']) if c==0))
    return after


def run(out,scratch):
    if scratch.is_relative_to(ROOT) or out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'views_\d{3}',out.name):
        raise ContractError('Use an external scratch directory and new views_NNN')
    rules,ct,arms,paths=pl.prepare();policy=SelectionPolicy(rules,reference_at(ROOT));gpu=se.probe_gpu();records={r['key']:r for r in ct['records']}
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp=Path(tmp);stage=tmp/'result';stage.mkdir()
        for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','XDG_CONFIG_HOME'):os.environ[k]=str(tmp)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch,torchvision,cv2,ultralytics
        from ultralytics.nn.tasks import load_checkpoint
        versions=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('locked packages',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions)
        torch.set_num_threads(4);screen={k:{} for k in MODES};warnings={k:{} for k in MODES};groups={k:[] for k in (*MODES,'shifted_minus_aligned')}
        for seed in se.SEEDS:
            model,_=load_checkpoint(str(ROOT/arms[seed]['path']),device='cuda:0',fuse=True);model.half().eval().requires_grad_(False)
            check_equal('detector classes',NAMES,list(model.names.values()));base=dm.read_rows(paths[seed]);check_equal('exact dev keys',set(records),set(pf._dev_index(base)))
            observations=[]
            for r in base:
                rec=records[r['key']]
                for f in ('image_sha256','label_sha256','group_id','width','height'):check_equal(f,rec[f],r[f])
                img=cv2.imdecode(np.fromfile(ct['root']/rec['image'],np.uint8),cv2.IMREAD_COLOR);check_equal('image shape',(r['height'],r['width']),img.shape[:2])
                boxes=[b for c,s,b in pf._detections(r['pred']) if c==0];scores=evidence(model,img,boxes) if boxes else np.empty((0,4))
                observations.append({'key':r['key'],'supports':scores.tolist()})
            check_equal('unchanged checkpoint',arms[seed]['sha256'],sha256(ROOT/arms[seed]['path']));del model;torch.cuda.empty_cache()
            # Compact scores + sealed seed_002 objects reproduce even the rejected arms without copying full outputs.
            folder=stage/str(seed);folder.mkdir();dm.write_rows(folder/'supports.jsonl.gz',observations)
            base_metrics=read_json(pl.BASE/str(seed)/'metrics.json');before=dm.read_rows(pl.BASE/str(seed)/'matches.jsonl.gz');matches={}
            for j,mode in enumerate(MODES):
                rows=[updated(r,o['supports'],j) for r,o in zip(base,observations,strict=True)];record,after,curves=summarize(rows,rules);checks,guards=policy.assess(record)
                for name in NAMES[1:]:check_equal(name+' original workpoint',base_metrics['workpoints'][name],record['workpoints'][name])
                trans,grp=pl.transitions(rows,before,after);groups[mode].append(grp);matches[mode]=after
                record.pop('nms');record.update(checks=checks,guards=guards,intervals=recall_intervals(rows,after)['intervals'],
                       transitions=trans,paired_recall=pl.paired_interval([grp]),metric_scope='source_dev; locked geometry, 960 dense support '+mode)
                old={r['slice']:r['recall50'] for r in base_metrics['slices'] if r['class']=='pedestrian'}
                warnings[mode][seed]=[r['slice'] for r in record['slices'] if r['class']=='pedestrian' and r['slice'] in ('area_S','width_le8','small_thin_union') and old[r['slice']]-r['recall50']>rules['slice_drop']]
                if base_metrics['metrics']['per_class']['pedestrian']['ap75']-record['metrics']['per_class']['pedestrian']['ap75']>rules['ap75_drop']:warnings[mode][seed].append('ap75')
                screen[mode][seed]=record;work=tmp/mode/str(seed);work.mkdir(parents=True);dm.write_rows(work/'objects.jsonl.gz',rows);dm.write_rows(work/'matches.jsonl.gz',after);np.savez_compressed(work/'pr.npz',**curves)
                print('VIEW',seed,mode,record['metrics']['per_class']['pedestrian']['ap50'],record['workpoints']['pedestrian'],flush=True)
            _,grp=pl.transitions(base,matches['aligned'],matches['shifted']);groups['shifted_minus_aligned'].append(grp)
        summary={k:se.summarize_seeds(screen[k]) for k in MODES};chosen=next((k for k in MODES if not any(any(not fail.startswith('tricycle/') for fail in fails) for fails in summary[k]['failed_by_seed'].values()) and not any(warnings[k].values())),None)
        if chosen:
            for seed in se.SEEDS:
                for f in (tmp/chosen/str(seed)).iterdir():shutil.move(str(f),stage/str(seed)/f.name)
        write_json(stage/'comparison.json',{'selected':chosen,'modes':summary,'warnings':warnings,'paired_recall':{k:pl.paired_interval(v) for k,v in groups.items()}});write_json(stage/'screen.json',screen)
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_dev','ready':pf.READY_SHA,'seed_002':pl.SEAL,'arms':arms,'dev_manifest':ct['manifest'],'rules':rules['sha256'],
                  'versions':versions,'gpu_initial':gpu,'detector_forwards_per_image':4,'offsets_xy':OFFSETS,'floor':pf.FLOOR,'cloud_cny':0,'training_performed':False,'final_test_executed':False,
                  'implementation':{n:sha256(ROOT/'scripts'/n) for n in ('person_phase.py','person_local.py','person_fusion.py','source_policy.py','seed_eval.py','detection_metrics.py','evaluate.py')},
                  'files':{f.relative_to(stage).as_posix():{'bytes':f.stat().st_size,'sha256':sha256(f)} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
