"""Open only the locked S7 union; replay both arms from sparse matching evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

import numpy as np

import detection_metrics as dm
import night_supplement as blind
from night_contract import ROOT, _no_reparse, _path, read_sealed_bytes, read_sealed_json
from source_contract import NAMES, check_equal, sha256, write_json

LOCK='9c4ccf604bf91058bb57a7bd6cc6ae2f3b58195b95c6fae36ae39da53983a83b'
PARENTS={'expand_010':'24975bacea26e9349d27886c624c97a3131176095af42950cf02275a1b68a2d2',
         'block_002':'4f75b291970d79b0d476bc81c6375b10a6d3e01d650463be7026a6f0f1139765',
         'agree_001':'2d013efb2e4e8aabdea8cca7f14d2495f3d1ca1b192a6cec4d292baf40823e36'}
OUT=blind.RUNS/'supp_001'
GAPS={'tram_car':(0,0,0,2,0),'tram_truck':(1,0,0,0,0),'truck_fp':(0,0,1,2,1)}
ROLES=tuple(r for r,_,_ in blind.ROLES)


def scope():
    folder=blind.RUNS/'pool_008';sel=blind.verified_selection(folder,LOCK); inv=blind.inventory()
    keys=sorted(set(sel['random'])|set(sel['targeted']))
    check_equal('locked union',(40,40,76,76),(len(sel['targeted']),len(sel['random']),len(keys),len({inv[k]['group_id'] for k in keys})))
    return sel, {k:inv[k] for k in keys}


def label_path(record):
    image=PurePosixPath(record['image']);check_equal('only pool images',('images','night_pool'),image.parts[:-1])
    return 'hidden_ground_truth/night_pool/'+image.with_suffix('.txt').name


def parent(name,field=None):
    root=blind.RUNS/name;s=read_sealed_json(root,'manifest.json',PARENTS[name],[])
    return blind.sealed(root,s,field) if field else s


def thresholds():
    b=parent('block_002','metrics.json');a=parent('agree_001','metrics.json')
    check_equal('same historic threshold parents',(PARENTS['expand_010'],PARENTS['agree_001']),(a['base_sha256'],b['parents']['agree_001']))
    def series(rows,get):
        check_equal('all original folds',list(range(5)),[r['fold'] for r in rows]);return [get(r) for r in rows]
    return {'truck_fp':series(b['results']['truck'],lambda x:x['boundary']['threshold']),
            **{n+'_low':series(a['results'][n]['folds'],lambda x:x['low_score_max']) for n in ('truck','tram')}}


def restrict_dataset(records):
    """The capture child may only open union payloads, once each; never enumerate data."""
    allowed={_path(blind.NIGHT,p) for r in records.values() for p in (r['image'],label_path(r))}
    opened=[];data=ROOT/'data'
    def audit(event,args):
        if event not in ('open','os.listdir','os.scandir') or not isinstance(args[0],(str,bytes)):return
        p=Path(os.path.abspath(os.fsdecode(args[0])))
        if not p.is_relative_to(data):return
        if event!='open' or p not in allowed or p in opened or args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            raise PermissionError('Outside single-open material whitelist: '+str(p))
        opened.append(p)
    sys.addaudithook(audit)
    return opened


def tail_row(row,anchors):
    keep=[i for i,c in enumerate(row['pred']['cls']) if c in (3,4)];ids={p:i for i,p in enumerate(keep)}
    return {'gt':row['gt'],'pred':{s:[v[i] for i in keep] for s,v in row['pred'].items()},
            'anchors':[anchors[i] for i in keep],'iou_edges':[(g,ids[p],v) for g,p,v in row['iou_edges'] if p in ids]}


def capture(box):
    import cv2
    import torch
    import torchvision
    import ultralytics
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.ops import xywh2xyxy
    from ultralytics.utils.metrics import box_iou
    import night_train as train
    import night_frontier as frontier
    from night_capture import VERSIONS
    from seed_eval import probe_gpu

    _no_reparse(box);blind.require(not OUT.exists() and not OUT.is_symlink() and not box.is_relative_to(ROOT) and not any(box.iterdir()),'Fresh external capture only')
    gpu=probe_gpu();_,records=scope();a=parent('agree_001','metrics.json')
    for file,field in ((frontier,'native_candidate_sha256'),(train,'prepare_sha256'),(dm,'matcher_sha256')):
        check_equal('historical native pipeline source',a[field],sha256(Path(file.__file__)))
    check_equal('historical capture versions',VERSIONS,(torch.__version__,torchvision.__version__,ultralytics.__version__))
    folder=blind.RUNS/'pool_008';s=read_sealed_json(folder,'manifest.json',LOCK,[])
    boxes=json.loads(gzip.decompress(read_sealed_bytes(folder,'a_boxes.json.gz',s['files']['a_boxes.json.gz']['sha256'],[])))
    check_equal('only opened union has A boxes',set(records),set(boxes))
    p=_path(blind.RUNS/'expand_010','43/last.pt');ws=a['weight_sha256']
    check_equal('fixed last EMA',(ws,ws),(parent('expand_010')['files']['43/last.pt']['sha256'],sha256(p)))
    cv2.setNumThreads(1);torch.set_num_threads(4);model,ck=load_checkpoint(str(p),device='cuda:0',fuse=False)
    check_equal('original training-side EMA',(200,NAMES),(ck['updates'],list(model.names.values())));model.half().eval().requires_grad_(False)
    opened=restrict_dataset(records); receipts=[];rows={}
    with torch.inference_mode():
        for k,r in records.items():
            data=read_sealed_bytes(blind.NIGHT,r['image'],r['image_sha256'],[]);im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            blind.require(im is not None,'Unreadable selected image');lp=label_path(r);body=_path(blind.NIGHT,lp).read_bytes()
            receipts.append(dict(key=k,group_id=r['group_id'],image=r['image'],image_sha256=r['image_sha256'],label=lp,label_sha256=hashlib.sha256(body).hexdigest(),bytes=len(body)))
            labels=train._labels(body);x,targets=train.fit_image(im,labels,False,1.,640);cls=torch.tensor([v[0] for v in labels],device='cuda',dtype=torch.long)
            gt=xywh2xyxy(torch.as_tensor(targets,device='cuda')*640.);x=torch.from_numpy(x).cuda().half().div_(255)[None]
            raw,pred=model(torch.cat([x,x])); row,ids,_=frontier.native_row(raw[1:],pred['scores'][1],gt,cls)
            b=boxes[k];filtered=cls<5;bb=torch.as_tensor([y['box'] for y in b],device='cuda',dtype=torch.float32).reshape(-1,4)
            iou=box_iou(gt[filtered].float(),bb).cpu().numpy();ag={'gt':{'cls':cls[filtered].tolist()},'pred':{'cls':[y['class_id'] for y in b],'conf':[y['score'] for y in b]},
                'iou_edges':[(int(i),int(j),float(iou[i,j])) for i,j in zip(*iou.nonzero(),strict=True)]}
            rows[k]={'expand':tail_row(row,ids),'a':tail_row(ag,[y['anchor'] for y in b])}
    check_equal('actual opens exactly follow grant',[blind.NIGHT/p for r in records.values() for p in (r['image'],label_path(r))],opened)
    blind.write_exclusive(box/'matches.json.gz',rows,True)
    write_json(box/'openings.json',dict(lock=LOCK,parents=PARENTS,weight_sha256=ws,versions=VERSIONS,gpu=gpu,
               inference='expand_010/43 fixed EMA, unfused FP16, two identical original 640 inputs, night slot 1',
               actual_opens=len(opened),receipts=receipts,updates=0,cloud=False,review_read=False,final_read=False))


def boxes_with_truth(row):
    gc,pc,score,iou=dm.arrays(row);result=[]
    for c in (3,4):
        _,pairs,inds=dm.image_point(row,c,.001);matched={int(p):float(iou[g,p]) for g,p in pairs}
        own=np.flatnonzero(gc==c);other=np.flatnonzero(gc!=c)
        for i in inds:
            g=int(other[np.argmax(iou[other,i])]) if len(other) else None
            result.append(dict(c=c,anchor=row['anchors'][i],score=float(score[i]),match=matched.get(int(i)),
                               same=float(iou[own,i].max()) if len(own) else 0.,other=float(iou[g,i]) if g is not None else 0.,
                               other_class=int(gc[g]) if g is not None and iou[g,i]>0 else None))
    return result


def actual_role(p,cut,f):
    c,s=p['c'],p['score'];m=p['match']
    if m is not None:
        name='truck_low' if c==3 else 'tram_low'
        return name if m>=.75 and s<=cut[name][f] else None
    if c==3 and s>=cut['truck_fp'][f] and p['same']<.50:return 'truck_fp'
    if c==4 and s>=.05 and p['same']<.30 and p['other']>=.50:return {2:'tram_car',3:'tram_truck'}.get(p['other_class'])
    return None


def summarize(rows,records,sel,cut):
    check_equal('material identities',set(records),set(rows));roles={};trigger={}; assigned={v['key']:v for v in sel['reasons']}
    for k,r in records.items():
        f=blind.fold(r['group_id']); roles[k]=sorted({v for p in boxes_with_truth(rows[k]['expand']) if (v:=actual_role(p,cut,f))})
        why=assigned.get(k)
        if not why or why['role']=='diversity':continue
        role=why['role'];c=3 if role in ('truck_low','truck_fp') else 4
        (p,)=[v for v in boxes_with_truth(rows[k]['a']) if v['c']==c and v['anchor']==why['anchor']]
        check_equal('locked trigger score',why['native_score'],p['score'])
        if role=='truck_fp':hit=p['match'] is None and p['same']<.5
        elif role.endswith('_low'):hit=p['match'] is not None and p['match']>=.75
        else:hit=actual_role(p,cut,f)==role
        trigger[k]=dict(role=role,hit=hit,**p)
    def count(keys):
        by={r:[len({records[k]['group_id'] for k in keys if r in roles[k] and blind.fold(records[k]['group_id'])==f}) for f in range(5)] for r in ROLES}
        totals={r:sum(v) for r,v in by.items()};missing={r:[max(0,g-n) for g,n in zip(GAPS[r],by[r])] for r in GAPS}
        return dict(by_fold=by,totals=totals,role_groups=sum(totals.values()),any_role_groups=sum(bool(roles[k]) for k in keys),missing=missing)
    target=count(sel['targeted']);random=count(sel['random']);diversity=count([k for k in assigned if assigned[k]['role']=='diversity'])
    delta=target['role_groups']-random['role_groups'];filled=not any(any(v) for v in target['missing'].values())
    return dict(targeted=target,random=random,diversity=diversity,overlap=count(sel['overlap']),delta_role_groups=delta,all_gaps_filled=filled,
                gate_passed=filled and delta>0,labels=len(records),triggers=trigger,roles=roles,
                trigger_hits={r:sum(v['role']==r and v['hit'] for v in trigger.values()) for r in ROLES},
                intended_image_hits={r:sum(why['role']==r and r in roles[k] for k,why in assigned.items()) for r in ROLES})


def replay(folder,seal_sha=None):
    if seal_sha:
        from night_train import _receipt
        s=read_sealed_json(folder,'manifest.json',seal_sha,[]);check_equal('complete material receipt',(LOCK,_receipt(folder)),(s['lock'],s['files']))
    sel,records=scope();rec=json.loads((folder/'openings.json').read_bytes());cut=thresholds()
    check_equal('opening scope',(LOCK,PARENTS,len(records)*2,False,False,False,0),tuple(rec[k] for k in ('lock','parents','actual_opens','cloud','review_read','final_read','updates')))
    check_equal('opening receipt order',list(records),[v['key'] for v in rec['receipts']]);
    for v in rec['receipts']:
        r=records[v['key']];check_equal('original image and label pairing',(r['image'],r['image_sha256'],r['group_id'],label_path(r)),tuple(v[k] for k in ('image','image_sha256','group_id','label')))
        blind.require(len(bytes.fromhex(v['label_sha256']))==32,'Invalid label receipt')
    rows=json.loads(gzip.decompress((folder/'matches.json.gz').read_bytes()));report=summarize(rows,records,sel,cut)
    if seal_sha:check_equal('full material replay',report,json.loads((folder/'metrics.json').read_bytes()))
    return report


def run(scratch):
    from night_train import _receipt
    _no_reparse(scratch);_no_reparse(OUT.parent);blind.require(not OUT.exists() and not OUT.is_symlink(),'Refusing to reopen a sealed batch')
    blind.require(scratch.resolve()==scratch and scratch.drive==ROOT.drive and not scratch.is_relative_to(ROOT),'External same-drive scratch required')
    with tempfile.TemporaryDirectory(dir=scratch,prefix='supp-open-') as tmp:
        box=Path(tmp);out=box/'results';out.mkdir();env=dict(os.environ,YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','TEMP','TMP','TMPDIR','XDG_CONFIG_HOME'):
            p=box/k.lower();p.mkdir();env[k]=str(p)
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'_capture',str(out)],cwd=box,env=env,check=True,timeout=600)
        m=replay(out);write_json(out/'metrics.json',m);write_json(out/'manifest.json',dict(status='sealed',lock=LOCK,source_sha256=sha256(Path(__file__)),files=_receipt(out)))
        digest=sha256(out/'manifest.json');out.rename(OUT)
    print('Material seal:',digest,'unique labels:',m['labels'],'role x groups:',m['targeted']['role_groups'],m['random']['role_groups'],'gate:',m['gate_passed'])


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='_capture':capture(Path(sys.argv[2]))
    else:
        a=argparse.ArgumentParser(description=__doc__);subs=a.add_subparsers(dest='mode',required=True)
        subs.add_parser('run').add_argument('--scratch',required=True,type=Path);subs.add_parser('verify').add_argument('--manifest-sha256',required=True);args=a.parse_args()
        if args.mode=='run':run(args.scratch.absolute())
        else:
            m=replay(OUT,args.manifest_sha256);print('Verified without reopening labels:',json.dumps({k:m[k] for k in ('labels','targeted','random','diversity','trigger_hits','intended_image_hits','delta_role_groups','gate_passed')}))
