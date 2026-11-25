"""S7 blind supplement: freeze both 40-image arms before opening any new label."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time

import numpy as np

from night_contract import (POOL, POOL_MANIFEST, ROOT, _no_reparse, _records,
                            read_sealed_bytes, read_sealed_json)
from source_contract import ContractError, canonical_hash, check_equal, sha256, write_json

RESERVE = '92b0a07efd68f62ef9294308ad459d85a01bf5c0dd3e5b7b1ff77d7a469b9d46'
POOL_SHA = '04e000c0fb125c2eb7518658a94bbfeaba24cb2885347029af17d637fd7bd4e0'
PROTOCOL = '0a62c4e54bd4b2ad86e8b4e644fc04d71c03c3e1b0c0039e80359f4463e96605'
NIGHT = ROOT/'data/soda10m/splits/night_v1'
RUNS = ROOT/'runs'
ROLES = (('tram_car',8,((3,4),)), ('tram_truck',8,((0,4),)),
         ('truck_fp',12,((2,2),(3,4),(4,2))), ('truck_low',4,()), ('tram_low',4,()))


def hashed(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def fold(g):
    return int(hashed('tail-support/43/'+g)[:16],16)%5


def require(ok, msg):
    if not ok: raise ContractError(msg)


def sealed(d, seal, n):
    return read_sealed_json(d,n,seal['files'][n]['sha256'],[])


def inventory():
    return {x['key']:x for x in _records(read_sealed_json(NIGHT,POOL_MANIFEST,POOL_SHA,[]),POOL,994,images_only=True)}


def reservation():
    d=RUNS/'reserve_001'; seal=read_sealed_json(d,'manifest.json',RESERVE,[])
    roles=sealed(d,seal,'roles.json'); inv=inventory(); base=roles['base']; keys=roles['candidates']
    require((roles['pool_sha256'],roles['protocol_sha256'])==(POOL_SHA,PROTOCOL),'Wrong study')
    require(len(base)==len(set(base))==300 and len(keys)==len(set(keys))==175,'Wrong selection budget')
    review=set(roles['review_groups']); eligible=set(roles['eligible_groups']); groups={inv[k]['group_id'] for k in keys}
    require(len(groups)==175 and len(review)==50 and review.isdisjoint(groups) and groups|review==eligible,'Invalid reserved group isolation')
    require({inv[k]['group_id'] for k in base}.isdisjoint(eligible),'Historical training group exposed')
    require(all(sum(fold(g)==f for g in review)==10 for f in range(5)),'Invalid reserve fold allocation')
    return inv,base,keys


def references(rows,c):
    """Rows: (key, group, predicted class, anchor, score); never ground truth."""
    best={}
    for x in sorted((x for x in rows if x[2]==c and x[4]>=.50),key=lambda x:(-x[4],x[0],x[3])):
        best.setdefault(x[1],x)
    result=list(best.values())[:20]
    require(len(result)>=5,f'Class {c} has only {len(result)} eligible reference groups; blind arm cannot complete')
    return result


def random_control(records, selected):
    cnt=Counter(fold(records[k]['group_id']) for k in selected)
    require(len({r['group_id'] for r in records.values()})==len(records),'One representative per group required')
    return [k for f in range(5) for k in sorted((k for k,x in records.items() if fold(x['group_id'])==f),
            key=lambda k:(hashed('supp-random/43/'+records[k]['group_id']),records[k]['group_id']))[:cnt[f]]]


def logit(p):
    p=max(1e-6,min(1.-1e-6,p));return math.log(p)-math.log1p(-p)


def role_scores(row, high):
    c,s=row['class_id'],row['score']; car,truck,tram=row['similarity']
    if s<=.001:return {}
    if c==4:
        return {'tram_car':car-tram,'tram_truck':truck-tram} if s>=high['4'] else {'tram_low':tram}
    if c==3:
        return {'truck_fp':1-truck+.5*max(0.,car-truck)-.1*abs(logit(s)-logit(high['3']))} if s>=high['3'] else {'truck_low':truck}
    return {}


def select_from_scores(rows, keys, records, high, initial, distances):
    check_equal('candidate image output identities',set(keys),set(rows)); chosen=[]; reasons=[]
    for role,quota,preferences in ROLES:
        ranked=[]
        for k in keys:
            if k in chosen: continue
            vals=[(-v,-o['score'],k,o['anchor']) for o in rows[k] for tag,v in role_scores(o,high).items() if tag==role]
            if vals: ranked.append(min(vals))
        ranked.sort();start=len(chosen)
        def take(x):
            chosen.append(x[2]);reasons.append(dict(key=x[2],role=role,anchor=x[3],rank_score=-x[0],native_score=-x[1]))
        for f,n in preferences:
            for x in [x for x in ranked if fold(records[x[2]]['group_id'])==f and x[2] not in chosen][:n]:take(x)
        for x in ranked:
            if len(chosen)-start==quota:break
            if x[2] not in chosen:take(x)
        if len(chosen)-start!=quota:
            return dict(stopped='role_quota',role=role,available=len(chosen)-start,required=quota,unique_label_budget=0)
    a=np.asarray(initial); D=np.asarray(distances);require(a.shape==(len(keys),) and D.shape==(len(keys),len(keys)) and np.isfinite(D).all() and np.isfinite(a).all(),'Invalid image-distance matrix')
    for _ in range(4):
        taken=[keys.index(k) for k in chosen];near=np.minimum(a,D[:,taken].min(axis=1))
        k=min((k for k in keys if k not in chosen),key=lambda k:(-near[keys.index(k)],k));chosen.append(k);reasons.append(dict(key=k,role='diversity',distance=float(near[keys.index(k)])))
    control=random_control(records,chosen)
    require(len(set(chosen))==len(set(control))==40,'Incorrect blind arm size')
    return dict(targeted=chosen,random=control,reasons=reasons,fold_counts=[sum(fold(records[k]['group_id'])==f for k in chosen) for f in range(5)],
                overlap=sorted(set(chosen)&set(control)),unique_label_budget=len(set(chosen)|set(control)))


def write_exclusive(path,obj,compressed=False):
    data=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    if compressed:data=gzip.compress(data,mtime=0)
    with path.open('xb') as f:f.write(data)


def worker(box, grant_sha):
    import cv2
    import torch
    import torchvision
    import ultralytics
    from torchvision.ops import roi_align
    from ultralytics.utils.nms import non_max_suppression
    from night_capture import _open_model, VERSIONS, A_SHA
    from night_strategy import worker_files
    from night_train import fit_image
    check_equal('locked blind capture versions',VERSIONS,(torch.__version__,torchvision.__version__,ultralytics.__version__))
    grant=read_sealed_json(box,'grant.json',grant_sha,[]);inv={x['key']:x for x in _records(grant['manifest'],POOL,475,images_only=True)}
    base,keys=grant['base'],grant['keys'];require(len(base)==len(set(base))==300 and len(keys)==len(set(keys))==175 and not set(base)&set(keys) and set(base+keys)==set(inv),'Wrong blind grant')
    torch.set_num_threads(4);cv2.setNumThreads(1);model=_open_model(torch);tap=[];scene=[]
    head=model.model[-1];check_equal('native head strides',[8.,16.,32.],head.stride.tolist())
    def head_tap(_,args): tap.append(args[0][0])
    def scene_tap(_,__ ,result): scene.append(result[0].float().mean((1,2)))
    handles=[head.register_forward_pre_hook(head_tap),model.model[10].register_forward_hook(scene_tap)]
    def infer(img):
        pixels,_=fit_image(img,[],False,1.,640);x=torch.from_numpy(pixels).to('cuda:0',torch.float16).div_(255)[None]
        raw,dec=model(x);require(len(tap)==len(scene)==1 and raw.shape==(1,10,8400),'Capture shape mismatch')
        feature=tap.pop().float(); global_z=scene.pop();require(feature.shape[-2:]==(80,80),'Incorrect ROI map')
        (det,),(ids,)=non_max_suppression(raw.clone(),.001,.7,multi_label=True,nc=6,max_det=300,max_nms=30000,return_idxs=True)
        ids=ids.flatten().long(); cls=det[:,5].long();require(torch.equal(det[:,4],dec['scores'][0,cls,ids].sigmoid().to(det.dtype)),'Score/anchor mismatch')
        boxes=det[:,:4].float().clamp(0,640);valid=(cls>=2)&(cls<=4)&(boxes[:,2:]>boxes[:,:2]).all(1);boxes=boxes[valid]
        roi=roi_align(feature,[boxes],(2,2),spatial_scale=1/8,sampling_ratio=2,aligned=True).flatten(1)
        lens=torch.linalg.vector_norm(roi,dim=1);good=lens>0;roi=roi[good]/lens[good,None]
        require(torch.isfinite(roi).all().item() and not feature.requires_grad and global_z.norm().item()>0,'Invalid frozen descriptor')
        return (cls[valid][good].tolist(),ids[valid][good].tolist(),det[valid,4][good].tolist(),det[valid,:4][good].float().cpu().tolist(),roi.cpu().numpy(),(global_z/global_z.norm()).cpu().numpy())
    out=box/'output';out.mkdir(); files=['blind.json.gz','selection.json','capture.json','a_boxes.json.gz']; imdir=box/'stage'; t0=time.monotonic()
    with torch.inference_mode():infer(np.zeros((320,640,3),np.uint8))
    reads=worker_files(readable=[imdir/x['image'] for x in inv.values()],writable=[out/n for n in files]); orig=[]; vectors={};scenes={};best_scores={};archives={};stub={}
    with torch.inference_mode():
        for is_base, bucket in ((True,base),(False,keys)):
            if not is_base:
                support={str(c):len({r[1] for r in orig if r[2]==c}) for c in (2,3,4)}
                if min(support.values())<5:
                    check_equal('reference-only guard reads',[imdir/inv[k]['image'] for k in base],reads)
                    write_exclusive(out/files[0],dict(source_rows=orig),True)
                    write_exclusive(out/files[1],dict(stopped='reference_groups',support=support,required=5,unique_label_budget=0))
                    write_exclusive(out/files[2],dict(versions=VERSIONS,model_sha256=A_SHA,reads=[p.relative_to(imdir).as_posix() for p in reads],labels_read=False,review_read=False,train=False))
                    return
                bank={c:references(orig,c) for c in (2,3,4)}; proto={c:np.stack([vectors[x[0],x[2],x[3]] for x in bank[c]]).astype(np.float64) for c in bank}
                high={str(c):max(.05,median(best_scores[k][c] for k in base)) for c in (3,4)}
            for k in bucket:
                require(time.monotonic()-t0<600,'Ten-minute blind capture cap')
                rr=inv[k];raw=read_sealed_bytes(imdir,rr['image'],rr['image_sha256'],[]);im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);require(im is not None,'Invalid image')
                classes,ids,ss,bb,zz,scenes[k]=infer(im)
                if is_base:
                    best_scores[k]={c:max((s for cat,s in zip(classes,ss) if cat==c),default=0.) for c in (3,4)}
                    for c,i,s,z in zip(classes,ids,ss,zz,strict=True):
                        if s>=.5:orig.append((k,rr['group_id'],c,i,s));vectors[k,c,i]=z.copy()
                else:
                    tail=[j for j,c in enumerate(classes) if c in (3,4)];z=zz[tail].astype(np.float64)
                    sim=[np.sort(z@proto[c].T,axis=1)[:,-5:].mean(1) for c in (2,3,4)]
                    stub[k]=[dict(class_id=classes[j],anchor=ids[j],score=ss[j],similarity=[float(s[t]) for s in sim]) for t,j in enumerate(tail)]
                    archives[k]=[dict(class_id=c,anchor=i,score=s,box=b) for c,i,s,b in zip(classes,ids,ss,bb,strict=True)]
            print('Frozen A image-only pass:',('references' if is_base else 'candidates'),len(bucket),flush=True)
    for h in handles:h.remove()
    base_z=np.stack([scenes[k] for k in base]).astype(np.float64);zz=np.stack([scenes[k] for k in keys]).astype(np.float64)
    initial=(1.-zz@base_z.T).min(1);distances=1.-zz@zz.T
    selection=select_from_scores(stub,keys,{k:inv[k] for k in keys},high,initial,distances);union=sorted(set(selection.get('targeted',[]))|set(selection.get('random',[])))
    check_equal('exact image-only worker reads',[imdir/inv[k]['image'] for k in base+keys],reads)
    ev=dict(keys=keys,source_rows=orig,bank=bank,reference_maxima=best_scores,high=high,rank_inputs=stub,initial=initial.tolist(),distances=distances.tolist())
    write_exclusive(out/files[0],ev,True);write_exclusive(out/files[1],selection)
    write_exclusive(out/files[2],dict(seconds=time.monotonic()-t0,versions=VERSIONS,model_sha256=A_SHA,reads=[p.relative_to(imdir).as_posix() for p in reads],labels_read=False,review_read=False,train=False))
    if union:write_exclusive(out/files[3],{k:archives[k] for k in union},True)


def verified_selection(output, digest):
    inv,base,keys=reservation();s=read_sealed_json(output,'manifest.json',digest,[])
    for n,meta in s['files'].items():
        p=output/n;_no_reparse(p);require(p.stat().st_size==meta['bytes'] and sha256(p)==meta['sha256'],'File receipt mismatch: '+n)
    require(s['status']=='sealed' and s['reserve']==RESERVE and s['protocol_sha256']==PROTOCOL,'Unbound selection')
    e=json.loads(gzip.decompress((output/'blind.json.gz').read_bytes())); observed=sealed(output,s,'selection.json'); capture=sealed(output,s,'capture.json')
    from night_capture import A_SHA, VERSIONS
    check_equal('only original frozen A', (A_SHA,list(VERSIONS),False,False,False), (capture['model_sha256'],capture['versions'],capture['labels_read'],capture['review_read'],capture['train']))
    require(all(k in base and g==inv[k]['group_id'] and c in (2,3,4) and v>=.5 for k,g,c,_,v in e['source_rows']),'Invalid reference source')
    support={str(c):len({r[1] for r in e['source_rows'] if r[2]==c}) for c in (2,3,4)}
    if min(support.values())<5:
        check_equal('stopped before candidate inference',([inv[k]['image'] for k in base],dict(stopped='reference_groups',support=support,required=5,unique_label_budget=0)),(capture['reads'],observed))
        return observed
    check_equal('exact replay input order',(keys,set(base),[inv[k]['image'] for k in base+keys]),(e['keys'],set(e['reference_maxima']),capture['reads']))
    check_equal('reference bank replay',e['bank'],{str(c):references(e['source_rows'],c) for c in (2,3,4)})
    high={str(c):max(.05,float(np.median([e['reference_maxima'][k][str(c)] for k in base]))) for c in (3,4)}
    check_equal('image-only proxy boundaries',high,e['high'])
    check_equal('both arm orders replay exactly',observed,select_from_scores(e['rank_inputs'],keys,{k:inv[k] for k in keys},high,e['initial'],e['distances']))
    return observed


def run_select(scratch):
    from seed_eval import probe_gpu
    from night_train import _receipt
    gpu=probe_gpu();inv,base,keys=reservation();out=RUNS/'pool_008';require(not out.exists() and not out.is_symlink(),'Already selected');_no_reparse(out.parent);_no_reparse(scratch)
    require(scratch.resolve()==scratch and scratch.drive==ROOT.drive and not scratch.is_relative_to(ROOT),'External same-drive scratch required')
    with tempfile.TemporaryDirectory(dir=scratch,prefix='supp-blind-') as temp:
        box=Path(temp);stage=box/'stage';rr=[inv[k] for k in base+keys]
        for row in rr:
            f=stage/row['image'];f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(read_sealed_bytes(NIGHT,row['image'],row['image_sha256'],[]))
        grant=box/'grant.json';write_json(grant,dict(base=base,keys=keys,manifest={'purpose':POOL,'records':rr}));env=dict(os.environ,YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','TEMP','TMP','TMPDIR','XDG_CONFIG_HOME'):
            f=box/k.lower();f.mkdir();env[k]=str(f)
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'_worker',str(box),sha256(grant)],cwd=stage,env=env,check=True,timeout=650)
        e=box/'output';write_json(e/'manifest.json',dict(status='sealed',protocol_sha256=PROTOCOL,reserve=RESERVE,gpu=gpu,files=_receipt(e),source_sha256=sha256(Path(__file__))))
        digest=sha256(e/'manifest.json');selection=verified_selection(e,digest);e.rename(out)
    print('Blind replay:',digest,selection if 'stopped' in selection else {k:selection[k] for k in ('fold_counts','overlap','unique_label_budget')})


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='_worker':worker(Path(sys.argv[2]),sys.argv[3])
    else:
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('select','verify'));parser.add_argument('--scratch',type=Path);parser.add_argument('--manifest-sha256');args=parser.parse_args()
        if args.mode=='select' and args.scratch and not args.manifest_sha256:run_select(args.scratch.absolute())
        elif args.mode=='verify' and args.manifest_sha256 and not args.scratch:
            s=verified_selection(RUNS/'pool_008',args.manifest_sha256);print('Verified:',s if 'stopped' in s else {k:s[k] for k in ('fold_counts','overlap','unique_label_budget')})
        else:parser.error('select takes --scratch; verify takes --manifest-sha256')
