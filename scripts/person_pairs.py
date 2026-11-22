"""Read-only equal-recall comparison and group-disjoint, score-matched pedestrian probes."""
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
from person_local import ROOT, pf, se, dm
from source_contract import NAMES, ContractError, check_equal, sha256, read_json, write_json
from evaluate import verify_subset, original_gt

PILOT=ROOT/'runs/teacher_002'
SIGNAL=ROOT/'runs/teacher_001'
ARMS=('origin','ordinary','teacher')


def order(s):return hashlib.sha256(('pairs29:'+s).encode()).hexdigest()


def sources():
    rules,dev,arms,paths=pl.prepare();m=read_json(PILOT/'manifest.json'); sm=read_json(SIGNAL/'manifest.json')
    check_equal('pilot signal',m['signal_sha256'],sha256(SIGNAL/'manifest.json'))
    check_equal('deployed seed 29',arms[29],m['frozen_initial']);check_equal('dev manifest',dev['manifest'],m['dev_manifest'])
    for f,info in m['files'].items():check_equal('pilot '+f,info['sha256'],sha256(PILOT/f))
    check_equal('training sample',sm['files']['targets.jsonl.gz']['sha256'],sha256(SIGNAL/'targets.jsonl.gz'))
    tr=verify_subset(pf.SOURCE,pf.READY_SHA,'source_train');check_equal('same train manifest',tr['manifest'],m['train_manifest'])
    check_equal('exclusion identity',sm['exclusions_sha256'],sha256(ROOT/'configs/rerank.json'))
    excluded={v['key'] for v in read_json(ROOT/'configs/rerank.json')['candidates']}
    seen={r['group_id'] for r in dm.read_rows(SIGNAL/'targets.jsonl.gz')};check_equal('pilot groups',512,len(seen))
    tr['records']=[r for r in tr['records'] if r['group_id'] not in seen and r['key'] not in excluded]
    groups=sorted({r['group_id'] for r in tr['records']},key=order)
    if set(groups)&{r['group_id'] for r in dev['records']}:raise ContractError('Train/dev groups overlap')
    halves={g:('discovery','replication')[i%2] for i,g in enumerate(groups)}
    weights={'origin':ROOT/arms[29]['path'],**{a:PILOT/a/'last.pt' for a in ARMS[1:]}}
    return tr,halves,weights,dev,paths[29],rules


def gpu_tools(tmp):
    for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','XDG_CONFIG_HOME'):os.environ[k]=str(tmp)
    os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
    import torch,torchvision,ultralytics,cv2
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils import ops
    versions=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('packages',
                      ['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions);torch.set_num_threads(4)
    def load(weight):
        model,_=load_checkpoint(str(weight),device='cuda:0',fuse=True);check_equal('model classes',NAMES,list(model.names.values()))
        return model.half().eval()
    def predict(model,img,side,suppress):
        pad=LetterBox((side,side),auto=True,stride=32)(image=img);tensor=torch.from_numpy(np.ascontiguousarray(pad[:,:,::-1].transpose(2,0,1))).cuda().half()[None]/255
        with torch.inference_mode():p=model(tensor)[0][0]
        if not p.isfinite().all() or p.shape[0]!=10:raise ContractError('Invalid dense output')
        boxes=ops.xywh2xyxy(p[:4].T.float());score=p[4].float();ids=torch.arange(len(score),device='cuda')
        if suppress:
            ids=ids[score>=.20];ids=ids[torchvision.ops.nms(boxes[ids],score[ids],.7)]
        boxes=ops.scale_boxes(tensor.shape[-2:],boxes[ids],img.shape[:2]);return ids.cpu().numpy(),score[ids].cpu().numpy(),boxes.cpu().numpy()
    def image(tr,r):
        a=cv2.imdecode(np.fromfile(tr['root']/r['image'],np.uint8),cv2.IMREAD_COLOR);check_equal('geometry',(r['height'],r['width']),a.shape[:2]);return a
    return load,predict,image,versions


def near_person(box,gt):
    pad=.5*(box[3]-box[1]);b=box+np.array([-pad,-pad,pad,pad]);return any(c in (0,1) and
        min(b[2],g[2])>max(b[0],g[0]) and min(b[3],g[3])>max(b[1],g[1]) for c,g in gt)


def sample(tr,halves,weight,tmp):
    load,predict,image,versions=gpu_tools(tmp);model=load(weight);candidates={'pos':[],'neg':[]}
    for sequence,rec in enumerate(tr['records'],1):
        ids,score,boxes=predict(model,image(tr,rec),640,True);gt=original_gt(rec,tr['root']);gc=np.array([c for c,_ in gt],int)
        iou=dm.box_iou([b for c,b in gt],boxes);matched={p:g for g,p in dm.native_pairs(np.zeros(len(ids),int),gc,iou,.5)}
        for j in np.flatnonzero(score<=.50):
            box=boxes[j];g=matched.get(j);best=float(iou[:,j].max(initial=0))
            if g is not None and int((iou[gc==0,j]>=.5).sum())==1 and iou[g,j]>=best:kind='pos'
            elif best<.1 and not near_person(box,gt):kind='neg'
            else:continue
            if box[3]-box[1]<1 or box[2]-box[0]<1:continue
            candidates[kind].append({'key':rec['key'],'group':rec['group_id'],'batch':halves[rec['group_id']],
                     'anchor':int(ids[j]),'score':float(score[j]),'box':box.tolist(),'max_gt_iou':best,
                     'gt':None if g is None else int(g),'image_sha256':rec['image_sha256'],'label_sha256':rec['label_sha256']})
        if sequence%500==0:print('TRAIN_PROPOSALS',sequence,len(tr['records']),flush=True)
    pair=[];used=set()
    for batch in ('discovery','replication'):
        pos=[r for r in candidates['pos'] if r['batch']==batch];negs=sorted((r for r in candidates['neg'] if r['batch']==batch),key=lambda r:order(f"{r['key']}:{r['anchor']}"));n=0
        for neg in negs:
            if neg['group'] in used:continue
            viable=[];height=lambda r:r['box'][3]-r['box'][1]
            for v in pos:
                if v['group'] in used or v['group']==neg['group']:continue
                delta=abs(v['score']-neg['score']);lh=abs(np.log(height(v)/height(neg)))
                if delta<=.05 and lh<=np.log(1.5):viable.append((delta/.05+lh/np.log(1.5),v['key'],v['anchor'],v))
            if not viable:continue
            v=min(viable,key=lambda x:x[:3])[-1];used.update((v['group'],neg['group']));n+=1
            pair.append({'id':f'{batch[0]}{n:02}','batch':batch,'pos':v,'neg':neg})
            if n==48:break
    return {'train_images':len(tr['records']),'groups':len(halves),'pool':{b:{k:sum(r['batch']==b for r in ls) for k,ls in candidates.items()} for b in ('discovery','replication')},'versions':versions,'pairs':pair}


def prepare(scratch):
    if scratch.is_relative_to(ROOT) or not scratch.is_dir() or any(scratch.iterdir()):raise ContractError('Expected empty external scratch')
    tr,halves,weights,*_=sources();gpu=se.probe_gpu()
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        pool=sample(tr,halves,weights['origin'],Path(tmp))
    pool.update(ready=pf.READY_SHA,train_manifest=tr['manifest'],initial_gpu=gpu,weights={k:sha256(f) for k,f in weights.items()})
    write_json(scratch/'sample.json',pool);print('SAMPLE',pool['train_images'],pool['groups'],pool['pool'],Counter(p['batch'] for p in pool['pairs']),flush=True)


def equal_recall(dev,origin,rules):
    shared=read_json(pl.BASE/'29/metrics.json')['workpoints']['pedestrian']['threshold'];results={};group_counts={}
    contracted={r['key']:r for r in dev['records']}
    for arm in ARMS:
        path=origin if arm=='origin' else PILOT/arm/'locked/objects.jsonl.gz'
        rows=dm.read_rows(path);check_equal('complete source_dev',set(contracted),set(pf._dev_index(rows)))
        for row in rows:
            for f in ('group_id','image_sha256','label_sha256'):check_equal(f,contracted[row['key']][f],row[f])
        curve=dm.work_curve(rows,0);need=int(np.ceil(curve['gt']*rules['classes']['pedestrian']['recall']));check_equal('target TP',291,need)
        ix=np.flatnonzero(curve['tp']==need);limit=int(np.floor(need*(1/rules['classes']['pedestrian']['precision']-1)+1e-8))
        allowed=curve['fp']<=limit;results[arm]={'fixed_fp':{'limit':limit,'max_tp':int(curve['tp'][allowed].max(initial=0))}};thresholds={'shared':shared}
        if len(ix):thresholds['equal_tp']=float(curve['threshold'][min(ix,key=lambda j:(curve['fp'][j],-curve['threshold'][j]))])
        else:results[arm]['equal_tp']=None
        for key,threshold in thresholds.items():
            counts=Counter();fp=Counter();bygroup={};boxes=[]
            for r in rows:
                stat,pairs,selected=dm.image_point(r,0,threshold);used=set(pairs[:,1]);g=bygroup.setdefault(r['group_id'],[0,0]);g[0]+=stat['fp'];g[1]+=1
                for k in ('tp','fp','fn'):counts[k]+=stat[k]
                for i in selected:
                    if i in used:continue
                    reason=dm.error_type(r,i,pairs);kind=reason['type'];name=NAMES[r['gt']['cls'][reason['best_gt']]] if kind=='classification' else None
                    label=f'wrong_{name}' if name else kind;fp[label]+=1;boxes.append([r['key'],int(i),label])
            check_equal('all FP accounted',counts['fp'],sum(fp.values()))
            if key=='equal_tp':check_equal('rematched exact TP',need,counts['tp'])
            results[arm][key]={'counts':dm.counts(**counts,threshold=threshold),'fp_types':fp,'fp_ids':boxes};group_counts[arm,key]=bygroup
    ci={}
    for k in ('shared','equal_tp'):
        if any((a,k) not in group_counts for a in ARMS[1:]):continue
        b,t=[group_counts[a,k] for a in ARMS[1:]];groups=sorted(b);check_equal('risk groups',groups,sorted(t))
        z=np.array([[t[g][0]-b[g][0],t[g][1]] for g in groups]);draws=np.stack([v for _,v in dm.cluster_draws(groups,2000,20261026)]);d=draws@z
        ci[k]={'groups':len(groups),'delta_fp_per_100_images':float(100*z[:,0].sum()/z[:,1].sum()),
               'ci95':np.quantile(100*d[:,0]/d[:,1],[.025,.975]).tolist(),'fixed_arm_thresholds':True,'repeats':2000,'seed':20261026}
    return {'arms':results,'teacher_minus_ordinary':ci}


def replay(pairs,review,tr,weights,tmp):
    load,predict,image,_=gpu_tools(tmp);models={k:load(v) for k,v in weights.items()};records={r['key']:r for r in tr['records']};out=[]
    for pair in pairs:
        jud=review[pair['id']]
        if jud['positive']!='confirmed' or jud['negative']!='structure':continue
        result={'id':pair['id'],'batch':pair['batch']}
        for side in ('pos','neg'):
            cand=pair[side];rec=records[cand['key']];a=image(tr,rec);i=cand['anchor'];values={}
            for arm in ARMS:
                ix,s,b=predict(models[arm],a,640,False);check_equal('same dense index',i,int(ix[i]))
                check_equal('frozen candidate geometry',cand['box'],b[i].tolist());values[arm]=float(s[i])
            check_equal('same original score',cand['score'],values['origin'])
            _,s,b=predict(models['origin'],a,960,False)
            # Same reference for both sides: the *640 candidate*, never GT, defines the 960 query.
            ov=dm.box_iou([cand['box']],b)[0];good=(ov>=.5)&(s>pf.FLOOR);values['scale960']=float(s[good].max()) if good.any() else 0.
            result[side]=values
        out.append(result)
    return out


def margins(rows):
    res={'pairs':len(rows)}
    if not rows:return res
    margins={}
    for arm in (*ARMS,'scale960'):
        x=np.array([r['pos'][arm]-r['neg'][arm] for r in rows]);margins[arm]=x
        res[arm]={'mean_margin':float(x.mean()),'positive_wins':int((x>0).sum()),'ties':int((x==0).sum()),
                  'mean_positive':float(np.mean([r['pos'][arm] for r in rows])), 'mean_negative':float(np.mean([r['neg'][arm] for r in rows]))}
    idx=np.random.default_rng(20261026).integers(0,len(rows),(2000,len(rows)))
    for name,a,b in [('teacher_minus_ordinary','teacher','ordinary'),('scale960_minus_origin','scale960','origin')]:
        d=margins[a]-margins[b];res[name]={'delta_margin':float(d.mean()),'ci95':np.quantile(d[idx].mean(axis=1),[.025,.975]).tolist(),
             'pos_raised':sum(r['pos'][a]>r['pos'][b] for r in rows),'neg_raised':sum(r['neg'][a]>r['neg'][b] for r in rows)}
    return res


def seal(scratch,out):
    if out.exists() or out.parent!=ROOT/'runs' or out.name!='teacher_003' or scratch.is_relative_to(ROOT):raise ContractError('Expected new teacher_003 and external scratch')
    tr,halves,weights,dev,origin,rules=sources();pool=read_json(scratch/'sample.json');review=read_json(scratch/'review.json')
    check_equal('sample provenance',(pf.READY_SHA,tr['manifest'],{k:sha256(v) for k,v in weights.items()}),(pool['ready'],pool['train_manifest'],pool['weights']))
    check_equal('blinded review bound to sample',sha256(scratch/'sample.json'),review['sample_sha256']);dec=review['decisions'];pairs=pool['pairs']
    check_equal('review contains all candidates',{p['id'] for p in pairs},set(dec));used=set();rx={r['key']:r for r in tr['records']}
    for pair in pairs:
        j=dec[pair['id']]
        if j['negative'] not in ('structure','possible_person','uncertain') or j['positive'] not in ('confirmed','uncertain') or not j['note']:
            raise ContractError('Missing review decision or rationale')
        for name in ('pos','neg'):
            c=pair[name];r=rx[c['key']];check_equal('group split',(r['group_id'],halves[r['group_id']],r['image_sha256'],r['label_sha256']),
                    (c['group'],pair['batch'],c['image_sha256'],c['label_sha256']))
            if c['group'] in used:raise ContractError('Pairs reuse a scene group')
            used.add(c['group'])
    se.probe_gpu()
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);rows=replay(pairs,dec,tr,weights,tmp);fixed=equal_recall(dev,origin,rules)
        summary={k:margins([r for r in rows if r['batch']==k]) for k in ('discovery','replication')}
        enough=summary['discovery']['pairs']>=20 and summary['replication']['pairs']>=12
        stage=tmp/'result';stage.mkdir();shutil.copyfile(scratch/'sample.json',stage/'sample.json');shutil.copyfile(scratch/'review.json',stage/'review.json')
        write_json(stage/'comparison.json',{'equal_recall':fixed,'paired_train':summary,'responses':rows,'pair_support_pass':enough})
        check_equal('weight identities after inference',pool['weights'],{k:sha256(v) for k,v in weights.items()})
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_train matched probe; source_dev offline fixed-TP comparison',
            'training_performed':False,'cloud_cny':0,'final_test_executed':False,'ready':pf.READY_SHA,'seed_002_sha256':pl.SEAL,
            'teacher_002_sha256':sha256(PILOT/'manifest.json'),'teacher_001_sha256':sha256(SIGNAL/'manifest.json'),
            'dev_manifest':dev['manifest'],'train_manifest':tr['manifest'],'rules':rules['sha256'],'weights':pool['weights'],
            'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('person_pairs.py','person_local.py','seed_eval.py','person_fusion.py','detection_metrics.py','source_contract.py','evaluate.py')},
            'files':{f.name:{'bytes':f.stat().st_size,'sha256':sha256(f)} for f in stage.iterdir()}});shutil.copytree(stage,out)
    print('PAIR_SUMMARY',summary,'support',enough)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',type=Path);g.add_argument('--seal',type=Path);p.add_argument('--output',type=Path,default=ROOT/'runs/teacher_003')
    a=p.parse_args()
    if a.prepare:prepare(a.prepare.resolve())
    else:seal(a.seal.resolve(),a.output.resolve())
