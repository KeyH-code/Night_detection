"""One checkpoint per seed: evaluate last EMA and a within-trajectory best/last midpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

import person_local as pl
from person_local import ROOT,dm,pf,se
from source_contract import ContractError,NAMES,check_equal,sha256,read_json,write_json
from source_policy import SelectionPolicy,recall_intervals,reference_at,summarize
from evaluate import stage_subset


def endpoint(seed,best):
    pt=ROOT/best['path'];last=pt.with_name('last.pt');side=read_json(last.with_name('last.pt.sha256.json'))
    manifest=read_json((ROOT/'runs/a_001' if seed==17 else se.REPLICA)/'manifest.json');rec=manifest['files'][('' if seed==17 else str(seed)+'/')+'weights/last.pt']
    check_equal('same-seed last seal',(rec['sha256'],rec['bytes']),(sha256(last),last.stat().st_size));check_equal('last sidecar',rec['sha256'],side['sha256'])
    return {'path':last.relative_to(ROOT).as_posix(),'sha256':rec['sha256'],'epoch':side['epoch_0based']+1}


def midpoint(first,last,out):
    import torch
    from ultralytics.nn.tasks import load_checkpoint
    a,_=load_checkpoint(str(first),device='cpu');b,_=load_checkpoint(str(last),device='cpu')
    check_equal('six classes',(NAMES,NAMES),(list(a.names.values()),list(b.names.values())))
    z=b.state_dict();ref=a.state_dict();check_equal('model tensor names',set(ref),set(z));changed=0
    with torch.no_grad():
        for k,x in ref.items():
            y=z[k];check_equal('tensor structure '+k,(x.shape,x.dtype),(y.shape,y.dtype))
            if x.is_floating_point():
                changed+=int(not torch.equal(x,y));x.copy_((x+y)*.5)
                if not torch.isfinite(x).all():raise ContractError(k+' contains non-finite coefficients')
            else:x.copy_(y)
    if not changed:raise ContractError('identical endpoints')
    a.half().eval();torch.save({'model':a,'train_args':a.args,'epoch':-1,'scope':'within-seed equal EMA endpoint mean'},out)
    reload,_=load_checkpoint(str(out),device='cpu'); other=reload.state_dict()
    if not all(torch.equal(v.float(),other[k].float()) for k,v in a.state_dict().items()):raise ContractError('midpoint reload mismatch')
    return changed


def grade(rows,rules,policy,base):
    rec,ma,curves=summarize(rows,rules);checks,guards=policy.assess(rec);rec.update(checks=checks,guards=guards,intervals=recall_intervals(rows,ma)['intervals'])
    failed=sorted(k for k,v in (checks|guards).items() if not v and not k.startswith('tricycle/'))
    old={v['slice']:v['recall50'] for v in base['slices'] if v['class']=='pedestrian'}
    warnings=[v['slice'] for v in rec['slices'] if v['class']=='pedestrian' and v['slice'] in ('area_S','width_le8','small_thin_union') and old[v['slice']]-v['recall50']>rules['slice_drop']]
    if base['metrics']['per_class']['pedestrian']['ap75']-rec['metrics']['per_class']['pedestrian']['ap75']>rules['ap75_drop']:warnings.append('ap75')
    return rec,ma,curves,failed,warnings


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or not re.fullmatch(r'seed_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Choose new seed_NNN and external scratch')
    rules,ct,arms,paths=pl.prepare();ct['rules']=rules;gpu=se.probe_gpu();policy=SelectionPolicy(rules,reference_at(ROOT))
    ends={s:endpoint(s,arms[s]) for s in se.SEEDS};native,person=pf.PROFILES['a1'][2:];sizes,mode=se.tf.PROFILES['low416-or'];chosen=None
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        temp=Path(d);stage=temp/'result';stage.mkdir()
        for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','XDG_CONFIG_HOME','TEMP','TMP','TMPDIR'):
            q=temp/k.lower();q.mkdir();os.environ[k]=str(q)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch,torchvision,ultralytics
        torch.set_num_threads(4);versions=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions)
        data=stage_subset(ct,temp/'data');summary={};effects={}
        for recipe in ('midpoint','last'):
            values={};groups={}
            for s in (29,17,43):
                folder=stage/recipe/str(s);folder.mkdir(parents=True);capture=temp/f'{recipe}-{s}';capture.mkdir();weight=ROOT/ends[s]['path'];info={'path':ends[s]['path']}
                if recipe=='midpoint':
                    weight=folder/'model.pt';info={'path':weight.relative_to(stage).as_posix(),'averaged_differing_tensors':midpoint(ROOT/arms[s]['path'],ROOT/ends[s]['path'],weight)}
                info['sha256']=sha256(weight);views={};obs=[]
                for size,flip in dict.fromkeys((native,*person,*[(z,False) for z in sizes])):
                    rows,v=pf._capture(torch,ct,data,weight,capture,size,flip);views[size,flip]=rows;obs.append(v)
                a,pc=pf.evaluate_rows(views[native],tuple(views[v] for v in person));fused,tc=se.tf.merge_dev(a,tuple(views[z,False] for z in sizes),mode)
                base=read_json(pl.BASE/str(s)/'metrics.json');baseline=dm.read_rows(paths[s]);check_equal('frozen sample and GT',[(r['key'],pf._identity(r)) for r in baseline],[(r['key'],pf._identity(r)) for r in fused])
                rec,ma,curves,failed,warn=grade(fused,rules,policy,base);single,_,_=summarize(views[native],rules);sc,sg=policy.assess(single)
                moved,grp=pl.transitions(fused,dm.read_rows(pl.BASE/str(s)/'matches.jsonl.gz'),ma);groups[s]=grp
                rec.update(failed_five=failed,relative_warnings=warn,weight=info,observations=obs,transitions=moved,paired_recall=pl.paired_interval([grp]),caps={'person':pc,'tricycle':tc},single640={'metrics':single['metrics'],'workpoints':single['workpoints'],'checks':sc,'guards':sg})
                values[s]=rec;write_json(folder/'metrics.json',rec);dm.write_rows(folder/'objects.jsonl.gz',fused);dm.write_rows(folder/'matches.jsonl.gz',ma);np.savez_compressed(folder/'pr.npz',**curves)
                check_equal('weight after',info['sha256'],sha256(weight));w=rec['workpoints']['pedestrian'];print('ENDPOINT',recipe,s,'AP50',rec['metrics']['per_class']['pedestrian']['ap50'],'TP/FP',w['tp'],w['fp'],'failed',failed,'warnings',warn,flush=True)
                if failed:break
            summary[recipe]={s:{'failed_five':r['failed_five'],'warnings':r['relative_warnings']} for s,r in values.items()};effects[recipe]=pl.paired_interval(list(groups.values()))
            if len(values)==3 and not any(v['failed_five'] for v in values.values()) and not any(sum(k in v['relative_warnings'] for v in values.values())>=2 for k in ('area_S','width_le8','small_thin_union','ap75')):
                chosen=recipe;break
            # A failed midpoint is reproducible directly from its two sealed inputs.
            for s in values:
                if recipe=='midpoint':(stage/recipe/str(s)/'model.pt').unlink()
        write_json(stage/'comparison.json',{'selected':chosen,'screen':summary,'conditional_paired_recall_on_evaluated_seeds':effects})
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_dev','baseline':pl.SEAL,'ready':pf.READY_SHA,'rules':rules['sha256'],'dev_manifest':ct['manifest'],
            'best':arms,'last':ends,'protocol':{'person':person,'native':native,'tricycle':sizes,'score':mode},'gpu_initial':gpu,'environment':versions,'selected':chosen,
            'training_performed':False,'final_test_executed':False,'cloud_cny':0,'code_sha256':{f:sha256(ROOT/'scripts'/f) for f in ('source_endpoints.py','person_local.py','person_fusion.py','tricycle_fusion.py','seed_eval.py','source_policy.py','detection_metrics.py','evaluate.py')},
            'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
