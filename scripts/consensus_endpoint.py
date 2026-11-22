"""One prespecified interaction: sealed consensus scorer applied to within-seed weight midpoints."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import person_local as pl
from person_consistency import SPECS,apply,clusters
from person_local import ROOT,dm,pf,se
from evaluate import stage_subset
from source_contract import ContractError,NAMES,check_equal,read_json,sha256,write_json
from source_policy import SelectionPolicy,reference_at,summarize
from source_endpoints import grade,midpoint,endpoint

TRAIN=ROOT/'runs/views_007'
MID=ROOT/'runs/seed_003'


def run(out,scratch):
    if out.exists() or out.parent!=ROOT/'runs' or not pl.re.fullmatch(r'views_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Expected fresh views_NNN and external scratch')
    check_equal('sealed two parent experiments',('e9afee54b40aea56dcce33bf72fcedd5257f59b452c4b3225bfd95d317950e2a','66931584f2a0e5c612b0e0af862c501db15234a23d10a6a65e7a89c0c58a022e'),(sha256(TRAIN/'manifest.json'),sha256(MID/'manifest.json')))
    for root,name in ((TRAIN,'fit.json'),(MID,'midpoint/29/objects.jsonl.gz')):check_equal(name,read_json(root/'manifest.json')['files'][name]['sha256'],sha256(root/name))
    fit=read_json(TRAIN/'fit.json');check_equal('internal qualification',True,fit['passed']);rules,ct,arms,paths=pl.prepare();ct['rules']=rules;policy=SelectionPolicy(rules,reference_at(ROOT));gpu=se.probe_gpu()
    native=pf.PROFILES['a1'][2];sizes,mode=se.tf.PROFILES['low416-or'];screen={};groups=[]
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'result';stage.mkdir()
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled');import torch,torchvision,ultralytics
        vers=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],vers);torch.set_num_threads(4);data=stage_subset(ct,tmp/'dev')
        for s in (29,17,43):
            f=stage/'joint'/str(s);f.mkdir(parents=True);weight=f/'model.pt';last=endpoint(s,arms[s]);changed=midpoint(ROOT/arms[s]['path'],ROOT/last['path'],weight);cap=tmp/f'capture{s}';cap.mkdir();views={};obs=[]
            # For 29 the sealed nine-view midpoint is already available; check its four pedestrian views before reuse.
            for sp in (SPECS if s==29 else tuple(dict.fromkeys((native,*SPECS,*[(z,False) for z in sizes])))):
                v,item=pf._capture(torch,ct,data,weight,cap,*sp);views[sp]=pf._dev_index(v);obs.append(item)
            if s==29:mid=dm.read_rows(MID/'midpoint/29/objects.jsonl.gz')
            else:
                a,_=pf.evaluate_rows(list(views[native].values()),tuple(list(views[v].values()) for v in SPECS));mid,_=se.tf.merge_dev(a,tuple(list(views[z,False].values()) for z in sizes),mode)
            orig=dm.read_rows(paths[s]);check_equal('frozen images and GT',[(r['key'],pf._identity(r)) for r in orig],[(r['key'],pf._identity(r)) for r in mid]);rows=[]
            for r in mid:
                vs=[[(score,box) for c,score,box in pf._detections(views[sp][r['key']]['pred']) if c==0] for sp in SPECS];ds,x=clusters(vs);old=[(score,box) for c,score,box in pf._detections(r['pred']) if c==0];check_equal('same midpoint person boxes and scores',old,ds[:len(old)])
                z,n=pl.replace(r,apply(old,x[:len(old)],fit['model']),False);check_equal('no new cap',0,n);keep=lambda a:[q for q in pf._detections(a['pred']) if q[0]!=0];check_equal('nonperson unchanged',keep(r),keep(z));rows.append(z)
            ref=read_json(pl.BASE/str(s)/'metrics.json');m,ma,curve,fails,warn=grade(rows,rules,policy,ref);trans,g=pl.transitions(rows,dm.read_rows(pl.BASE/str(s)/'matches.jsonl.gz'),ma);groups.append(g)
            control,_,_=summarize(mid,rules)
            for n in NAMES[1:]:check_equal('protected workpoint',control['workpoints'][n],m['workpoints'][n])
            m.update(failed_five=fails,relative_warnings=warn,transitions=trans,paired_recall=pl.paired_interval([g]),weight={'sha256':sha256(weight),'best':arms[s],'last':last,'changed_tensors':changed},midpoint={'metrics':control['metrics'],'workpoints':control['workpoints']},observations=obs)
            write_json(f/'metrics.json',m);dm.write_rows(f/'objects.jsonl.gz',rows);dm.write_rows(f/'matches.jsonl.gz',ma);np.savez_compressed(f/'pr.npz',**curve);screen[s]={'failed_five':fails,'warnings':warn};print('JOINT',s,m['metrics']['per_class']['pedestrian']['ap50'],m['workpoints']['pedestrian'],fails,warn,flush=True)
            if fails:break
        chosen='joint' if len(screen)==3 and all(not v['failed_five'] for v in screen.values()) and not any(sum(t in v['warnings'] for v in screen.values())>=2 for t in ('area_S','width_le8','small_thin_union','ap75')) else None
        if not chosen:
            for s in screen:(stage/'joint'/str(s)/'model.pt').unlink()
        write_json(stage/'comparison.json',{'selected':chosen,'screen':screen,'paired_recall':pl.paired_interval(groups)});write_json(stage/'manifest.json',{'status':'complete','scope':'source_dev','baseline':pl.SEAL,'ready':pf.READY_SHA,'rules':rules['sha256'],'dev_manifest':ct['manifest'],'arms':arms,'parent_seals':[sha256(z/'manifest.json') for z in (TRAIN,MID)],'frozen_calibration_sha256':sha256(TRAIN/'fit.json'),'gpu':gpu,'versions':vers,'cloud_cny':0,'training_performed':False,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('consensus_endpoint.py','person_consistency.py','source_endpoints.py','person_fusion.py','person_local.py','seed_eval.py','detection_metrics.py','source_policy.py','evaluate.py')},'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('JOINT_SEALED',out,chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.scratch.resolve())
