"""Evaluate one downloaded, completed source_alias best on the locked views after cloud shutdown."""
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
from person_prior import evaluate
from gpu_queue import verify_inventory
from evaluate import stage_subset
from source_policy import SelectionPolicy,reference_at,summarize
from source_contract import ContractError,check_equal,read_json,sha256,write_json,NAMES


def run(base,scratch):
    base=base.resolve();scratch=scratch.resolve();out=base/'dev';cloud=base/'run'
    if base.parent!=ROOT/'runs' or not re.fullmatch(r'alias_\d{3}',base.name) or out.exists() or scratch.is_relative_to(ROOT):raise ContractError('Expected completed alias run and external scratch')
    receipt=read_json(base.with_suffix('.queue.json'));seal=read_json(cloud/'manifest.json');seed=seal['seed'];info=seal['run'];actual=sha256(cloud/'manifest.json');weight=cloud/str(seed)/'weights/best.pt'
    check_equal('cloud execution',('download_verified','Stopped',seed,actual),(receipt['status'],receipt['platform_state'],receipt['formal_seed'],receipt['verified_remote']['sha256']))
    check_equal('alias training identity',('complete',False,seed,info['best_sha256'],True,pf.READY_SHA),(seal['status'],seal['final_test_executed'],info['seed'],sha256(weight),seal['initialization']['destination_other_rows_exact'],seal['ready']))
    verify_inventory(cloud,seal);rules,ct,arms,paths=pl.prepare();ct['rules']=rules;gpu=se.probe_gpu();policy=SelectionPolicy(rules,reference_at(ROOT))
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'result';stage.mkdir()
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch,torchvision,ultralytics
        versions=(torch.__version__,torchvision.__version__,ultralytics.__version__);check_equal('inference packages',('2.14.0+cu130','0.29.0+cu130','8.4.171'),versions);torch.set_num_threads(4);data=stage_subset(ct,tmp/'data')
        failed,warnings,_=evaluate(seed,weight,stage,ct,data,tmp/'views',policy,rules,paths);check_equal('checkpoint after',info['best_sha256'],sha256(weight))
        m=read_json(stage/'metrics.json');rec,ma,curves=summarize(dm.read_rows(stage/'objects.jsonl.gz'),rules);check_equal('recomputed all metrics',rec['metrics'],m['metrics']);check_equal('recomputed points',rec['workpoints'],m['workpoints']);check_equal('recomputed matches',ma,dm.read_rows(stage/'matches.jsonl.gz'))
        with np.load(stage/'pr.npz') as z:
            if set(z.files)!=set(curves) or not all(np.array_equal(z[k],v,equal_nan=True) for k,v in curves.items()):raise ContractError('PR curve export differs')
        checks,guards=policy.assess(rec);check_equal('recomputed verdict',(checks,guards),(m['checks'],m['guards']));five=all(v for k,v in (checks|guards).items() if k.split('/')[0] in NAMES[:5]);check_equal('five class results',five,not failed)
        write_json(stage/'comparison.json',{'seed':seed,'five_pass':five,'failed_five':failed,'relative_warnings':warnings,'all_six_numeric':all((checks|guards).values()),'paired_recall':m['paired_recall'],'metric_scope':'500 source_dev; locked views, per-class constrained precision/recall','not_final_test':True})
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_dev','seed':seed,'source_training_seal':actual,'checkpoint':{'path':weight.relative_to(ROOT).as_posix(),'sha256':info['best_sha256']},'original_control':arms[seed],'dev_manifest':ct['manifest'],'ready':pf.READY_SHA,'rules':rules['sha256'],'versions':versions,'gpu':gpu,'training_performed':False,'cloud_cny':0,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('alias_eval.py','person_prior.py','source_endpoints.py','person_local.py','person_fusion.py','tricycle_fusion.py','evaluate.py','source_policy.py','detection_metrics.py')},'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in stage.iterdir()}});shutil.copytree(stage,out)
        for n in NAMES:
            a=rec['metrics']['per_class'][n];w=rec['workpoints'][n];print('ALIAS_RESULT',seed,n,a['ap50'],a['ap50_95'],w['precision'],w['recall'],w['tp'],w['fp'],flush=True)
        print('ALIAS_FIVE',seed,five,failed,warnings,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.result,a.scratch)
