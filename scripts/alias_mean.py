"""Test a within-seed parameter mean of the original-A and name-remapped source trajectories."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import person_local as pl
from person_local import ROOT,pf,se,dm
from person_prior import evaluate
from source_endpoints import midpoint
from source_contract import ContractError,read_json,sha256,check_equal,write_json
from source_policy import SelectionPolicy,reference_at
from gpu_queue import verify_inventory
from evaluate import stage_subset


def run(alias,out,scratch):
    if alias.parent!=ROOT/'runs' or out.parent!=ROOT/'runs' or out.exists() or not pl.re.fullmatch(r'seed_\d{3}',out.name) or scratch.is_relative_to(ROOT):raise ContractError('Invalid run location')
    cloud=read_json(alias/'run/manifest.json');dev=read_json(alias/'dev/manifest.json');receipt=read_json(alias.with_suffix('.queue.json'));s=cloud['seed'];dseal=sha256(alias/'dev/manifest.json');verify_inventory(alias/'dev',dev)
    check_equal('verified training and local eval',('Stopped',sha256(alias/'run/manifest.json'),s,cloud['run']['best_sha256'],False),(receipt['platform_state'],dev['source_training_seal'],dev['seed'],dev['checkpoint']['sha256'],dev['final_test_executed']))
    rules,ct,arms,paths=pl.prepare();ct['rules']=rules;policy=SelectionPolicy(rules,reference_at(ROOT));new=alias/'run'/str(s)/'weights/best.pt';check_equal('alias best',cloud['run']['best_sha256'],sha256(new));gpu=se.probe_gpu()
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);stage=tmp/'result';folder=stage/'parallel_mean'/str(s);folder.mkdir(parents=True)
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled');import torch,torchvision,ultralytics
        versions=[torch.__version__,torchvision.__version__,ultralytics.__version__];check_equal('versions',['2.14.0+cu130','0.29.0+cu130','8.4.171'],versions);torch.set_num_threads(4);weight=folder/'model.pt';tensors=midpoint(ROOT/arms[s]['path'],new,weight)
        fail,warn,_=evaluate(s,weight,folder,ct,stage_subset(ct,tmp/'data'),tmp/'views',policy,rules,paths);ma=dm.read_rows(folder/'matches.jsonl.gz');rows=dm.read_rows(folder/'objects.jsonl.gz');trans,g=pl.transitions(rows,dm.read_rows(alias/'dev/matches.jsonl.gz'),ma);m=read_json(folder/'metrics.json')
        write_json(stage/'comparison.json',{'seed':s,'numeric_five':not fail,'failed_five':fail,'warnings':warn,'vs_alias':{'transitions':trans,'paired_recall':pl.paired_interval([g])},'vs_original':m['paired_recall'],'qualification_scope':'this seed only'})
        if fail:weight.unlink()
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_dev','seed':s,'source_control':arms[s],'alias':dev['checkpoint'],'alias_dev_seal':dseal,'differing_tensors':tensors,'output_weight_sha256':m['checkpoint_sha256'],'ready':pf.READY_SHA,'rules':rules['sha256'],'dev_manifest':ct['manifest'],'baseline':pl.SEAL,'gpu':gpu,'versions':versions,'cloud_cny':0,'training_performed':False,'final_test_executed':False,'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in ('alias_mean.py','source_endpoints.py','person_prior.py','seed_eval.py','source_policy.py','person_fusion.py','tricycle_fusion.py','detection_metrics.py','evaluate.py')},'files':{f.relative_to(stage).as_posix():{'sha256':sha256(f),'bytes':f.stat().st_size} for f in stage.rglob('*') if f.is_file()}});shutil.copytree(stage,out)
    print('ALIAS_MEAN',s,m['metrics']['per_class']['pedestrian']['ap50'],m['workpoints']['pedestrian'],fail,warn,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--alias',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();run(a.alias.resolve(),a.output.resolve(),a.scratch.resolve())
