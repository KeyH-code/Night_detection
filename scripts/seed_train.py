"""Repeat the original source-A training recipe for seeds 29 and 43 only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time

from source_contract import ContractError, check_equal, read_json, sha256, stage_dataset, verify_source, write_json

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (29, 43)
READY = '72e75ebd99014c830e0227061e390e6aed768ae724b532a5c2c90fd8dec93e85'
COCO = 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'
ORIGINAL_CONFIG = 'runs/a_001/code/configs/source_a.yaml'
ORIGINAL_ARGS = 'runs/a_001/train/args.yaml'
ORIGINAL_FILES = {ORIGINAL_CONFIG: 'dc7656794556aa8a19b9158998338f8ccfeccd400596db8aded764c197ee80e8',
                  ORIGINAL_ARGS: 'b907370f0b8f26d0a6965dce9df68d383bbb3aae15b5570d76413416bfc34fc0'}
PATH_FIELDS = {'model', 'data', 'project', 'name', 'save_dir'}
KEEP = ('args.yaml','results.csv','source_metrics.jsonl','source_run.json','best_dev_metrics.json',
        'weights/best.pt','weights/last.pt','weights/best.pt.sha256.json','weights/last.pt.sha256.json',
        'audit_initial/config_resolved.json','audit_initial/defaults_locked.json',
        'audit_initial/trainer_effective.json','audit_initial/trainer_passed.json',
        'audit_initial/optimizer_effective.json','audit_initial/runtime.json','audit_initial/checkpoint_reload.json')


def replica_config(seed, root=ROOT):
    import yaml
    if seed not in SEEDS:
        raise ContractError('Only missing source-A seeds 29 or 43 are permitted')
    for name, value in ORIGINAL_FILES.items():
        check_equal('sealed original '+name,value,sha256(root/name))
    config=yaml.safe_load((root/ORIGINAL_CONFIG).read_text(encoding='utf-8'))
    baseline=yaml.safe_load((root/ORIGINAL_ARGS).read_text(encoding='utf-8'))
    check_equal('original non-S3 trainer','source_a',config['experiment'])
    check_equal('original select and stop',('pedestrian_car_ap50_95_macro','retain_earlier',20),
                (config['selection']['metric'],config['selection']['tie_policy'],config['training']['patience']))
    check_equal('no supplementary epoch observer',None,config.get('monitor'))
    for name,v in config['training'].items():
        check_equal('original training '+name,baseline[name],v)
    config['training']['seed']=seed
    return config,baseline


def verify_effective(actual, baseline, seed):
    check_equal('repeat seed',seed,actual['seed'])
    check_equal('effective option names',set(baseline)-PATH_FIELDS,set(actual)-PATH_FIELDS)
    for field in sorted(set(baseline)-PATH_FIELDS-{'seed'}):
        a,b=baseline[field],actual[field]
        check_equal('effective '+field,str(a) if field=='device' else a,str(b) if field=='device' else b)


def atomic(path, record):
    temp=path.with_suffix('.tmp'); write_json(temp,record); temp.replace(path)


def inventory(root):
    return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}


def collect(training, target, baseline, seed):
    verify_effective(read_json(training/'audit_initial/trainer_effective.json'),baseline,seed)
    actual=read_json(training/'audit_initial/runtime.json')
    check_equal('finite batch 16 replay',(True,16),(actual['finite_loss'],actual['batch']))
    if not 1 <= actual['epoch_1based'] <= 100 or actual['optimizer_steps'] < 1:
        raise ContractError('Missing valid optimizer and stopping evidence')
    for role in ('best','last'):
        name=f'weights/{role}.pt'; record=read_json(training/(name+'.sha256.json'))
        check_equal('exact completed checkpoint '+role,record['sha256'],sha256(training/name))
        if role=='last': check_equal('completed last epoch',actual['epoch_1based'],record['epoch_0based']+1)
    target.mkdir()
    for name in KEEP:
        p=target/name; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(training/name,p)
        check_equal('copied '+name,sha256(training/name),sha256(p))
    return {'status':'complete','seed':seed,'epochs':actual['epoch_1based'], 'optimizer_steps':actual['optimizer_steps'],
            'best_epoch':read_json(target/'weights/best.pt.sha256.json')['epoch_0based']+1,
            'best_sha256':sha256(target/'weights/best.pt'),'last_sha256':sha256(target/'weights/last.pt')}


def clean_promoted(working, parent):
    if working.is_symlink() or working.resolve()!=parent.resolve()/working.name:
        raise ContractError('Refusing to delete an unexpected working directory')
    for p in working.rglob('*'):
        if p.is_symlink() or not (stat.S_ISREG(p.lstat().st_mode) or stat.S_ISDIR(p.lstat().st_mode)):
            raise ContractError('Unexpected link or special file in the working run')
    shutil.rmtree(working)


def run(a):
    from train_source import configure_paths, environment, run_segment, code_identity
    from source_resources import shared_memory_risk, snapshot
    import psutil
    source,weights,scratch,out=(Path(p).resolve() for p in (a.source,a.weights,a.scratch,a.output))
    if out.exists() or scratch.is_relative_to(ROOT) or not weights.is_file():
        raise ContractError('Expected a new output, external scratch, and existing official COCO weights')
    check_equal('frozen official COCO YOLO11m',COCO,sha256(weights))
    cfg, baseline=replica_config(29)
    exact={s:replica_config(s)[0] for s in SEEDS}
    check_equal('cloud worker platform','linux',sys.platform)
    quotas=snapshot(os.cpu_count(),psutil.virtual_memory().total)
    if (quotas['cpu_count']<cfg['cloud']['min_cpus'] or quotas['ram_bytes']/2**30<cfg['cloud']['min_ram_gib']
            or shared_memory_risk(quotas)):
        raise ContractError('Effective CPU, RAM, or /dev/shm fails original resource gate')
    contract=verify_source(source,READY)
    check_equal('frozen train/dev counts',(4000,500),tuple(len(contract['manifests'][p]) for p in ('source_train','source_dev')))
    with tempfile.TemporaryDirectory(prefix='a-replica-',dir=scratch) as name:
        temp=Path(name); os.environ['NIGHT_SOURCE_SCRATCH']=str(temp); configure_paths()
        out.mkdir(); record={'plan':'source_a_original','requested_seeds':list(SEEDS),'ready':READY,'pretrained_sha256':COCO,
             'original':ORIGINAL_FILES,'active_code':code_identity(),'entry_sha256':sha256(Path(__file__)),
             'resource_limits':quotas,'final_test_executed':False,'runs':{},'status':'working'}
        status=out/'status.json'; atomic(status,record)
        try:
            environment(cfg,out/'environment',cloud=True)
            for name in (*record['active_code'],*ORIGINAL_FILES,'scripts/seed_train.py',
                         'environment/requirements-linux.lock','specs/s3/replica.md'):
                target=out/'code'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
                check_equal('training snapshot '+name,sha256(ROOT/name),sha256(target))
            from ultralytics.cfg import DEFAULT_CFG_DICT
            for seed in SEEDS:
                actual=dict(DEFAULT_CFG_DICT,**exact[seed]['training'])
                actual.update(device=0,val=True,split='val',save=True,save_period=-1,exist_ok=True)
                verify_effective(actual,baseline,seed)
            dataset,_=stage_dataset(contract,temp/'data')
            work=out.parent/'work'; work.mkdir(exist_ok=False)
            for seed in SEEDS:
                if time.time()+a.min_seed_minutes*60+a.reserve_minutes*60>=a.deadline:
                    raise ContractError(f'Not enough authorized time for seed {seed}; preserved completed work')
                record.update(active_seed=seed); atomic(status,record)
                config=exact[seed]; config['model']['path']=str(weights)
                original=work/str(seed)
                run_segment(config,contract,{'sha256':COCO},original,dataset,16,'formal')
                record['runs'][str(seed)]=collect(original,out/str(seed),baseline,seed)
                atomic(status,record)
                # Only this freshly created working copy; promoted artifacts have passed byte checks.
                clean_promoted(original,work)
            check_equal('pretrained weights unchanged',COCO,sha256(weights))
            check_equal('frozen READY unchanged',READY,sha256(source/'READY.json'))
            record.update(status='complete',active_seed=None)
        except BaseException as exc:
            record.update(status='partial' if record['runs'] else 'failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            atomic(status,record)
            atomic(out/'manifest.json',{**record,'files':inventory(out)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','weights','scratch','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--min-seed-minutes',type=int,default=75)
    p.add_argument('--reserve-minutes',type=int,default=12)
    a=p.parse_args()
    import math
    if not math.isfinite(a.deadline) or a.deadline<=time.time() or not 60<=a.min_seed_minutes<=150 or not 8<=a.reserve_minutes<=30:
        p.error('Require a valid deadline and conservative per-seed/transfer window')
    run(a)


if __name__=='__main__': main()
