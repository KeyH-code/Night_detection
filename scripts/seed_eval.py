"""Evaluate the three sealed original-A best checkpoints on the locked source-dev views."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
from statistics import fmean
import subprocess
import tempfile

import numpy as np

import detection_metrics as dm
import person_fusion as pf
import tricycle_fusion as tf
from evaluate import stage_subset, verify_subset
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json
from source_policy import SelectionPolicy, load_rules, recall_intervals, reference_at, summarize

ROOT=pf.ROOT
SEEDS=(17,29,43)
BASELINE=ROOT/'runs/tricycle_001'
REPLICA=ROOT/'runs/replica_002/replica'
BASELINE_SHA='421d509660de5cc455433c0f156ef4cf704e1fce37b28fdc162955b46412623e'
REPLICA_SHA='dcdd0dc56976720441b2103b99d764c76ecd85e0b01f29930bd16896181dd9a9'


def summarize_seeds(arms):
    if set(arms)!=set(SEEDS):
        raise ContractError('Exactly the original-A seeds 17, 29 and 43 are required')
    failed={s:sorted(k for k,v in (arms[s]['checks']|arms[s]['guards']).items() if not v) for s in SEEDS}
    classes={}
    for n in NAMES:
        cls=classes[n]={}
        for k in ('ap50','ap75','ap50_95','precision','recall'):
            values=[]
            for a in arms.values():
                w=a['workpoints'][n]
                values.append(a['metrics']['per_class'][n][k] if k.startswith('ap') else
                              w[k] if w['t_rule'] is not None else None)
            cls[k]=dict.fromkeys(('mean','spread','min','max')) if any(v is None for v in values) else {
                'mean':fmean(values),'spread':max(values)-min(values),'min':min(values),'max':max(values)}
    return {'classes':classes,'failed_by_seed':failed,'all_seeds_pass_numerically':not any(failed.values())}


def parse_gpu_probe(text):
    rows=text.strip().splitlines()
    try:
        total,free,util=map(int,rows[0].split(','))
    except (IndexError,ValueError) as e:
        raise ContractError('GPU probe did not yield memory and usage numbers') from e
    if len(rows)!=1 or not 0<=free<=total or not 0<=util<=100:
        raise ContractError('Expected exactly one valid local GPU')
    if free<4096 or util>35:
        raise ContractError(f'GPU busy or insufficient free memory: {free} MiB, utilization {util}%')
    return {'total_mib':total,'free_mib':free,'utilization_pct':util}


def probe_gpu():
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'],encoding='utf-8',timeout=10)
    except (subprocess.SubprocessError,OSError) as e:
        raise ContractError('Cannot verify the local GPU; refusing to start') from e
    return parse_gpu_probe(output)


def inputs():
    check_equal('locked tricycle and pedestrian evaluation seal',BASELINE_SHA,sha256(BASELINE/'manifest.json'))
    check_equal('complete original-A cloud run seal',REPLICA_SHA,sha256(REPLICA/'manifest.json'))
    base=read_json(BASELINE/'manifest.json'); copy=read_json(REPLICA/'manifest.json')
    rules=load_rules(ROOT/'RULES.md')
    check_equal('frozen evaluation identity',(pf.READY_SHA,rules['sha256']),(base['ready'],base['rules']))
    check_equal('original-A replica identity',('source_a_original','complete',False,pf.READY_SHA),
                (copy['plan'],copy['status'],copy['final_test_executed'],copy['ready']))
    for name,digest in base['code_sha256'].items(): check_equal('locked evaluation code '+name,digest,sha256(ROOT/name))
    anchor=read_json(tf.LOCKED/'manifest.json')
    check_equal('locked pedestrian view identity',tf.LOCKED_SHA,sha256(tf.LOCKED/'manifest.json'))
    _,_,native,person=pf.PROFILES['a1']; tris,mode=tf.PROFILES['low416-or']
    check_equal('authorized fusion views',(native,person,tris,mode),
                (tuple(anchor['anchor_view']),tuple(tuple(x) for x in anchor['person_views']),tuple(base['sizes']),base['score_mode']))
    evidence=BASELINE/'source_dev/objects.jsonl.gz'
    check_equal('cached seed 17 objects',base['files']['source_dev/objects.jsonl.gz']['sha256'],sha256(evidence))
    arms={17:{**base['checkpoint'],'epoch':80,'source':'tricycle_001_cached'}}
    for seed in SEEDS[1:]:
        item=copy['runs'][str(seed)]; name=f'{seed}/weights/best.pt'
        check_equal('sealed training role',('complete',seed,item['best_sha256']),
                    (item['status'],item['seed'],copy['files'][name]['sha256']))
        arms[seed]={'path':(REPLICA/name).relative_to(ROOT).as_posix(),'epoch':item['best_epoch'],
                    'sha256':item['best_sha256'],'source':'replica_002_best'}
    for s,arm in arms.items(): check_equal(f'seed {s} best SHA',arm['sha256'],sha256(ROOT/arm['path']))
    contract=verify_subset(pf.SOURCE,pf.READY_SHA,'source_dev')
    check_equal('evaluation only and exact 500-image split',(500,base['dev_manifest']),(len(contract['records']),contract['manifest']))
    return rules,contract,arms,evidence,native,person,tris,mode


def evaluate(out,scratch):
    out,scratch=out.resolve(),scratch.resolve()
    if out.parent!=ROOT/'runs' or not re.fullmatch(r'seed_\d{3}',out.name) or out.exists():
        raise ContractError('Choose a new runs/seed_NNN output; no overwrites')
    if scratch.is_relative_to(ROOT) or not scratch.is_dir():
        raise ContractError('Choose an existing task scratch directory outside the repository')
    rules,ct,arms,original,native,person,tris,mode=inputs()
    initial_gpu=probe_gpu()
    ct['rules']=rules
    with tempfile.TemporaryDirectory(prefix='seed-eval-',dir=scratch) as p:
        tmp=Path(p)
        for var in ('TEMP','TMP','TMPDIR','YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','XDG_CONFIG_HOME'):
            path=tmp/var.lower(); path.mkdir(); os.environ[var]=str(path)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
        import torch
        import torchvision
        import ultralytics
        check_equal('locked inference packages',('2.14.0+cu130','0.29.0+cu130','8.4.171'),
                    (torch.__version__,torchvision.__version__,ultralytics.__version__))
        torch.set_num_threads(4)
        data=stage_subset(ct,tmp/'data'); stage=tmp/'results'; stage.mkdir()
        policy=SelectionPolicy(rules,reference_at(ROOT)); values={}
        expected_keys={r['key'] for r in ct['records']}
        specs=tuple(dict.fromkeys((native,*person,*[(s,False) for s in tris])))
        ref_rows=dm.read_rows(original)
        ref_index=pf._dev_index(ref_rows)
        check_equal('seed 17 exact keys',expected_keys,set(ref_index))
        fixed,_,_=summarize(ref_rows,rules)
        for seed in SEEDS:
            location=stage/str(seed); location.mkdir()
            caps={}; obs=[]
            if seed==17:
                rows=ref_rows
            else:
                probe_gpu()  # Recheck at each arm; do not silently use CPU or start against another workload.
                capture=tmp/f'views-{seed}'; capture.mkdir(); views={}
                weight=ROOT/arms[seed]['path']
                for size,flip in specs:
                    z,item=pf._capture(torch,ct,data,weight,capture,size,flip)
                    views[size,flip]=z; obs.append(item)
                    print('SEED_VIEW',seed,size,flip,len(z),flush=True)
                a,pc=pf.evaluate_rows(views[native],tuple(views[x] for x in person))
                rows,tc=tf.merge_dev(a,tuple(views[s,False] for s in tris),mode)
                caps={'person':pc,'tricycle':tc}
                for before,after in zip(views[native],rows,strict=True):
                    keep=lambda r:[x for x in pf._detections(r['pred']) if x[0] not in (0,5)]
                    check_equal('protected native outputs '+before['key'],keep(before),keep(after))
                check_equal('post-eval checkpoint '+str(seed),arms[seed]['sha256'],sha256(weight))
            check_equal('exact new dev keys',expected_keys,set(pf._dev_index(rows)))
            for row in rows: check_equal('frozen image identity '+row['key'],pf._identity(ref_index[row['key']]),pf._identity(row))
            record,matches,curves=summarize(rows,rules); checks,guards=policy.assess(record)
            values[seed]={**record,'checks':checks,'guards':guards,'intervals':recall_intervals(rows,matches),
                          'fixed_seed_17_thresholds':{n:dm.aggregate_point(rows,i,fixed['workpoints'][n]['threshold'])
                            for i,n in enumerate(NAMES)},'observations':obs,'caps':caps}
            write_json(location/'metrics.json',values[seed]); dm.write_rows(location/'matches.jsonl.gz',matches)
            np.savez_compressed(location/'pr.npz',**curves)
            if seed!=17: dm.write_rows(location/'objects.jsonl.gz',rows)
            for name in NAMES:
                fail=[k for k,v in (checks|guards).items() if k.startswith(name+'/') and not v]
                print('SEED_RESULT',seed,name,record['workpoints'][name],fail,flush=True)
        comparison=summarize_seeds(values)
        write_json(stage/'comparison.json',comparison)
        write_json(stage/'manifest.json',{'status':'complete','scope':'source_dev','images':500,'arms':arms,
           'baseline_manifest_sha256':BASELINE_SHA,'replica_manifest_sha256':REPLICA_SHA,'ready':pf.READY_SHA,
           'dev_manifest':ct['manifest'],'rules':rules['sha256'],'protocol':{'person':person,'tricycle':tris,'score':mode,'native':native},
           'initial_gpu_probe':initial_gpu,'training_performed':False,'final_test_executed':False,
           'cloud_cny':0,'code_sha256':sha256(Path(__file__)),'reference_objects':str(original.relative_to(ROOT)),
           'files':{z.relative_to(stage).as_posix():{'sha256':sha256(z),'bytes':z.stat().st_size} for z in stage.rglob('*') if z.is_file()}})
        shutil.copytree(stage,out)
    return comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='verify weights, data and GPU availability; no inference')
    p.add_argument('--scratch',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'runs/seed_002')
    a=p.parse_args()
    try:
        if a.check:
            _,ct,arms,*_=inputs()
            print('VERIFIED_INPUTS',len(ct['records']),{s:(x['epoch'],x['sha256'][:12]) for s,x in arms.items()},flush=True)
            print('GPU_READY',probe_gpu(),flush=True)
        else:
            if a.scratch is None: p.error('--scratch is required for inference')
            report=evaluate(a.output,a.scratch)
            print('SEED_COMPLETE',a.output,report['failed_by_seed'],flush=True)
    except (ContractError,OSError,ValueError) as exc:
        p.exit(1,f'Seed evaluation: {exc}\n')


if __name__=='__main__': main()
