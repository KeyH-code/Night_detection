"""Recheck all three sealed align_001 results; optionally replay one checkpoint from source_dev pixels."""
import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

import person_local as pl
from person_local import ROOT, dm, pf, se
from evaluate import original_gt, stage_subset
from source_contract import ContractError, NAMES, canonical_hash, check_equal, read_json, sha256
from source_policy import SelectionPolicy, recall_intervals, reference_at, summarize
from train_scene import schedule

RUN = ROOT/'runs/align_001'


def verify():
    m = read_json(RUN/'manifest.json')
    rules, dev, arms, paths = pl.prepare()
    check_equal('run inputs', ('complete', pf.READY_SHA, rules['sha256'], pl.SEAL, dev['manifest'], False),
                (m['status'], m['ready'], m['rules'], m['baseline'], m['source_dev'], m['final_test_executed']))
    for n,v in m['files'].items():
        f=RUN/n; check_equal(n,(v['sha256'],v['bytes']),(sha256(f),f.stat().st_size))
    for n,v in m['code_sha256'].items(): check_equal(n,v,sha256(ROOT/'scripts'/n))
    tr=pf.SOURCE/m['source_train']['path'];check_equal('train list SHA',m['source_train']['sha256'],sha256(tr))
    train=read_json(tr)['records'];idx={r['key']:r for r in dev['records']}
    check_equal('separate whitelists',(4000,500,set()),(len(train),len(idx),{r['group_id'] for r in train}&{r['group_id'] for r in idx.values()}))
    gt={k:original_gt(r,dev['root']) for k,r in idx.items()};policy=SelectionPolicy(rules,reference_at(ROOT));passes=0;paired=[];warnings={}
    print('FILE_SEALS',len(m['files']),'CODE_SEALS',len(m['code_sha256']),'MANIFEST',sha256(RUN/'manifest.json'))
    for s in se.SEEDS:
        f=RUN/str(s); d=f/'dev'; hist=read_json(f/'train.json'); saved=read_json(d/'metrics.json'); ck=sha256(f/'last.pt')
        check_equal('training identity',(s,arms[s],m['recipe'],m['source_train'],ck,ck,'complete',750),
                    (hist['identity']['seed'],hist['identity']['source_best'],hist['identity']['recipe'],hist['identity']['train_manifest'],hist['checkpoint_sha256'],saved['checkpoint_sha256'],hist['status'],hist['updates']))
        check_equal('epochs',6,len(hist['history']))
        for j,h in enumerate(hist['history']):
            seq=schedule(s,j);ids=[entry[0] for batch in seq for entry in batch]
            check_equal('train sequence',(list(range(4000)),canonical_hash(seq),canonical_hash(ids),4000,125),
                        (sorted(ids),h['augmentation_schedule_sha256'],h['consumption_sha256'],h['main_images'],h['updates']))
        rows=dm.read_rows(d/'objects.jsonl.gz');check_equal('dev keys',set(idx),set(pf._dev_index(rows)))
        for r in rows:
            k=r['key'];ref=idx[k]
            for field in ('group_id','purpose','width','height','image_sha256','label_sha256'):check_equal(k+' '+field,ref[field],r[field])
            check_equal(k+' GT',gt[k],list(zip(r['gt']['cls'],r['gt']['original_boxes'],strict=True)))
            flags,pairs=dm.ap_flags(r);check_equal(k+' per-IoU assignment',(flags.tolist(),pairs),(r['ap_tp'],r['ap_pairs']))
        rec,match,curve=summarize(rows,rules);checks,guards=policy.assess(rec)
        for k in ('metrics','workpoints','slices'):check_equal('recomputed '+k,saved[k],rec[k])
        check_equal('recomputed evidence',(checks,guards,match,recall_intervals(rows,match)['intervals']),
                    (saved['checks'],saved['guards'],dm.read_rows(d/'matches.jsonl.gz'),saved['intervals']))
        with np.load(d/'pr.npz') as z:
            check_equal('PR fields',set(curve),set(z.files))
            if not all(np.array_equal(z[k],v,equal_nan=True) for k,v in curve.items()):raise ContractError('PR curve mismatch')
        old=read_json(pl.BASE/str(s)/'metrics.json');trans,g=pl.transitions(rows,dm.read_rows(pl.BASE/str(s)/'matches.jsonl.gz'),match);paired.append(g)
        check_equal('matched GT transitions and CI',(trans,pl.paired_interval([g])),(saved['transitions'],saved['paired_recall']))
        old_slices={v['slice']:v['recall50'] for v in old['slices'] if v['class']=='pedestrian'}
        warnings[s]=[v['slice'] for v in rec['slices'] if v['class']=='pedestrian' and v['slice'] in ('area_S','width_le8','small_thin_union') and old_slices[v['slice']]-v['recall50']>rules['slice_drop']]
        if old['metrics']['per_class']['pedestrian']['ap75']-rec['metrics']['per_class']['pedestrian']['ap75']>rules['ap75_drop']:warnings[s].append('ap75')
        check_equal('relative warnings',warnings[s],saved['relative_warnings'])
        for n in NAMES:
            a=rec['metrics']['per_class'][n];w=rec['workpoints'][n];gates={k:v for k,v in (checks|guards).items() if k.startswith(n+'/')};ok=all(gates.values())
            passes+=int(n in NAMES[:5] and ok)
            print(s,n,'PASS' if ok else 'FAIL','AP50/AP50:95/P/R',*(f'{v*100:.4f}' for v in (a['ap50'],a['ap50_95'],w['precision'],w['recall'])),f"TP/FP/FN={w['tp']}/{w['fp']}/{w['fn']}")
        print('RECOMPUTED',s,len(rows),'images',len(curve),'PR fields; slice flags:',warnings[s])
    c=read_json(RUN/'comparison.json');check_equal('paired group result',c['paired_recall_on_evaluated_seeds'],pl.paired_interval(paired))
    recurring={n for v in warnings.values() for n in v if sum(n in a for a in warnings.values())>=2}
    check_equal('recommendation',passes==15 and not recurring,c['recommended']);print('FIVE_CLASSES',f'{passes}/15','recurring_slice_flags',sorted(recurring))
    return rules,dev,paths


def replay(s,inputs,scratch):
    from person_prior import evaluate
    if not scratch.is_dir() or scratch.is_relative_to(ROOT):raise ContractError('Use an existing external scratch directory')
    se.probe_gpu();rules,ct,paths=inputs;ct['rules']=rules;sealed=RUN/str(s)/'dev'
    with tempfile.TemporaryDirectory(dir=scratch) as name:
        tmp=Path(name);folder=tmp/'results';folder.mkdir()
        for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','YOLO_CONFIG_DIR','TORCH_HOME','XDG_CONFIG_HOME'):
            p=tmp/k.lower();p.mkdir();os.environ[k]=str(p)
        os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled');import torch,torchvision,ultralytics
        check_equal('packages',read_json(RUN/'manifest.json')['versions'],[torch.__version__,torchvision.__version__,ultralytics.__version__]);torch.set_num_threads(4)
        evaluate(s,RUN/str(s)/'last.pt',folder,ct,stage_subset(ct,tmp/'data'),tmp/'views',SelectionPolicy(rules,reference_at(ROOT)),rules,paths)
        left=dm.read_rows(sealed/'objects.jsonl.gz');right=dm.read_rows(folder/'objects.jsonl.gz');check_equal('replay image count',500,len(right))
        for a,b in zip(left,right,strict=True):check_equal('replayed image '+a['key'],a,b)
        m,r=read_json(sealed/'metrics.json'),read_json(folder/'metrics.json')
        for k in ('metrics','workpoints','checks','guards','single640','transitions'):check_equal('replay '+k,m[k],r[k])
        print('ORIGINAL_PIXEL_REPLAY',s,'500/500 identical fused objects and assignments; single640 and locked-view metrics identical')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replay',type=int,choices=se.SEEDS);parser.add_argument('--scratch',type=Path);args=parser.parse_args()
    if args.replay and not args.scratch:parser.error('--replay needs --scratch')
    values=verify()
    if args.replay:replay(args.replay,values,args.scratch.resolve())
