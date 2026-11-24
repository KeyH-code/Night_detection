"""S7: nested 300-image random control, then matched E1 and E2 development contrasts."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import detection_metrics as dm
import night_accept as accept
import night_adapt as adapt
import night_select as selector
import night_strategy as strategy
import night_train as train
from night_contract import POOL, POOL_MANIFEST, ROOT, _no_reparse, _records, read_sealed_bytes, read_sealed_json, verify_dev
from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256, write_json

KINDS = ('d','e1','e2')
MIX_QUOTAS = (133,100,67)


def random_prefix(keys, base, seed):
    if len(keys)!=994 or len(set(keys))!=994:
        raise ContractError('Expected complete frozen image-only pool')
    ordered=sorted(keys)
    ordered=[ordered[i] for i in np.random.Generator(np.random.PCG64(seed)).permutation(len(ordered))[:300]]
    check_equal('D100 ordered prefix',base,ordered[:100])
    return ordered


def align_slots(d,e):
    if len(d)!=300 or len(e)!=300 or len(set(d))!=300 or len(set(e))!=300:
        raise ContractError('Expected 300 unique images in each arm')
    check_equal('fixed paired anchors',d[:20],e[:20])
    shared=set(e); original=set(d)
    fill=iter(k for k in e if k not in original)
    return [k if k in shared else next(fill) for k in d]


def mix_slots(random, balanced, selected):
    if any(len(x)!=300 or len(set(x))!=300 for x in (random,balanced,selected)):
        raise ContractError('Expected unique 300-image slot inventories')
    chosen=set(selected); slots=[None]*300
    for i,key in enumerate(balanced):
        if key in chosen: slots[i]=key
    placed=chosen & set(balanced)
    for i,key in enumerate(random):
        if slots[i] is None and key in chosen and key not in placed:
            slots[i]=key; placed.add(key)
    remainder=iter(k for k in selected if k not in placed)
    slots=[next(remainder) if k is None else k for k in slots]
    check_equal('exact mechanism image set',chosen,set(slots))
    return slots,{'balanced_shared':len(chosen & set(balanced)), 'random_shared':len(chosen & set(random)),
                  'random_slot_changes':[k for i,k in enumerate(random) if k in chosen and slots[i]!=k]}


def coverage_stats(keys,z,groups,balanced,selected,additional):
    if len(groups)!=len(keys) or not balanced or not selected:
        raise ContractError('Expected both fixed representative coverage populations')
    z=strategy._validate_vectors(keys,z)
    ix={k:i for i,k in enumerate(keys)}
    _,g=np.unique(groups,return_inverse=True)
    sizes=np.bincount(g)

    def cover(items):
        sims=(1+np.clip(z@z[[ix[k] for k in items]].T,-1,1))/2
        return np.bincount(g,weights=sims.max(axis=1))/sizes

    base=cover(balanced); cand=cover(selected)
    low=np.argsort(base,kind='stable')[:max(1,len(base)//4)]
    def stats(x):
        return {'group_mean':float(x.mean()),'group_p10':float(np.quantile(x,.10)),
                'fixed_low_quartile_mean':float(x[low].mean()),'fixed_low_quartile_groups':len(low)}
    b,c=stats(base),stats(cand)
    c['fixed_low_quartile_change']=c['fixed_low_quartile_mean']-b['fixed_low_quartile_mean']
    a=z[[ix[k] for k in additional]]
    c['additional_pairwise_cosine_mean']=float((a@a.T)[np.triu_indices(len(a),1)].mean()) if len(a)>1 else None
    return b,c


def check_mix_prefix(arms,total):
    previous=[]
    for n,arm in sorted(arms.items()):
        steps=arm['steps']
        check_equal('ordered stage budgets',['representative']*n+['diverse']*(total-n),[x['kind'] for x in steps])
        now=[x['key'] for x in steps[:n]]
        check_equal('shorter representative prefix',previous,now[:len(previous)])
        previous=now


def mix_configuration(digest,expand_digest):
    raw=read_sealed_bytes(ROOT,'configs/e2_300.json',digest,[])
    spec=json.loads(raw)
    check_equal('approved three-arm E2 internal mixture',(1,expand_digest,list(MIX_QUOTAS),200),
                (spec['schema_version'],spec['expand_sha256'],spec['representatives'],spec['additional']))
    return raw,spec


def margin_configuration(digest, expand_digest):
    raw = read_sealed_bytes(ROOT, 'configs/margin.json', digest, [])
    spec = json.loads(raw)
    keys = ('schema_version', 'expand_sha256', 'seed', 'classes', 'weight', 'margin')
    check_equal('registered margin intervention', (1, expand_digest, 43, [2, 3, 4], .5, .75),
                tuple(spec[k] for k in keys))
    return raw, spec


def expanded_plan(old_plan, slots, seed, salt):
    if len(old_plan)!=2000 or len(slots)!=300 or len(set(slots))!=300:
        raise ContractError('Expected 2000 paired microbatches and 300 input slots')
    rng=np.random.default_rng(np.random.SeedSequence([seed,salt,1]))
    order=[k for _ in range(7) for k in rng.permutation(len(slots)).tolist()][:2000]
    return [(d,(slots[i],*n[1:])) for (d,n),i in zip(old_plan,order,strict=True)]


def configuration(digest):
    raw=read_sealed_bytes(ROOT,'configs/expand.json',digest,[])
    cfg=json.loads(raw)
    check_equal('registered single-seed expansion',(1,43,300,200,35,{'e1','e2'}),
                (cfg['schema_version'],cfg['seed'],cfg['night_images'],cfg['updates'],cfg['max_minutes'],set(cfg['parent_runs'])))
    _,pair,old=adapt._config(cfg['pair_sha256'])
    return raw,cfg,pair,old


def context(cfg,pair,old,arm):
    seed=cfg['seed']
    old_tasks,protocol,identity=adapt.alignment(pair,old,arms=(() if arm=='d' else (arm,)),seeds=(seed,))
    digest=protocol['data']['pool_images_sha256']
    raw=read_sealed_json(train.NIGHT,POOL_MANIFEST,digest,[])
    inv={r['key']:r for r in _records(raw,POOL,994,images_only=True)}
    base=ROOT/'runs/pool_001'
    seal=read_sealed_json(base,'manifest.json',pair['pool_manifest_sha256']['d'],[])
    draw=adapt._json(base,seal,'random.json')
    d100=[r['key'] for r in adapt.draw_records('d',seed,draw,digest,inv)]
    d300=random_prefix(list(inv),d100,seed)
    days=adapt._day_keys(identity['source_train'])
    baseplan=adapt._schedule(seed,days,d100,old)
    dseal=read_sealed_json(ROOT/'runs/d_001','manifest.json',pair['d_manifest_sha256'],[])
    historical=adapt._json(ROOT/'runs/d_001',dseal,f'{seed}/train.json')
    check_equal('actual D100 training schedule',historical['augment_schedule_sha256'],canonical_hash(baseplan))
    return protocol,identity,digest,inv,d100,d300,baseplan,(None if arm=='d' else old_tasks[arm][seed]['slots']['selection_order'])


def _selection_worker(box,digest,seed,base):
    approved=box/POOL_MANIFEST
    reads=selector.guard_reads(approved)
    data=read_sealed_json(box,POOL_MANIFEST,digest,[])
    records=_records(data,POOL,994,images_only=True)
    keys=random_prefix([r['key'] for r in records],base,seed)
    check_equal('only image manifest opened',[approved],reads)
    print(json.dumps({'keys':keys,'reads':[POOL_MANIFEST]}),flush=True)


def select(arm,seed,inv,digest,d100,d300,parent,pair,scratch,*,representatives=100):
    if arm=='d':
        approved=read_sealed_bytes(train.NIGHT,POOL_MANIFEST,digest,[])
        with tempfile.TemporaryDirectory(dir=scratch,prefix='expand-blind-') as td:
            box=Path(td); path=box/POOL_MANIFEST
            path.parent.mkdir(); path.write_bytes(approved)
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'_draw',str(box),digest,str(seed),json.dumps(d100)]
            a,b=[json.loads(subprocess.check_output(cmd,cwd=box,timeout=60)) for _ in range(2)]
        check_equal('two isolated random replays identical',a,b)
        check_equal('random result matches frozen first 100 and extends to 300',d300,a['keys'])
        return {'keys':d300,'reads':a['reads'],'parent_keys':d100,'steps':None}
    folder=ROOT/'runs/pool_002'
    seal=read_sealed_json(folder,'manifest.json',pair['pool_manifest_sha256']['e1'],[])
    params=({'shortlist':128,'quota':40} if arm=='e1' else {'representatives':representatives,'diverse':200-representatives,'neighbors':5})
    files=('features.npz','scores.npz') if arm=='e1' else ('features.npz',)
    with tempfile.TemporaryDirectory(dir=scratch,prefix='expand-rank-') as td:
        box=Path(td)
        for rel in files:
            (box/rel).write_bytes(accept._checked_file(folder,seal,rel,[]))
        keys,z=strategy.load_features(box/'features.npz')
        check_equal('exact feature and pool key identity',sorted(inv),keys)
        metadata={'keys':keys,'anchors':{str(seed):parent},'params':params}
        if arm=='e2': metadata['groups']=[inv[k]['group_id'] for k in keys]
        def rank():
            return strategy.run_rank_worker(arm,box/'features.npz',metadata,scores=(box/'scores.npz' if arm=='e1' else None))
        a=rank(); check_equal('two isolated ranking replays',a,rank())
    check_equal('ranking file scope',list(files),a['reads'])
    pick=a['arms'][str(seed)]
    check_equal('original strategy list retained exactly',(300,300,parent),
                (len(pick['keys']),len(set(pick['keys'])),pick['keys'][:100]))
    if arm=='e2': check_equal('E2 risk-group exclusion',300,len({inv[k]['group_id'] for k in pick['keys']}))
    return {**pick,'parent_keys':parent,'reads':a['reads'],'parameters':params}


def prepare(box,digest,grant_sha):
    _,cfg,pair,old=configuration(digest)
    _no_reparse(box)
    check_equal('fresh label preparation sandbox',['grant.json'],sorted(p.name for p in box.iterdir()))
    grant=read_sealed_json(box,'grant.json',grant_sha,[])
    check_equal('grant purpose and seed',(cfg['seed'],300),(grant['seed'],len(grant['rows'])))
    if grant['arm'] not in KINDS: raise ContractError('Unregistered 300-image arm')
    proto=train.load_protocol()
    rows=_records({'purpose':POOL,'records':grant['rows']},POOL,300,images_only=True)
    im=read_sealed_json(train.NIGHT,POOL_MANIFEST,proto['data']['pool_images_sha256'],[])
    inventory={r['key']:r for r in _records(im,POOL,994,images_only=True)}
    check_equal('only official frozen image records can receive labels',rows,[inventory.get(r['key']) for r in rows])
    ready=read_sealed_json(train.NIGHT,'READY.json',proto['data']['night_ready_sha256'],[])
    entry=ready['held_out']['night_pool']
    check_equal('selected metadata receipt',('manifests/night_pool.json',994),(entry['path'],entry['count']))
    managed=read_sealed_json(train.NIGHT,entry['path'],entry['sha256'],[])
    check_equal('selected metadata scope',('night_pool',994),(managed['purpose'],len(managed['records'])))
    selected=train.selected_records(rows,managed['records'],count=300)
    staged,reads=train._stage(train.NIGHT,box,selected)
    check_equal('only the 300 selected image/label files', [r[k] for r in selected for k in ('image','label')],reads)
    write_json(box/'inputs.json',{'records':staged,'source_reads':reads,'grant_sha256':grant_sha,'management_sha256':entry['sha256']})


def load_run(folder,digest=None):
    _no_reparse(folder)
    seal_path=folder/'manifest.json'
    seal=read_sealed_json(folder,'manifest.json',digest or sha256(seal_path),[])
    check_equal('complete evidence file set',('sealed',seal['files']),(seal['status'],train._receipt(folder)))
    state=adapt._json(folder,seal,'run.json')
    check_equal('complete local expansion run',('complete',False,False),(state['status'],state['cloud_used'],state['final_test_executed']))
    return seal,state


def mix_reference(spec,digest):
    p=ROOT/'runs/expand_003'; seal,state=load_run(p,spec['balanced_sha256'])
    check_equal('exact balanced expansion identity',('e2',43,digest,{'run':'expand_001','sha256':spec['random_sha256']},None),
                (state['arm'],state['seed'],state['config_sha256'],state['d300_parent'],state.get('e2_mix')))
    return p,seal,adapt._json(p,seal,'selection.json')


def mix_selections(spec,digest,seed,pair,pool,inventory,d100,d300,parent,baseplan,old,scratch):
    balanced_path,balanced_seal,balanced=mix_reference(spec,digest)
    check_equal('verified history is the approved E2 reference','e2',verify(balanced_path,spec['balanced_sha256'])['arm'])
    arms={n:select('e2',seed,inventory,pool,d100,d300,parent,pair,scratch,representatives=n) for n in MIX_QUOTAS}
    check_mix_prefix(arms,spec['additional'])
    check_equal('balanced ordered lists and stages match release',arms[100],{k:balanced[k] for k in arms[100]})
    check_equal('balanced slots and actual schedule re-created',
                (balanced['slots'],balanced['schedule_sha256']),
                (align_slots(d300,arms[100]['keys']),canonical_hash(expanded_plan(baseplan,balanced['slots'],seed,old['schedule_salt']))))
    e1=ROOT/'runs/expand_002'; seal,_=load_run(e1,spec['e1_receipt_sha256'])
    old_union=set(d300)|set(balanced['keys'])|set(adapt._json(e1,seal,'selection.json')['keys'])
    keys,z=strategy.load_features(ROOT/'runs/pool_002/features.npz')
    groups=[inventory[k]['group_id'] for k in keys]
    for n,arm in arms.items():
        x=set(arm['keys']); slots,alignment=mix_slots(d300,balanced['slots'],arm['keys'])
        coverage_ref,stats=coverage_stats(keys,z,groups,balanced['keys'],arm['keys'],arm['keys'][100:])
        other=arms[200-n]['keys']; other_slots=mix_slots(d300,balanced['slots'],other)[0]
        arm['mix_audit']={**alignment,'balanced_replayed':True,'coverage':stats,'coverage_reference':coverage_ref,
                          'overlap_other_pole':len(x & set(other)) if n!=100 else None,
                          'other_pole_common_slot_changes':sum(k in set(other_slots) and k!=other_slots[i] for i,k in enumerate(slots)) if n!=100 else None,
                          'potential_new_vs_s7_union':len(x-old_union),'same_set_as':[]}
    for n in (133,67):
        check=[('balanced',arms[100]['keys']),('random',d300)]
        if n==67: check.append(('representative',arms[133]['keys']))
        arms[n]['mix_audit']['same_set_as']=[name for name,ls in check if set(ls)==set(arms[n]['keys'])]
    return arms,balanced['slots']


def compare_balanced(result,spec,digest,sd,approved,night_pair,policy):
    folder,seal,_=mix_reference(spec,digest)
    rel=f'{sd}/dev/night'
    rows=accept.sealed_objects(folder,seal,f'{rel}/objects.jsonl.gz','night_dev',approved,[])
    metrics=adapt._json(folder,seal,f'{rel}/metrics.json')
    balanced_day=adapt._json(folder,seal,f'{sd}/dev/evaluation.json')['assessment']['day_protection']
    day=result['day_candidate_vs_a']
    result['mix_vs_balanced']=effects([night_pair,(rows,metrics['workpoints'])],[1,-1],policy)
    result['mix_day_vs_balanced']={'macro':day['macro_delta']-balanced_day['macro_delta'],
                                   'per_class':{k:day['per_class'][k]['delta']-v['delta'] for k,v in balanced_day['per_class'].items()}}
    return result


def effects(arms,coefficients,policy):
    first=arms[0][0]
    identity=accept._pair_identity(first)
    groups=sorted({r['group_id'] for r in first})
    draws=accept._draws(groups,policy)
    values=[]; samples=[]
    for rows,points in arms:
        check_equal('common night development groups and GT',identity,accept._pair_identity(rows))
        counts=accept._group_counts(rows,points,groups)
        values.append(accept._recall_samples(counts.sum(axis=0)))
        samples.append(accept._recall_samples(np.einsum('rg,gct->rct',draws,counts)))
    result=accept._effect(np.tensordot(coefficients,values,axes=1),np.tensordot(coefficients,samples,axes=1))
    return {'effects':result,'groups':len(groups),'repeats':len(draws),'fixed_thresholds':True,
            'signs':coefficients}


def compare(arm,phase,out,cfg,pair,policy,ref,baseline,night_approved,d_pair,candidate,d_path,d_seal):
    seed=cfg['seed']; sd=str(seed)
    night=candidate[2:]; state=read_json(phase/'dev/evaluation.json')['assessment']
    result={'arm':arm,'seed':seed,'day_candidate_vs_a':state['day_protection'],'assessment':state}
    if arm=='d':
        result['scale_delta']=effects([night,d_pair],[1,-1],policy)
        old=baseline['D'][sd]['assessment']['day_protection']
    else:
        base=ROOT/'runs'/f'{arm}_001'
        seal=read_sealed_json(base,'manifest.json',cfg['parent_runs'][arm],[])
        prior=accept.sealed_objects(base,seal,f'{sd}/dev/night/objects.jsonl.gz','night_dev',night_approved,[])
        pm=adapt._json(base,seal,f'{sd}/dev/night/metrics.json')['workpoints']
        old=adapt._json(base,seal,f'{sd}/dev/evaluation.json')['assessment']['day_protection']
        rand=accept.sealed_objects(d_path,d_seal,f'{sd}/dev/night/objects.jsonl.gz','night_dev',night_approved,[])
        rm=adapt._json(d_path,d_seal,f'{sd}/dev/night/metrics.json')['workpoints']
        result.update(scale_delta=effects([night,(prior,pm)],[1,-1],policy),
                      new_strategy_gap=effects([night,(rand,rm)],[1,-1],policy),
                      old_strategy_gap=effects([(prior,pm),d_pair],[1,-1],policy),
                      gap_change=effects([night,(rand,rm),(prior,pm),d_pair],[1,-1,-1,1],policy))
        dr=adapt._json(d_path,d_seal,f'{sd}/dev/evaluation.json')['assessment']['day_protection']
        result['relative_day_gap_change']=((state['day_protection']['macro_delta']-dr['macro_delta'])
                                           -(old['macro_delta']-baseline['D'][sd]['assessment']['day_protection']['macro_delta']))
    result['day_parent100_vs_a']=old
    result['day_scale_macro_delta']=state['day_protection']['macro_delta']-old['macro_delta']
    return result


def margin_reference(spec, digest, random_parent):
    folder = ROOT/'runs/expand_003'
    seal, state = load_run(folder, spec['balanced_manifest_sha256'])
    check_equal('exact untreated balanced E2', ('e2', 43, digest, random_parent, None, None, None),
                (state['arm'], state['seed'], state['config_sha256'], state['d300_parent'],
                 state.get('e2_mix'), state.get('margin'), state.get('crop')))
    return folder, seal, adapt._json(folder, seal, 'selection.json')


def compare_margin(result, spec, digest, random_parent, approved, candidate, policy, *, variant='margin'):
    folder, seal, _ = margin_reference(spec, digest, random_parent)
    rel = '43/dev/night'
    rows = accept.sealed_objects(folder, seal, f'{rel}/objects.jsonl.gz', 'night_dev', approved, [])
    base = adapt._json(folder, seal, f'{rel}/metrics.json')['workpoints']
    effects_ = effects([(rows, base), candidate], [-1, 1], policy)
    day = adapt._json(folder, seal, '43/dev/evaluation.json')['assessment']['day_protection']
    names = ('truck', 'tram')
    gains = {n: candidate[1][n]['tp'] - base[n]['tp'] for n in names}
    result[f'{variant}_vs_balanced'] = {'night': effects_, 'tail_gain_tp': gains,
                                    'day_macro_delta': result['day_candidate_vs_a']['macro_delta'] - day['macro_delta'],
                                    'continue_gate': result['day_candidate_vs_a']['met'] and min(gains.values()) >= 3 and
                                    any(effects_['effects'][n]['ci95'][0] > 0 for n in names)}


def run(arm,digest,scratch,out,d_path=None,d_digest=None,*,mix_sha=None,representatives=None,margin_sha=None,crop_sha=None):
    from night_run import CODE
    from seed_eval import probe_gpu

    start=time.monotonic(); raw,cfg,pair,old=configuration(digest)
    seed=cfg['seed']; sd=str(seed); until=start+cfg['max_minutes']*60
    protocol,identity,pool,inventory,d100,d300,baseplan,parent=context(cfg,pair,old,arm)
    d_seal=None
    if arm!='d':
        d_seal,ds=load_run(d_path,d_digest)
        check_equal('D300 must be completed first with same policy',('d',seed,digest),(ds['arm'],ds['seed'],ds['config_sha256']))
        dc=verify(d_path,d_digest)
        check_equal('verified random 300 arm','d',dc['arm'])
    mix=None; margin=None; crop=None
    if margin_sha or crop_sha:
        if arm != 'e2' or mix_sha or representatives or (margin_sha and crop_sha):
            raise ContractError('Expected one isolated E2 intervention')
        if crop_sha:
            import night_crop

            crop_raw, spec, support = night_crop.configuration(crop_sha, digest)
            crop = {'sha256': crop_sha, 'balanced': spec['balanced_manifest_sha256']}
        else:
            margin_raw, spec = margin_configuration(margin_sha, digest)
            margin = {'sha256': margin_sha, 'balanced': spec['balanced_manifest_sha256']}
        random_parent = {'run': d_path.name, 'sha256': d_digest}
        _, _, chosen = margin_reference(spec, digest, random_parent)
        slots = align_slots(d300, chosen['keys'])
        check_equal('unchanged balanced slots', chosen['slots'], slots)
        check_equal('balanced E2 already independently replayed', 'e2',
                    verify(ROOT/'runs/expand_003', spec['balanced_manifest_sha256'])['arm'])
    elif mix_sha:
        mix_raw,spec=mix_configuration(mix_sha,digest)
        check_equal('registered internal E2 run and locked random reference',('e2',True,ROOT/'runs/expand_001',spec['random_sha256']),
                    (arm,representatives in (133,67),d_path,d_digest))
        arms,balanced_slots=mix_selections(spec,digest,seed,pair,pool,inventory,d100,d300,parent,baseplan,old,scratch)
        chosen=arms[representatives]
        check_equal('duplicate sets are not retrained',[],chosen['mix_audit']['same_set_as'])
        slots,_=mix_slots(d300,balanced_slots,chosen['keys'])
        mix={'sha256':mix_sha,'representatives':representatives}
    else:
        chosen=select(arm,seed,inventory,pool,d100,d300,parent,pair,scratch)
        slots=align_slots(d300,chosen['keys'])
    plan=expanded_plan(baseplan,slots,seed,old['schedule_salt'])
    if margin or crop:
        check_equal('uses the exact untreated schedule', chosen['schedule_sha256'], canonical_hash(plan))
    daykeys=[d[0] for d,_ in plan]
    check_equal('unchanged A weight',identity['A_sha256'],sha256(train.CHECKPOINT))
    versions={name:version(name) for name in train.LIBS}
    check_equal('locked environment',identity['versions'],versions)
    gpu=probe_gpu()
    tr=train.verify_subset(train.SOURCE,protocol['data']['source_ready_sha256'],'source_train')
    dev=train.verify_subset(train.SOURCE,protocol['data']['source_ready_sha256'],'source_dev')
    night=verify_dev(train.NIGHT,protocol['data']['night_ready_sha256'])
    for purpose,ct in (('source_train',tr),('source_dev',dev),('night_dev',night)):
        check_equal('historical purpose identity',identity[purpose],ct['manifest'])
    source={r['key']:r for r in tr['records']}; selected=[inventory[k] for k in slots]
    check_equal('train/development risk groups disjoint',set(),
                ({source[k]['group_id'] for k in daykeys}|{r['group_id'] for r in selected}) &
                {r['group_id'] for r in dev['records']+night['records']})
    night_approved={r['key']:r for r in night['records']}
    out.mkdir(exist_ok=False); phase=out/sd; phase.mkdir()
    error=None; complete=False
    try:
        (out/'protocol.json').write_bytes(raw)
        if mix: (out/'mix.json').write_bytes(mix_raw)
        if margin: (out/'margin.json').write_bytes(margin_raw)
        if crop: (out/'crop.json').write_bytes(crop_raw)
        (out/'code').mkdir()
        code = set(CODE)|{'night_expand.py','night_select.py','night_strategy.py','night_adapt.py','night_train.py','night_accept.py',
                          'source_policy.py','tricycle_fusion.py'}
        if margin or crop:
            code.update({'night_focus.py','night_rank.py', 'night_margin.py' if margin else 'night_crop.py'})
        for name in sorted(code):
            shutil.copyfile(ROOT/'scripts'/name,out/'code'/name)
        write_json(out/'selection.json',{**chosen,'arm':arm,'records':selected,'slots':slots,
                                        'records_sha256':canonical_hash(selected),'schedule_sha256':canonical_hash(plan),
                                        'pool_sha256':pool,'group_count':len({r['group_id'] for r in selected}),
                                        'shared_d300':len(set(d300)&set(slots)),'labels_read_in_selection':False})
        print(f'{arm.upper()}300{" crop" if crop else " margin" if margin else ""}: locked 300 unique night images, parent prefix 100, '
              f'{len(set(d300)&set(slots))} shared D slots.',flush=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='expand-') as td:
            box=Path(td)
            for k in ('YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','TEMP','TMP','TMPDIR','XDG_CONFIG_HOME'):
                p=box/k.lower(); p.mkdir(); os.environ[k]=str(p)
            os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            import cv2
            import torch
            torch.set_num_threads(4); cv2.setNumThreads(1)
            n=box/'target'; n.mkdir(); grant=n/'grant.json'
            write_json(grant,{'seed':seed,'arm':arm,'rows':selected}); grant_sha=sha256(grant)
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'_prepare',str(n),digest,grant_sha],
                           check=True,timeout=180)
            rec=read_json(n/'inputs.json')
            check_equal('exact grant and gradient order',(grant_sha,slots),(rec['grant_sha256'],[r['key'] for r in rec['records']]))
            staged,_=train._stage(train.SOURCE,box/'source',[source[k] for k in daykeys])
            d_payload=train.training_rows(box/'source',staged,'source_train')
            n_payload=train.training_rows(n,rec['records'],'night_pool')
            check_equal('exact unique staged two-domain inputs',(2000,300),(len(d_payload),len(n_payload)))
            write_json(phase/'inputs.json',{**rec,'source_keys':daykeys})
            objective=None
            build=None
            if margin:
                from night_margin import VehicleMargin
                objective=VehicleMargin(spec['weight'],spec['margin'])
            if crop:
                build=night_crop.CropBatches(plan,set(d_payload),{row['key']:row['group_id'] for row in selected})
            with train.telemetry() as monitor:
                if objective is None and build is None:
                    weight,record=train._fit(torch,cv2,d_payload|n_payload,plan,seed,old,phase,200,until)
                else:
                    from night_focus import fit_native
                    weight,record=fit_native(torch,cv2,{'day':d_payload,'night':n_payload},plan,seed,old,
                                              {'updates':200},until,phase,classification_bce=objective,batch_builder=build)
            if objective is not None:
                signal=objective.summary()
                check_margin_signal(signal)
                record['margin_signal']=signal
            if build is not None:
                record['crop_signal']=build.summary()
                night_crop.check_signal(record['crop_signal'],support)
            check_equal('exact actual expanded schedule and exposure',
                        (canonical_hash(plan),{k:1 for k in daykeys},dict(Counter(n[0] for _,n in plan))),
                        (record['augment_schedule_sha256'],record['counts']['source'],record['counts']['night']))
            instances=Counter(c for row in n_payload.values() for c,*_ in row[3])
            labels=Counter(c for row in n_payload.values() for c in {r[0] for r in row[3]})
            write_json(phase/'train.json',{**record,'night_instances_by_class':instances,'night_images_by_class':labels,'gpu':monitor})
            del n_payload,d_payload
            torch.cuda.empty_cache(); torch.use_deterministic_algorithms(False)
            policy,ref,baseline,d_pairs,a_day=adapt._reference(pair,protocol,dev,night,[seed])
            candidate,assessment=adapt._evaluate(torch,weight,phase/'dev',box/'eval',night,dev,policy,ref,d_pairs[seed],a_day,until)
            result=compare(arm,phase,out,cfg,pair,policy,ref,baseline,night_approved,d_pairs[seed],candidate,d_path,d_seal)
            if mix: compare_balanced(result,spec,digest,sd,night_approved,candidate[2:],policy)
            if margin: compare_margin(result,spec,digest,random_parent,night_approved,candidate[2:],policy)
            if crop: compare_margin(result,spec,digest,random_parent,night_approved,candidate[2:],policy,variant='crop')
            write_json(out/'comparison.json',result)
            complete=True
            print(f'{arm.upper()}300: night {assessment["classes_met"]}/5; day={assessment["day_protection"]["met"]}; '
                  f'day macro delta={assessment["day_protection"]["macro_delta"]*100:.6f} pp.',flush=True)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(out/'run.json',{'arm':arm,'seed':seed,'status':'complete' if complete else 'incomplete','error':error,
                                   'seconds':time.monotonic()-start,'config_sha256':digest,'a_sha256':identity['A_sha256'],
                                    **({'e2_mix':mix} if mix else {}),
                                    **({'margin':margin} if margin else {}),
                                    **({'crop':crop} if crop else {}),
                                   'd300_parent':None if d_path is None else {'run':d_path.name,'sha256':d_digest},
                                   'versions':versions,'initial_gpu':gpu,'python':sys.version,'cloud_used':False,
                                   'final_test_executed':False,'source_train':tr['manifest'],'source_dev':dev['manifest'],'night_dev':night['manifest']})
        write_json(out/'manifest.json',{'status':'sealed','files':train._receipt(out)})


def check_margin_signal(signal):
    check_equal('one coupled objective per night microbatch', 2000, signal['calls'])
    positive, loss = signal['positive_anchors'], signal['extra_unnormalized']
    if len(positive) != 3 or len(loss) != 3 or min(positive) <= 0 or not all(np.isfinite(x) and x > 0 for x in loss):
        raise ContractError('Expected positive finite coupled-loss signal in every registered vehicle class')


def verify(folder,digest=None):
    seal,s=load_run(folder,digest)
    raw=adapt._json(folder,seal,'protocol.json'); _,cfg,pair,old=configuration(s['config_sha256'])
    check_equal('unchanged registered protocol',cfg,raw)
    arm=s['arm']; seed=s['seed']; sd=str(seed)
    proto,identity,pool,inv,d100,d300,baseplan,prior=context(cfg,pair,old,arm)
    choice=adapt._json(folder,seal,'selection.json')
    mix=s.get('e2_mix'); margin=s.get('margin'); crop=s.get('crop'); reps=100
    if margin:
        _, margin_spec=margin_configuration(margin['sha256'],s['config_sha256'])
        check_equal('sealed margin protocol and condition', (margin_spec, 'e2', None, margin_spec['balanced_manifest_sha256']),
                    (adapt._json(folder,seal,'margin.json'),arm,mix,margin['balanced']))
        _, _, parent_choice=margin_reference(margin_spec,s['config_sha256'],s['d300_parent'])
        check_equal('exact same verified balanced treatment inputs', parent_choice, choice)
    if crop:
        import night_crop

        _, crop_spec, support=night_crop.configuration(crop['sha256'],s['config_sha256'])
        check_equal('sealed crop protocol and condition', (crop_spec,'e2',None,None,crop_spec['balanced_manifest_sha256']),
                    (adapt._json(folder,seal,'crop.json'),arm,mix,margin,crop['balanced']))
        _,_,parent_choice=margin_reference(crop_spec,s['config_sha256'],s['d300_parent'])
        check_equal('exact same verified balanced crop inputs',parent_choice,choice)
    if mix:
        _,spec=mix_configuration(mix['sha256'],s['config_sha256'])
        check_equal('sealed mix protocol',spec,adapt._json(folder,seal,'mix.json'))
        reps=mix['representatives']
        check_equal('sealed mix arm and quotas',('e2',True,{'representatives':reps,'diverse':200-reps,'neighbors':5}),
                    (arm,reps in (133,67),choice['parameters']))
        check_equal('locked random mix reference',{'run':'expand_001','sha256':spec['random_sha256']},s['d300_parent'])
    check_equal('approved seal and earlier ordered first hundred',(pool,False,d100 if arm=='d' else prior),
                (choice['pool_sha256'],choice['labels_read_in_selection'],choice['keys'][:100]))
    if arm=='d': check_equal('verified original random prefix',d300,choice['keys'])
    else:
        pool_dir=ROOT/'runs/pool_002'; pool_seal=read_sealed_json(pool_dir,'manifest.json',pair['pool_manifest_sha256']['e1'],[])
        check_equal('verified ranking feature identity',pool_seal['files']['features.npz']['sha256'],sha256(pool_dir/'features.npz'))
        keys,z=strategy.load_features(pool_dir/'features.npz')
        if arm=='e1':
            check_equal('verified E1 scores SHA',pool_seal['files']['scores.npz']['sha256'],sha256(pool_dir/'scores.npz'))
            rank=strategy.rank_e1(keys,z,strategy._load_e1_scores(pool_dir/'scores.npz',keys),prior,shortlist=128,quota=40)
        else: rank=strategy.rank_e2(keys,[inv[k]['group_id'] for k in keys],z,prior,representatives=reps,diverse=200-reps,neighbors=5)
        check_equal('identity of independent ranking',rank,{k:choice[k] for k in rank})
    if mix:
        _,_,balanced=mix_reference(spec,s['config_sha256'])
        slots,audit=mix_slots(d300,balanced['slots'],choice['keys'])
        check_equal('independent priority slot replay',audit,{k:choice['mix_audit'][k] for k in audit})
        h,b=coverage_stats(keys,z,[inv[k]['group_id'] for k in keys],balanced['keys'],choice['keys'],choice['keys'][100:])
        check_equal('independent group coverage replay',(h,b),(choice['mix_audit']['coverage_reference'],choice['mix_audit']['coverage']))
    else: slots=align_slots(d300,choice['keys'])
    plan=expanded_plan(baseplan,slots,seed,old['schedule_salt'])
    inputs=adapt._json(folder,seal,f'{sd}/inputs.json'); t=adapt._json(folder,seal,f'{sd}/train.json')
    if margin: check_margin_signal(t['margin_signal'])
    if crop: night_crop.check_signal(t['crop_signal'],support)
    check_equal('verified 300 authorized input schedules and actual consumption',
                (slots,canonical_hash([inv[k] for k in slots]),[d[0] for d,_ in plan],canonical_hash(plan),canonical_hash(plan),
                 [[d[0],n[0]] for d,n in plan],dict(Counter(n[0] for _,n in plan)),200,0,seal['files'][f'{sd}/last.pt']['sha256']),
                ([r['key'] for r in inputs['records']],choice['records_sha256'],inputs['source_keys'],choice['schedule_sha256'],
                 t['augment_schedule_sha256'],t['microbatches'],t['counts']['night'],t['updates'],t['skipped_updates'],t['last_sha256']))
    af=ROOT/'runs/accept_001'; aseal=read_sealed_json(af,'manifest.json',pair['accept_manifest_sha256'],[])
    policy,ref,base=(adapt._json(af,aseal,k+'.json') for k in ('protocol','reference','metrics'))
    approved={}; results={}; rows={}
    ev=adapt._json(folder,seal,f'{sd}/dev/evaluation.json')
    for purpose,domain in (('source_dev','day'),('night_dev','night')):
        version_name,count=accept.DEV[purpose]; info=s[purpose]; expected=policy['development'][purpose]
        check_equal('approved development manifest receipt',(f'manifests/{purpose}.json',count,expected),
                    (info['path'],info['count'],info['sha256']))
        doc=read_sealed_json(ROOT/'data/soda10m/splits'/version_name,info['path'],expected,[])
        approved[purpose]=accept._index(doc['records'],purpose)
        rel=f'{sd}/dev/{domain}'
        rows[domain]=accept.sealed_objects(folder,seal,f'{rel}/objects.jsonl.gz',purpose,approved[purpose],[])
        m=accept.evaluate_dev(rows[domain],policy)
        m['capture']=accept._audit_views(ev[f'{domain}_views'],ev[f'{domain}_fusion_caps'])
        check_equal('recomputed full development metrics',m,adapt._json(folder,seal,f'{rel}/metrics.json'))
        results[domain]=m
    assessment=accept._assessment(ref,results['day'],results['night'],policy)
    check_equal('recomputed protection and recovery',assessment,ev['assessment'])
    d_folder=ROOT/'runs/d_001'; ds=read_sealed_json(d_folder,'manifest.json',pair['d_manifest_sha256'],[])
    historical=accept.sealed_objects(d_folder,ds,f'{sd}/dev/night/objects.jsonl.gz','night_dev',approved['night_dev'],[])
    d_pair=(historical,base['D'][sd]['night']['workpoints'])
    d_path=d_seal=None
    if arm!='d':
        up=s['d300_parent']; d_path=checked_output(ROOT/'runs'/up['run'],existing=True)
        d_seal,d_state=load_run(d_path,up['sha256'])
        check_equal('verified D300 parent binding',('d',seed,s['config_sha256']),(d_state['arm'],d_state['seed'],d_state['config_sha256']))
    recomputed=compare(arm,folder/sd,folder,cfg,pair,policy,ref,base,approved['night_dev'],d_pair,
                       (*d_pair,rows['night'],results['night']['workpoints']),d_path,d_seal)
    if mix: compare_balanced(recomputed,spec,s['config_sha256'],sd,approved['night_dev'],(rows['night'],results['night']['workpoints']),policy)
    if margin:
        compare_margin(recomputed,margin_spec,s['config_sha256'],s['d300_parent'],approved['night_dev'],
                       (rows['night'],results['night']['workpoints']),policy)
    if crop:
        compare_margin(recomputed,crop_spec,s['config_sha256'],s['d300_parent'],approved['night_dev'],
                       (rows['night'],results['night']['workpoints']),policy,variant='crop')
    check_equal('complete paired comparison independently recomputed',recomputed,adapt._json(folder,seal,'comparison.json'))
    return {'arm':arm,'seed':seed,'files_verified':len(seal['files']),'manifest_sha256':sha256(folder/'manifest.json'),
            'night_counts':{n:[p['tp'],p['fp']] for n,p in results['night']['workpoints'].items()},'comparison':recomputed,
            'seconds':s['seconds'], **({'e2_mix':mix} if mix else {}), **({'margin':margin} if margin else {}),
            **({'crop':crop} if crop else {})}


def checked_output(path,*,existing=False):
    p=Path(path).absolute()
    if p.parent!=ROOT/'runs' or not re.fullmatch(r'expand_\d{3}',p.name) or p.exists()!=existing or p.is_symlink():
        raise ContractError('Expected a registered new or exact existing runs/expand_NNN')
    _no_reparse(p if existing else p.parent)
    return p


def mix_pair(diverse,representative):
    checked=[verify(p) for p in (diverse,representative)]
    a,b=[v['e2_mix'] for v in checked]
    check_equal('direct polar pair shares policy and ordered 67,133',[a['sha256'],67,133],
                [b['sha256'],a['representatives'],b['representatives']])
    records=[]; excluded=set(); policy=None; approved=None
    nights=[]
    for p in (diverse,representative):
        m,s=load_run(p); rel=f'{s["seed"]}/dev'
        _,cfg,pair,_=configuration(s['config_sha256'])
        af=ROOT/'runs/accept_001'; aseal=read_sealed_json(af,'manifest.json',pair['accept_manifest_sha256'],[])
        pol=adapt._json(af,aseal,'protocol.json'); policy=policy or pol
        check_equal('identical E2 policy',policy,pol)
        if approved is None:
            entry=s['night_dev']
            doc=read_sealed_json(train.NIGHT,entry['path'],entry['sha256'],[])
            approved=accept._index(doc['records'],'night_dev')
        rows=accept.sealed_objects(p,m,f'{rel}/night/objects.jsonl.gz','night_dev',approved,[])
        points=adapt._json(p,m,f'{rel}/night/metrics.json')['workpoints']
        nights.append((rows,points))
        ev=adapt._json(p,m,f'{rel}/evaluation.json')
        local={d['key'] for d in ev['night_fusion_caps']}
        local.update(k for v in ev['night_views'] for kind in ('pre_nms_truncated','max_det_truncated') for k in v['nms'][kind])
        excluded.update(local); records.append(len(local))
    result=effects(nights,[1,-1],policy)
    subset=[]
    for rows,points in nights:
        kept=[row for row in rows if row['key'] not in excluded]
        if not kept: break
        subset.append((kept,{k:dm.aggregate_point(kept,i,points[k]['threshold']) for i,k in enumerate(accept.CLASSES)}))
    sensitivity=None if len(subset)!=2 else effects(subset,[1,-1],policy)
    return {'contrast':'diverse_minus_representative','formal':result,'truncation_sensitivity':sensitivity,
            'excluded_images':len(excluded),'per_arm_excluded_images':records,
            'manifest_sha256':[v['manifest_sha256'] for v in checked]}


def main():
    if len(sys.argv)==6 and sys.argv[1]=='_draw':
        return _selection_worker(Path(sys.argv[2]).absolute(),sys.argv[3],int(sys.argv[4]),json.loads(sys.argv[5]))
    if len(sys.argv)==5 and sys.argv[1]=='_prepare':
        return prepare(Path(sys.argv[2]).absolute(),sys.argv[3],sys.argv[4])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('run','verify','mix-check','mix-pair')); p.add_argument('--arm',choices=KINDS)
    p.add_argument('--config-sha256'); p.add_argument('--scratch',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--d300',type=Path); p.add_argument('--d300-sha256')
    p.add_argument('--mix-sha256'); p.add_argument('--representatives',type=int,choices=(133,67))
    p.add_argument('--margin-sha256')
    p.add_argument('--crop-sha256')
    p.add_argument('--diverse-run',type=Path); p.add_argument('--representative-run',type=Path)
    a=p.parse_args()
    if a.mode=='mix-pair':
        if not (a.diverse_run and a.representative_run) or any((a.output,a.arm,a.scratch,a.config_sha256,a.d300,a.d300_sha256,a.mix_sha256,a.representatives,a.margin_sha256,a.crop_sha256)):
            p.error('mix-pair requires only --diverse-run and --representative-run')
        print(json.dumps(mix_pair(checked_output(a.diverse_run,existing=True),checked_output(a.representative_run,existing=True)),allow_nan=False))
        return
    if a.diverse_run or a.representative_run: p.error('contrast runs are only for mix-pair')
    if a.mode=='verify':
        if not a.output or any((a.arm,a.scratch,a.config_sha256,a.d300,a.d300_sha256,a.mix_sha256,a.representatives,a.margin_sha256,a.crop_sha256)): p.error('verify accepts only --output')
        print(json.dumps(verify(checked_output(a.output,existing=True)),allow_nan=False))
        return
    if not (a.config_sha256 and a.scratch): p.error('expected config digest and external scratch')
    scratch=a.scratch.absolute(); _no_reparse(scratch)
    if not scratch.is_dir() or scratch.resolve()!=scratch or scratch.is_relative_to(ROOT): p.error('invalid external scratch')
    if a.mode=='mix-check':
        if not a.mix_sha256 or any((a.arm,a.output,a.d300,a.d300_sha256,a.representatives,a.margin_sha256,a.crop_sha256)): p.error('expected exact expansion and mix protocols, no output')
        _,cfg,pair,old=configuration(a.config_sha256); _,spec=mix_configuration(a.mix_sha256,a.config_sha256)
        _,_,pool,inv,d100,d300,plan,parent=context(cfg,pair,old,'e2')
        arms,_=mix_selections(spec,a.config_sha256,cfg['seed'],pair,pool,inv,d100,d300,parent,plan,old,scratch)
        print(json.dumps({n:{**a['mix_audit'],'images':len(a['keys']),'groups':len({inv[k]['group_id'] for k in a['keys']}),
                              'reads':a['reads'],'list_sha256':canonical_hash(a['keys'])} for n,a in arms.items()},allow_nan=False))
        return
    if not (a.arm and a.output): p.error('run requires arm and output')
    if bool(a.mix_sha256)!=bool(a.representatives) or (a.mix_sha256 and a.arm!='e2'):
        p.error('mix training requires matching E2 mixture digest and one new quota')
    if (a.margin_sha256 or a.crop_sha256) and (a.arm!='e2' or a.mix_sha256 or (a.margin_sha256 and a.crop_sha256)):
        p.error('crop and margin are isolated E2 treatments, not mixes')
    if (a.arm=='d' and (a.d300 or a.d300_sha256)) or (a.arm!='d' and not(a.d300 and a.d300_sha256)):
        p.error('E1/E2 require the completed random D300 receipt; random does not accept a parent')
    run(a.arm,a.config_sha256,scratch,checked_output(a.output),
        None if not a.d300 else checked_output(a.d300,existing=True),a.d300_sha256,
          mix_sha=a.mix_sha256,representatives=a.representatives,margin_sha=a.margin_sha256,crop_sha=a.crop_sha256)


if __name__=='__main__':
    main()
