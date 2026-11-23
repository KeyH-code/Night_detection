"""Extend the exact 200-step E1 trajectory with one fixed additional optimization phase."""
from __future__ import annotations

import math

import night_adapt as adapt
import night_focus
import night_rank
from night_contract import ROOT,read_sealed_json
from source_contract import canonical_hash,check_equal


def extended_schedule(plan,old,cfg):
    check_equal('registered balanced 600-step extension',(200,10,2000,600),
                (old['updates'],old['accumulate'],len(plan),cfg['updates']))

    def cosine(n,lo,hi):
        return [lo+(hi-lo)*(1+math.cos(math.pi*step/(n-1)))/2 for step in range(n)]

    lrs=cosine(200,old['lr_end'],old['lr_start'])+cosine(400,cfg['continue_lr_end'],cfg['continue_lr_start'])
    return list(plan)*3,lrs


def validate(cfg):
    check_equal('registered extra phase only',(2000,600,.0004,.00004,45),
                tuple(cfg[k] for k in ('day_images','updates','continue_lr_start','continue_lr_end','max_minutes')))


def fit(torch,cv2,payloads,plan,seed,original,cfg,until,phase):
    new,rates=extended_schedule(plan,original,cfg)
    parent=ROOT/'runs/e1_001'
    seal=read_sealed_json(parent,'manifest.json',cfg['parent_manifest_sha256'],[])
    historical=adapt._json(parent,seal,f'{seed}/train.json')
    check_equal('historical E1 uses exact base schedule',(200,canonical_hash(plan)),
                (historical['updates'],historical['augment_schedule_sha256']))
    check_equal('verified historical cosine update order',[r['lr'] for r in historical['trace']],rates[:200])

    def verify_prefix(trace):
        check_equal('extended trajectory first 200 losses exactly reproduce E1',
                    canonical_hash(historical['trace']),canonical_hash(trace))
        print(f'long seed {seed}: exact first 200 updates verified against sealed E1; continuing fixed 400.',flush=True)

    path,record=night_focus.fit_native(torch,cv2,payloads,new,seed,original,cfg,until,phase,rates=rates,prefix_verify=verify_prefix)
    check_equal('three exact cycles of each original night and source image',
                ({k:3 for k in payloads['day']},{k:60 for k in payloads['night']}),
                (record['counts']['source'], record['counts']['night']))
    return path,dict(record,baseline_updates=200,additional_updates=400,original_schedule_sha256=canonical_hash(plan),
                    exact_prefix_trace_sha256=canonical_hash(historical['trace']),
                    cosine_phases=[[200,original['lr_start'],original['lr_end']],
                                   [400,cfg['continue_lr_start'],cfg['continue_lr_end']]])


if __name__=='__main__':
    night_rank.main(kind='long',fit=fit,validate=validate,description=__doc__)
