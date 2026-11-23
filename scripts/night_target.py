"""Remove the source gradient, preserving the exact E1 night sample and transform stream."""
from __future__ import annotations

import night_focus
import night_rank
from source_contract import ContractError,canonical_hash,check_equal


def night_batches(plan,keys):
    if not plan or any(len(m)!=2 for m in plan) or {m[1][0] for m in plan}!=set(keys):
        raise ContractError('Target-only must match the same night whitelist exactly')
    return [(n,) for _,n in plan]


def validate(cfg):
    check_equal('registered target-only identity',(0,200,35),
                tuple(cfg[k] for k in ('day_images','updates','max_minutes')))


def fit(torch,cv2,payloads,plan,seed,original,cfg,until,phase):
    check_equal('source gradient payload is empty',{},payloads['day'])
    batch=night_batches(plan,payloads['night'])
    check_equal('locked original night consumption',2000,len(batch))
    path,record=night_focus.fit_native(torch,cv2,payloads,batch,seed,original,cfg,until,phase,target_only=True)
    check_equal('only night gradients observed with original weights and exposures',
                ({},{k:20 for k in payloads['night']},[]),
                (record['counts']['source'],record['counts']['night'],record['focus_classes']))
    return path,dict(record,original_schedule_sha256=canonical_hash(plan),
                     domain_gradient_batches={'source':0,'night':len(batch)},source_staged=False)


if __name__=='__main__':
    night_rank.main(kind='target',fit=fit,validate=validate,description=__doc__)
