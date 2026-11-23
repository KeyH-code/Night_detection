"""Reweight exposure using only the labels of the fixed E1 training whitelist."""
from __future__ import annotations

from collections import Counter

import numpy as np

import night_rank
import night_train as train
from source_contract import ContractError, canonical_hash, check_equal, sha256


def reweight_schedule(plan, classes, seed, weights, salt):
    keys = sorted(classes)
    coverage = Counter(n[0] for _,n in plan[:1000])
    if len(plan) != 2000 or len(keys) != 100 or coverage != {k:10 for k in keys} or {n[0] for _,n in plan} != set(keys):
        raise ContractError('Repeat needs exactly 100 fixed night training images over 2000 exposures')
    if len(weights) != 3 or any(type(w) is not int or w <= 0 for w in weights):
        raise ContractError('Expected three positive integer weights')
    per_key = {k:weights[2] if 4 in classes[k] else weights[1] if 3 in classes[k] else weights[0] for k in keys}
    rng = np.random.default_rng(np.random.SeedSequence([seed,salt]))
    prob = np.asarray(list(per_key.values()),dtype=np.float64)
    drawn = rng.choice(keys,size=1000,replace=True,p=prob/prob.sum()).tolist()
    changed = list(plan[:1000])+[(d,(key,*n[1:])) for (d,n),key in zip(plan[1000:],drawn,strict=True)]
    return changed, per_key


def validate(cfg):
    check_equal('registered repeat treatment', (2000, 200, [1,2,6], 20261123, 40),
                tuple(cfg[k] for k in ('day_images','updates','night_weights','salt','max_minutes')))


def fit(torch,cv2,payloads,plan,seed,original,cfg,until,phase):
    labels = {k:[v[0] for v in payload[3]] for k,payload in payloads['night'].items()}
    changed, weights = reweight_schedule(plan,labels,seed,cfg['night_weights'],cfg['salt'])
    weight, record = train._fit(torch,cv2,payloads['day'] | payloads['night'],changed,seed,original,phase,cfg['updates'],until)
    check_equal('reweighted actual gradient schedule',canonical_hash(changed),record['augment_schedule_sha256'])

    def class_exposure(schedule):
        images, boxes = Counter(), Counter()
        for _,n in schedule:
            classes=labels[n[0]]
            images.update(set(classes)); boxes.update(classes)
        return {'images':images,'instances':boxes}

    return weight, dict(record, initializer_sha256=sha256(train.CHECKPOINT), original_schedule_sha256=canonical_hash(plan),
                        original_label_exposures=class_exposure(plan), actual_label_exposures=class_exposure(changed),
                        night_sampling_weights=weights,
                        checkpoint_roles={'last':'last.pt','best':{'path':'last.pt','selected_on_dev':False}})


if __name__ == '__main__':
    night_rank.main(kind='repeat',fit=fit,validate=validate,description=__doc__)
