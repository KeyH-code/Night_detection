"""Test source-only low-light replay with fixed E1/D sample and update controls."""
from __future__ import annotations

import night_rank
import night_train as train
from source_contract import ContractError, canonical_hash, check_equal, sha256


def day_replay_plan(plan, accumulate, multiplier):
    if not plan or accumulate <= 0 or accumulate % 2 or len(plan) % accumulate:
        raise ContractError('Dark replay requires whole, even-sized optimizer update groups')
    if not 0 < multiplier < 1:
        raise ContractError('Source multiplier must reduce brightness')
    return [((key, flip, gain*multiplier if i % 2 else gain), night)
            for i, ((key, flip, gain), night) in enumerate(plan)]


def validate(cfg):
    check_equal('registered source replay treatment', (2000, 200, .55, 40),
                tuple(cfg[k] for k in ('day_images', 'updates', 'dark_multiplier', 'max_minutes')))


def fit(torch, cv2, payloads, plan, seed, original, cfg, until, phase):
    acc = original['accumulate']
    check_equal('original update schedule', (10, 200, 2000), (acc, original['updates'], len(plan)))
    check_equal('staged source order', list(payloads['day']), [d[0] for d, _ in plan])
    changed = day_replay_plan(plan, acc, cfg['dark_multiplier'])
    weight, result = train._fit(torch, cv2, payloads['day'] | payloads['night'], changed, seed,
                                original, phase, cfg['updates'], until)
    check_equal('exact source-only intervention schedule', canonical_hash(changed), result['augment_schedule_sha256'])
    check_equal('exact observed two-domain exposure', [2000, 2000], [sum(r.values()) for r in result['counts'].values()])
    return weight, dict(result, initializer_sha256=sha256(train.CHECKPOINT),
                        original_schedule_sha256=canonical_hash(plan),
                        synthetic_source_exposures=len(changed)//2, dark_multiplier=cfg['dark_multiplier'],
                        synthetic_source_per_update=acc//2,
                        checkpoint_roles={'last': 'last.pt', 'best': {'path': 'last.pt', 'selected_on_dev': False}})


if __name__ == '__main__':
    night_rank.main(kind='replay', fit=fit, validate=validate, description=__doc__)
