"""Protect mixed-domain budgets, label isolation and the fitted box geometry."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from source_contract import ContractError


def test_random_schedule_exact_coverage_replays_and_matches_different_night_lists():
    from night_train import schedule
    day = [f'd:{i:04}' for i in range(4000)]
    night = [f'n:{i:03}' for i in range(100)]
    a = schedule(17, day, night)
    assert len(a) == 2000 and a == schedule(17, list(reversed(day)), night)
    assert len({pair[0][0] for pair in a}) == 2000
    assert all({pair[1][0] for pair in a[i:i+100]} == set(night) for i in range(0, 2000, 100))
    b = schedule(17, day, ['other:'+k for k in night])
    assert all(p[0] == q[0] and p[1][1:] == q[1][1:] for p,q in zip(a,b,strict=True))
    with pytest.raises(ContractError,match='100 unique'):
        schedule(17, day, ['dup']*100)


def test_metadata_receipt_picks_only_locked_keys_and_rejects_wrong_paths():
    from night_train import selected_records
    def rec(i):
        return {'purpose':'night_pool','key':f'v:{i}','group_id':'g','image':f'images/night_pool/{i}.jpg',
                'image_sha256':'a'*64,'label':f'hidden_ground_truth/night_pool/{i}.txt',
                'label_sha256':'b'*64,'width':320,'height':160,'classes':[0,2]}
    rows=[rec(i) for i in range(4)]
    keep=[{k:rows[i][k] for k in ('key','group_id','image','image_sha256')} for i in (2,0)]
    rows[1]['label']='labels/night_test/1.txt'
    selected=selected_records(keep,rows,count=2)
    assert [r['key'] for r in selected]==['v:2','v:0']
    assert all(r['purpose']=='night_pool' for r in selected)
    rows[2]['label']='hidden_ground_truth/night_pool/1.txt'
    with pytest.raises(ContractError,match='label pairing'):
        selected_records(keep,rows,count=2)


def test_letterbox_uses_actual_padded_geometry_and_flips_normalized_box():
    from night_train import fit_image
    img=np.zeros((40,80,3),np.uint8)
    tensor,boxes=fit_image(img,[(0,.25,.5,.2,.4)],True,1.,640)
    assert tensor.shape==(3,640,640)
    assert np.array_equal(tensor[:,0,0],np.full(3,114,dtype=np.uint8))
    np.testing.assert_allclose(boxes,[[.75,.5,.2,.2]],atol=1e-7)


def test_dev_and_test_purposes_refuse_before_training_payload_is_read(tmp_path):
    from night_train import training_rows
    for forbidden in ('source_dev','night_dev','night_test','source_test'):
        with pytest.raises(ContractError,match='gradient purpose'):
            training_rows(tmp_path,[{'purpose':forbidden,'key':'bad'}],'source_train')


def test_paired_slots_preserve_every_shared_image_and_all_augmentation():
    from night_adapt import aligned_slots, paired_schedule
    from night_train import schedule
    d = [f'val:{i}' for i in range(100)]
    day = [f'train:{i}' for i in range(2000)]
    e = [*d[:20], 'val:200', *d[21:70], 'val:201', *d[71:][::-1]]
    slots = aligned_slots(d, e)
    assert slots[:20] == d[:20]
    assert [slots[20], slots[70]] == ['val:200', 'val:201']
    assert all(slots[d.index(k)] == k for k in set(d)&set(e))
    cfg = {'schedule_salt': 20261122, 'flip_probability': .5, 'brightness': [.85, 1.15]}
    old = schedule(17, day, d)
    new = paired_schedule(17, day, d, slots, cfg)
    for a, b in zip(old, new, strict=True):
        assert a[0] == b[0] and a[1][1:] == b[1][1:]
        if a[1][0] in e:
            assert a == b
    assert paired_schedule(17, day, d, aligned_slots(d, [*d[:20], *d[20:][::-1]]), cfg) == old
    with pytest.raises(ContractError, match='anchor'):
        aligned_slots(d, [d[1], d[0], *d[2:]])


def test_strategy_receipt_rejects_another_arm_and_unapproved_records():
    from night_adapt import draw_records
    from source_contract import canonical_hash
    rows = [{'key': f'val:{i}', 'group_id': f'g{i}', 'image': f'images/night_pool/{i}.jpg', 'image_sha256': 'a'*64}
            for i in range(100)]
    manifest = {'purpose': 'night_pool_strategy_e2', 'images_sha256': 'b'*64,
                'arms': {str(s): {'keys': [r['key'] for r in rows], 'records': rows,
                                 'record_sha256': canonical_hash(rows)} for s in (17, 29, 43)}}
    inventory = {r['key']: r for r in rows}
    assert draw_records('e2', 17, manifest, 'b'*64, inventory) == rows
    with pytest.raises(ContractError, match='selection identity'):
        draw_records('e1', 17, manifest, 'b'*64, inventory)
    with pytest.raises(ContractError, match='approved image'):
        draw_records('e2', 17, manifest, 'b'*64, dict(inventory, **{'val:1': dict(rows[1], image_sha256='c'*64)}))


def test_label_worker_grant_is_bound_to_one_arm_and_seed(tmp_path):
    from night_adapt import granted_rows
    from source_contract import sha256, write_json
    rows = [{'key': f'val:{i}', 'group_id': f'g{i}', 'image': f'images/night_pool/{i}.jpg', 'image_sha256': 'a'*64}
            for i in range(100)]
    grant = tmp_path/'grant.json'
    write_json(grant, {'arm': 'e1', 'seed': 17, 'records': rows})
    digest = sha256(grant)
    assert granted_rows(tmp_path, 'e1', 17, digest) == rows
    for arm, seed in [('e2', 17), ('e1', 29)]:
        with pytest.raises(ContractError, match='grant identity'):
            granted_rows(tmp_path, arm, seed, digest)
    with pytest.raises(ContractError, match='SHA-256'):
        granted_rows(tmp_path, 'e1', 17, 'b'*64)


def test_ratio_slot_task_checks_more_than_the_original_twenty_anchors():
    from night_adapt import ratio_slot_task
    from night_train import schedule
    from source_contract import canonical_hash

    d = [f'val:{i}' for i in range(100)]
    days = [f'train:{i}' for i in range(2000)]
    cfg = {'schedule_salt': 20261122, 'flip_probability': .5, 'brightness': [.85, 1.15]}
    previous = {'d_input_slots': d, 'd_schedule_sha256': canonical_hash(schedule(43, days, d))}
    picks = [*d[:60], *[f'val:{i}' for i in range(200, 240)]]
    task = ratio_slot_task(43, 60, days, [{'key': k} for k in picks], cfg, previous)
    assert task['slots']['anchors'] == d[:60] and task['slots']['common_images'] == 60
    assert [r['key'] for r in task['rows']] == picks
    picks[58], picks[59] = picks[59], picks[58]
    with pytest.raises(ContractError, match='prefix'):
        ratio_slot_task(43, 60, days, [{'key': k} for k in picks], cfg, previous)
