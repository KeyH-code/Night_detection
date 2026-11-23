"""Synthetic regressions for the two blind night-pool selectors."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from source_contract import ContractError, canonical_hash, sha256, write_json


@pytest.mark.parametrize('random_count,e1,e2', [(20, 16, 40), (40, 12, 30), (60, 8, 20)])
def test_ratio_changes_only_the_number_of_new_picks(random_count, e1, e2):
    from night_strategy import ratio_parameters

    assert ratio_parameters('e1', random_count) == {'shortlist': 64, 'quota': e1}
    assert ratio_parameters('e2', random_count) == {'representatives': e2, 'diverse': e2, 'neighbors': 5}
    assert ratio_parameters('e1', 100) is None
    with pytest.raises(ContractError, match='ratio'):
        ratio_parameters('e2', 30)


def test_ratio_pick_checks_entire_prefix_and_e2_group_exclusion():
    from night_strategy import ratio_pick_rows

    keys = [f'val:{i}' for i in range(100)]
    rows = {k: {'key': k, 'group_id': k} for k in keys}
    # The tail fixtures mimic actual E2 step order, with 30 + 30 new picks.
    steps = [{'key': k, 'group_id': k, 'kind': 'representative' if i < 30 else 'diverse'}
             for i, k in enumerate(keys[40:])]
    assert ratio_pick_rows('e2', 40, keys, dict(keys=keys, steps=steps), rows) == list(rows.values())
    with pytest.raises(ContractError, match='prefix'):
        ratio_pick_rows('e2', 60, keys, dict(keys=[*keys[:20], keys[21], keys[20], *keys[22:]]), rows)
    rows[keys[-1]] = {'key': keys[-1], 'group_id': keys[0]}
    with pytest.raises(ContractError, match='risk group'):
        ratio_pick_rows('e2', 40, keys, dict(keys=keys, steps=steps), rows)


def test_e1_class_flip_votes_for_both_classes_and_uses_per_class_maximum():
    from night_strategy import score_e1
    original = [[0, 0, 2, 2, .8, 0], [9, 9, 10, 10, .3, 0], [20, 20, 30, 30, .99, 5]]
    flip_class = [[0, 0, 2, 2, .5, 1]]
    expected = [(1.3 / 3 + .8 + .3) / 3, (1.3 / 3) / 3, 0, 0, 0]
    # Compare flip_class, an empty view, and a copy of the original without
    # its second pedestrian. That lost box alone contributes .3.
    assert score_e1([original, flip_class, [], original[:1]]) == pytest.approx(expected)
    assert score_e1([[], [], [], []]).tolist() == [0] * 5


def test_e1_geometry_matching_is_stable_and_not_class_greedy():
    from night_strategy import view_scores
    ref = [[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .7, 2]]
    alt = [[0, 0, 10, 10, .8, 2]]
    expected = [1.1 / 3, 0., .7, 0., 0.]
    assert view_scores(ref, alt) == pytest.approx(expected)
    assert view_scores(list(reversed(ref)), alt) == pytest.approx(expected)


def test_e1_keeps_ordered_anchors_and_fills_shortage_from_shortlist_union():
    from night_strategy import rank_e1
    keys = ['z', 'b', 'c', 'd', 'e', 'f', 'g']
    vectors = np.eye(len(keys))
    scores = np.zeros((len(keys), 5))
    scores[1:, 0] = np.linspace(.1, .6, 6)
    scores[2, 1] = .7
    scores[1, 2] = .8
    result = rank_e1(keys, vectors, scores, ['z'], shortlist=6, quota=1)
    assert result['keys'] == ['z', 'g', 'c', 'b', 'f', 'e']
    assert result['shortfalls'] == [0, 0, 0, 1, 1]
    assert rank_e1(keys[::-1], vectors[::-1], scores[::-1], ['z'], shortlist=6, quota=1) == result
    with pytest.raises(ContractError, match='shortlist'):
        rank_e1(keys, vectors, scores, ['z'], shortlist=1, quota=1)


def test_e2_favors_group_coverage_then_diversity_without_reusing_a_group():
    from night_strategy import rank_e2
    keys = ['anchor', 'repeat', 'b', 'c', 'd', 'e']
    groups = ['0', '0', '1', '1', '2', '3']
    z = np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, .8, .6], [-1, 0, 0]])
    found = rank_e2(keys, groups, z, ['anchor'], representatives=1, diverse=1, neighbors=2)
    assert found['keys'] == ['anchor', 'b', 'e']
    # .5 from the duplicate pair counted as one group, .4 from d,
    # and .5 from e (it is opposite to the anchor, orthogonal to b).
    assert found['steps'][0]['gain'] == pytest.approx(1.4)
    assert found['steps'][1]['distance'] == pytest.approx(1.)
    assert found == rank_e2(keys[::-1], groups[::-1], z[::-1], ['anchor'], representatives=1, diverse=1, neighbors=2)
    with pytest.raises(ContractError, match='risk groups'):
        rank_e2(keys, groups, z, ['anchor'], representatives=3, diverse=1, neighbors=2)


def test_e2_preserves_both_fixed_anchors_even_in_one_risk_group():
    from night_strategy import rank_e2
    keys = ['x', 'y', 'j', 'k']
    result = rank_e2(keys, ['g', 'g', 'h', 'i'], np.eye(4), ['y', 'x'], representatives=1, diverse=1, neighbors=2)
    assert result['keys'] == ['y', 'x', 'j', 'k']


def test_both_selectors_meet_locked_budgets_for_three_synthetic_arms():
    from night_strategy import rank_e1, rank_e2

    keys = [f'v:{i:03}' for i in range(135)]
    groups = [f'g{i:03}' if i >= 15 else f'g{i//3:03}' for i in range(len(keys))]
    rng = np.random.default_rng(23)
    z = rng.normal(size=(len(keys), 16))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    u = np.zeros((len(keys), 5))
    for c in range(5):
        u[c::5, c] = rng.uniform(.1, 1., size=len(keys)//5)
    group_for = dict(zip(keys, groups, strict=True))
    for seed in (17, 29, 43):
        # Four anchors per class, leaving every shortlist enough eligible keys.
        anchors = [keys[5*i+c] for i in np.random.default_rng(seed).choice(27, 4, replace=False) for c in range(5)]
        e1, e2 = rank_e1(keys, z, u, anchors), rank_e2(keys, groups, z, anchors)
        for arm in (e1, e2):
            assert arm['keys'][:20] == anchors
            assert len(arm['keys']) == len(set(arm['keys'])) == 100
        assert e1['per_class'] == [16]*5 and e1['shortfalls'] == [0]*5
        assert [step['kind'] for step in e2['steps']] == ['representative']*40 + ['diverse']*40
        extra_groups = {group_for[k] for k in e2['keys'][20:]}
        assert len(extra_groups) == 80
        assert extra_groups.isdisjoint(group_for[k] for k in anchors)


def test_blind_pool_rejects_metadata_and_opens_only_by_key(tmp_path):
    from night_strategy import BlindPool
    image = tmp_path / 'images/night_pool/图.jpg'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'picture')
    manifest = tmp_path / 'selection/night_pool_images.json'
    info = {'key': 'val:10', 'group_id': 'g', 'image': image.relative_to(tmp_path).as_posix(), 'image_sha256': sha256(image)}
    write_json(manifest, {'purpose': 'night_pool_selection_images_only', 'records': [info]})
    pool = BlindPool(tmp_path, sha256(manifest), 1)
    assert pool.read('val:10') == b'picture'
    with pytest.raises(ContractError, match='not in'):
        pool.read('hidden_ground_truth/night_pool/图.txt')
    image.write_bytes(b'tampered')
    with pytest.raises(ContractError, match='SHA-256'):
        pool.read('val:10')
    (tmp_path / 'labels').mkdir()
    with pytest.raises(ContractError, match='sandbox inventory'):
        BlindPool(tmp_path, sha256(manifest), 1)


def test_worker_file_boundary_blocks_gt_dev_and_offstage_writes(tmp_path):
    picture = tmp_path / 'only.jpg'
    picture.write_bytes(b'allowed')
    forbidden = [tmp_path / x for x in ('gt.txt', 'night_dev.jpg', 'audit.json', 'train.json')]
    for path in forbidden:
        path.write_bytes(b'never read')
    script = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from night_strategy import worker_files
from source_contract import ContractError
picture, *forbidden = map(Path, sys.argv[2:])
reads = worker_files(readable=[picture])
assert picture.read_bytes() == b'allowed'
for f in forbidden:
    try: f.read_bytes()
    except ContractError: pass
    else: raise AssertionError('off-list read')
try: picture.write_bytes(b'blocked')
except ContractError: pass
else: raise AssertionError('input write')
assert reads == [picture]
"""
    done = subprocess.run([sys.executable, '-B', '-c', script, str(SCRIPTS), str(picture), *map(str, forbidden)],
                          capture_output=True, text=True, encoding='utf-8')
    assert done.returncode == 0, done.stderr


def test_e2_feature_payload_declines_scores_before_ranking(tmp_path):
    from night_strategy import load_features
    p = tmp_path / 'features.npz'
    np.savez(p, keys=np.array(['c', 'b']), vectors=np.eye(2), scores=np.zeros((2, 5)))
    with pytest.raises(ContractError, match='features fields'):
        load_features(p)
    np.savez(p, keys=np.array(['c', 'b']), vectors=np.eye(2))
    keys, vectors = load_features(p)
    assert keys == ['c', 'b']
    np.testing.assert_array_equal(vectors, np.eye(2))


def test_e1_rectangular_pixels_and_mirror_round_trip_match_native_resize():
    import cv2
    from ultralytics.data.augment import LetterBox
    from night_capture import restore, transform, rect_pixels
    img = np.random.default_rng(17).integers(0, 256, (73, 109, 3), dtype=np.uint8)
    scaled = cv2.resize(img, (640, 429), interpolation=cv2.INTER_LINEAR)
    native = LetterBox(new_shape=(int(np.ceil(73/109*640/32+.5))*32, 672), scaleup=False)(image=scaled)
    pixels, geom = rect_pixels(img)
    assert pixels.shape == (3, 448, 672)
    np.testing.assert_array_equal(pixels.transpose(1, 2, 0)[:, :, ::-1], native)
    old = np.array([[3, 7, 19, 30, .79, 0]], dtype=float)
    _, _, sy, sx, dy, dx = geom
    mirror = old.copy()
    mirror[:, [0, 2]] = (109-old[:, [2, 0]])*sx+dx
    mirror[:, [1, 3]] = old[:, [1, 3]]*sy+dy
    np.testing.assert_allclose(restore(mirror, geom, True), old)
    assert np.array_equal(transform(img, 'flip'), img[:, ::-1])
    gray = np.full((1, 1, 3), 128, np.uint8)
    assert int(transform(gray, 'bright')[0,0,0]) > 128 > int(transform(gray, 'dark')[0,0,0])


def test_native_multilabel_nms_and_actual_output_cap():
    import torch
    import torchvision  # Native validation selects this NMS backend when imported.
    from night_capture import native_detections
    raw = torch.zeros((1, 10, 3), dtype=torch.float32)
    raw[0,:4,:] = torch.tensor([[4,4,10],[4,4,10],[3,3,2],[3,3,2]], dtype=torch.float32)
    raw[0,4:6,0] = torch.tensor([.9,.8])
    raw[0,4,1], raw[0,4,2], raw[0,9,2] = .7,.05,.6
    rows = native_detections(raw)
    assert rows[:,5].tolist() == [0,1,5]
    dense = torch.zeros((1, 10, 301), dtype=torch.float32)
    dense[0,0] = 4*torch.arange(301, dtype=torch.float32)
    dense[0,1:4] = 2
    dense[0,4] = .7
    with pytest.raises(ContractError, match='output truncation'):
        native_detections(dense)


def test_rank_worker_e2_never_needs_images_or_e1_scores(tmp_path):
    from night_strategy import run_rank_worker
    keys = ['anchor','b','c','d']
    features = tmp_path/'features.npz'
    np.savez(features, keys=np.asarray(keys), vectors=np.eye(4))
    score_path = tmp_path/'scores.npz'
    score_path.write_bytes(b'not an E2 input')
    metadata = dict(keys=keys, groups=keys, anchors={'17':['anchor']}, params={'representatives':1,'diverse':1,'neighbors':2})
    output = run_rank_worker('e2', features, metadata)
    assert output['arms']['17']['keys'] == ['anchor','b','c']
    assert output['reads'] == ['features.npz']
    with pytest.raises(ContractError, match='E2 cannot accept scores'):
        run_rank_worker('e2', features, {}, scores=score_path)
    with pytest.raises(ContractError, match='ranking metadata fields'):
        run_rank_worker('e2', features, dict(metadata, scores=[.2,.5]))
    with pytest.raises(ContractError, match='ranking parameter fields'):
        run_rank_worker('e2', features, dict(metadata, params=dict(metadata['params'], scores=[.2,.5])))


def test_rank_worker_e1_replays_in_memory_with_two_exact_npz_files(tmp_path):
    from night_strategy import rank_e1, run_rank_worker
    keys = [f'c{i}' for i in range(7)]
    features, u = tmp_path/'features.npz', tmp_path/'scores.npz'
    z = np.eye(7)
    scores = np.zeros((7,5))
    scores[1:,0] = np.linspace(.1,.6,6)
    np.savez_compressed(features, keys=np.asarray(keys), vectors=z)
    np.savez_compressed(u, keys=np.asarray(keys), scores=scores)
    metadata = dict(keys=keys, anchors={'17':['c0']}, params={'shortlist':6,'quota':1})
    result = run_rank_worker('e1', features, metadata, scores=u)
    assert result['reads'] == ['features.npz','scores.npz']
    assert result['arms']['17'] == rank_e1(keys,z,scores,['c0'],shortlist=6,quota=1)
    assert result == run_rank_worker('e1', features, metadata, scores=u)


def test_preflight_stage_needs_only_eight_image_files(tmp_path):
    from hashlib import sha256 as hash_bytes
    from night_strategy import BlindPool, _stage_preflight
    root = tmp_path/'frozen'
    approved = [i*15//7 for i in range(8)]
    records = []
    for i in range(16):
        image = f'images/night_pool/{i:02}.jpg'
        data = f'{i}'.encode()
        records.append(dict(key=f'v:{i:02}',group_id=f'g{i}',image=image,image_sha256=hash_bytes(data).hexdigest()))
        if i in approved:
            p = root/image
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(data)
    manifest = root/'selection/night_pool_images.json'
    write_json(manifest, {'purpose':'night_pool_selection_images_only','records':records[::-1]})
    out = _stage_preflight(root, {'images_sha256':sha256(manifest),'pool_count':16}, tmp_path/'sandbox')
    assert out['reads'] == ['selection/night_pool_images.json',*[records[i]['image'] for i in approved]]
    pool = BlindPool(out['root'],out['manifest_sha256'],8)
    assert [pool.read(k) for k in pool.records] == [str(i).encode() for i in approved]


def test_capture_receipt_requires_both_ordered_payloads_and_shape_coverage(tmp_path):
    from night_capture import FEATURE_LAYER, INFERENCE, VIEWS
    from night_contract import POOL_MANIFEST
    from night_strategy import _checked_capture

    records = [{'key': 'a', 'image': 'images/night_pool/a.jpg'}, {'key': 'b', 'image': 'images/night_pool/b.jpg'}]
    receipt = dict(purpose='blind E1 capture', reads=[r['image'] for r in records], manifest_read=POOL_MANIFEST,
                   views=list(VIEWS), feature_layer=FEATURE_LAYER, inference=INFERENCE, input_shapes={'352x672': 2})
    write_json(tmp_path / 'capture.json', receipt)
    np.savez(tmp_path / 'features.npz', keys=['a', 'b'], vectors=np.eye(2, 512))
    np.savez(tmp_path / 'scores.npz', keys=['b', 'a'], scores=np.zeros((2, 5)))
    with pytest.raises(ContractError, match='exact E1 score keys'):
        _checked_capture(tmp_path, records)
    np.savez(tmp_path / 'scores.npz', keys=['a', 'b'], scores=np.full((2, 5), np.nan))
    with pytest.raises(ContractError, match='Invalid E1 scores'):
        _checked_capture(tmp_path, records)
    np.savez(tmp_path / 'scores.npz', keys=['a', 'b'], scores=np.zeros((2, 5)))
    assert _checked_capture(tmp_path, records) == receipt
    write_json(tmp_path / 'capture.json', dict(receipt, input_shapes={'352x672': 1}))
    with pytest.raises(ContractError, match='capture shape coverage'):
        _checked_capture(tmp_path, records)


@pytest.fixture
def e2_handoff(tmp_path):
    from night_contract import POOL, POOL_MANIFEST

    project, scratch = tmp_path / 'project', tmp_path / 'scratch'
    code = project / 'scripts'
    code.mkdir(parents=True)
    scratch.mkdir()
    for name in ('night_strategy.py', 'night_contract.py', 'night_select.py', 'source_contract.py'):
        shutil.copyfile(SCRIPTS / name, code / name)
    keys = [f'v:{i:03}' for i in range(994)]
    records = [{'key': key, 'group_id': f'g{max(1, i)}', 'image': f'images/night_pool/{i}.jpg',
                'image_sha256': 'a'*64} for i, key in enumerate(keys)]
    images = project / 'data/soda10m/splits/night_v1' / POOL_MANIFEST
    write_json(images, {'purpose': POOL, 'records': records[::-1]})
    anchors, draws = {}, {}
    for seed in (17, 29, 43):
        rows = [*records[:2], *[records[i] for i in np.random.default_rng(seed).permutation(np.arange(2, len(keys)))[:98]]]
        anchors[str(seed)] = [r['key'] for r in rows[:20]]
        draws[str(seed)] = {'records': rows, 'record_sha256': canonical_hash(rows)}
    random = project / 'runs/pool_001/random.json'
    write_json(random, {'purpose': 'night_pool_random_control', 'image_manifest_sha256': sha256(images),
                        'budget_per_seed': 100, 'arms': draws})
    config = project / 'configs/strategy.json'
    write_json(config, {'schema_version': 1, 'pool_count': 994, 'anchor_count': 20, 'seeds': [17, 29, 43],
                        'images_sha256': sha256(images), 'random_sha256': sha256(random), 'checkpoint_sha256': 'b'*64,
                        'e1': {'shortlist': 64, 'quota': 16}, 'e2': {'representatives': 40, 'diverse': 40, 'neighbors': 5}})
    parent = project / 'runs/pool_002'
    parent.mkdir()
    f = parent / 'features.npz'
    z = np.zeros((len(keys), 512))
    z[:, :16] = np.random.default_rng(7).normal(size=(len(keys), 16))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    np.savez_compressed(f, keys=keys, vectors=z)
    seal = parent / 'manifest.json'
    write_json(seal, {'status': 'sealed', 'scope': 'E1 and shared feature capture only', 'files': {
        'features.npz': {'sha256': sha256(f), 'bytes': f.stat().st_size},
        'strategy.json': {'sha256': sha256(config), 'bytes': config.stat().st_size}}})
    output = project / 'runs/pool_003'
    command = [sys.executable, '-B', str(code / 'night_strategy.py'), 'e2', '--protocol-sha256', sha256(config),
               '--scratch', str(scratch), '--output', str(output), '--e1-run', str(parent), '--e1-manifest-sha256', sha256(seal)]
    return command, project, output, parent, anchors, records


def test_e2_seals_three_arms_without_scores_detector_or_images(e2_handoff):
    command, project, output, parent, anchors, records = e2_handoff
    parent_before = {p.name: sha256(p) for p in parent.iterdir()}
    done = subprocess.run(command, cwd=project, capture_output=True, text=True, encoding='utf-8', timeout=120)
    assert done.returncode == 0, done.stderr
    assert {p.name: sha256(p) for p in parent.iterdir()} == parent_before
    data = json.loads((output / 'e2.json').read_text(encoding='utf-8'))
    assert set(data['arms']) == set(anchors)
    group = {r['key']: r['group_id'] for r in records}
    for seed, arm in data['arms'].items():
        keys = arm['keys']
        assert keys[:20] == anchors[seed] and len(keys) == len(set(keys)) == 100
        assert len({group[k] for k in keys}) == arm['groups'] == 99
        assert [s['kind'] for s in arm['steps']] == ['representative']*40 + ['diverse']*40
    audit = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert audit['ranking_worker_reads'] == ['features.npz'] and audit['replay_exact']
    assert audit['source_reads'] == ['configs/strategy.json', 'runs/pool_002/manifest.json', 'runs/pool_002/features.npz',
                                     'data/soda10m/splits/night_v1/selection/night_pool_images.json', 'runs/pool_001/random.json']
    assert audit['feature_source']['features_sha256'] == parent_before['features.npz']
    seal = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert not {'features.npz', 'scores.npz', 'e1.json'} & set(seal['files'])
    assert all(sha256(output / rel) == f['sha256'] for rel, f in seal['files'].items())


def test_e2_rejects_tampered_features_without_publishing(e2_handoff):
    command, project, output, parent, _, _ = e2_handoff
    feature = parent / 'features.npz'
    raw = feature.read_bytes()
    feature.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    done = subprocess.run(command, cwd=project, capture_output=True, text=True, encoding='utf-8', timeout=20)
    assert done.returncode != 0 and 'features.npz SHA-256' in done.stderr
    assert not output.exists()
