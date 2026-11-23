"""Regression gates for the five-class development acceptance protocol."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from source_contract import ContractError, NAMES, sha256, write_json

CLASSES = NAMES[:5]
POLICY = {'fppi': '0.02', 'recovery_ratio': '0.80', 'day': {'macro_drop': '0.01', 'class_drop': '0.02'},
          'interval': {'repeats': 64, 'seed': 20261026, 'width_review': '0.20'}}


def row(key, gt=(), conf=(), edges=(), *, classes=None, group=None, purpose='night_dev'):
    return {'key': key, 'purpose': purpose, 'group_id': group or key, 'gt': {'cls': list(gt)},
            'pred': {'cls': [0]*len(conf) if classes is None else list(classes),
                     'conf': np.asarray(conf, dtype=np.float32).tolist()}, 'iou_edges': list(edges)}


def test_workpoint_counts_all_images_and_rematches_before_applying_tied_events():
    from night_accept import evaluate_dev

    # The lower-confidence but higher-IoU predicted box would steal the match
    # if full-list TP flags were reused. Its tied companion makes that point cost 2 FP.
    first = row('hit', [0], [.9, .8, .8], [(0, 0, .6), (0, 1, .9)])
    rest = [row(str(i)) for i in range(49)]
    p = evaluate_dev([first, *rest], POLICY)['workpoints']['pedestrian']
    assert (p['fp_budget'], p['tp'], p['fp'], p['threshold']) == (1, 1, 0, first['pred']['conf'][0])
    assert p['budget_sensitivity']['plus_one']['fp_budget'] == 2


def test_workpoints_never_borrow_fp_budget_and_keep_missing_class_undefined():
    from night_accept import evaluate_dev

    rows = [row('hit', [0, 1], [.9, .8, .8, .9], [(0, 1, .9), (1, 3, .8)], classes=[0, 0, 0, 1])]
    rows += [row(str(i)) for i in range(49)]
    p = evaluate_dev(rows, POLICY)['workpoints']
    assert p['pedestrian']['empty_output'] and p['pedestrian']['recall'] == 0
    assert p['cyclist']['recall'] == 1 and p['cyclist']['fppi'] == 0
    assert p['car']['recall'] is None and p['car']['interval']['valid'] == 0


@pytest.mark.parametrize('candidate,macro,each', [([.47, .51, .51, .51, .51], True, False),
                                                 ([.485]*5, False, True), ([.49]*5, True, True),
                                                 ([.48, .5, .5, .5, .5], True, True),
                                                 ([None, .51, .51, .51, .51], None, None)])
def test_day_protection_needs_both_limits(candidate, macro, each):
    from night_accept import day_protection

    def metrics(values):
        return {name: {'ap50_95': v} for name, v in zip(CLASSES, values, strict=True)}
    result = day_protection(metrics([.5]*5), metrics(candidate), POLICY)
    assert result['macro_pass'] is macro and result['class_pass'] is each
    assert result['met'] is (None if macro is None else macro and each)


def point(tp, fn, fp=0, *, width=.1):
    return {'tp': tp, 'fn': fn, 'fp': fp, 'fp_budget': 6, 'interval': {'ci95': [0, width], 'valid': 64},
            'budget_sensitivity': {'minus_one': {'tp': tp}, 'plus_one': {'tp': tp}}}


def test_recovery_uses_fixed_reference_exact_boundary_and_review_gates():
    from night_accept import recovery

    a = {name: point(5, 5) for name in CLASSES}
    b = {name: point(4, 6) for name in CLASSES}
    a['truck'] = point(0, 10)
    b['car'] = point(4, 6, 7)
    b['tram'] = point(4, 6, width=.3)
    b['cyclist']['budget_sensitivity']['minus_one']['tp'] = 3
    got = recovery(a, b, POLICY)
    assert got['pedestrian']['met'] is True and got['pedestrian']['evidence_pass'] is True
    assert got['pedestrian']['target'] == float(Fraction(2, 5))
    assert got['car']['met'] is False and got['car']['budget_pass'] is False
    assert got['truck']['met'] is None and 'zero_day_reference' in got['truck']['review']
    assert got['tram']['met'] is True and not got['tram']['evidence_pass']
    assert 'budget_sensitivity' in got['cyclist']['review']


def test_paired_draws_share_groups_skip_missing_and_do_not_retune_thresholds():
    from night_accept import evaluate_dev, paired_recall

    a = [row('1', [0], [.8], [(0, 0, 1.)]), row('2'), row('3', [1], [.9], [(0, 0, 1.)], classes=[1])]
    b = [row('1', [0]), row('2'), row('3', [1], [.5], [(0, 0, 1.)], classes=[1])]
    pa, pb = (evaluate_dev(r, POLICY)['workpoints'] for r in (a, b))
    same = paired_recall({17: (a, pa, a, pa)}, POLICY)
    out = paired_recall({17: (a, pa, b, pb), 29: (a, pa, b, pb), 43: (a, pa, b, pb)}, POLICY)
    assert same['mean']['pedestrian']['ci95'] == [0., 0.]
    assert out['mean']['pedestrian']['ci95'] == [-1., -1.]
    assert 2 <= out['mean']['pedestrian']['valid'] < POLICY['interval']['repeats']
    assert out['mean']['cyclist']['delta'] == 0 and out['mean']['macro']['ci95'] is None
    with pytest.raises(ContractError, match='paired'):
        paired_recall({17: (a, pa, [dict(b[0], group_id='wrong'), *b[1:]], pb)}, POLICY)


def test_evaluation_rejects_unapproved_purpose():
    from night_accept import evaluate_dev

    with pytest.raises(ContractError, match='development'):
        evaluate_dev([row('x', purpose='night_pool')], POLICY)


def test_paired_identity_projects_only_the_excluded_class():
    from night_accept import evaluate_dev, paired_recall

    person = [0., 0., 2., 2.]
    a = [dict(row('same', [0, 5], [.9], [(0, 0, 1.)]),
              gt={'cls': [0, 5], 'original_boxes': [person, [3., 3., 4., 4.]]})]
    b = [dict(a[0], gt={'cls': [0], 'original_boxes': [person]})]
    pa, pb = (evaluate_dev(r, POLICY)['workpoints'] for r in (a, b))
    assert paired_recall({17: (a, pa, b, pb)}, POLICY)['mean']['pedestrian']['delta'] == 0
    wrong = [dict(b[0], gt={'cls': [0], 'original_boxes': [[0., 0., 2., 2.001]]})]
    with pytest.raises(ContractError, match='paired'):
        paired_recall({17: (a, pa, wrong, pb)}, POLICY)


def test_sealed_objects_reject_wrong_split_or_modified_bytes(tmp_path):
    from night_accept import sealed_objects

    path = tmp_path / 'day.jsonl.gz'
    identity = {'group_id': 'g', 'image': 'approved.jpg', 'label': 'approved.txt', 'image_sha256': 'b'*64,
                'label_sha256': 'c'*64, 'width': 16, 'height': 16}
    obj = {**row('k', [0], [.75], [(0, 0, 1.)], purpose='source_dev'), **identity}
    with gzip.open(path, 'wt', encoding='utf-8') as dst:
        dst.write(json.dumps(obj)+'\n')
    seal = {'files': {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}}}
    manifest = {'k': obj}
    reads = []
    assert sealed_objects(tmp_path, seal, path.name, 'source_dev', manifest, reads) == [json.loads(json.dumps(obj))]
    assert reads == [path.name]
    with pytest.raises(ContractError, match='development'):
        sealed_objects(tmp_path, seal, path.name, 'source_test', manifest, [])
    with pytest.raises(ContractError, match='object identity'):
        sealed_objects(tmp_path, seal, path.name, 'source_dev', {'k': dict(obj, group_id='other')}, [])
    path.write_bytes(b'modified')
    with pytest.raises(ContractError, match='SHA-256'):
        sealed_objects(tmp_path, seal, path.name, 'source_dev', manifest, [])


def test_locked_entry_refuses_unapproved_configuration_before_any_run(tmp_path):
    import night_accept

    root = tmp_path / 'repo'
    scripts = root / 'scripts'
    scripts.mkdir(parents=True)
    for name in ('night_accept.py', 'source_contract.py', 'night_contract.py', 'detection_metrics.py'):
        shutil.copyfile(Path(night_accept.__file__).parent / name, scripts / name)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (root / 'runs').mkdir()
    write_json(root / 'configs/accept.json', {'unapproved': True})
    result = subprocess.run([sys.executable, '-B', str(scripts/'night_accept.py'), 'replay', '--protocol-sha256', 'a'*64,
                             '--scratch', str(scratch), '--output', str(root/'runs/accept_001')], cwd=root,
                            capture_output=True, encoding='utf-8', timeout=15)
    assert result.returncode != 0 and 'configs/accept.json SHA-256' in result.stderr
    assert not any((root/'runs').iterdir()) and not any(scratch.iterdir())
