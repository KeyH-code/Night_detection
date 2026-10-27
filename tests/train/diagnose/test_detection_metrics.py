"""Regression protection for matching, real operating points, slices and risk-group CIs."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
import detection_metrics as dm


def row(gt=None, pred=None, key='train:1', group='g1'):
    gt = gt or []
    pred = pred or []
    result = {'key': key, 'purpose': 'source_dev', 'group_id': group, 'width': 1920, 'height': 1080,
              'image': 'images/source_dev/a.jpg', 'image_sha256': 'a'*64,
              'gt': {'cls': [g[0] for g in gt], 'boxes': [g[1] for g in gt],
                     'original_boxes': [g[1] for g in gt]},
              'pred': {'cls': [p[0] for p in pred], 'conf': [p[1] for p in pred],
                       'boxes': [p[2] for p in pred], 'original_boxes': [p[2] for p in pred]}}
    flags, pairs = dm.ap_flags(result)
    result.update(ap_tp=flags.tolist(), ap_pairs=pairs)
    return result


def points(threshold=.5):
    return {n: {'threshold': threshold, 'origin': 'rule', 't_rule': threshold} for n in dm.NAMES}


def test_native_pairing_reproduces_locked_validator_including_ties():
    import torch
    from ultralytics.engine.validator import BaseValidator
    validator = BaseValidator.__new__(BaseValidator)
    validator.iouv = torch.linspace(.5, .95, 10)
    rng = np.random.default_rng(17)
    for _ in range(30):
        gc = rng.integers(0, 6, 10)
        pc = rng.integers(0, 6, 25)
        iou = rng.choice(np.array([0, .3, .5, .55, .75, .95, 1], dtype=np.float32), (10, 25))
        native = validator.match_predictions(torch.tensor(pc), torch.tensor(gc), torch.tensor(iou)).numpy()
        actual = np.zeros_like(native)
        for j, t in enumerate(validator.iouv.tolist()):
            pairs = dm.native_pairs(pc, gc, iou, t)
            actual[pairs[:, 1], j] = True
        assert np.array_equal(native, actual)


def test_low_confidence_high_iou_requires_rematch_after_filtering():
    # Do not assume a consumer's prediction rows are already confidence-sorted.
    sample = row([(0, [0, 0, 10, 10])], [(0, .2, [0, 0, 10, 10]), (0, .9, [0, 0, 9, 9])])
    assert np.array(sample['ap_tp'])[:, 0].tolist() == [True, False]
    stats, pairs, _ = dm.image_point(sample, 0, .8)
    assert (stats['tp'], stats['fp'], stats['fn']) == (1, 0, 0)
    assert pairs.tolist() == [[0, 1]]


def test_exact_event_curve_equals_brute_force_and_whole_score_ties():
    samples = [row([(0, [0, 0, 10, 10])], [(0, .9, [0, 0, 9, 9]), (0, .2, [0, 0, 10, 10]),
                                                   (0, .2, [30, 30, 40, 40])]),
               row([], [(0, .9, [0, 0, 10, 10])], key='train:2', group='g2')]
    curve = dm.work_curve(samples, 0)
    assert len(curve['threshold']) == 2
    for i, threshold in enumerate(curve['threshold']):
        brute = dm.aggregate_point(samples, 0, threshold)
        assert (curve['tp'][i], curve['fp'][i], curve['fn'][i]) == (brute['tp'], brute['fp'], brute['fn'])
    assert curve['tp'].tolist() == [1, 1]
    assert curve['fp'].tolist() == [1, 3]


def test_rule_uses_real_counts_infeasible_and_recall_below_rule():
    data = [row([(0, [0, 0, 10, 10]), (0, [20, 20, 30, 30])], [(0, .9, [0, 0, 10, 10])])]
    point = dm.choose_point(dm.work_curve(data, 0), .75, .6)
    assert point['status'] == 'recall_below_rule' and point['recall'] == .5
    false = [row([(0, [0, 0, 10, 10])], [(0, .9, [20, 20, 30, 30])])]
    point = dm.choose_point(dm.work_curve(false, 0), .75, .6)
    assert point['t_rule'] is None and point['origin'] == 'diagnostic_f1'
    empty = dm.choose_point(dm.work_curve(data, 1), .8, .65)
    assert empty['precision'] is None and empty['t_rule'] is None


def test_train_uses_dev_point_instead_of_optimizing_itself():
    dev = [row([(0, [0, 0, 10, 10])], [(0, .8, [0, 0, 10, 10])])]
    point = dm.choose_point(dm.work_curve(dev, 0), .75, .6)
    train = [row([(0, [0, 0, 10, 10])], [(0, .6, [0, 0, 10, 10])])]
    assert dm.aggregate_point(train, 0, point['threshold'])['recall'] == 0
    assert dm.choose_point(dm.work_curve(train, 0), .75, .6)['recall'] == 1


def test_empty_images_contribute_false_positives_and_undefined_values():
    samples = [row([(0, [0, 0, 10, 10])], []), row([], [(0, .8, [0, 0, 10, 10])], key='empty')]
    summary, _ = dm.summarize_ap(samples)
    assert summary['images'] == 2 and summary['empty_images'] == 1
    assert summary['per_class']['pedestrian']['ap50'] == 0
    assert summary['per_class']['cyclist']['ap50'] is None
    result = dm.aggregate_point(samples, 0, .5)
    assert (result['tp'], result['fp'], result['fn']) == (0, 1, 1)
    assert dm.aggregate_point(samples, 0, .99)['precision'] is None


def test_native_ap_integration_matches_locked_ap_per_class():
    from ultralytics.utils.metrics import ap_per_class
    rng = np.random.default_rng(17)
    gt = np.repeat(np.arange(6), 10)
    pc = rng.integers(0, 6, 80)
    cf = rng.random(80).astype(np.float32)
    tp = rng.random((80, 10)) > .7
    native = ap_per_class(tp, cf, pc, gt)[5]
    order = np.argsort(-cf)
    for c in range(6):
        ours = dm.ap_from_rank(tp[order][pc[order] == c], 10)
        assert np.allclose(ours, native[c], rtol=0, atol=1e-12)


def test_integer_weighted_ap_matches_explicit_rank_replication():
    rng = np.random.default_rng(17)
    for _ in range(40):
        flags = rng.random((12, 10)) > .5
        weights = rng.integers(0, 4, len(flags))
        weighted = dm.ap_from_rank(flags, 40, weights)
        explicit = dm.ap_from_rank(np.repeat(flags, weights, axis=0), 40)
        assert np.allclose(weighted, explicit, rtol=0, atol=1e-12)


def test_slice_boundaries_and_full_image_matching_do_not_create_fp():
    sample = row([(0, [0, 0, 32, 32]), (0, [100, 100, 196, 196])],
                 [(0, .9, [100, 100, 196, 196])])
    tags = dm.slice_tags(sample)
    assert 'area_M' in tags[0] and 'area_L' in tags[1]
    assert 'area_S' not in tags[0]
    matches = [dm.match_record(sample, points())]
    records = dm.slice_statistics([sample], matches, 'dev')
    medium = next(r for r in records if r['class'] == 'pedestrian' and r['slice'] == 'area_M')
    large = next(r for r in records if r['class'] == 'pedestrian' and r['slice'] == 'area_L')
    assert medium['recall50'] == 0 and large['recall50'] == 1
    assert matches[0]['classes']['pedestrian']['fp_items'] == []


def test_fp_categories_and_count_conservation():
    sample = row([(0, [0, 0, 10, 10]), (2, [20, 20, 30, 30])],
                 [(0, .9, [0, 0, 10, 10]), (0, .8, [0, 0, 10, 10]),
                  (0, .7, [20, 20, 30, 30]), (0, .6, [50, 50, 60, 60])])
    result = dm.match_record(sample, points())['classes']['pedestrian']
    assert result['tp'] + result['fp'] == 4 and result['tp'] + result['fn'] == 1
    assert CounterType(result['fp_items']) == {'duplicate', 'classification', 'background_candidate'}


def test_sub_micropixel_roundoff_does_not_move_exact_reference_boundaries():
    sample = row([(0, [.1, .1, 24.1000000000001, 96.1000000000001])])
    tags = dm.slice_tags(sample)[0]
    assert 'width_le8' in tags and 'height_le32' in tags


def CounterType(errors):
    return {e['type'] for e in errors}


def test_cluster_draws_keep_singletons_and_repeat_whole_groups_deterministically():
    first = list(dm.cluster_draws(['a', 'a', 'b', 'c'], repeats=20))
    second = list(dm.cluster_draws(['c', 'b', 'a', 'a'], repeats=20))
    assert first[0][0] == ['a', 'b', 'c']
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, second))
    assert all(int(counts.sum()) == 3 for _, counts in first)
    assert any(int(counts.max()) > 1 for _, counts in first)


def test_bootstrap_fixed_threshold_counts_and_null_no_gt_repeats():
    samples = [row([(0, [0, 0, 10, 10])], [(0, .9, [0, 0, 10, 10])], group='related'),
               row([(0, [0, 0, 10, 10])], [], key='2', group='related'),
               row([], [], key='3', group='singleton')]
    result = dm.bootstrap(samples, [dm.match_record(r, points()) for r in samples], repeats=30)
    interval = result['intervals']['pedestrian/recall']
    assert interval['lower'] == interval['upper'] == .5
    assert interval['undefined'] > 0
    assert result['singletons'] == 1
    assert result['intervals']['cyclist/ap50']['lower'] is None


def test_sparse_native_iou_is_authoritative_for_boundaries():
    sample = row([(0, [0, 0, 10, 10])], [(0, .9, [0, 0, 10, 10])])
    sample['iou_edges'] = [[0, 0, .49999997]]
    stats, _, _ = dm.image_point(sample, 0, .5)
    assert stats['tp'] == 0
