from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from night_consensus import check_nested, score_rows
from source_contract import ContractError


def objects(*items):
    return {k: list(v) for k, v in zip(('cls', 'conf', 'original_boxes'), zip(*items))} if items else {
        'cls': [], 'conf': [], 'original_boxes': []}


BOX = [0, 0, 20, 20]
AWAY = [30, 30, 50, 50]


def test_tail_scores_keep_original_boxes_class_order_and_other_classes():
    anchor = {'key': 'a', 'gt': {'ids': [1]}, 'iou_edges': [[0, 2, .6]], 'pred': objects(
        (2, .75, BOX), (0, .25, BOX), (3, .5, BOX), (4, .25, AWAY))}
    mirror = {'a': objects((3, .625, BOX), (4, .5, BOX), (4, .25, AWAY), (3, .875, AWAY))}
    scale = {'a': objects((3, .875, AWAY), (3, .25, BOX), (2, .5, BOX))}
    out = score_rows([anchor], [mirror, scale])
    plain, margin = (out[k][0] for k in ('consensus', 'competition'))
    assert plain == {**anchor, 'pred': {**anchor['pred'], 'conf':
                    np.float32([.75, .25, (.5 + .625 + .25)/3, (.25 + .25)/3]).tolist()}}
    assert margin['pred']['conf'] == np.float32([.75, .25, (.5*.25 + .625*.5 + .25*.5)/3,
                                                  (.25 + .25*(1-.875))/3]).tolist()
    assert anchor['pred']['conf'] == [.75, .25, .5, .25]


def test_refuses_incomplete_view_instead_of_confusing_missing_images_with_missing_boxes():
    with pytest.raises(ContractError, match='view keys'):
        score_rows([{'key': 'a', 'pred': objects()}], [{}, {'a': objects()}])


def test_parent_lineage_requires_the_exact_ordered_prefix_not_just_a_subset():
    keys = list(map(str, range(300)))
    check_nested(keys, keys[:100])
    with pytest.raises(ContractError, match='ordered parent prefix'):
        check_nested(keys, keys[:100][::-1])
