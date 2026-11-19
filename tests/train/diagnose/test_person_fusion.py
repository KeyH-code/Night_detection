"""Keep multiscale votes independent and protected native outputs unchanged."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))


def predictions(*detections):
    return {k: list(a) for k,a in zip(('cls','conf','original_boxes'), zip(*detections))} if detections else {
        'cls': [], 'conf': [], 'original_boxes': []}


def test_fusion_votes_only_once_per_scale_and_penalizes_a_missing_view():
    from person_fusion import fuse_predictions

    base = predictions((0,.9,[0,0,8,8]), (0,.3,[0,0,8,8]), (1,.91,[0,0,8,8]))
    close = predictions((0,.6,[1,0,9,8]), (1,.99,[30,0,38,8]))
    far = predictions((0,.8,[100,0,108,8]))
    pred, dropped = fuse_predictions(base,(base,close,far))
    assert dropped == 0
    assert pred['cls'] == [1,0,0]
    assert pred['conf'] == pytest.approx([.91,.5,.8/3])
    assert pred['original_boxes'][1] == pytest.approx([.4,0,8.4,8])
    assert pred['original_boxes'][2] == [100,0,108,8]


def test_only_people_can_be_dropped_at_the_shared_output_limit():
    from person_fusion import MAX_DET, fuse_predictions

    anchor = predictions(*((1,.0011,[1,0,2,3]) for _ in range(MAX_DET-1)))
    views = (predictions((0,.4,[0,0,4,8]), (0,.8,[20,0,24,8])),) * 3
    out, dropped = fuse_predictions(anchor, views)
    assert {k:v[:-1] for k,v in out.items()} == anchor
    assert dropped == 1
    assert out['conf'][-1] == pytest.approx(.8)


def test_empty_people_do_not_change_original_nonperson_output():
    from person_fusion import fuse_predictions

    anchor = predictions((5,.1,[0,0,4,8]))
    assert fuse_predictions(anchor,(predictions(),)) == (anchor,0)


def test_mirrored_corner_coordinates_round_trip_without_swapping_y():
    from person_fusion import unmirror_boxes

    a = np.array([[1.,2.,5.,6.],[0.,0.,16.,8.]])
    expected = [[11.,2.,15.,6.],[0.,0.,16.,8.]]
    np.testing.assert_equal(unmirror_boxes(a.copy(),16), expected)
    np.testing.assert_equal(unmirror_boxes(unmirror_boxes(a.copy(),16),16), a)


def row(k, pred, *, purpose='source_dev', gp='g'):
    from detection_metrics import IOUS

    return {'key':k, 'group_id':gp, 'purpose':purpose, 'image_sha256':k, 'label_sha256':k, 'width':32, 'height':32,
            'gt':{'cls':[0,1], 'original_boxes':[[0,0,4,8],[10,10,14,18]],'boxes':[[0,0,4,8],[10,10,14,18]]},
            'pred':pred | {'boxes':pred['original_boxes']}, 'iou_thresholds':IOUS.tolist(),
            'iou_edges':[[1,i,.6] for i,c in enumerate(pred['cls']) if c==1]}


def test_fused_evidence_aligns_by_image_key_and_preserves_native_nonperson_edges():
    from person_fusion import evaluate_rows

    ped, rider = (0,.9,[0,0,4,8]),(1,.8,[10,10,14,18])
    a = row('a',predictions(ped,rider)); b = row('b',predictions(rider))
    alt_a = row('a',predictions(ped)); alt_b = row('b',predictions())
    result, caps = evaluate_rows([a,b],([alt_b,alt_a],[alt_a,alt_b]))
    assert not caps
    assert [r['key'] for r in result] == ['a','b']
    assert result[0]['pred']['cls'] == [1,0]
    assert [edge for edge in result[0]['iou_edges'] if edge[1]==0] == [[1,0,.6]]
    assert result[0]['ap_tp'][1] == [True]*10
    assert result[1]['pred']['cls'] == [1]


def test_fused_evidence_rejects_wrong_groups_and_non_dev_purposes():
    from person_fusion import evaluate_rows
    from source_contract import ContractError

    a = row('a',predictions())
    with pytest.raises(ContractError,match='identity'):
        evaluate_rows([a],([row('a',predictions(),gp='different')],))
    out = row('a',predictions(),purpose='source_test')
    with pytest.raises(ContractError,match='source_dev'):
        evaluate_rows([out],([out],))
