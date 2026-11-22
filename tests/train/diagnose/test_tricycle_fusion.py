"""Keep tricycle-only fusion isolated from the locked five-class output."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))


def preds(*entries):
    return {k:[e[i] for e in entries] for i,k in enumerate(('cls','conf','original_boxes'))}


def test_tricycle_votes_ignore_other_classes_and_missing_views_count_zero():
    from tricycle_fusion import predict

    a, b, far = [0,0,10,8], [1,0,11,8], [40,10,50,18]
    out = predict((preds((5,.90,a),(5,.36,a),(1,.99,far)), preds((5,.60,b)), preds((1,.8,a))))
    assert len(out)==1 and out[0][0]==5
    assert out[0][1]==pytest.approx(.5)
    assert out[0][2]==pytest.approx([.4,0,10.4,8])


def test_tricycle_merge_never_uses_slots_from_locked_five_classes():
    from tricycle_fusion import replace_tricycle

    keep=(0,.002,[0,0,4,8]); tri=(5,.8,[10,10,16,20]); other=(5,.7,[20,10,26,20])
    base={'key':'a','gt':{'cls':[0,5],'original_boxes':[keep[2],tri[2]]},
          'pred':preds(*([keep]*299),other), 'iou_edges':[[0,0,.87],[1,299,1.0]]}
    fused,dropped=replace_tricycle(base,[tri,other])
    assert dropped==1
    assert fused['pred']['cls']==[0]*299+[5]
    assert fused['pred']['conf']==[.002]*299+[.8]
    assert fused['pred']['original_boxes']==[keep[2]]*299+[tri[2]]
    assert fused['iou_edges']==[[0,0,.87],[1,299,1.0]]


def test_tricycle_dev_join_refuses_a_different_label_or_held_out_view():
    from source_contract import ContractError
    from tricycle_fusion import merge_dev

    original={'key':'a','purpose':'source_dev','group_id':'fixed','image_sha256':'i','label_sha256':'l',
              'width':32,'height':32,'gt':{'cls':[],'original_boxes':[]},'pred':preds(),'iou_edges':[]}
    with pytest.raises(ContractError,match='frozen data identity'):
        merge_dev([original],([dict(original,label_sha256='changed')],),'consensus')
    with pytest.raises(ContractError,match='source_dev'):
        merge_dev([original],([dict(original,purpose='source_test')],),'consensus')


def test_bounded_union_scores_each_view_once_without_changing_fused_coordinates():
    from tricycle_fusion import predict

    a,b=[0,0,10,8],[1,0,11,8]
    views=(preds((5,.90,a),(5,.36,a)),preds((5,.60,b)),preds())
    mean=predict(views,'consensus')
    result=predict(views,'or')
    assert len(result)==1 and result[0][0]==5
    assert result[0][1]==pytest.approx(.96)
    assert result[0][2]==mean[0][2]
