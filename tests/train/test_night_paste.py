from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def example(night_box):
    images=torch.zeros(2,3,640,640)
    images[0].fill_(.7)
    images[1].fill_(.25)
    return {'img':images,'cls':torch.tensor([[3.],[1.]]),
            'bboxes':torch.tensor([[.5,.5,.12,.20],night_box]),
            'batch_idx':torch.tensor([0.,1.])}


def test_paste_keeps_original_night_tensor_and_labels_beside_composite():
    from night_paste import composite

    before=example([.1,.35,.09,.1])
    batch,event=composite(before,0,[20,28,40,56])
    assert event['class_id']==3
    assert torch.equal(batch['img'][1],before['img'][1])
    assert torch.equal(batch['bboxes'][batch['batch_idx']==1],before['bboxes'][1:])
    assert batch['cls'][batch['batch_idx']==0].flatten().tolist()==[1.,3.]
    assert int((batch['img'][0]!=before['img'][1]).sum())>0


def test_paste_refuses_to_cover_any_existing_night_annotation():
    from night_paste import composite

    before=example([.5,.5,1.,.56])
    batch,event=composite(before,0,[20,28,40,56])
    assert event is None and batch is before
