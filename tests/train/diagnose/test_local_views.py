"""Coordinate ownership and class isolation for the pedestrian experiments."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import person_local as local
from person_teacher import representative
from person_verify import roi


def test_overlapping_crops_restore_coordinates_and_give_seam_to_one_owner():
    # Same original box [45,0,55,10], seen in both horizontal mirror crops.
    scores=np.array([.7]);left=np.array([[5,0,15,10]]);right=np.array([[45,0,55,10]])
    assert not local.remap(left,scores,0,60,100,False,True)
    assert local.remap(right,scores,40,60,100,True,True)==[(.7,[45,0,55,10])]
    # For a box at the bottom-right image edge the verification context shifts, not the original box.
    x,y,u,v=roi([1900,980,1920,1080],1920,1080)
    assert (u-x,v-y,u,v)==(800,600,1920,1080)


def test_one_teacher_candidate_cannot_supervise_two_overlapping_people_or_a_rider():
    gt=[(0,[0,0,10,10]),(0,[1,0,11,10]),(1,[60,0,70,10]),(0,[62,0,72,10])]
    output=(np.array([.8,.95],np.float32),np.array([[0,0,10,10],[60,0,70,10]],np.float32))
    out=representative(output,gt)
    assert out[0]['iou']==1 and out[1] is None and out[3] is None and 2 not in out


def test_competition_only_changes_person_scores_and_keeps_native_nonperson_iou():
    row={'gt':{'cls':[1],'original_boxes':[[0,0,10,10]]},
         'pred':{'cls':[1,2,0],'original_boxes':[[0,0,10,10],[20,0,32,10],[0,0,10,10]],'conf':[.8,.9,.6]},
         'iou_edges':[[0,0,.81],[0,2,1.0]]}
    out,cut=local.replace(row,[(.6,[0,0,10,10])],True)
    assert not cut and out['pred']['cls']==[1,2,0] and out['pred']['conf'][:2]==[.8,.9]
    assert out['iou_edges'][0]==[0,0,.81] and np.isclose(out['pred']['conf'][-1],.12)
