"""Train-scene supervision must still refer to the same pixels after resize/pad/flip."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from train_scene import geometry,transformed,schedule


@pytest.mark.parametrize('scale',[640,960,1280])
@pytest.mark.parametrize('flip',[True,False])
def test_rectangular_letterbox_restores_border_and_subpixel_targets(scale,flip):
    w,h=1920,1080;nw,nh,ow,oh,l,t=geometry(w,h,scale)
    assert ow%32==oh%32==0 and nw==scale and nh<h
    for b in [(0,0,6,17),(1867,990,1920,1080),(750.35,600.67,769.9,658.31)]:
        cx,cy,wx,hy=transformed(b,w,h,scale,flip);cx=1-cx if flip else cx
        restored=[((cx-wx/2)*ow-l)*w/nw,((cy-hy/2)*oh-t)*h/nh,((cx+wx/2)*ow-l)*w/nw,((cy+hy/2)*oh-t)*h/nh]
        np.testing.assert_allclose(restored,b,rtol=0,atol=1e-9)


def test_full_train_coverage_with_distinct_seed_and_epoch_and_matched_batch_shapes():
    b=schedule(29,0);idx=[j for pair in b for j,s,f,g in pair];assert sorted(idx)==list(range(4000));assert b==schedule(29,0) and b!=schedule(29,1) and b!=schedule(43,0)
    assert all(len(pair)==2 and len({s for _,s,_,_ in pair})==1 for pair in b)
