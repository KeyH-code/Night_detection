from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_slot_is_fixed_and_alternates_tail_targets():
    from night_crop import slot_class

    assert [slot_class(i) for i in range(12)] == [None]*3+[3]+[None]*3+[4]+[None]*3+[3]


def test_centered_crop_updates_every_retained_label_and_drops_outside():
    from night_crop import geometry

    labels = [(3,.5,.5,.05,.1),(2,.6,.5,.05,.1),(0,.05,.5,.02,.02)]
    status, crop = geometry(1920,1080,labels,3)
    assert status == 'applied' and crop['window'] == (192,108,1728,972)
    assert [r[0] for r in crop['labels']] == [3,2]
    assert crop['labels'][0][1:] == pytest.approx((.5,.5,.0625,.125))
    assert crop['labels'][1][1:] == pytest.approx((.625,.5,.0625,.125))


def test_cross_boundary_box_below_visibility_interrupts_crop():
    from night_crop import geometry

    labels = [(4,.5,.5,.05,.1),(0,.1,.5,.10,.1)]
    assert geometry(1920,1080,labels,4) == ('partial_object',None)
