from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_floor_only_raises_selected_night_tram_and_detaches_reference():
    torch = pytest.importorskip('torch')
    from night_floor import lower_bound_loss

    z = torch.zeros((2,6,4),requires_grad=True)
    ref = torch.tensor([2.,-1.],requires_grad=True)
    value = lower_bound_loss(z,torch.tensor([0,1]),ref)
    g,r = torch.autograd.grad(value,(z,ref),allow_unused=True)
    assert value.item() == pytest.approx(.1)
    assert g[1,4,0] == pytest.approx(-.05) and g.count_nonzero().item() == 1
    assert r is None


def test_reference_nms_and_shared_geometry_identify_only_supported_floor_crossings():
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from night_floor import observations

    gt = torch.tensor([[10.,10.,22.,22.],[34.,34.,46.,46.],[54.,54.,66.,66.]])
    coords = torch.tensor([[16.,16.,12.,12.],[16.,16.,12.,12.],[40.,40.,12.,12.],
                           [60.,60.,12.,12.],[60.,60.,12.,12.],[110.,110.,12.,12.],
                           [111.,110.,12.,12.]]).T
    ref = torch.full((6,7),-12.); ref[4,:4] = torch.tensor([2.,1.,2.,2.])
    teacher = torch.cat((coords,ref.sigmoid()))[None]
    student_coords = coords.clone(); student_coords[0,2] = 80.
    z = torch.full((6,7),-12.); z[4,4]=2.
    student = torch.cat((student_coords,z.sigmoid()))[None]
    kept, refs, ids, crossing, _ = observations(teacher,ref,student,gt)
    assert kept.tolist()==[0,3] and refs.tolist()==[2.,2.] and ids.tolist()==[0,2]
    assert crossing.tolist()==[True,False]  # The other student anchor already covers GT 2.


@pytest.mark.parametrize(('stats','expected'), [((12,4),[]), ((11,4),['support']), ((12,3),['floor_signal'])])
def test_floor_gate_requires_distinct_gt_support_and_crossings(stats,expected):
    pytest.importorskip('torch')
    from night_floor import gate

    assert gate({'eligible_gt':stats[0],'below_floor_gt':stats[1]})==expected


def test_coverage_requires_nondecrease_at_dense_floor_and_final_stages():
    from night_expand import floor_coverage

    baseline=[9,8,8,8,8]
    rows={f'g{n}':[int(n<x) for x in [9,9,9,8,7]] for n in range(10)}
    c=floor_coverage(rows,baseline)
    assert c['candidate']==[9,9,9,8,7] and not c['met']
    rows['g7'][-1]=1
    assert floor_coverage(rows,baseline)['met']
