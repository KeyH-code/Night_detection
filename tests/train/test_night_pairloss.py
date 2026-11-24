from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_cross_group_pairs_only_move_current_tail_logits_in_the_right_directions():
    torch = pytest.importorskip('torch')
    from night_pairloss import Queue, Candidate

    queue = Queue()
    logits = torch.full((2, 6, 4), 0., requires_grad=True)
    p = Candidate(3, 0, 'tp', .2, -1., ('x', 0))
    n = Candidate(3, 1, 'classification', .9, 2., None)
    assert queue.push(logits, 'a', [p,n])[1] == []
    assert queue.push(logits, 'a', [p,n])[1] == []
    loss, pairs = queue.push(logits, 'b', [p,n])
    gradient, = torch.autograd.grad(loss, logits)
    assert [q[3] for q in pairs] == [True, False, False] and loss.item() > 0
    assert gradient[1,3,0] < 0 and gradient[1,3,1] > 0
    assert gradient.count_nonzero().item() == 2


def test_queue_expires_by_microbatch_even_when_it_has_no_candidates():
    torch = pytest.importorskip('torch')
    from night_pairloss import Queue, Candidate

    queue = Queue()
    logits = torch.zeros(2,6,4)
    neg = Candidate(4,1,'localization',.9,2.,None)
    pos = Candidate(4,0,'tp',.2,-1.,('x',0))
    queue.push(logits,'a',[neg])
    for _ in range(32): queue.push(logits,'b',[])
    assert queue.push(logits,'b',[pos])[1] == []


def test_native_nms_indices_select_the_matching_live_class_rows():
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from night_pairloss import candidates

    logits = torch.full((2, 6, 10), -10.)
    logits[1,3,0] = 2.
    logits[1,3,1] = 1.
    logits[1,4,2] = 3.
    boxes = torch.tensor([[16.,16.,12.,12.],[16.,16.,12.,12.],[40.,40.,12.,12.]]+[[60.,60.,8.,8.]]*7)
    raw = torch.cat((boxes.T[None],logits[1:].sigmoid()),dim=1)
    result = candidates(raw, logits[1], torch.tensor([[10.,10.,22.,22.],[34.,34.,46.,46.]]), torch.tensor([3,2]), 'k')
    assert {(x.cls,x.anchor,x.kind,x.gt) for x in result}=={(3,0,'tp',('k',0)),(4,2,'classification',None)}


def test_training_receipt_requires_both_tail_classes_and_original_microbatch_count():
    from night_expand import check_pairloss_signal
    from source_contract import ContractError

    q = {'classes':{'truck':{'pairs_by_bucket':{'classification':2}},'tram':{'pairs_by_bucket':{'localization':1}}},
         'calls':2000, 'active_batches':3, 'gradient_checked_batches':0}
    check_pairloss_signal(q)
    with pytest.raises(ContractError): check_pairloss_signal(dict(q,calls=1999))
    with pytest.raises(ContractError): check_pairloss_signal(dict(q,classes={**q['classes'],'tram':{'pairs_by_bucket':{}}}))
