"""Guard the source-A multi-view comparison against selective checkpoint reuse."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))


def test_comparison_cannot_hide_failed_seed_or_out_of_range_class():
    from seed_eval import summarize_seeds
    from source_contract import ContractError, NAMES

    def row(seed):
        value=.61 if seed!=29 else .59
        return {'checks':{'pedestrian/workpoint':seed!=29},'guards':{},
                'metrics':{'per_class':{n:{k:.7 for k in ('ap50','ap75','ap50_95')} for n in NAMES}},
                'workpoints':{n:{'precision':.8,'recall':value,'t_rule':.3} for n in NAMES}}
    arms={s:row(s) for s in (17,29,43)}
    summary=summarize_seeds(arms)
    assert summary['classes']['pedestrian']['recall']['mean']==pytest.approx((.61+.59+.61)/3)
    assert summary['failed_by_seed'][29]==['pedestrian/workpoint']
    assert not summary['all_seeds_pass_numerically']
    with pytest.raises(ContractError,match='seed'):
        summarize_seeds({17:row(17),43:row(43)})


def test_no_silent_gpu_work_when_local_gpu_is_busy():
    from seed_eval import parse_gpu_probe
    from source_contract import ContractError

    assert parse_gpu_probe('8192, 4870, 3\n')['free_mib']==4870
    for text in ('8192, 2250, 94\n','8192, 7000, 95\n','8192, 2020, 12\n','unknown'):
        with pytest.raises(ContractError,match='GPU'):
            parse_gpu_probe(text)


def test_no_feasible_working_point_does_not_enter_mean_as_a_valid_recall():
    from seed_eval import summarize_seeds
    from source_contract import NAMES

    ap={n:{'ap50':.8,'ap75':.5,'ap50_95':.6} for n in NAMES}
    points={n:{'precision':.85,'recall':.65,'t_rule':.5} for n in NAMES}
    passing={'checks':{'tricycle/workpoint':True},'guards':{},'metrics':{'per_class':ap},'workpoints':points}
    rejected={**passing,'checks':{'tricycle/workpoint':False},'workpoints':{
        **points,'tricycle':{'precision':.5,'recall':.7,'t_rule':None}}}
    result=summarize_seeds({17:passing,29:rejected,43:passing})
    assert result['classes']['tricycle']['recall']=={'mean':None,'spread':None,'min':None,'max':None}
