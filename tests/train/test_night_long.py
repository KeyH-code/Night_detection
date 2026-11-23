from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_long_control_preserves_the_original_prefix_and_fixed_exposure_support():
    from night_long import extended_schedule

    plan = [(f'd-{n}',f'n-{n%100}') for n in range(2000)]
    old = {'updates':200,'accumulate':10,'lr_start':.0008,'lr_end':.00008}
    cfg = {'updates':600,'continue_lr_start':.0004,'continue_lr_end':.00004}
    result, rates = extended_schedule(plan,old,cfg)
    assert result == plan+plan+plan
    assert len(rates)==600
    assert (rates[0],rates[199],rates[200],rates[-1])==pytest.approx((.0008,.00008,.0004,.00004))
    assert all(x>=y for x,y in zip(rates[200:],rates[201:]))


def test_long_config_rejects_unequal_cycle_instead_of_reweighting_some_images():
    from source_contract import ContractError
    from night_long import extended_schedule

    with pytest.raises(ContractError):
        extended_schedule([None]*2000,{'updates':200,'accumulate':10,'lr_start':1.,'lr_end':0.},
                          {'updates':500,'continue_lr_start':.0004,'continue_lr_end':.00004})
