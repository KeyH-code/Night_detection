from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def fixture():
    keys = [f'n-{i}' for i in range(100)]
    plan = [((f'd-{i}', False, 1.), (keys[i % 100], i % 2 == 0, 1.1)) for i in range(2000)]
    classes = {key: [4] if i < 20 else [3] if i < 50 else [2] for i,key in enumerate(keys)}
    return plan, classes


def test_repeat_retains_all_original_images_and_never_changes_day_or_augmentation():
    from night_repeat import reweight_schedule

    original, classes = fixture()
    new, weights = reweight_schedule(original, classes, 43, [1, 2, 6], 20261123)
    assert original[:1000] == new[:1000]
    assert [d for d,_ in new] == [d for d,_ in original]
    assert [n[1:] for _,n in new] == [n[1:] for _,n in original]
    assert set(classes) == {n[0] for _,n in new}
    assert new == reweight_schedule(original, classes, 43, [1, 2, 6], 20261123)[0]
    assert sum(weights[n[0]] == 6 for _,n in new[1000:]) > 400


def test_repeat_rejects_a_payload_extension_before_sampling():
    from night_repeat import reweight_schedule
    from source_contract import ContractError

    plan, classes = fixture()
    classes['not-on-the-whitelist'] = [4]
    with pytest.raises(ContractError, match='exactly 100'):
        reweight_schedule(plan, classes, 43, [1, 2, 6], 20261123)
