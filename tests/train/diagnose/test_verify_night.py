from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))


def test_aggregation_rejects_the_same_seed_twice():
    from source_contract import ContractError
    from verify_night import seed_assessments

    with pytest.raises(ContractError, match='Duplicate seed'):
        seed_assessments([{43: {}}, {43: {}}])


def test_all_five_points_do_not_satisfy_stability_if_evidence_needs_review():
    from verify_night import seed_assessments

    passed = {'point_met': True, 'review': [], 'classes_met': 5, 'day_protection': {'met': True}}
    flagged = dict(passed, review=['class_evidence_review'])
    res = seed_assessments([{17: passed}, {29: passed, 43: flagged}])
    assert res['all_three_point_pass'] is True
    assert res['all_stable_recovered'] is False


def test_fit_config_does_not_treat_different_budgets_as_the_same_experiment():
    from verify_night import recipe_identity

    one = {'schema_version': 1, 'seeds': [43], 'max_minutes': 30, 'updates': 120}
    replica = {**one, 'seeds': [17, 29], 'max_minutes': 60}
    assert recipe_identity(one) == recipe_identity(replica)
    assert recipe_identity(one) != recipe_identity({**replica, 'updates': 121})
