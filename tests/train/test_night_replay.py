from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_dark_replay_changes_only_half_the_source_gains_in_every_update():
    from night_replay import day_replay_plan

    original = [((f'day-{i}', i%3 == 0, .9), (f'night-{i}', i%2 == 0, 1.1)) for i in range(20)]
    actual = day_replay_plan(original, 10, .55)
    assert [n for _, n in actual] == [n for _, n in original]
    assert [d[:2] for d, _ in actual] == [d[:2] for d, _ in original]
    assert [d[2] for d,_ in actual] == pytest.approx([.9, .9*.55]*10)
    assert [d[2] for d,_ in original] == [.9]*20


def test_replay_rejects_an_update_that_cannot_have_the_locked_exposure_fraction():
    from night_replay import day_replay_plan
    from source_contract import ContractError

    with pytest.raises(ContractError, match='whole, even'):
        day_replay_plan([(('d', False, 1.), ('n', False, 1.))]*3, 3, .55)


def test_snapshot_includes_the_executed_replay_entrypoint(tmp_path):
    from night_rank import ROOT, snapshot_sources

    snapshot_sources(tmp_path, 'night_replay.py')
    for name in ('night_replay.py', 'night_train.py', 'night_rank.py', 'night_accept.py'):
        assert (tmp_path/'code'/name).read_bytes() == (ROOT/'scripts'/name).read_bytes()
