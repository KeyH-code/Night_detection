from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_target_night_stream_has_no_source_rows_and_keeps_exact_augments():
    from night_target import night_batches

    plan=[(('d1',True,.7),('n1',False,.9)),(('d2',False,1.1),('n2',True,1.15))]
    assert night_batches(plan, {'n1','n2'}) == [(plan[0][1],),(plan[1][1],)]


def test_target_rejects_stream_with_extra_night_payload():
    from source_contract import ContractError
    from night_target import night_batches

    with pytest.raises(ContractError,match='same night whitelist'):
        night_batches([(('a',True,.9),('b',False,.9))],{'b','unapproved'})
