from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def test_presence_uses_overlap_and_minimum_scores_only_for_same_class():
    from night_presence import score_presence

    orig = [[0, 0, 10, 10, .8, 3], [30, 30, 40, 40, .9, 4], [50, 50, 60, 60, .95, 0]]
    alt = [[0, 0, 10, 10, .7, 3], [30, 30, 40, 40, .8, 2], [50, 50, 60, 60, .03, 0]]
    z = score_presence(orig, [alt, []])
    assert z == pytest.approx([0, 0, 0, .7, 0])
    assert score_presence([], [alt]).tolist() == [0]*5


def test_half_score_stage_picks_highest_positive_even_if_less_diverse():
    from night_tail import rank_quota

    keys = [f'x{i:03d}' for i in range(13)]
    scores = np.ones((13,5))*.1
    for i in range(5):
        scores[i+1,i] = .9
    result = rank_quota(keys,np.eye(13),scores,['x000'],12,[2]*5,first_score=True)
    assert result['keys'][:6] == keys[:6]
    assert result['per_class'] == [2]*5
