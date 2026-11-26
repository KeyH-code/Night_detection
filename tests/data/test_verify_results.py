"""Public verification using only the bundled statistics."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import verify_results


def test_bundled_evidence_recomputes_report_metrics():
    counts, results = verify_results.verify(verify_results.EVIDENCE)
    assert counts == {'night_300': 300, 'source': 4000}
    assert results['day']['passed']
    assert round(results['day']['macro_pp'], 2) == -0.64
    assert round(results['day']['worst_pp'], 2) == -0.93
    assert [round(results['night_300'][name]['gap_pp'], 2)
            for name in ('pedestrian', 'cyclist', 'car')] == [5.30, 11.57, 10.79]
    assert [round(results['night_340_targeted'][name]['gap_pp'], 2)
            for name in ('truck', 'tram')] == [-8.58, -18.53]


def test_evidence_with_changed_bytes_fails_before_recomputation(tmp_path):
    tampered = tmp_path / 'evidence.json.gz'
    tampered.write_bytes(verify_results.EVIDENCE.read_bytes() + b'\x00')
    with pytest.raises(verify_results.ContractError, match='evidence SHA-256'):
        verify_results.verify(tampered)
