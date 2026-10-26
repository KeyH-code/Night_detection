import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from source_resources import effective_limits, shared_memory_risk
from source_cloud_gate import resource_rejections
import pytest


def test_real_bitahub_limits_do_not_report_host_as_allocation():
    assert effective_limits(128, 503*1024**3, 13, '1300000 100000', str(56*1024**3)) == {
        'cpu_count': 13.0, 'ram_bytes': 56*1024**3}


def test_fractional_quota_is_not_rounded_up_and_affinity_is_respected():
    assert effective_limits(128, 100, 13, '750000 100000', 'max')['cpu_count'] == 7.5
    assert effective_limits(128, 100, 4, '1300000 100000', 'max')['cpu_count'] == 4


def test_unlimited_and_host_memory_caps_are_handled():
    assert effective_limits(16, 1000, None, 'max 100000', 'max') == {'cpu_count': 16., 'ram_bytes': 1000}
    assert effective_limits(16, 1000, 16, '1600000 100000', '2000')['ram_bytes'] == 1000


def test_shared_memory_risk_boundary_does_not_claim_observed_crash():
    assert shared_memory_risk({'shared_memory_total_bytes': 64*1024**2})
    assert not shared_memory_risk({'shared_memory_total_bytes': 1024**3})
    assert not shared_memory_risk({'shared_memory_total_bytes': None})


def test_gate_rejects_underallocated_cpu_and_ram_even_when_host_is_large():
    resource = effective_limits(128, 503*1024**3, 13, '750000 100000', str(29*1024**3))
    resource['shared_memory_total_bytes'] = 2*1024**3
    reasons = resource_rejections(resource)
    assert len(reasons) == 2
    assert 'CPU' in reasons[0] and 'RAM' in reasons[1]


def test_gate_accepts_true_minima_and_rejects_shared_memory_risk_without_recipe_change():
    r = {'cpu_count': 8., 'ram_bytes': 30*1024**3, 'shared_memory_total_bytes': 1024**3}
    assert resource_rejections(r) == []
    r['shared_memory_total_bytes'] = 64*1024**2
    reasons = resource_rejections(r)
    assert len(reasons) == 1 and 'no DataLoader crash' in reasons[0]


@pytest.fixture
def identity_tree(tmp_path, monkeypatch):
    import source_cloud_gate as gate
    from source_contract import sha256, write_json
    for rel in (*gate.BASE_CODE, 'scripts/source_resources.py', 'scripts/source_cloud_gate.py',
                'data/source_v1/READY.json', 'configs/source_a.yaml', 'weights/yolo11m.pt'):
        p = tmp_path/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('fixture bytes '+rel, encoding='utf-8')
    monkeypatch.setattr(gate, 'BASE_CODE', {p: sha256(tmp_path/p) for p in gate.BASE_CODE})
    monkeypatch.setattr(gate, 'READY', sha256(tmp_path/'data/source_v1/READY.json'))
    monkeypatch.setattr(gate, 'BASE_CONFIG', sha256(tmp_path/'configs/source_a.yaml'))
    weight_hash = sha256(tmp_path/'weights/yolo11m.pt')
    monkeypatch.setattr(gate, 'PRETRAINED', weight_hash)
    write_json(tmp_path/'weights/yolo11m.provenance.json', {'sha256': weight_hash})
    return tmp_path


@pytest.mark.parametrize('relative', ['scripts/train_source.py', 'data/source_v1/READY.json',
                                     'configs/source_a.yaml', 'weights/yolo11m.pt'])
def test_identity_rejects_changed_baseline_core_data_recipe_or_weight(identity_tree, relative):
    from source_cloud_gate import runtime_identity
    from source_contract import ContractError
    runtime_identity(identity_tree)
    (identity_tree/relative).write_bytes(b'tampered')
    with pytest.raises(ContractError, match='expected='):
        runtime_identity(identity_tree)


def test_supplemental_gate_code_is_in_full_identity(identity_tree):
    from source_cloud_gate import runtime_identity
    before = runtime_identity(identity_tree)
    assert 'scripts/source_resources.py' in before['code_identity']
    assert 'scripts/source_cloud_gate.py' in before['code_identity']
    (identity_tree/'scripts/source_resources.py').write_bytes(b'changed resource gate')
    after = runtime_identity(identity_tree)
    assert after['identity_sha256'] != before['identity_sha256']
