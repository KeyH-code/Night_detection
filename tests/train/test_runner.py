import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from source_contract import ContractError, read_json
from run_source_cloud_job import run_guarded
from run_source_cloud_job import check_preflight_receipt
from source_contract import sha256, write_json


def commands(tmp_path, gate_exit=0, gate_status='passed', identity='pinned', train_exit=0):
    job = tmp_path/'job'
    gate = tmp_path/'fake_gate.py'
    gate.write_text('import json,sys\nfrom pathlib import Path\n'+
                    f'Path({str(job/"resource_gate.json")!r}).write_text(json.dumps('+repr({
                        'status':gate_status,'runtime_identity':{'identity_sha256':identity}})+'))\n'+
                    f'sys.exit({gate_exit})\n', encoding='utf-8')
    train = tmp_path/'fake_training.py'
    train.write_text('import sys\nfrom pathlib import Path\n'+
                     f'Path({str(tmp_path/"TRAINING_WAS_CALLED")!r}).write_text("called")\n'+
                     f'sys.exit({train_exit})\n', encoding='utf-8')
    return [sys.executable,str(gate)], [sys.executable,str(train)], job


@pytest.mark.parametrize('mode',['preflight','formal','resume'])
def test_nonzero_gate_never_starts_training_for_any_mode(tmp_path,mode):
    gate, train, job = commands(tmp_path,gate_exit=2)
    assert run_guarded(gate,train,job,'pinned',mode,'runner') == 2
    assert not (tmp_path/'TRAINING_WAS_CALLED').exists()
    state = read_json(job/'status.json')
    assert state['status']=='rejected_by_gate' and not state['training_started']


@pytest.mark.parametrize('status,identity',[('rejected','pinned'),('passed','tampered')])
def test_zero_exit_but_rejected_or_wrong_identity_never_starts_training(tmp_path,status,identity):
    gate,train,job = commands(tmp_path,gate_status=status,identity=identity)
    with pytest.raises(ContractError):
        run_guarded(gate,train,job,'pinned','preflight','runner')
    assert not (tmp_path/'TRAINING_WAS_CALLED').exists()


def test_success_records_gate_and_training_in_same_job(tmp_path):
    gate,train,job = commands(tmp_path)
    assert run_guarded(gate,train,job,'pinned','preflight','runner')==0
    assert (tmp_path/'TRAINING_WAS_CALLED').exists()
    state = read_json(job/'status.json')
    assert state['status']=='command_passed' and state['training_started']
    assert state['gate_exit_code']==0 and state['training_exit_code']==0
    assert (job/'gate.log').exists() and (job/'training.log').exists()
    assert not (job/'status.json.part').exists()


def test_training_failure_retains_nonzero_exit(tmp_path):
    gate,train,job = commands(tmp_path,train_exit=9)
    assert run_guarded(gate,train,job,'pinned','preflight','runner')==9
    assert read_json(job/'status.json')['status']=='training_failed'


@pytest.mark.parametrize('case',['hash','runtime','runner'])
def test_preflight_receipt_cannot_be_reused_with_other_bytes_or_runtime(tmp_path,case):
    receipt = tmp_path/'receipt.json'
    write_json(receipt, {'status':'passed','runtime_identity_sha256':'runtime','runner_sha256':'runner'})
    expected_hash = '0'*64 if case=='hash' else sha256(receipt)
    with pytest.raises(ContractError):
        check_preflight_receipt(receipt, expected_hash,
                               'wrong' if case=='runtime' else 'runtime',
                               'wrong' if case=='runner' else 'runner')


@pytest.mark.parametrize('gate_exit',[0,2])
@pytest.mark.parametrize('monitor_present', [True, False])
def test_actual_cli_forces_gate_arguments_and_receipt_binding_without_gpu(tmp_path,gate_exit,monitor_present):
    import subprocess
    from run_source_cloud_job import BASE_CODE, READY
    runner = Path(__file__).resolve().parents[2]/'scripts/run_source_cloud_job.py'
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    gate = scripts/'source_cloud_gate.py'
    gate.write_text('''import sys,json
from pathlib import Path
a=sys.argv
assert '--enforce' in a
assert a[a.index('--expected-identity-sha256')+1] == 'f'*64
out=Path(a[a.index('--output')+1])
out.write_text(json.dumps({'status':STATUS,'runtime_identity':{'identity_sha256':'f'*64}}))
Path(a[a.index('--identity-output')+1]).write_text(json.dumps({'identity_sha256':'f'*64}))
sys.exit(EXIT)
'''.replace('STATUS',repr('passed' if gate_exit==0 else 'rejected')).replace('EXIT',str(gate_exit)), encoding='utf-8')
    fake_core = scripts/'train_source.py'
    recipe = {'status':'frozen','identity':{'code':BASE_CODE,'ready_sha256':READY,'batch':16}}
    monitor_code = '''
run=out/'preflight_batch16';run.mkdir()
identity={'fixture':'observation'}
(run/'source_run.json').write_text(json.dumps({'identity':identity}))
(run/'train_monitor.json').write_text(json.dumps({'source_identity':identity,'overhead_ratio':0.1,'evaluation_seconds':0.5}))
(run/'source_metrics.jsonl').write_text(json.dumps({'epoch':1})+'\\n'+json.dumps({'epoch':2})+'\\n')
(run/'train_metrics.csv').write_text('epoch,images\\n1,4000\\n2,4000\\n')
''' if monitor_present else ''
    fake_core.write_text('import sys,json\nfrom pathlib import Path\n'+
        f'Path({str(tmp_path/"TRAINING_WAS_CALLED")!r}).write_text("called")\n'+
        'out=Path(sys.argv[sys.argv.index("--output")+1]);out.mkdir(parents=True)\n'+
        f'(out/"FROZEN_RECIPE.json").write_text(json.dumps({recipe!r}))\n'+monitor_code,encoding='utf-8')
    job,output = tmp_path/'job',tmp_path/'output'
    result = subprocess.run([sys.executable,str(runner),'preflight','--root',str(tmp_path),
        '--job-dir',str(job),'--output',str(output),'--expected-identity-sha256','f'*64,
        '--expected-runner-sha256',sha256(runner),'--hourly-price','1.68','--currency','CNY'],
        capture_output=True,text=True,encoding='utf-8')
    if gate_exit == 0 and not monitor_present:
        assert result.returncode != 0
        assert not (job/'PREFLIGHT_RECEIPT.json').exists()
        assert read_json(job/'status.json')['status'] == 'post_command_validation_failed'
        return
    assert result.returncode==gate_exit, result.stderr
    assert (tmp_path/'TRAINING_WAS_CALLED').exists() == (gate_exit==0)
    if gate_exit==0:
        receipt = read_json(job/'PREFLIGHT_RECEIPT.json')
        assert receipt['runtime_identity_sha256']=='f'*64
        assert receipt['runner_sha256']==sha256(runner)
        assert receipt['recipe_sha256']==sha256(output/'FROZEN_RECIPE.json')
        assert receipt['train_observation']['epochs'] == 2
        assert read_json(job/'status.json')['status']=='preflight_passed_pending_controller_review'
    else:
        assert not (job/'PREFLIGHT_RECEIPT.json').exists()
