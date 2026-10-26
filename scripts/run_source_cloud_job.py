"""Foreground worker for tmux: mandatory resource/identity gate, then exactly one training command."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from source_contract import ContractError, check_equal, read_json, safe_path, sha256, write_json
from source_cloud_gate import BASE_CODE, READY

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix+'.part')
    write_json(tmp, obj)
    tmp.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def execute_logged(command, log, on_pid):
    with log.open('w', encoding='utf-8') as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        on_pid(process.pid)
        return process.wait()


def run_guarded(gate_command, training_command, job_dir, expected_identity, mode,
                runner_sha256, executor=execute_logged):
    """All modes share this gate. Nonzero or rejected/mismatched reports never reach training."""
    job_dir.mkdir(parents=True, exist_ok=False)
    state = {'mode': mode, 'status': 'gate_running', 'runner_pid': os.getpid(),
             'runner_sha256': runner_sha256, 'expected_runtime_identity_sha256': expected_identity,
             'started_utc': utc_now(), 'gate_command': gate_command,
             'training_command': training_command, 'training_started': False}
    status_path = job_dir/'status.json'
    atomic_json(status_path, state)
    (job_dir/'runner.pid').write_text(str(os.getpid())+'\n', encoding='utf-8')
    def child(pid):
        state['child_pid'] = pid
        atomic_json(status_path, state)
    try:
        code = executor(gate_command, job_dir/'gate.log', child)
        state['gate_exit_code'] = code
        if code:
            state.update(status='rejected_by_gate', finished_utc=utc_now())
            atomic_json(status_path, state)
            return code
        gate = read_json(job_dir/'resource_gate.json')
        check_equal('successful resource gate report', 'passed', gate['status'])
        check_equal('enforced runtime identity', expected_identity,
                    gate['runtime_identity']['identity_sha256'])
        state.update(status='training_running', training_started=True)
        atomic_json(status_path, state)
        code = executor(training_command, job_dir/'training.log', child)
        state.update(training_exit_code=code, status='command_passed' if code == 0 else 'training_failed',
                     finished_utc=utc_now())
        atomic_json(status_path, state)
        return code
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_utc=utc_now())
        atomic_json(status_path, state)
        raise


def check_preflight_receipt(path, expected_hash, runtime_hash, runner_hash):
    check_equal('preflight receipt SHA', expected_hash, sha256(path))
    receipt = read_json(path)
    check_equal('preflight receipt status', 'passed', receipt['status'])
    check_equal('preflight full runtime identity', runtime_hash, receipt['runtime_identity_sha256'])
    check_equal('preflight runner identity', runner_hash, receipt['runner_sha256'])
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight','formal','resume'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--job-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-identity-sha256', required=True)
    p.add_argument('--expected-runner-sha256', required=True)
    p.add_argument('--hourly-price', type=float)
    p.add_argument('--currency')
    p.add_argument('--preflight-receipt', type=Path)
    p.add_argument('--preflight-receipt-sha256')
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    root, job, output = args.root.resolve(), args.job_dir.resolve(), args.output.resolve()
    own_hash = sha256(Path(__file__))
    check_equal('external runner SHA', args.expected_runner_sha256, own_hash)
    for dest in (job, output):
        if not dest.is_relative_to(root) or dest.is_relative_to(root/'data'):
            raise ContractError(f'Job/output must stay inside project and outside frozen data: {dest}')
    if args.mode != 'resume' and output.exists():
        raise ContractError(f'Refusing existing output: {output}')
    if args.mode != 'preflight' and (not args.preflight_receipt or not args.preflight_receipt_sha256):
        p.error('formal/resume require external preflight receipt and its SHA-256')
    receipt = None
    if args.mode != 'preflight':
        receipt = check_preflight_receipt(args.preflight_receipt, args.preflight_receipt_sha256,
                                          args.expected_identity_sha256, own_hash)
    # Same process-level cache isolation as the fixed original entry; no model import here.
    from train_source import configure_paths
    configure_paths()
    gate_command = [sys.executable, str(root/'scripts/source_cloud_gate.py'), '--root', str(root),
                    '--output', str(job/'resource_gate.json'), '--identity-output', str(job/'runtime_identity.json'),
                    '--expected-identity-sha256', args.expected_identity_sha256, '--enforce']
    data = root/'data/source_v1'
    if not data.exists() and (root/'data/soda10m/splits/source_v1').is_dir():
        data = root/'data/soda10m/splits/source_v1'
    train_command = [sys.executable, str(root/'scripts/train_source.py'), args.mode,
                     '--data-root', str(data), '--config', str(root/'configs/source_a.yaml'),
                     '--ready-sha256', READY, '--output', str(output)]
    if args.mode == 'preflight':
        if args.hourly_price is not None:
            if args.hourly_price <= 0 or not args.currency:
                p.error('provided estimate must be positive and include currency; otherwise omit price')
            train_command += ['--hourly-price', str(args.hourly_price), '--currency', args.currency]
    elif args.mode == 'formal':
        recipe = safe_path(root, receipt['recipe_relative_path'])
        check_equal('preflight recipe unchanged', receipt['recipe_sha256'], sha256(recipe))
        train_command += ['--recipe', str(recipe), '--recipe-sha256', receipt['recipe_sha256']]
    else:
        if not args.checkpoint:
            p.error('resume requires --checkpoint')
        checkpoint_info = read_json(args.checkpoint.with_suffix('.pt.sha256.json'))
        if checkpoint_info['identity']['scope'] not in ('preflight','formal'):
            raise ContractError('Cloud runner refuses smoke checkpoint')
        train_command += ['--checkpoint', str(args.checkpoint.resolve())]
    code = run_guarded(gate_command, train_command, job, args.expected_identity_sha256,
                       args.mode, own_hash)
    if code:
        raise SystemExit(code)
    try:
        if args.mode == 'preflight':
            recipe = output/'FROZEN_RECIPE.json'
            parsed = read_json(recipe)
            check_equal('preflight frozen recipe', 'frozen', parsed['status'])
            check_equal('preflight core identity', BASE_CODE, parsed['identity']['code'])
            check_equal('preflight frozen data identity', READY, parsed['identity']['ready_sha256'])
            from train_monitor import audit_observation
            train_observation = audit_observation(output/'preflight_batch16' if parsed['identity']['batch'] == 16
                                                   else output/'preflight_batch8')
            receipt = {'status':'passed', 'scope':'preflight only; formal A has not started',
                       'runtime_identity_sha256':args.expected_identity_sha256, 'runner_sha256':own_hash,
                       'recipe_relative_path':recipe.relative_to(root).as_posix(), 'recipe_sha256':sha256(recipe),
                       'price_source':'user estimate, not verified bill' if args.hourly_price is not None else 'unknown',
                       'hourly_price':args.hourly_price, 'currency':args.currency,
                       'finished_utc':utc_now(), 'train_observation':train_observation,
                       'cost_review': 'Use measured full-train evaluation overhead in controller timing/budget review'}
            atomic_json(job/'PREFLIGHT_RECEIPT.json', receipt)
            atomic_json(job/'PREFLIGHT_RECEIPT.sha256.json', {'sha256':sha256(job/'PREFLIGHT_RECEIPT.json')})
        state = read_json(job/'status.json')
        if args.mode in ('formal', 'resume'):
            from train_monitor import audit_observation
            train_dir = output/'train' if args.mode == 'formal' else args.checkpoint.resolve().parents[1]
            state['train_observation'] = audit_observation(train_dir)
        state.update(status='preflight_passed_pending_controller_review' if args.mode == 'preflight'
                     else 'guarded_command_complete', finished_utc=utc_now())
        atomic_json(job/'status.json', state)
    except BaseException as exc:
        state = read_json(job/'status.json')
        state.update(status='post_command_validation_failed', error=f'{type(exc).__name__}: {exc}',
                     finished_utc=utc_now())
        atomic_json(job/'status.json', state)
        raise
    print(f'{args.mode.upper()}_GUARDED_COMMAND_COMPLETE')


if __name__ == '__main__':
    main()
