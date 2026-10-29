"""Promote only a byte-verified archive with a successful relocation receipt."""
from pathlib import Path

from source_contract import ContractError, S3_EVALUATION, check_equal, read_json, safe_path, sha256, write_json


def audit_training_roles(run_dir):
    """A valid S3 negative run need not contain best.pt; roles never imply a freeze."""
    run_dir = Path(run_dir)
    source = read_json(run_dir/'source_run.json')
    roles = read_json(run_dir/'checkpoint_roles.json')
    if not {'last', 'diagnostic'} <= set(roles) or set(roles)-{'last', 'diagnostic', 'best'}:
        raise ContractError('Missing/unknown S3 checkpoint role')
    for role, entry in roles.items():
        path = safe_path(run_dir, entry['path'])
        check_equal(role+' weight bytes', entry['sha256'], sha256(path))
        sidecar = read_json(path.with_suffix('.pt.sha256.json'))
        check_equal(role+' sidecar bytes', entry['sha256'], sidecar['sha256'])
        check_equal(role+' checkpoint identity', source['identity'], sidecar['identity'])
        check_equal(role+' epoch', entry['epoch'], sidecar['epoch_0based']+1)
        if role == 'best':
            check_equal('best qualification', 'single_run_eligible_pending_repeats', entry['status'])
    final = read_json(run_dir/'final_eval/run.json')
    selected = 'best' if 'best' in roles else 'diagnostic'
    check_equal('final selected role', selected, final['selected_role'])
    check_equal('final role identity', roles, final['roles'])
    check_equal('final tests held out', False, final['final_tests_run'])
    check_equal('last reload identity', roles['last']['sha256'], final['last_dev_checkpoint_sha256'])
    if final.get('status') == 'pending_local_evaluation':
        check_equal('deferred evaluation identity', S3_EVALUATION, source['identity'].get('evaluation_policy'))
        check_equal('handoff weights reloaded', True, final['weight_reload_checked'])
        check_equal('no fabricated final results', {}, final['results'])
        return {'selected_role': selected, 'eligible_checkpoint_exists': 'best' in roles,
                'qualification': final['qualification'], 'roles': roles,
                'evaluation_status': 'pending_local_evaluation'}
    if source['identity'].get('evaluation_policy') == S3_EVALUATION:
        check_equal('local evaluation completed', 'complete_local', final.get('status'))
        check_equal('original cloud handoff bytes', final['cloud_handoff_sha256'], sha256(run_dir/'final_eval/cloud.json'))
    for purpose, images in [('source_train', 4000), ('source_dev', 500)]:
        check_equal('final full '+purpose, images, final['results'][purpose]['images'])
        check_equal('final selected '+purpose+' weight', roles[selected]['sha256'],
                    final['results'][purpose]['checkpoint_sha256'])
        for file in ('metrics.json', 'objects.jsonl.gz', 'matches.jsonl.gz', 'pr.npz'):
            if not (run_dir/'final_eval'/purpose/file).is_file():
                raise ContractError('Missing final evidence: '+purpose+'/'+file)
    return {'selected_role': selected, 'eligible_checkpoint_exists': 'best' in roles,
            'qualification': final['qualification'], 'roles': roles,
            'evaluation_status': final.get('status', 'complete')}


def deduplicate_roles(run_dir):
    """Within a newly generated S3 run only, reference identical last bytes once."""
    run_dir = Path(run_dir)
    if 's3_policy' not in read_json(run_dir/'source_run.json')['identity']:
        raise ContractError('Role cleanup is restricted to S3 output')
    roles = read_json(run_dir/'checkpoint_roles.json')
    last = roles['last']
    for role in ('best', 'diagnostic'):
        if role in roles and roles[role]['sha256'] == last['sha256']:
            roles[role]['path'] = last['path']
    used = {r['path'] for r in roles.values()}
    for name in ('best.pt', 'diagnostic.pt'):
        rel = 'weights/'+name
        path = safe_path(run_dir, rel)
        if rel not in used and path.exists():
            path.unlink()
            path.with_suffix('.pt.sha256.json').unlink(missing_ok=True)
    write_json(run_dir/'checkpoint_roles.json', roles)
    return roles


def seal_training(run_dir, scope):
    """Retain hashes of actual S3 evidence, without pretending a local/preflight pass is training quality."""
    run_dir = Path(run_dir)
    from train_monitor import audit_observation
    observation = audit_observation(run_dir)
    check_equal('S3 completed epoch budget', 2 if scope == 'preflight' else 100, observation['epochs'])
    roles = observation['roles']
    body = {'schema_version': 1, 'status': 'completed_'+scope, 'scope': scope,
            'identity': read_json(run_dir/'source_run.json')['identity'],
            'qualification': roles['qualification'], 'selected_role': roles['selected_role'],
            'evaluation_status': roles['evaluation_status'],
            'recipe_accepted': False, 'A_frozen': False, 'final_tests_run': False,
            'files': {p.relative_to(run_dir).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                      for p in sorted(run_dir.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    write_json(run_dir/'manifest.json', body)
    return body


def publish_ready(archive: Path, sidecar_path: Path, qa_path: Path):
    receipt = read_json(qa_path)
    if receipt.get('status') != 'passed':
        raise ContractError('Relocation must pass before READY_FOR_CLOUD publication')
    actual = sha256(archive)
    sidecar = read_json(sidecar_path)
    check_equal('relocation archive SHA', actual, receipt['archive_sha256'])
    check_equal('sidecar archive SHA', actual, sidecar['sha256'])
    check_equal('relocation READY identity', sidecar['ready_sha256'], receipt['ready_sha256'])
    check_equal('final tests not run', False, receipt['final_tests_run'])
    sidecar.update(status='READY_FOR_CLOUD', relocation_qa=str(qa_path), sha256=actual)
    temporary = sidecar_path.with_suffix(sidecar_path.suffix + '.part')
    write_json(temporary, sidecar)
    temporary.replace(sidecar_path)
    return sidecar
