"""Promote only a byte-verified archive with a successful relocation receipt."""
from pathlib import Path

from source_contract import ContractError, check_equal, read_json, sha256, write_json


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
