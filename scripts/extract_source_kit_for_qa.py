"""Local relocation QA: strict path/type checks, separate Chinese directory, then verify."""
import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

from source_contract import ContractError, read_json, sha256, write_json

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--destination', type=Path, required=True)
a = p.parse_args()
a.archive = a.archive.resolve()
a.destination = a.destination.resolve()
if a.destination.exists():
    raise ContractError(f'Relocation destination already exists: {a.destination}')
with tarfile.open(a.archive, 'r:gz') as archive:
    names = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if not member.isfile() or path.is_absolute() or '..' in path.parts or '\\' in member.name:
            raise ContractError(f'Unsafe archive member: {member.name}')
        if not member.name.startswith('source_a_kit/') or member.name in names:
            raise ContractError(f'Invalid archive member: {member.name}')
        names.add(member.name)
    a.destination.mkdir(parents=True)
    archive.extractall(a.destination, filter='data')
kit = a.destination / 'source_a_kit'
result = subprocess.run([sys.executable, str(kit/'scripts/verify_source_bundle.py'), '--root', str(kit)],
                        cwd=kit, capture_output=True, text=True, encoding='utf-8')
if result.returncode:
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    raise ContractError(f'Relocated verifier failed with exit={result.returncode}')
manifest = read_json(kit / 'PACKAGE_MANIFEST.json')
write_json(root/'artifacts/source_a/relocation_qa.json', {
    'status': 'passed', 'platform': sys.platform, 'scope': 'Windows local extraction to relocated Chinese path; not Linux execution',
    'archive_sha256': sha256(a.archive), 'destination': str(kit.resolve()),
    'archive_members': len(names), 'verifier_stdout': result.stdout.strip(),
    'ready_sha256': manifest['ready_sha256'],
    'held_out_files_present': False, 'final_tests_run': False})
from source_release import publish_ready
publish_ready(a.archive, a.archive.with_suffix(a.archive.suffix + '.json'),
              root/'artifacts/source_a/relocation_qa.json')
print(result.stdout)
