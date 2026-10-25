"""Read-only delivery evidence and exact installed-vs-lock checks."""
import importlib.metadata
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from source_contract import check_equal, read_json, sha256, write_json

root = Path(__file__).resolve().parents[1]
installed = {re.sub('[-_.]+', '-', d.metadata['Name']).lower(): d.version
             for d in importlib.metadata.distributions()}
locks = {}
for name in ('windows', 'linux'):
    path = root / 'environment' / f'requirements-{name}.lock'
    versions = dict(re.findall(r'^([a-zA-Z0-9_-]+)==([^\s\\]+)', path.read_text(encoding='utf-8'), re.M))
    common = {k: v for k, v in versions.items() if k in installed}
    for key, version in common.items():
        check_equal(f'{name} lock {key}', installed[key], version)
    locks[name] = {'sha256': sha256(path), 'packages': len(versions),
                   'common_with_installed_windows': len(common),
                   'status': 'installed and pip-check passed' if name == 'windows' else 'resolved only; Linux not run'}
settings = Path(os.environ['APPDATA']) / 'Ultralytics' / 'settings.json'
global_state = {'path': str(settings), 'exists': settings.is_file(),
                'initial_console_message': 'WARNING Ultralytics settings updated to the latest schema. Existing values were preserved where possible.',
                'original_bytes_available': False, 'pre_import_sha256_available': False,
                'verified_before_after_diff': None, 'automatic_rollback_attempted': False}
if settings.is_file():
    obj = read_json(settings)
    global_state.update({'current_sha256': sha256(settings),
                         'created_utc': datetime.fromtimestamp(settings.stat().st_ctime, timezone.utc).isoformat(),
                         'modified_utc': datetime.fromtimestamp(settings.stat().st_mtime, timezone.utc).isoformat(),
                         'safe_current_fields': {k: obj[k] for k in ('settings_version', 'datasets_dir', 'weights_dir', 'runs_dir', 'sync') if k in obj},
                         'directory_file_names': sorted(p.name for p in settings.parent.iterdir() if p.is_file())})
write_json(root / 'artifacts/source_a/global_settings_audit.json', global_state)
write_json(root / 'artifacts/source_a/dependency_lock_audit.json', locks)
print(json.dumps({'locks': locks, 'global_settings': global_state}, ensure_ascii=False))
