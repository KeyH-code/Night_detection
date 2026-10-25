"""Small read-only preparation snapshot; never traverses held-out manifests."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'data/soda10m/splits/source_v1'
ready = json.loads((root / 'READY.json').read_text(encoding='utf-8'))
protocol = json.loads((root / ready['grouping_protocol']['path']).read_text(encoding='utf-8'))
print(json.dumps({k: v for k, v in ready.items() if k not in ('files', 'distributions')}, ensure_ascii=False, indent=2))
print('protocol keys', list(protocol))
print(json.dumps({k: v for k, v in protocol.items() if 'fingerprint' in k or 'hash' in k or 'source' in k}, ensure_ascii=False, indent=2))
for purpose in ('source_train', 'source_dev'):
    manifest = json.loads((root / ready['manifests'][purpose]['path']).read_text(encoding='utf-8'))
    print(purpose, 'manifest keys', list(manifest), 'first record', manifest['records'][0])
