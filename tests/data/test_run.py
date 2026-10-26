"""Regression checks for retained-run integrity and strict model selection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from source_contract import ContractError, sha256
from verify_run import verify


class RunTests(unittest.TestCase):
    def fixture(self, root):
        def save(relative, value):
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value) if not isinstance(value, str) else value, encoding='utf-8')
        save('weights/best.pt', 'best bytes')
        save('weights/last.pt', 'last bytes')
        best, last = sha256(root / 'weights/best.pt'), sha256(root / 'weights/last.pt')
        save('job/COMPLETION_AUDIT.json', {'status': 'FORMAL_A_COMPLETE', 'final_test_executed': False,
             'night_data_used': False, 'epochs_completed': 2, 'best_epoch': 1,
             'selection_dev_primary': .2, 'weights': {'best.pt': {'sha256': best}, 'last.pt': {'sha256': last}}})
        save('job/exit', '0')
        save('train/results.csv', 'epoch,train/box_loss\n1,1.0\n2,0.5\n')
        scores = [{'epoch': i, 'per_class_AP50-95': {'pedestrian': .1, 'car': .3},
                   'fitness': .2, 'strict_improvement': i == 1} for i in (1, 2)]
        save('train/source_metrics.jsonl', '\n'.join(json.dumps(r) for r in scores))
        save('train/best_dev_metrics.json', {'checkpoint_sha256': best, 'fitness': .2,
             'metrics': {'metrics/pedestrian_AP50-95': .1, 'metrics/car_AP50-95': .3}})
        self.manifest(root)

    def manifest(self, root):
        files = {p.relative_to(root).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (root / 'manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')

    def test_valid_tie_keeps_earlier_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='中文证据_') as tmp:
            root = Path(tmp); self.fixture(root)
            self.assertEqual(verify(root)['best_epoch'], 1)
            (root / 'weights/best.pt').write_bytes(b'changed')
            with self.assertRaises(ContractError):
                verify(root)

    def test_nonfinite_loss_and_false_strict_improvement_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            p = root / 'train/results.csv'
            original = p.read_text(encoding='utf-8')
            p.write_text(original.replace('0.5', 'nan'), encoding='utf-8'); self.manifest(root)
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                verify(root)
            p.write_text(original, encoding='utf-8')
            scores = root / 'train/source_metrics.jsonl'
            scores.write_text(scores.read_text(encoding='utf-8').replace('false', 'true'), encoding='utf-8')
            self.manifest(root)
            with self.assertRaisesRegex(ContractError, 'strict improvement'):
                verify(root)
