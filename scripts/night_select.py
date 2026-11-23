"""Freeze the blind S5 random arm; ranking/strategy is not part of this entry."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

import numpy as np
from numpy.random import Generator, PCG64  # Load the RNG before installing the worker's read guard.

from night_contract import (POOL, POOL_MANIFEST, ROOT, _no_reparse, _records,
                            read_sealed_bytes, read_sealed_json, stage_selection)
from source_contract import ContractError, canonical_hash, check_equal, read_json, sha256, write_json

ALGORITHM = 'lexicographic_key_pcg64_permutation_prefix'
CODE = ('night_select.py', 'night_contract.py', 'source_contract.py')


def _policy(settings):
    check_equal('random policy fields', {'schema_version', 'algorithm', 'pool_count', 'budget', 'seeds'}, set(settings))
    check_equal('random policy identity', (1, ALGORITHM), (settings['schema_version'], settings['algorithm']))
    n, budget, seeds = (settings[k] for k in ('pool_count', 'budget', 'seeds'))
    if (type(n) is not int or type(budget) is not int or not 1 <= budget <= n or not isinstance(seeds, list)
            or not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ContractError('Invalid random pool, budget, or seeds')
    return n, budget, seeds


def guard_reads(approved):
    """Runtime guard for a disposable Python worker, not an OS permission sandbox."""
    approved = Path(approved).absolute()
    reads = []
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def hook(event, args):
        if event != 'open':
            return
        target, _, flags = args
        if (not isinstance(target, (str, bytes)) or Path(os.fsdecode(target)).absolute() != approved
                or flags & write_flags):
            raise ContractError('Read guard refused file outside the allowed image manifest or a write attempt')
        reads.append(approved)

    sys.addaudithook(hook)
    return reads


def _expect_inventory(parent, names, *, dirs):
    try:
        with os.scandir(parent) as entries:
            actual = {entry.name: entry.stat(follow_symlinks=False) for entry in entries}
    except OSError as exc:
        raise ContractError('Cannot inspect required sandbox inventory') from exc
    if set(actual) != names:
        raise ContractError('Unexpected sandbox inventory: expected exact image-only layout')
    expected_type = stat.S_ISDIR if dirs else stat.S_ISREG
    for info in actual.values():
        if not expected_type(info.st_mode) or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ContractError('Unexpected type or reparse point in sandbox inventory')


def _draw(sandbox, manifest_sha, settings):
    n, budget, seeds = _policy(settings)
    sandbox = Path(sandbox).absolute()
    _no_reparse(sandbox)
    _expect_inventory(sandbox, {'images', 'selection'}, dirs=True)
    _expect_inventory(sandbox / 'images', {'night_pool'}, dirs=True)
    _expect_inventory(sandbox / 'selection', {'night_pool_images.json'}, dirs=False)
    file_reads = guard_reads(sandbox / POOL_MANIFEST)
    manifest = read_sealed_json(sandbox, POOL_MANIFEST, manifest_sha, [])
    records = _records(manifest, POOL, n, images_only=True)
    _expect_inventory(sandbox / 'images/night_pool', {Path(r['image']).name for r in records}, dirs=False)
    ordered = sorted(records, key=lambda r: r['key'])
    arms = {}
    for seed in seeds:
        draw = Generator(PCG64(seed)).permutation(n)[:budget]
        selected = [ordered[i] for i in draw]
        arms[str(seed)] = {'records': selected, 'record_sha256': canonical_hash(selected),
                          'groups': len({r['group_id'] for r in selected})}
    reads = [p.relative_to(sandbox).as_posix() for p in file_reads]
    check_equal('worker data files opened', [POOL_MANIFEST], reads)
    return {'arms': arms, 'reads': reads}


def run_worker(sandbox, manifest_sha, settings):
    """Spawn the same worker used in production; it receives no frozen data root."""
    worker = Path(__file__).resolve()
    args = [sys.executable, '-B', str(worker), '_draw', str(Path(sandbox).absolute()), manifest_sha,
            json.dumps(settings, separators=(',', ':'))]
    try:
        result = subprocess.run(args, cwd=sandbox, capture_output=True, check=True, timeout=45,
                                text=True, encoding='utf-8',
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
    except subprocess.CalledProcessError as exc:
        raise ContractError(f'Random worker rejected input: {exc.stderr.strip()}') from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError('Could not finish the isolated random worker') from exc
    return json.loads(result.stdout)


def run_random(protocol_sha, manifest_sha, scratch, output):
    scratch, output = (Path(p).absolute() for p in (scratch, output))
    if (not scratch.is_dir() or scratch.resolve() != scratch or scratch.is_relative_to(ROOT)
            or scratch.drive != output.drive or output.parent != ROOT / 'runs'
            or output.parent.resolve() != output.parent or output.exists() or output.is_symlink()
            or not re.fullmatch(r'pool_\d{3}', output.name)):
        raise ContractError('Use a clean external scratch and a new same-volume runs/pool_NNN')
    raw = read_sealed_bytes(ROOT, 'configs/select.json', protocol_sha, [])
    settings = json.loads(raw.decode('utf-8'))
    n, budget, seeds = _policy(settings)
    source = ROOT / 'data/soda10m/splits/night_v1'
    hashes = {name: sha256(ROOT / 'scripts' / name) for name in CODE}
    print('Scope: frozen image-only pool manifest and its verified images; no labels, prediction, or training.', flush=True)
    with tempfile.TemporaryDirectory(prefix='pool-random-', dir=scratch) as tmpdir:
        temp = Path(tmpdir)
        staged = stage_selection(source, manifest_sha, temp / 'images-only', count=n)
        verified_paths = [POOL_MANIFEST, *(row['image'] for row in staged['records'])]
        check_equal('staging input paths', verified_paths, staged['reads'])
        result = run_worker(staged['root'], manifest_sha, settings)
        check_equal('repeat from isolated sandbox', result, run_worker(staged['root'], manifest_sha, settings))
        arms = result['arms']
        check_equal('drawn seed arms', {str(seed) for seed in seeds}, set(arms))
        approved = {r['key']: r for r in staged['records']}
        memberships = {}
        for seed, arm in arms.items():
            selected = arm['records']
            keys = {r['key'] for r in selected}
            check_equal(f'arm {seed} draw budget', (budget, budget), (len(selected), len(keys)))
            check_equal(f'arm {seed} approved inputs', selected, [approved.get(r['key']) for r in selected])
            check_equal(f'arm {seed} fingerprint', arm['record_sha256'], canonical_hash(selected))
            memberships[seed] = keys
        pair_overlaps = {f'{a}/{b}': len(memberships[a] & memberships[b]) for a, b in combinations(memberships, 2)}
        folder = temp / 'evidence'
        folder.mkdir()
        (folder / 'protocol.json').write_bytes(raw)
        write_json(folder / 'random.json', {'schema_version': 1, 'purpose': 'night_pool_random_control',
                                            'image_manifest_sha256': manifest_sha, 'budget_per_seed': budget,
                                            'arms': arms})
        write_json(folder / 'run.json', {
            'status': 'random_only', 'protocol_sha256': protocol_sha,
            'image_manifest': {'path': POOL_MANIFEST, 'sha256': manifest_sha, 'images': n},
            'groups_per_seed': {seed: arm['groups'] for seed, arm in arms.items()},
            'overlap_images': pair_overlaps, 'unique_images_across_seeds': len(set().union(*memberships.values())),
            'staging_verified_paths': staged['reads'], 'worker_reads_per_draw': result['reads'],
            'worker_replay_equal': True, 'candidate_labels_read': False, 'prediction_performed': False,
            'training_performed': False, 'final_test_executed': False, 'strategy_executed': False,
            'code_sha256': hashes, 'environment': {'python': sys.version.split()[0], 'numpy': np.__version__},
        })
        for name, digest in hashes.items():
            path = folder / 'code/scripts' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'scripts' / name, path)
            check_equal(f'archived {name}', digest, sha256(path))
        files = {f.relative_to(folder).as_posix(): {'bytes': f.stat().st_size, 'sha256': sha256(f)}
                 for f in sorted(folder.rglob('*')) if f.is_file()}
        write_json(folder / 'manifest.json', {'status': 'sealed', 'scope': 'random-only, image manifest', 'files': files})
        check_equal('random policy unchanged', protocol_sha, sha256(ROOT / 'configs/select.json'))
        check_equal('image manifest unchanged', manifest_sha, sha256(source / POOL_MANIFEST))
        if output.exists():
            raise ContractError('Refusing to replace a random run')
        folder.rename(output)
        check_equal('persisted run manifest', files, read_json(output / 'manifest.json')['files'])
        for name, seal in files.items():
            p = output / name
            check_equal(f'published {name}', (seal['bytes'], seal['sha256']), (p.stat().st_size, sha256(p)))
        print(f'{output.name}: {len(arms)} × {budget} random images, replayed and sealed', flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_draw':
        try:
            _, sandbox, manifest_sha, settings = argv
            result = _draw(sandbox, manifest_sha, json.loads(settings))
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--images-sha256', required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    run_random(args.protocol_sha256, args.images_sha256, args.scratch, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
