"""Recompute published development metrics from the bundled statistical evidence."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from detection_metrics import ap_from_rank
from night_accept import CLASSES, day_protection
from source_contract import ContractError, check_equal


EVIDENCE = Path(__file__).resolve().parents[1] / 'reports/evidence.json.gz'
EVIDENCE_SHA256 = '23ea775a762362188dce3f9d9f418cf9fe65172accb32a1294449407cedd2cd4'


def percent(value):
    return round(float(value) * 100, 8)


def day_ap(dataset):
    result = {}
    for name in CLASSES:
        row = dataset['classes'][name]
        mask = np.asarray(row['ap_rank'], dtype=np.uint16)
        flags = (mask[:, None] & (1 << np.arange(10))) != 0
        result[name] = {'ap50_95': float(ap_from_rank(flags, row['gt']).mean())}
    return result


def workpoints(dataset, fppi):
    budget = int(Fraction(fppi) * dataset['images'])
    result = {}
    for name in CLASSES:
        row = dataset['classes'][name]
        # Events are in descending threshold order. An empty prediction also competes.
        feasible = ((tp, fp) for tp, fp in row['events'] if fp <= budget)
        tp, fp = max(feasible, key=lambda point: (point[0], -point[1]), default=(0, 0))
        if tp == 0:
            fp = 0
        result[name] = {'tp': tp, 'fp': fp, 'gt': row['gt']}
    return result


def night_results(reference, current, ratio):
    results = {}
    for name in CLASSES:
        base, point = reference[name], current[name]
        day = Fraction(base['tp'], base['gt'])
        recall = Fraction(point['tp'], point['gt'])
        target = Fraction(ratio) * day
        results[name] = {**point, 'day_recall_pct': percent(day),
                         'night_recall_pct': percent(recall), 'target_pct': percent(target),
                         'gap_pp': percent(recall - target)}
    return results


def recompute(evidence):
    policy = evidence['policy']
    datasets = evidence['datasets']
    source = datasets['source_day']
    candidate = datasets['adapted_day']
    protection = day_protection(day_ap(source), day_ap(candidate), policy)
    per_class = {name: percent(result['delta']) for name, result in protection['per_class'].items()}
    reference = workpoints(source, policy['fppi'])
    summary = {'day': {'macro_pp': percent(protection['macro_delta']),
                       'worst_pp': min(per_class.values()), 'per_class_pp': per_class,
                       'passed': protection['met']}}
    for key in ('night_300', 'night_340_targeted'):
        points = workpoints(datasets[key], policy['fppi'])
        summary[key] = night_results(reference, points, policy['recovery_ratio'])
    return summary


def verify(path):
    # Read once so the hashed bytes and the parsed bytes are the same.
    compressed = path.read_bytes()
    check_equal('evidence SHA-256', EVIDENCE_SHA256, hashlib.sha256(compressed).hexdigest())
    evidence = json.loads(gzip.decompress(compressed))
    check_equal('evidence schema', 1, evidence['schema'])
    summary = recompute(evidence)
    check_equal('archived development metrics', evidence['expected'], summary)
    return evidence['training_images'], summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    args = parser.parse_args()
    try:
        counts, summary = verify(args.evidence)
    except (OSError, ContractError) as error:
        parser.exit(1, f'{error}\n')
    day = summary['day']
    labeled, trained = counts['night_300'], counts['source']
    print(f'Night labels / daytime training images: {labeled}/{trained} = {labeled/trained:.1%}')
    print(f'Daytime AP50:95 change: overall {day["macro_pp"]:+.2f} pp, '
          f'worst class {day["worst_pp"]:+.2f} pp')
    for experiment in ('night_300', 'night_340_targeted'):
        print(experiment)
        for name, row in summary[experiment].items():
            print(f'  {name}: recall {row["night_recall_pct"]:.2f}%, '
                  f'target {row["target_pct"]:.2f}%, gap {row["gap_pp"]:+.2f} pp '
                  f'(TP={row["tp"]}, FP={row["fp"]}, GT={row["gt"]})')
    print('PASS: Development metrics match the archived results.')


if __name__ == '__main__':
    main()
