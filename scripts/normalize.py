"""Normalize COCO boxes without changing raw labels or an existing version."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import json

from prepare_soda10m import sha256, write_json


def clip_box(bbox, width, height):
    x, y, w, h = bbox
    x1, y1 = min(width, max(0, x)), min(height, max(0, y))
    x2, y2 = min(width, max(0, x + w)), min(height, max(0, y + h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Clipping would create an empty box; manual review required")
    return [x1, y1, x2 - x1, y2 - y1]


def normalize(root):
    root = root.resolve()
    if any((root / 'splits').glob('*/READY.json')):
        raise ValueError('Frozen data exists; normalize a new data version instead')
    output = root / 'normalized/annotations'
    if output.exists():
        raise FileExistsError('Never overwrite existing normalized annotations')
    changes, receipt = [], {}
    for split in ('train', 'val'):
        raw = root / f'raw/SSLAD-2D/labeled/annotations/instance_{split}.json'
        original = json.loads(raw.read_text(encoding='utf-8'))
        result = deepcopy(original)
        images = {r['id']: r for r in result['images']}
        count = 0
        for a in result['annotations']:
            im = images[a['image_id']]
            clipped = clip_box(a['bbox'], im['width'], im['height'])
            if clipped != a['bbox']:
                changes.append({'official_split': split, 'image_id': a['image_id'],
                                'annotation_id': a['id'], 'category_id': a['category_id'],
                                'original_bbox': a['bbox'], 'normalized_bbox': clipped,
                                'original_area': a['area'], 'normalized_area': clipped[2]*clipped[3]})
                count += 1
            a.update(bbox=clipped, area=clipped[2]*clipped[3])
        dest = output / raw.name
        write_json(dest, result)
        receipt[split] = {'original_sha256': sha256(raw), 'normalized_sha256': sha256(dest),
                          'images': len(result['images']), 'annotations': len(result['annotations']),
                          'clipped_boxes': count, 'dropped_boxes': 0,
                          'changed_category_ids': 0, 'changed_period_labels': 0}
    write_json(root / 'audit/box_normalization_changes.json', changes)
    write_json(root / 'audit/normalization.json', {
        'version': 'clip_xyxy_to_image_bounds_v1', 'raw_unchanged': True,
        'operation': 'independent xyxy clipping, xywh/area recomputed', 'splits': receipt})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/soda10m'))
    print(json.dumps(normalize(parser.parse_args().data_root), indent=2))
