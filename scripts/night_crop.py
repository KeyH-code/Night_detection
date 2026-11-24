"""Apply the registered tail-centred crop to only predeclared night slots."""
from __future__ import annotations

from collections import Counter
import json

import numpy as np

import night_train as train
from night_accept import _checked_file
from night_contract import ROOT, read_sealed_bytes, read_sealed_json
from source_contract import ContractError, check_equal


def slot_class(index):
    return (3, 4)[index//4 % 2] if index % 4 == 3 else None


def geometry(w, h, labels, cid):
    scale = min(640/w, 640/h)
    eligible = [(r[3], i) for i, r in enumerate(labels) if r[0] == cid and 8 < r[3]*w*scale <= 64]
    if not eligible:
        return 'no_target', None
    _, index = min(eligible)
    _, x, y, _, _ = labels[index]
    cw, ch = round(w*.8), round(h*.8)
    left = min(max(round(x*w-cw/2), 0), w-cw)
    top = min(max(round(y*h-ch/2), 0), h-ch)
    right, bottom = left+cw, top+ch
    kept = []
    for j, (c, x, y, bw, bh) in enumerate(labels):
        raw = ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)
        cut = (max(raw[0], left), max(raw[1], top), min(raw[2], right), min(raw[3], bottom))
        if j == index and cut != raw:
            return 'target_cut', None
        x0, y0, x1, y1 = cut
        area = max(0., x1-x0) * max(0., y1-y0)
        if area == 0:
            continue
        if area/((raw[2]-raw[0])*(raw[3]-raw[1])) < .70:
            return 'partial_object', None
        kept.append((c, ((x0+x1)/2-left)/cw, ((y0+y1)/2-top)/ch, (x1-x0)/cw, (y1-y0)/ch))
    return 'applied', {'window': (left, top, right, bottom), 'labels': kept, 'scale': w/cw}


class CropBatches:
    def __init__(self, plan, source_keys, night_group):
        self.plan = plan
        self.source_keys, self.night_group = source_keys, night_group
        self.calls = 0
        self.status = {3: Counter(), 4: Counter()}
        self.used = {3: set(), 4: set()}
        self.original, self.retained = Counter(), Counter()
        self.width_factors = []

    def __call__(self, torch, cv2, payload, micro, size, index):
        check_equal('exact historical crop slot', (self.calls, self.plan[self.calls], 640), (index, micro, size))
        self.calls += 1
        day, night = micro
        if day[0] not in self.source_keys or night[0] not in self.night_group:
            raise ContractError('Expected source and approved target in configured order')
        key, flip, gain = night
        path, w, h, labels = payload[key]
        self.original.update(r[0] for r in labels)
        cls = slot_class(index)
        cut = None
        if cls is not None:
            status, cut = geometry(w, h, labels, cls)
            self.status[cls][status] += 1
        if cut is None:
            self.retained.update(r[0] for r in labels)
            return train._batch(torch, cv2, payload, micro, size)
        self.retained.update(r[0] for r in cut['labels'])
        self.used[cls].add(key)
        self.width_factors.append(cut['scale'])
        source = train._batch(torch, cv2, payload, [day], size)
        night_image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        check_equal('night training crop original shape', (h,w), None if night_image is None else night_image.shape[:2])
        x0,y0,x1,y1 = cut['window']
        crop = cv2.resize(night_image[y0:y1, x0:x1], (w,h), interpolation=cv2.INTER_LINEAR)
        pixels, boxes = train.fit_image(crop, cut['labels'], flip, gain, size)
        source['img'] = torch.cat((source['img'],torch.from_numpy(pixels).cuda().float().div_(255)[None]))
        source['cls'] = torch.cat((source['cls'],torch.tensor([r[0] for r in cut['labels']], device='cuda', dtype=torch.float32)[:, None]))
        source['bboxes'] = torch.cat((source['bboxes'],torch.as_tensor(boxes,device='cuda')))
        source['batch_idx'] = torch.cat((source['batch_idx'],torch.ones(len(boxes),device='cuda')))
        return source

    def summary(self):
        return {'budget_slots': self.calls, 'reserved_slots': sum(sum(q.values()) for q in self.status.values()),
                'targets': {train.NAMES[c]: {'counts': dict(self.status[c]), 'images': len(self.used[c]),
                            'groups': len({self.night_group[k] for k in self.used[c]})} for c in (3,4)},
                'exposures': {train.NAMES[c]: {'reference': self.original[c], 'retained': self.retained[c],
                              'retention': self.retained[c]/self.original[c]} for c in range(5)},
                'min_width_factor': min(self.width_factors, default=None), 'max_width_factor': max(self.width_factors, default=None)}


def configuration(digest, expand_digest):
    raw = read_sealed_bytes(ROOT, 'configs/crop.json', digest, [])
    cfg = json.loads(raw)
    check_equal('registered single crop treatment', (1, expand_digest, 43, {'size':.8, 'visible':.7, 'width':[8,64], 'stride':4}),
                tuple(cfg[k] for k in ('schema_version','expand_sha256','seed','estimate')))
    root = ROOT/'runs/crop_001'
    seal = read_sealed_json(root, 'manifest.json', cfg['support_manifest_sha256'], [])
    support = json.loads(_checked_file(root, seal, 'metrics.json', []))
    check_equal('sealed preflight used only authorized labels and passed',
                ('sealed',cfg['balanced_manifest_sha256'],300,False,False,False,False,True),
                (seal['status'],*(support[k] for k in ('parent_manifest_sha256','reads','image_read','development_read',
                                                       'holdout_read','train_performed','met'))))
    return raw, cfg, support


def check_signal(signal, support):
    check_equal('exact realized crop and label exposure identical to sealed geometry probe',
                {k: support[k] for k in ('budget_slots','reserved_slots','targets','exposures','min_width_factor','max_width_factor')},
                signal)
