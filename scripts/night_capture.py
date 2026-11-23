"""E1-only CUDA image capture worker; no dataset, validator, or trainer opens files."""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
import time

import numpy as np

from night_contract import POOL_MANIFEST, ROOT, read_sealed_bytes
from night_strategy import BlindPool, score_e1, worker_files
from source_contract import ContractError, NAMES, check_equal

VIEWS = ('original', 'bright', 'dark', 'flip')
MODEL = 'runs/align_001/17/last.pt'
A_SHA = '45d212eb674aa60ef2a159652a448cf84eb897ce5a0820ade7dc591fe2afe659'
VERSIONS = ('2.14.0+cu130', '0.29.0+cu130', '8.4.171')
CONF = .05
MAX_NMS = 30000
MAX_DET = 300
SIDE = 640
STRIDE = 32
NMS_IOU = .7
FEATURE_LAYER = {'index': 10, 'type': 'C2PSA', 'channels': 512}
INFERENCE = {'device': 'cuda:0', 'precision': 'fp16', 'batch': 1, 'imgsz': SIDE, 'stride': STRIDE,
             'rect': True, 'augment': False, 'conf': CONF, 'iou': NMS_IOU, 'agnostic': False,
             'multi_label': True, 'max_det': MAX_DET, 'max_nms': MAX_NMS}
_GAMMA = {name: np.rint((np.arange(256)/255)**gamma*255).astype(np.uint8)
          for name, gamma in (('bright', .7), ('dark', 1.5))}


def transform(image, name):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape):
        raise ContractError('Expected decoded uint8 BGR image')
    if name == 'original':
        return image
    if name == 'flip':
        return np.ascontiguousarray(image[:, ::-1])
    if name in _GAMMA:
        return _GAMMA[name][image]
    raise ContractError('Unknown E1 view')


def rect_pixels(image):
    """Batch-1 geometry from the locked validator data path, without labels."""
    import cv2
    from ultralytics.data.augment import LetterBox

    h, w = image.shape[:2]
    gain = SIDE/max(h, w)
    rh, rw = min(math.ceil(h*gain), SIDE), min(math.ceil(w*gain), SIDE)
    res = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rh, rw) != (h, w) else image
    target = tuple(math.ceil(SIDE*x/max(h, w)/STRIDE + .5)*STRIDE for x in (h, w))
    box = LetterBox(new_shape=target, scaleup=False)
    payload = {'img': res}
    params = box.get_params(payload)
    res = box.apply_image(payload, params)['img']
    check_equal('padded input shape', target, res.shape[:2])
    check_equal('no unexpected second resize', (rw, rh), params['new_unpad'])
    geom = h, w, rh/h, rw/w, params['top'], params['left']
    return np.ascontiguousarray(res[:, :, ::-1].transpose(2, 0, 1)), geom


def restore(detections, geom, mirrored):
    h, w, sy, sx, dy, dx = geom
    out = np.array(detections, dtype=np.float64, copy=True).reshape(-1, 6)
    if not np.isfinite(out).all():
        raise ContractError('Nonfinite detections')
    out[:, [0, 2]] = (out[:, [0, 2]]-dx)/sx
    out[:, [1, 3]] = (out[:, [1, 3]]-dy)/sy
    if mirrored:
        out[:, [0, 2]] = w-out[:, [2, 0]]
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    if np.any(out[:, 2:4] <= out[:, :2]):
        raise ContractError('Zero-area detection after original-coordinate restore')
    if not np.isin(out[:, 5], np.arange(6)).all():
        raise ContractError('Invalid restored class ID')
    return out


def native_detections(prediction):
    import torch
    from ultralytics.utils import nms

    x = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
    if x.ndim != 3 or tuple(x.shape[:2]) != (1, 10) or x.shape[2] in (0, 6) or not x.isfinite().all():
        raise ContractError('Unknown six-class YOLO prediction layout or nonfinite output')
    if int((x[:, 4:] > CONF).sum().item()) > MAX_NMS:
        raise ContractError('E1 pre-NMS output truncation')

    class TimeoutGuard(logging.Handler):
        def emit(self, record):
            if 'NMS time limit' in record.getMessage():
                raise ContractError('E1 NMS time limit reached')

    opts = dict(conf_thres=CONF, iou_thres=NMS_IOU, agnostic=False, multi_label=True, nc=6, max_nms=MAX_NMS)
    guard = TimeoutGuard()
    nms.LOGGER.addHandler(guard)
    try:
        output = nms.non_max_suppression(x.clone(), max_det=MAX_DET, **opts)[0]
        if len(output) == MAX_DET:
            uncapped = nms.non_max_suppression(x.clone(), max_det=MAX_NMS, **opts)[0]
            if len(uncapped) > MAX_DET:
                raise ContractError('E1 post-NMS output truncation')
            if not torch.equal(output, uncapped):
                raise ContractError('Unverifiable E1 output at NMS cap')
    finally:
        nms.LOGGER.removeHandler(guard)
    return output.detach().float().cpu().numpy()


def _open_model(torch):
    # Only these trusted, hash-pinned bytes may be unpickled. No path-based
    # Ultralytics loader, which can resolve/download absent weight paths.
    raw = read_sealed_bytes(ROOT, MODEL, A_SHA, [])
    ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    model = ckpt.get('ema') or ckpt.get('model')
    if not isinstance(model, torch.nn.Module) or ckpt.get('modelopt'):
        raise ContractError('Invalid frozen model checkpoint')
    check_equal('frozen six-class IDs', dict(enumerate(NAMES)), model.names)
    layer = model.model[FEATURE_LAYER['index']]
    check_equal('frozen E1 feature layer', (FEATURE_LAYER['type'], STRIDE), (type(layer).__name__, int(model.stride.max())))
    # Match the native path-based backend: move FP32 to CUDA before fusion,
    # then convert to FP16. Do not send the path through its weight resolver.
    return model.float().eval().requires_grad_(False).to('cuda:0').fuse(verbose=False).half()


def capture(sandbox, manifest_sha, count, output):
    # Runtime paths are set by the supervisor to its external temporary folder.
    import cv2
    import torch
    import torchvision  # Select the same NMS backend as the locked validator.
    import ultralytics

    check_equal('locked capture dependencies', VERSIONS, (torch.__version__, torchvision.__version__, ultralytics.__version__))
    if not torch.cuda.is_available():
        raise ContractError('CUDA FP16 is required; refusing CPU fallback')
    if not output.is_dir() or any(output.iterdir()):
        raise ContractError('Capture needs a new empty output directory')
    pool = BlindPool(sandbox, manifest_sha, count)
    model = _open_model(torch)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    seen_feature = []
    def capture_layer(_layer, _inputs, result):
        if result.ndim != 4 or result.shape[:2] != (1, FEATURE_LAYER['channels']):
            raise ContractError('Frozen YOLO11m feature layer no longer has batch 1 with 512 channels')
        seen_feature.append(result.float().mean(dim=(2, 3))[0].detach().cpu().numpy())

    def infer(img, view, key):
        try:
            pixels, geom = rect_pixels(img)
            x = torch.from_numpy(pixels).unsqueeze(0).to(device='cuda:0', dtype=torch.float16).div_(255)
            return restore(native_detections(model(x)), geom, view == 'flip'), tuple(pixels.shape[1:])
        except ContractError as exc:
            raise ContractError(f'E1 capture {key} / {view}: {exc}') from exc

    features_path, scores_path = output / 'features.npz', output / 'scores.npz'
    # Initialize library lazy paths with a synthetic tensor before the blind
    # image boundary. This output does not enter scoring or any pool record.
    with torch.inference_mode():
        infer(np.full((320, 640, 3), 114, np.uint8), 'original', 'synthetic-warmup')
    np.savez_compressed(io.BytesIO(), value=np.ones(1))
    reads = worker_files(readable=[pool.root / r['image'] for r in pool.records.values()],
                         writable=[features_path, scores_path, output / 'capture.json'])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    z, scores, shapes, counts = [], [], {}, [0]*len(VIEWS)
    with torch.inference_mode():
        for key in pool.records:
            data = pool.read(key)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ContractError('Unreadable verified pool image: ' + key)
            handle = model.model[FEATURE_LAYER['index']].register_forward_hook(capture_layer)
            try:
                result, shape = infer(image, 'original', key)
            finally:
                handle.remove()
            shapes['x'.join(map(str, shape))] = shapes.get('x'.join(map(str, shape)), 0) + 1
            check_equal('original feature forwards', 1, len(seen_feature))
            v = seen_feature.pop().astype(np.float64)
            norm = np.linalg.norm(v)
            if not np.isfinite(norm) or norm <= 0:
                raise ContractError('Non-finite/zero frozen feature: ' + key)
            z.append(v/norm)
            sides = [result, *(infer(transform(image, name), name, key)[0] for name in VIEWS[1:])]
            counts = [a+len(b) for a, b in zip(counts, sides, strict=True)]
            scores.append(score_e1(sides))
    check_equal('captured allowed image opens', [pool.root / r['image'] for r in pool.records.values()], reads)
    keys = np.asarray(list(pool.records))
    for path, field, values in ((features_path, 'vectors', z), (scores_path, 'scores', scores)):
        with path.open('xb') as f:
            np.savez_compressed(f, keys=keys, **{field: np.asarray(values, dtype=np.float64)})
    info = {'purpose': 'blind E1 capture', 'reads': [x.relative_to(pool.root).as_posix() for x in reads],
            'manifest_read': POOL_MANIFEST, 'views': list(VIEWS), 'inference': INFERENCE, 'input_shapes': shapes,
            'box_counts_before_research_filter': counts, 'feature_layer': FEATURE_LAYER, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'libraries': list(VERSIONS), 'opencv': cv2.__version__,
            'numpy': np.__version__, 'seconds': time.perf_counter()-started,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
    # Exact non-overwriting path admitted by worker_files.
    with (output / 'capture.json').open('x', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sandbox', type=Path)
    p.add_argument('manifest_sha')
    p.add_argument('count', type=int)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
    capture(args.sandbox, args.manifest_sha, args.count, args.output)


if __name__ == '__main__':
    main()
