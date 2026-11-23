"""S6 random-only local adaptation and measured development evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

import detection_metrics as dm
from evaluate import verify_subset
from night_contract import ROOT, _path, read_sealed_bytes, read_sealed_json, verify_dev
from night_eval import load_protocol, recovery
from source_contract import ContractError, NAMES, canonical_hash, check_equal, read_json, sha256, write_json

SOURCE = ROOT / 'data/soda10m/splits/source_v1'
NIGHT = ROOT / 'data/soda10m/splits/night_v1'
BASE = ROOT / 'runs/night_001'
CHECKPOINT = ROOT / 'runs/align_001/17/last.pt'
FIELDS = ('key', 'group_id', 'image', 'image_sha256')
LIBS = ('torch', 'torchvision', 'ultralytics')


def schedule(seed, day, night, *, salt=20261122, flip=.5, brightness=(.85, 1.15)):
    if len(night) != 100 or len(set(night)) != 100 or len(day) < 2000 or len(set(day)) != len(day):
        raise ContractError('Schedule needs 100 unique night keys and >=2000 unique source_train keys')
    if set(day) & set(night):
        raise ContractError('Source and night keys must be disjoint')
    streams = [np.random.default_rng(np.random.SeedSequence([seed, salt, i])) for i in range(4)]
    day_ids = streams[0].permutation(sorted(day))[:2000].tolist()
    night_ids = [k for _ in range(20) for k in streams[1].permutation(night).tolist()]
    def transform(keys, rng):
        return [(k, bool(rng.random() < flip), float(rng.uniform(*brightness))) for k in keys]
    return list(zip(transform(day_ids, streams[2]), transform(night_ids, streams[3]), strict=True))


def selected_records(locked, managed, *, count=100):
    selected = {r['key']: r for r in locked}
    check_equal('locked night count', (count, count), (len(locked), len(selected)))
    found = {}
    for row in managed:
        key = row['key']
        if key not in selected:
            continue
        if key in found:
            raise ContractError('Duplicate selected key in frozen management metadata')
        check_equal('selected image identity', selected[key], {k: row[k] for k in FIELDS})
        check_equal('selected night purpose', 'night_pool', row['purpose'])
        name = PurePosixPath(row['image']).name
        check_equal('selected night image path', 'images/night_pool/' + name, row['image'])
        check_equal('selected night label pairing', 'hidden_ground_truth/night_pool/' + PurePosixPath(name).with_suffix('.txt').name,
                    row['label'])
        found[key] = {k: row[k] for k in (*FIELDS, 'purpose', 'label', 'label_sha256', 'width', 'height', 'classes')}
    check_equal('selected keys exactly matched', set(selected), set(found))
    return [found[r['key']] for r in locked]


def _labels(content):
    rows = []
    for line in content.decode('utf-8').splitlines():
        fields = line.split()
        try:
            c, x, y, w, h = int(fields[0]), *map(float, fields[1:])
        except (IndexError, ValueError) as exc:
            raise ContractError('Invalid YOLO label row') from exc
        if c not in range(len(NAMES)) or not (0 < w <= 1 and 0 < h <= 1 and
                -1e-8 <= x-w/2 <= x+w/2 <= 1+1e-8 and -1e-8 <= y-h/2 <= y+h/2 <= 1+1e-8):
            raise ContractError('Invalid label geometry or class')
        rows.append((c, x, y, w, h))
    return rows


def training_rows(root, records, purpose):
    if (purpose not in ('source_train', 'night_pool') or not records or
            any(r['purpose'] != purpose for r in records)):
        raise ContractError('Forbidden gradient purpose')
    payload = {}
    for r in records:
        key = r['key']
        if key in payload:
            raise ContractError('Duplicate allowed gradient image')
        check_equal('staged gradient payload paths', (f'images/{key.replace(":", "_")}.jpg', f'labels/{key.replace(":", "_")}.txt'),
                    (r['image'], r['label']))
        label = read_sealed_bytes(root, r['label'], r['label_sha256'], [])
        image = _path(root, r['image'])
        check_equal('staged training image hash', r['image_sha256'], sha256(image))
        parsed = _labels(label)
        check_equal('six-class label semantics', sorted(r['classes']), sorted({c for c, *_ in parsed}))
        payload[key] = image, r['width'], r['height'], parsed
    return payload


def fit_image(image, labels, flip, gain, side):
    import cv2
    h, w = image.shape[:2]
    scale = side / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (side-nw)//2, (side-nh)//2
    pixels = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lut = np.rint(np.clip(np.arange(256) * gain, 0, 255)).astype(np.uint8)
    pixels = cv2.LUT(pixels, lut)
    pixels = cv2.copyMakeBorder(pixels, top, side-nh-top, left, side-nw-left, cv2.BORDER_CONSTANT, value=(114,)*3)
    if flip:
        pixels = pixels[:, ::-1]
    boxes = []
    for _, x, y, bw, bh in labels:
        cx = (x*nw+left)/side
        boxes.append((1-cx if flip else cx, (y*nh+top)/side, bw*nw/side, bh*nh/side))
    return np.ascontiguousarray(pixels[:, :, ::-1].transpose(2, 0, 1)), np.asarray(boxes, np.float32).reshape(-1, 4)


def _stage(source, dest, records):
    selected, reads = [], []
    for r in records:
        key = r['key']
        if not re.fullmatch(r'(?:train|val):[0-9]+', key):
            raise ContractError('Invalid frozen image key')
        out = {k: r[k] for k in ('key', 'purpose', 'group_id', 'width', 'height', 'classes', 'image_sha256', 'label_sha256')}
        for kind, ext in (('image', 'jpg'), ('label', 'txt')):
            out[kind] = f'{kind}s/{key.replace(":", "_")}.{ext}'
            path = dest / out[kind]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(read_sealed_bytes(source, r[kind], r[kind+'_sha256'], reads))
        selected.append(out)
    return selected, reads


def _prepare_night(seed, dest, cfg):
    draw = read_sealed_json(ROOT / 'runs/pool_001', 'random.json', cfg['random_sha256'], [])
    protocol = load_protocol()
    check_equal('approved evaluation', cfg['evaluation_sha256'], protocol['sha256'])
    check_equal('image-only random input', protocol['data']['pool_images_sha256'], draw['image_manifest_sha256'])
    arm = draw['arms'][str(seed)]
    check_equal('locked random arm', arm['record_sha256'], canonical_hash(arm['records']))
    ready = read_sealed_json(NIGHT, 'READY.json', protocol['data']['night_ready_sha256'], [])
    entry = ready['held_out']['night_pool']
    check_equal('management receipt', ('manifests/night_pool.json', 994), (entry['path'], entry['count']))
    managed = read_sealed_json(NIGHT, entry['path'], entry['sha256'], [])
    check_equal('management scope', ('night_pool', 994), (managed['purpose'], len(managed['records'])))
    rows = selected_records(arm['records'], managed['records'])
    staged, reads = _stage(NIGHT, dest, rows)
    check_equal('selected pool payload access', [r[k] for r in rows for k in ('image', 'label')], reads)
    write_json(dest/'inputs.json', {'records': staged, 'source_reads': reads, 'management_sha256': entry['sha256'],
                                   'random_arm_sha256': arm['record_sha256']})


def _sample_gpu():
    line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], encoding='utf-8', timeout=4).splitlines()
    if len(line) != 1:
        raise ContractError('Expected exactly one local NVIDIA GPU')
    util, used, total = [int(v.strip()) for v in line[0].split(',')]
    return [util, used, total]


@contextmanager
def telemetry():
    start = time.monotonic()
    samples, errors = [[0., *_sample_gpu()]], []
    stop = threading.Event()
    def watch():
        while not stop.wait(2):
            try:
                samples.append([round(time.monotonic()-start, 3), *_sample_gpu()])
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                errors.append(str(exc))
    thread = threading.Thread(target=watch, daemon=True)
    log = {'columns': ['seconds', 'gpu_util_pct', 'device_memory_used_mib', 'device_memory_total_mib'],
           'samples': samples, 'errors': errors, 'interval_seconds': 2, 'device_wide': True}
    thread.start()
    try:
        yield log
    finally:
        stop.set()
        thread.join()
        log['seconds'] = time.monotonic()-start
        active = samples[1:] or samples
        log['mean_sampled_gpu_util_pct'] = sum(row[1] for row in active) / len(active)
        log['peak_sampled_device_memory_mib'] = max(row[2] for row in samples)


def _batch(torch, cv2, payload, planned, side):
    images, classes, boxes, batches = [], [], [], []
    for i, (key, flip, gain) in enumerate(planned):
        p, w, h, labels = payload[key]
        image = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        check_equal('training image dimensions', (h, w), None if image is None else image.shape[:2])
        pixels, transformed = fit_image(image, labels, flip, gain, side)
        images.append(pixels)
        classes.extend(row[0] for row in labels)
        boxes.extend(transformed)
        batches.extend([i]*len(labels))
    return {'img': torch.from_numpy(np.stack(images)).cuda().float().div_(255),
            'cls': torch.tensor(classes, device='cuda', dtype=torch.float32).reshape(-1, 1),
            'bboxes': torch.as_tensor(np.asarray(boxes, np.float32).reshape(-1, 4), device='cuda'),
            'batch_idx': torch.tensor(batches, device='cuda', dtype=torch.float32)}


def _fit(torch, cv2, payload, plan, seed, cfg, folder, limit, until, *, preview=False):
    from torch.nn.modules.batchnorm import _BatchNorm
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import ModelEMA

    started = time.monotonic()
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    m, _ = load_checkpoint(str(CHECKPOINT), device='cuda')
    m.train(); args = dict(m.args); m.args = SimpleNamespace(**args)
    check_equal('six class detector', NAMES, list(m.names.values()))
    for name, p in m.named_parameters():
        p.requires_grad_('.dfl.' not in name)
    bn = {}
    for name, layer in m.named_modules():
        if isinstance(layer, _BatchNorm):
            layer.eval().requires_grad_(False)
            bn.update({f'{name}.{k}': v.detach().clone() for k, v in layer.state_dict().items()})
    wd, free = [], []
    for layer in m.modules():
        for name, p in layer.named_parameters(recurse=False):
            if p.requires_grad:
                (wd if isinstance(layer, torch.nn.Conv2d) and name == 'weight' else free).append(p)
    optim = torch.optim.SGD([{'params': wd, 'weight_decay': cfg['weight_decay']}, {'params': free, 'weight_decay': 0.}],
                            lr=cfg['lr_start'], momentum=cfg['momentum'])
    scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    ema, crit = ModelEMA(m), v8DetectionLoss(m)
    acc, trace, seen = cfg['accumulate'], [], []
    for step in range(limit):
        if time.monotonic() >= until:
            raise ContractError('Local run exceeded its time budget')
        lr = cfg['lr_end'] + (cfg['lr_start'] - cfg['lr_end']) * (1 + math.cos(math.pi*step/(cfg['updates']-1))) / 2
        for group in optim.param_groups:
            group['lr'] = lr
        optim.zero_grad(set_to_none=True)
        sums = torch.zeros(3, device='cuda')
        for micro in plan[step*acc:(step+1)*acc]:
            b = _batch(torch, cv2, payload, micro, cfg['size'])
            seen.append([item[0] for item in micro])
            with torch.autocast('cuda', dtype=torch.float16):
                _, values, _ = crit.get_assigned_targets_and_loss(m(b['img']), b)
                loss = values.sum() / acc
            if not loss.isfinite():
                raise ContractError('Nonfinite mixed-domain training loss')
            scaler.scale(loss).backward()
            sums += values.detach().float()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(m.parameters(), cfg['clip'], error_if_nonfinite=True)
        scale = scaler.get_scale()
        scaler.step(optim); scaler.update()
        if scaler.get_scale() < scale:
            raise ContractError('AMP skipped optimizer update')
        ema.update(m)
        trace.append({'step': step+1, 'lr': lr, 'loss_terms': (sums/acc).cpu().tolist()})
        if (step+1) % 10 == 0 or (preview and step+1 == 2):
            print(f'{"preflight" if preview else "train"} seed={seed} updates={step+1}/{limit}', flush=True)
    check_equal('observed update count', limit, ema.updates)
    check_equal('exact gradient consumption', [[m[0][0],m[1][0]] for m in plan[:limit*acc]], seen)
    for k, v in bn.items():
        if not torch.equal(m.state_dict()[k], v):
            raise ContractError('Frozen BN changed')
        ema.ema.state_dict()[k].copy_(v)
    # Verify a complete optimizer/EMA state can be round-tripped in preflight only.
    if preview:
        snap = folder/'resume.pt'
        torch.save({'model':m.state_dict(),'ema':ema.ema.state_dict(),'optimizer':optim.state_dict(),
                    'scaler':scaler.state_dict(),'updates':ema.updates}, snap)
        sample = torch.load(snap,map_location='cpu',weights_only=True)
        check_equal('preflight serialized update count', limit, sample['updates'])
        if not all(torch.equal(v.detach().cpu(), sample['model'][k]) for k,v in m.state_dict().items()):
            raise ContractError('Preflight training snapshot values differ')
        del sample
        snap.unlink()
    last = folder/'last.pt'
    model = deepcopy(ema.ema).cpu().half().eval(); model.args = args
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ContractError('Nonfinite packaged EMA')
    torch.save({'ema': model, 'model':None, 'epoch': limit//10-1, 'updates':limit,
                'train_args':args, 'scope':'fixed last mixed adaptation'}, last)
    restored,_ = load_checkpoint(str(last),device='cpu')
    if not all(torch.equal(v.float(), restored.state_dict()[k].float()) for k,v in model.state_dict().items()):
        raise ContractError('EMA reloaded values differ')
    summaries = [{k: dict(Counter(r[i] for r in seen)) for k,i in [('source',0),('night',1)]}]
    return last, dict(updates=limit, skipped_updates=0, bn_frozen_buffers=len(bn), trace=trace,
                      microbatches=seen, augment_schedule_sha256=canonical_hash(plan[:limit*acc]),
                      counts=summaries[0], seconds=time.monotonic()-started,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                      last_sha256=sha256(last), role='fixed_last_ema', ema_reload_verified=True)


def _receipt(folder):
    return {p.relative_to(folder).as_posix(): {'sha256': sha256(p),'bytes':p.stat().st_size}
            for p in sorted(folder.rglob('*')) if p.is_file() and p != folder/'manifest.json'}


def _guard_manifest(folder, seal):
    m=read_sealed_json(folder, 'manifest.json', seal, [])
    check_equal('sealed baseline artifacts',m['files'],_receipt(folder))


def _paired(rows, old, point, old_point, protocol):
    groups = sorted({r['group_id'] for r in rows}); ids={k:i for i,k in enumerate(groups)}
    check_equal('paired evaluation keys', {r['key'] for r in rows}, {r['key'] for r in old})
    old = {r['key']:r for r in old}
    a = np.zeros((len(groups),2,3),dtype=np.int64)
    for row in rows:
        ref=old[row['key']]; check_equal('paired GT identity', (row['group_id'],row['gt']['cls'],row['gt']['original_boxes']),
                                        (ref['group_id'],ref['gt']['cls'],ref['gt']['original_boxes']))
        for j,c in enumerate((0,2)):
            new,_,_=dm.image_point(row,c,point); prev,_,_=dm.image_point(ref,c,old_point)
            a[ids[row['group_id']],j] += (new['tp'],prev['tp'],new['tp']+new['fn'])
    values=[]; iv=protocol['interval']
    for _, mult in dm.cluster_draws(groups,iv['repeats'],iv['seed']):
        total=np.tensordot(mult,a,axes=1)
        if np.all(total[:,2]): values.append((total[:,0]-total[:,1])/total[:,2])
    total=a.sum(axis=0); arr=np.asarray(values)
    return {NAMES[c]:{'delta':float((total[i,0]-total[i,1])/total[i,2]),
                      'interval_95':np.quantile(arr[:,i],[.025,.975]).tolist() if len(arr)>=2 else None,
                      'groups':len(groups),'valid_draws':len(arr)} for i,c in enumerate((0,2))}


def _evaluate(torch, checkpoint, path, tmp, ct, daytime, protocol, reference, baseline, old_rows):
    from night_run import _capture, _collect, save_predictions, LOCKED_VIEWS
    from evaluate import stage_subset
    import person_fusion as pf
    from night_eval import _five, evaluate_rows
    path.mkdir()
    with telemetry() as night_gpu:
        rows, views, caps = _collect(torch, ct, checkpoint, tmp/'night', 'night_dev')
        target = path/'night'; target.mkdir()
        night = save_predictions(target, rows, {r['key'] for r in ct['records']}, protocol, reference)
    p, q = (x['workpoints']['primary'] for x in (night, baseline))
    comparison = _paired(rows, old_rows, p['threshold'], q['threshold'], protocol)
    use_nms = ('pre_nms_truncated', 'max_det_truncated')
    if caps or any(v['nms'][k] for v in views for k in use_nms):
        night['review_reasons'].append('output_truncation_review')
    with telemetry() as day_gpu:
        tmp_day=tmp/'day'; tmp_day.mkdir(); data=stage_subset(daytime,tmp_day/'data'); captures=[]; day_views=[]
        for size,mirror in LOCKED_VIEWS:
            rows, obs=_capture(torch,daytime,data,checkpoint,tmp_day,size,mirror)
            captures.append(rows); day_views.append(obs)
        fused, day_caps=pf.evaluate_rows(captures[0],captures[1:])
        fused=[_five(r) for r in fused]
        day, curves=evaluate_rows(fused,protocol)
        native, _=evaluate_rows(captures[0],protocol)
        m=day['workpoints']['primary']; matches=[dm.match_record(r,{name:{'threshold':m['threshold']} for name in NAMES}) for r in fused]
        small=[s for s in dm.slice_statistics(fused,matches,'source_dev') if s['class']=='pedestrian']
        target=path/'day'; target.mkdir()
        dm.write_rows(target/'objects.jsonl.gz',fused); np.savez_compressed(target/'pr.npz',**curves)
        write_json(target/'metrics.json',dict(day,slices=small,native_640=native['per_class']))
    drops={n:day['per_class'][n]['ap50_95']-reference['per_class'][n]['ap50_95'] for n in ('pedestrian','car')}
    write_json(path/'evaluation.json', {'night_views':views,'night_fusion_caps':caps,'night_gpu':night_gpu,
                                        'day_views':day_views,'day_fusion_caps':day_caps,'day_gpu':day_gpu,
                                        'night_review_reasons':night['review_reasons'],'night_delta_vs_A':comparison,
                                        'white_day_ap50_95_delta':drops,'white_day_pass':all(v>=-.02 for v in drops.values()),
                                        'recovery':night['zero_shot_comparison']})


def _run(cfg, cfg_sha, scratch, out):
    start=time.monotonic(); until=start+60*60
    protocol=load_protocol(); check_equal('frozen evaluation protocol',cfg['evaluation_sha256'],protocol['sha256'])
    check_equal('authorized update budgets',(640,10,200,10,[17,29,43]),
                (cfg['size'],cfg['accumulate'],cfg['updates'],cfg['preflight_updates'],cfg['seeds']))
    check_equal('frozen A SHA',protocol['source_a']['checkpoint_sha256'],sha256(CHECKPOINT))
    _guard_manifest(BASE,cfg['baseline_manifest_sha256'])
    reference, baseline = (read_json(BASE/f) for f in ('reference.json','metrics.json'))
    old_rows=dm.read_rows(BASE/'objects.jsonl.gz')
    day=verify_subset(SOURCE,protocol['data']['source_ready_sha256'],'source_train')
    dev=verify_subset(SOURCE,protocol['data']['source_ready_sha256'],'source_dev')
    night=verify_dev(NIGHT,protocol['data']['night_ready_sha256'])
    check_equal('split coverage',(4000,500,331),(len(day['records']),len(dev['records']),len(night['records'])))
    from seed_eval import probe_gpu
    gpu=probe_gpu()
    out.mkdir(exist_ok=False)
    shutil.copyfile(ROOT/'configs/adapt.json',out/'protocol.json')
    code_names=('night_train.py','night_run.py','night_eval.py','night_contract.py','evaluate.py','detection_metrics.py',
                'person_fusion.py','seed_eval.py','source_contract.py')
    (out/'code').mkdir()
    for name in code_names: shutil.copyfile(ROOT/'scripts'/name,out/'code'/name)
    env={name:version(name) for name in LIBS}
    check_equal('local library identity',['2.14.0+cu130','0.29.0+cu130','8.4.171'],list(env.values()))
    identity={'scope':'S6 D only','config_sha256':cfg_sha,'A_sha256':sha256(CHECKPOINT),'base_sha256':cfg['baseline_manifest_sha256'],
              'source_train':day['manifest'],'source_dev':dev['manifest'],'night_dev':night['manifest'],
              'python':sys.version,'versions':env,'gpu_start':gpu,'started':time.strftime('%Y-%m-%d %H:%M:%S %z'),
              'strategy_executed':False,'final_test_executed':False,'cloud_used':False}
    completed=[]; failure=None
    try:
        with tempfile.TemporaryDirectory(prefix='d-',dir=scratch) as dr:
            tmp=Path(dr)
            for var in ('YOLO_CONFIG_DIR','TORCH_HOME','MPLCONFIGDIR','TEMP','TMP','TMPDIR','XDG_CONFIG_HOME'):
                path=tmp/var.lower(); path.mkdir(); os.environ[var]=str(path)
            os.environ.update(YOLO_AUTOINSTALL='false',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            import cv2, torch, torchvision
            torch.set_num_threads(4); cv2.setNumThreads(1)
            identity['gpu']=torch.cuda.get_device_name()
            day_by={r['key']:r for r in day['records']}
            for seed in cfg['seeds']:
                task=tmp/str(seed); task.mkdir(); nt=task/'night-input'; nt.mkdir()
                subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'prepare','--seed',str(seed),
                                '--scratch',str(nt),'--config-sha256',cfg_sha],check=True)
                source_receipt=read_json(nt/'inputs.json')
                night_records=source_receipt['records']; plan=schedule(seed,day_by,[r['key'] for r in night_records],
                    salt=cfg['schedule_salt'],flip=cfg['flip_probability'],brightness=cfg['brightness'])
                phase=out/str(seed); phase.mkdir()
                day_staged=task/'day-input'; day_staged.mkdir()
                stage_rows, _ = _stage(SOURCE,day_staged,[day_by[d[0]] for d,n in plan])
                payload={**training_rows(day_staged,stage_rows,'source_train'),**training_rows(nt,night_records,'night_pool')}
                check_equal('gradient payload covers both domains',2100,len(payload))
                check_equal('gradient and dev risk groups disjoint',set(),
                            {r['group_id'] for r in stage_rows+night_records}&{r['group_id'] for r in dev['records']+night['records']})
                public_receipt={k:v for k,v in source_receipt.items() if k!='records'}
                # Exact source identities can be reconstructed from the immutable source manifest and these ordered keys.
                write_json(phase/'inputs.json',dict(public_receipt,night=night_records,source_keys=[d[0] for d,n in plan]))
                if seed==17:
                    box=task/'preflight'; box.mkdir()
                    with telemetry() as pre_gpu:
                        _, pre=_fit(torch,cv2,payload,plan,seed,cfg,box,cfg['preflight_updates'],until,preview=True)
                    estimate=pre['seconds']*cfg['updates']/cfg['preflight_updates']*3 + 720
                    write_json(out/'preflight.json',{'training':pre,'gpu':pre_gpu,'rough_total_seconds':estimate,
                               'status':'passed','used_in_formal':False})
                    if estimate>until-time.monotonic():
                        raise ContractError('Measured local preflight projects past the 60-minute ceiling')
                    print(f'Preflight passed: {pre["seconds"]:.1f}s, projected total with evaluation allowance {estimate/60:.1f} min',flush=True)
                torch.cuda.empty_cache(); probe_gpu()
                with telemetry() as train_gpu:
                    weight, train=_fit(torch,cv2,payload,plan,seed,cfg,phase,cfg['updates'],until)
                write_json(phase/'train.json',dict(train,gpu=train_gpu,seed=seed,seed_scope='common-A fixed schedule',
                                                   checkpoint_roles={'last':'last.pt','best':{'path':'last.pt','selected_on_dev':False}}))
                del payload
                torch.cuda.empty_cache()
                torch.use_deterministic_algorithms(False)
                eval_tmp=task/'eval'; eval_tmp.mkdir()
                _evaluate(torch,weight,phase/'dev',eval_tmp,night,dev,protocol,reference,baseline,old_rows)
                completed.append(seed)
                print(f'Seed {seed} sealed: {train["seconds"]:.1f}s training plus dev evaluation',flush=True)
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(out/'run.json',dict(identity,status='complete' if completed==cfg['seeds'] else 'incomplete',
                                     completed_seeds=completed,error=failure,seconds=time.monotonic()-start))
        write_json(out/'manifest.json',{'status':'sealed','files':_receipt(out)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('run','prepare')); p.add_argument('--config-sha256',required=True)
    p.add_argument('--scratch',type=Path,required=True); p.add_argument('--output',type=Path); p.add_argument('--seed',type=int)
    a=p.parse_args(); scratch=a.scratch.absolute()
    if not scratch.is_dir() or scratch.resolve()!=scratch or scratch.is_relative_to(ROOT):
        p.error('scratch must be an existing clean external path')
    raw=read_sealed_bytes(ROOT,'configs/adapt.json',a.config_sha256,[]); cfg=json.loads(raw)
    if a.mode=='prepare':
        if a.seed not in cfg['seeds'] or a.output or any(scratch.iterdir()): p.error('prepare requires one locked seed and empty scratch')
        _prepare_night(a.seed,scratch,cfg)
        return
    if not a.output or a.seed is not None: p.error('run needs a new output, no seed override')
    out=a.output.absolute()
    if out.parent!=ROOT/'runs' or out.parent.resolve()!=out.parent or out.exists() or out.is_symlink() or not re.fullmatch(r'd_\d{3}',out.name):
        p.error('run output must be a new runs/d_NNN')
    _run(cfg,a.config_sha256,scratch,out)


if __name__=='__main__':
    main()
