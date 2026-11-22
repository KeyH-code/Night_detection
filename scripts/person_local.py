"""Paired source-dev experiment: local pedestrian views and cyclist competition."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np

import detection_metrics as dm
import person_fusion as pf
import seed_eval as se
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json
from source_policy import SelectionPolicy, recall_intervals, reference_at, summarize

ROOT = pf.ROOT
BASE = ROOT/'runs/seed_002'
SEAL = 'aade22f3d66b1b9adb7c8cbe5ee5878469dcae8ae28f5736153ab198cb975d9d'
MODES = ('competition', 'local', 'both')
WEIGHTS = (.5, .25, .25)


def prepare():
    check_equal('seed_002 manifest', SEAL, sha256(BASE/'manifest.json'))
    rules, ct, arms, original, *_ = se.inputs()
    seal = read_json(BASE/'manifest.json')
    check_equal('seed_002 checkpoints', {str(k): v for k, v in arms.items()}, seal['arms'])
    check_equal('seed_002 dev manifest', ct['manifest'], seal['dev_manifest'])
    paths = {17: original, **{s: BASE/str(s)/'objects.jsonl.gz' for s in (29,43)}}
    for name, info in seal['files'].items():
        check_equal(name, info['sha256'], sha256(BASE/name))
    return rules, ct, arms, paths


def remap(boxes, scores, start, span, width, right, mirror):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1,4).copy()
    if mirror:
        pf.unmirror_boxes(boxes, span)
    boxes[:, [0,2]] += start
    # Fixed ownership at the midpoint avoids double votes in the overlap.
    owner = (boxes[:,0]+boxes[:,2] >= width) == right
    return [(float(score), box.tolist()) for score, box in zip(scores[owner], boxes[owner], strict=True)]


def capture(ct, arms, scratch):
    import cv2
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import nms

    class PersonPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds,(tuple,list)) else preds
            if raw.shape[1] != 10 or not raw.isfinite().all():
                raise ContractError('Invalid six-class decoder output')
            person = raw[:, :5].contiguous()
            if int((person[:,4] > pf.FLOOR).sum(1).max()) >= 30000:
                raise ContractError('Unaccounted pre-NMS cap')
            output = nms.non_max_suppression(person, pf.FLOOR, pf.NMS_IOU, nc=1,
                                              max_det=pf.MAX_DET, max_time_img=10.)
            if any(len(x) >= pf.MAX_DET for x in output):
                raise ContractError('Unaccounted crop output cap')
            return self.construct_results(output, img, orig_imgs)

    env = {'torch':torch.__version__, 'torchvision':torchvision.__version__, 'ultralytics':ultralytics.__version__,
           'python':sys.version, 'opencv':cv2.__version__, 'gpu':torch.cuda.get_device_name(0)}
    check_equal('locked inference packages', ('2.14.0+cu130','0.29.0+cu130','8.4.171'),
                tuple(env[k] for k in ('torch','torchvision','ultralytics')))
    torch.set_num_threads(4)
    settings = dict(imgsz=960, rect=True, quantize=16, conf=pf.FLOOR, iou=pf.NMS_IOU, device=0,
                    augment=False, verbose=False, save=False, max_det=pf.MAX_DET, project=str(scratch/'predict'))
    observations = {}
    for seed in (29,17,43):
        started = time.perf_counter()
        model = YOLO(str(ROOT/arms[seed]['path']))
        check_equal('model classes', NAMES, list(model.names.values()))
        tiles = []
        for record in ct['records']:
            im = cv2.imdecode(np.fromfile(ct['root']/record['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
            h,w = im.shape[:2]
            check_equal(record['key']+' geometry', (record['height'],record['width']), (h,w))
            span = round(.6*w)
            inputs = [np.ascontiguousarray(im[:, start:start+span][:, ::-1 if flip else 1])
                      for flip in (False,True) for start in (0,w-span)]
            batch = model.predict(inputs, predictor=PersonPredictor, **settings)
            check_equal('four tiles', 4, len(batch))
            views = [[],[]]
            for j,r in enumerate(batch):
                b = r.boxes.data.float().cpu().numpy()
                views[j//2].extend(remap(b[:,:4], b[:,4], (w-span)*(j%2), span, w, bool(j%2), j>=2))
            tiles.append({'key':record['key'], 'views':views})
        check_equal('complete tile keys', {r['key'] for r in ct['records']}, {r['key'] for r in tiles})
        check_equal('unchanged checkpoint', arms[seed]['sha256'], sha256(ROOT/arms[seed]['path']))
        dm.write_rows(scratch/f'{seed}.gz', tiles)
        observations[seed] = {'seconds':time.perf_counter()-started, 'images':len(tiles)}
        print('LOCAL_CAPTURE', seed, observations[seed], flush=True)
        del model, batch
        torch.cuda.empty_cache()
    return {'environment':env, 'settings':{k:v for k,v in settings.items() if k!='project'}, 'arms':observations}


def collect(scratch):
    scratch = scratch.resolve()
    if not scratch.is_dir() or scratch.is_relative_to(ROOT) or any(scratch.iterdir()):
        raise ContractError('Capture requires a new empty external scratch directory')
    _, ct, arms, _ = prepare()
    gpu = se.probe_gpu()
    with tempfile.TemporaryDirectory(dir=scratch) as config:
        for k in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','XDG_CONFIG_HOME'):
            os.environ[k] = config
        os.environ.update(YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
        obs = capture(ct, arms, scratch)
    write_json(scratch/'capture.json', {**obs,'gpu_initial':gpu,'baseline':SEAL,
               'implementation':sha256(Path(__file__)), 'files':{p.name:sha256(p) for p in scratch.glob('*.gz')}})


def fused_people(views):
    entries = [(score, box, i) for i, view in enumerate(views) for score, box in view if score>pf.FLOOR]
    if not entries:
        return []
    entries.sort(key=lambda e:-e[0])
    n = len(entries)
    centers = np.zeros((n,4), np.float32)
    sums = np.zeros_like(centers)
    used = np.zeros((n,3), bool)
    mass, square = np.zeros((2,n), np.float32)
    size = 0
    for s, b, view in entries:
        box = np.asarray(b,np.float32)
        overlaps = pf._iou(box,centers[:size]); overlaps[used[:size,view]] = -1
        j = int(np.argmax(overlaps)) if size else -1
        if j < 0 or overlaps[j] < pf.CLUSTER_IOU:
            j = size; size += 1
        w = WEIGHTS[view]*s
        mass[j] += w; sums[j] += w*box; centers[j] = sums[j]/mass[j]
        square[j] += w*s; used[j,view] = True
    scores = np.sqrt(square[:size]); idx = np.argsort(-scores,kind='stable')
    output = []
    while len(idx):
        j, rest = idx[0], idx[1:]
        output.append((float(scores[j]),centers[j].tolist()))
        idx = rest[pf._iou(centers[j],centers[rest]) <= pf.NMS_IOU]
    return output


def replace(row, people, competition):
    pred = row['pred']; keep = [i for i,c in enumerate(pred['cls']) if c!=0]
    if competition and people:
        cyclists = [(s,b) for c,s,b in pf._detections(pred) if c==1]
        if cyclists:
            s,b = zip(*cyclists,strict=True)
            penalty = (dm.box_iou([p[1] for p in people], b)**2*np.asarray(s)).max(axis=1)
            people = [(float(score*(1-d)),box) for (score,box),d in zip(people,penalty,strict=True)]
    people = sorted((p for p in people if p[0]>pf.FLOOR), key=lambda p:-p[0])
    cap = max(0,len(people)+len(keep)-pf.MAX_DET)
    people = people[:pf.MAX_DET-len(keep)]
    index = {k:j for j,k in enumerate(keep)}
    edges = [[g,index[p],v] for g,p,v in row['iou_edges'] if p in index]
    boxes = [b for _,b in people]
    mat = dm.box_iou(row['gt']['original_boxes'],boxes)
    edges.extend([int(g),len(keep)+int(p),float(mat[g,p])] for g,p in zip(*np.nonzero(mat),strict=True))
    ds = [d for d in pf._detections(pred) if d[0]!=0]+[(0,s,b) for s,b in people]
    joint = {k:[d[j] for d in ds] for j,k in enumerate(pf.FIELDS)}
    joint.update(ids=list(range(len(ds))),boxes=joint['original_boxes'])
    out = dict(row, pred=joint, iou_edges=edges)
    flags,pairs = dm.ap_flags(out); out.update(ap_tp=flags.tolist(), ap_pairs=pairs)
    return out,cap


def diagnose(rows,matches,curve,rule):
    pos,small,fp,other = (Counter() for _ in range(4))
    cases = []
    for row,match in zip(rows,matches,strict=True):
        check_equal('diagnosis order',row['key'],match['key'])
        gc,pc,s,iou = dm.arrays(row); m = match['classes']['pedestrian']; tags = dm.slice_tags(row)
        for g in m['fn_ids']:
            ix = np.flatnonzero((pc==0)&(iou[g]>=.5)); top = iou[g,pc==0].max(initial=0)
            score = float(s[ix].max()) if len(ix) else None
            kind = ('assignment' if score >= m['threshold'] else 'below_threshold') if len(ix) else (
                    'localization' if top>=.1 else 'no_close_output')
            pos[kind]+=1
            if 'area_S' in tags[g] or 'width_le8' in tags[g]: small[kind]+=1
            cases.append([row['key'],g,kind,score,float(top)])
        for item in m['fp_items']:
            fp[item['type']]+=1
            if item['type']=='classification': other[NAMES[gc[item['best_gt']]]]+=1
    r = curve['pedestrian_work_recall']; p = curve['pedestrian_work_precision']
    feasible = np.flatnonzero(r>=rule['recall'])
    best = feasible[np.argmax(p[feasible])] if len(feasible) else None
    return {'fn':pos,'small_thin_fn':small,'fp':fp,'wrong_classes':other,'fn_detail':cases,
            'best_precision_at_target_recall':None if best is None else {k:curve['pedestrian_work_'+k][best].item()
                       for k in ('threshold','tp','fp','fn','precision','recall')}}


def transitions(rows,old,new):
    gains, losses = [], []
    group = {}
    for r,a,b in zip(rows,old,new,strict=True):
        check_equal('transition keys',(r['key'],r['key']),(a['key'],b['key']))
        prior,now = [{g for g,_ in m['classes']['pedestrian']['pairs']} for m in (a,b)]
        tags = dm.slice_tags(r)
        for diff,out in ((now-prior,gains),(prior-now,losses)):
            out.extend([r['key'],g,('area_S' in tags[g] or 'width_le8' in tags[g])] for g in sorted(diff))
        pair = group.setdefault(r['group_id'],[0,0]); pair[0] += len(now)-len(prior); pair[1] += r['gt']['cls'].count(0)
    return {'gained':gains,'lost':losses}, group


def paired_interval(groups):
    keys = sorted(groups[0]); vals = np.asarray([[d[k] for k in keys] for d in groups])
    draws = np.stack([v for _,v in dm.cluster_draws(keys,2000,20261026)])
    totals = np.einsum('rg,sgt->srt',draws,vals)
    delta = np.mean(totals[:,:,0]/totals[:,:,1],axis=0)
    return {'mean_delta':float(np.mean(vals[:,:,0].sum(1)/vals[:,:,1].sum(1))),
            'lower':float(np.quantile(delta,.025)), 'upper':float(np.quantile(delta,.975)),
            'groups':len(keys),'repeats':len(draws),'rng_seed':20261026,'fixed_workpoints':True}


def assess(scratch,out):
    out = out.resolve(); scratch = scratch.resolve()
    if out.parent!=ROOT/'runs' or not re.fullmatch(r'views_\d{3}',out.name) or out.exists() or scratch.is_relative_to(ROOT):
        raise ContractError('Expected new runs/views_NNN and external tile cache')
    rules,ct,arms,paths = prepare(); policy = SelectionPolicy(rules,reference_at(ROOT))
    obs = read_json(scratch/'capture.json'); check_equal('capture baseline', SEAL,obs['baseline'])
    check_equal('capture seeds', {f'{s}.gz' for s in se.SEEDS},set(obs['files']))
    checked = {r['key']:r for r in ct['records']}
    metrics,groups = {m:{} for m in MODES},{m:{} for m in MODES}
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        stage = Path(tmp); diagnosis = {}; screen = {}; warnings = {}
        for seed in se.SEEDS:
            base = dm.read_rows(paths[seed]); bidx = pf._dev_index(base)
            check_equal('exact dev keys',set(checked),set(bidx))
            for k,r in bidx.items():
                for field in ('image_sha256','label_sha256','width','height','group_id'):
                    check_equal(field,checked[k][field],r[field])
            old = read_json(BASE/str(seed)/'metrics.json'); matches = dm.read_rows(BASE/str(seed)/'matches.jsonl.gz')
            with np.load(BASE/str(seed)/'pr.npz') as curves:
                diagnosis[seed] = diagnose(base,matches,curves,rules['classes']['pedestrian'])
            cache = scratch/f'{seed}.gz'; check_equal('tile cache '+str(seed),obs['files'][cache.name],sha256(cache))
            tile_rows = dm.read_rows(cache); tiles = {r['key']:r['views'] for r in tile_rows}
            check_equal('tile coverage',(500,set(checked)),(len(tile_rows),set(tiles)))
            full = [[(s,b) for c,s,b in pf._detections(r['pred']) if c==0] for r in base]
            local = [fused_people((p,*tiles[r['key']])) for r,p in zip(base,full,strict=True)]
            screen[seed] = {}; warnings[seed] = {}
            for mode in MODES:
                results = [replace(r,p,mode!='local') for r,p in zip(base,full if mode=='competition' else local,strict=True)]
                rows = [r for r,_ in results]; record, after, curves = summarize(rows,rules)
                checks,guards = policy.assess(record); effect, groups[mode][seed] = transitions(rows,matches,after)
                for name in NAMES[1:]: check_equal('protected workpoint '+name,old['workpoints'][name],record['workpoints'][name])
                for a,b in zip(base,rows,strict=True):
                    others = lambda t:[x for x in pf._detections(t['pred']) if x[0]!=0]
                    check_equal('non-person outputs '+a['key'],others(a),others(b))
                intervals = recall_intervals(rows,after)['intervals']
                record.pop('nms'); record.update(checks=checks, guards=guards, intervals=intervals,
                    metric_scope='complete source_dev; seed_002 nonperson outputs; pedestrian local/competition ablation',
                    cap_removed=sum(n for _,n in results),transition=effect,
                    paired_recall=paired_interval([groups[mode][seed]]))
                prev = {r['slice']:r['recall50'] for r in old['slices'] if r['class']=='pedestrian'}
                warnings[seed][mode]=[r['slice'] for r in record['slices'] if r['class']=='pedestrian'
                    and r['slice'] in ('area_S','width_le8','small_thin_union') and prev[r['slice']]-r['recall50']>rules['slice_drop']]
                if old['metrics']['per_class']['pedestrian']['ap75']-record['metrics']['per_class']['pedestrian']['ap75']>rules['ap75_drop']:
                    warnings[seed][mode].append('ap75')
                screen[seed][mode]=record
                metrics[mode][seed]=record
                folder=stage/mode/str(seed); folder.mkdir(parents=True)
                dm.write_rows(folder/'objects.jsonl.gz',rows); dm.write_rows(folder/'matches.jsonl.gz',after)
                np.savez_compressed(folder/'pr.npz',**curves)
                print('LOCAL_RESULT',seed,mode,record['metrics']['per_class']['pedestrian']['ap50'],record['workpoints']['pedestrian'],flush=True)
        summary = {m:se.summarize_seeds(metrics[m]) for m in MODES}
        def eligible(m):
            return all(not any(not k.startswith('tricycle/') for k in summary[m]['failed_by_seed'][s]) for s in se.SEEDS) and not any(
                sum(tag in warnings[s][m] for s in se.SEEDS)>=2 for tag in ('area_S','width_le8','small_thin_union','ap75'))
        selected = next((m for m in MODES if eligible(m)),None)
        deliver = stage/'result'; deliver.mkdir()
        if selected:
            for seed in se.SEEDS: shutil.move(str(stage/selected/str(seed)),deliver/str(seed))
        write_json(deliver/'diagnosis.json',diagnosis); write_json(deliver/'screen.json',screen)
        write_json(deliver/'comparison.json',{'selected':selected,'modes':summary,'warnings':warnings,
            'conditional_paired_recall':{m:paired_interval([groups[m][s] for s in se.SEEDS]) for m in MODES}})
        snapshot = deliver/'code'; snapshot.mkdir(); shutil.copyfile(__file__,snapshot/'person_local.py')
        write_json(deliver/'manifest.json',{'status':'complete','scope':'source_dev','images':500,'arms':arms,
            'seed_002_sha256':SEAL,'ready':pf.READY_SHA,'dev_manifest':ct['manifest'],'rules':rules['sha256'],
            'observations':obs,'plan':'specs/s3/scale.md#行人局部视图与双报','cloud_cny':0,'training_performed':False,
            'final_test_executed':False,'code_sha256':{p:sha256(ROOT/'scripts'/p) for p in (
            'person_local.py','person_fusion.py','seed_eval.py','evaluate.py','source_policy.py','source_contract.py','detection_metrics.py')},
            'selected':selected,'files':{p.relative_to(deliver).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
                                         for p in deliver.rglob('*') if p.is_file()}})
        shutil.copytree(deliver,out)
    print('SEALED',out,selected,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--capture', type=Path); modes.add_argument('--evaluate', type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/views_002')
    args = parser.parse_args()
    if args.capture: collect(args.capture)
    else: assess(args.evaluate,args.output)
