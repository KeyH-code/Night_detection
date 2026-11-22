"""Native source-dev evaluation and paired contrasts of two training scales."""
from __future__ import annotations

import gzip
import math
from pathlib import Path
import tempfile
import time

import numpy as np

import detection_metrics as dm
from evaluate import evidence_validator_class, validator_args
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, write_json
from source_policy import load_rules, summarize
from train_scale import SIZES, evaluation_sizes, observation_checkpoint, observation_dir, run_observations

ROOT = Path(__file__).resolve().parents[1]


def raw_strict_coverage(raw, row, _):
    import torch
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou
    check_equal('expected decoded box and six class channels', 10, raw.shape[0])
    ped = [i for i, c in enumerate(row['gt']['cls']) if c==0]
    if not ped:
        return {}
    boxes = ops.xywh2xyxy(raw[:4].T).float()
    refs = torch.tensor([row['gt']['boxes'][i] for i in ped], device=raw.device, dtype=torch.float32)
    iou = box_iou(refs, boxes)
    ped_scores = raw[4].float().expand(len(ped), -1)
    score = ped_scores.masked_fill(iou<.75, -1).amax(dim=1).cpu().tolist()
    above_floor = ((iou>=.75) & (raw[4]>.001)).any(dim=1).cpu().tolist()
    return {g: [s if s>=0 else None, k] for g,s,k in zip(ped, score, above_floor, strict=True)}


def coverage(rows, traces, matches):
    result = []
    for row, raw, match in zip(rows, traces, matches, strict=True):
        tags = dm.slice_tags(row)
        ped = match['classes']['pedestrian']
        pid = {g for g, _ in ped['pairs']}
        pid75 = {g for g, _ in ped['pairs75']}
        output75 = {g for g, _ in row['ap_pairs'][5]}
        result.append({'key':row['key'], 'group':row['group_id'], 'image_sha256':row['image_sha256'],
                       'label_sha256':row['label_sha256'], 'targets':[
            {'gt':g, 'small_thin':'area_S' in tags[g] or 'width_le8' in tags[g], 'best_score75':s,
             'pre75':above_floor, 'output75':g in output75, 'work50':g in pid, 'work75':g in pid75}
            for g,(s,above_floor) in sorted(raw.items())]})
    return result


def coverage_summary(rows, feasible):
    people = [item for row in rows for item in row['targets']]
    def count(item):
        return dict(gt=len(item), pre75=sum(t['pre75'] for t in item), output75=sum(t['output75'] for t in item),
                    work50=sum(t['work50'] for t in item) if feasible else None,
                    work75=sum(t['work75'] for t in item) if feasible else None)
    return {'all':count(people), 'small_thin':count([t for t in people if t['small_thin']]),
            'other':count([t for t in people if not t['small_thin']])}


def evaluate(job, fit_size, size, epoch=None):
    import torch
    torch.set_num_threads(4)
    epoch = run_observations(job)[-1] if epoch is None else epoch
    arm = Path(job['output'])/str(fit_size)
    record = read_json(arm/'train.json')
    check_equal('completed arm', 'complete', record['status'])
    checkpoint, checkpoint_sha256 = observation_checkpoint(arm, record, epoch, job['updates_per_epoch'])
    destination = observation_dir(arm, job['identity'], epoch)/f'dev{size}'
    destination.parent.mkdir(exist_ok=True)
    rules = load_rules(ROOT/'RULES.md')
    args = validator_args(job['data'], imgsz=size)
    args.update(cache=False, fraction=1., single_cls=False, seed=17, deterministic=True)
    validator_type = evidence_validator_class(trace_fn=raw_strict_coverage)
    with tempfile.TemporaryDirectory(prefix='.eval-', dir=destination.parent) as scratch:
        stage = Path(scratch)/destination.name
        stage.mkdir()
        print(f'Evaluating train{fit_size} epoch {epoch} at {size}', flush=True)
        start = time.perf_counter()
        with gzip.open(stage/'objects.jsonl.gz', 'wt', encoding='utf-8') as stream:
            validator = validator_type(args=args, save_dir=Path(scratch)/'validator', records=job['dev'],
                                       source_root=Path(job['source_root']), stream=stream, trace_context=(True,))
            validator(model=str(checkpoint))
        check_equal('dev complete key set', {r['key'] for r in job['dev']}, validator.keys)
        check_equal('dev count', len(job['dev']), validator.seen)
        check_equal('inference size', size, validator.dataloader.dataset.imgsz)
        check_equal('no dev augmentation', False, validator.dataloader.dataset.augment)
        metrics, matches, curves = summarize(validator.rows, rules)
        for cid, native_ap in zip(validator.metrics.box.ap_class_index,validator.metrics.box.all_ap,strict=True):
            offline = metrics['metrics']['per_class'][NAMES[int(cid)]]['ap_iou']
            if not np.allclose(offline, native_ap, rtol=0, atol=1e-12):
                raise ContractError('Offline AP diverges from native validator')
        cover = coverage(validator.rows, validator.traces, matches)
        feasible = metrics['workpoints']['pedestrian']['t_rule'] is not None
        summary = coverage_summary(cover, feasible)
        if feasible:
            check_equal('GT coverage matches working point', metrics['workpoints']['pedestrian']['tp'], summary['all']['work50'])
        dm.write_rows(stage/'matches.jsonl.gz', matches)
        np.savez_compressed(stage/'pr.npz', **curves)
        write_json(stage/'coverage.json', cover)
        metrics.update(train_size=fit_size, inference_size=size, epoch=epoch,
                       updates=epoch*job['updates_per_epoch'], coverage=summary,
                       checkpoint_sha256=checkpoint_sha256, scope='smoke' if job['smoke'] else 'source_dev',
                       tensor_shapes=sorted({tuple(r['input_shape']) for r in validator.rows}),
                       seconds=time.perf_counter()-start, speed_ms=validator.speed,
                       recall_if_precision_passes=metrics['workpoints']['pedestrian']['recall'] if feasible else None,
                       settings={k:getattr(validator.args,k) for k in args if k!='data'})
        write_json(stage/'metrics.json', metrics)
        stage.rename(destination)
    print(f'train{fit_size} epoch {epoch} @ {size}: {summary}', flush=True)


def compare_epoch(job, epoch):
    run = Path(job['output'])
    sizes = evaluation_sizes(job['identity'])
    keys = [(x,y) for x in SIZES for y in sizes]
    cells, ledgers = {}, {}
    for x,y in keys:
        folder=observation_dir(run/str(x), job['identity'], epoch)/f'dev{y}'
        cells[x,y]=read_json(folder/'metrics.json')
        ledgers[x,y]=read_json(folder/'coverage.json')
        cell = cells[x,y]
        # Original eight-epoch metrics carried the update count, without a separate epoch key.
        recorded_epoch = cell.get('epoch', run_observations(job)[-1] if sizes == SIZES else None)
        check_equal('comparison checkpoint and geometry', (epoch, x, y, epoch*job['updates_per_epoch']),
                    (recorded_epoch, cell['train_size'], cell['inference_size'], cell['updates']))
        check_equal('comparison evaluation images', len(job['dev']), len(ledgers[x,y]))
    groups = sorted({r['group'] for r in ledgers[keys[0]]})
    lookup = {g:i for i,g in enumerate(groups)}
    strata, endpoints = ('all','small_thin','other'), ('pre75','output75','work50','work75')
    totals = np.zeros((len(groups),3),dtype=np.int32)
    counts = np.zeros((len(keys),len(groups),3,4),dtype=np.int32)
    gtsets = {k:{endpoint:set() for endpoint in endpoints} for k in keys}
    images = [ledgers[k] for k in keys]
    for rows in zip(*images, strict=True):
        a=rows[0]
        for r in rows[1:]:
            check_equal('same frozen image and label in paired cells',
                        [a[k] for k in ('key','group','image_sha256','label_sha256')],
                        [r[k] for k in ('key','group','image_sha256','label_sha256')])
        gt=lookup[a['group']]
        for items in zip(*(r['targets'] for r in rows),strict=True):
            anchor=items[0]
            slim=1 if anchor['small_thin'] else 2
            totals[gt,[0,slim]]+=1
            for i,t in enumerate(items):
                check_equal('paired GT identity', (anchor['gt'],anchor['small_thin']),(t['gt'],t['small_thin']))
                flags=[t[name] for name in endpoints]
                for j,on in enumerate(flags):
                    if on:
                        gtsets[keys[i]][endpoints[j]].add((a['key'],t['gt']))
                counts[i,gt,[0,slim]]+=flags
    if not job['smoke']:
        check_equal('paired pedestrian GT support', [484,195,289], totals.sum(0).tolist())
    draws = np.asarray([m for _,m in dm.cluster_draws(groups,repeats=2000,seed=20261026)])
    weighted_gt = draws@totals
    effects = {}
    if sizes == (960,):
        comparisons = {'train_effect_at_960':(-1,1)}
    else:
        comparisons = {'train_effect_at_640':(-1,0,1,0), 'train_effect_at_960':(0,-1,0,1),
                       'inference_effect_after_640':(-1,1,0,0), 'inference_effect_after_960':(0,0,-1,1),
                       'interaction':(1,-1,-1,1)}
    for name, coefficients in comparisons.items():
        differences = np.tensordot(coefficients, counts, axes=1)
        replicates=np.einsum('rg,gst->rst',draws,differences)
        signal={}
        allow_work=all(cells[key]['recall_if_precision_passes'] is not None
                       for key,c in zip(keys,coefficients,strict=True) if c)
        for i,stratum in enumerate(strata):
            keep=weighted_gt[:,i]>0
            total=totals[:,i].sum()
            if not total:
                signal[stratum] = dict.fromkeys(endpoints)
                continue
            full=differences[:,i].sum(0)/total
            rates=replicates[keep,i]/weighted_gt[keep,i,None]
            bounds=np.quantile(rates,[.025,.975],axis=0)
            signal[stratum]={metric:dict(delta=float(full[j]),lower=float(bounds[0,j]),upper=float(bounds[1,j]))
                             if not metric.startswith('work') or allow_work else None for j,metric in enumerate(endpoints)}
        ap={}
        for cls in NAMES:
            ap[cls]={metric:sum(c*cells[k]['metrics']['per_class'][cls][metric] for c,k in zip(coefficients,keys,strict=True))
                     for metric in ('ap50','ap75','ap50_95')}
        effects[name]={'coverage_and_recall':signal,'ap_point_delta':ap}
        if name!='interaction':
            a,b=keys[coefficients.index(-1)],keys[coefficients.index(1)]
            effects[name]['pedestrian_migration']={m:{'gained':sorted(gtsets[b][m]-gtsets[a][m]),
                                                       'lost':sorted(gtsets[a][m]-gtsets[b][m])}
                                                   if not m.startswith('work') or allow_work else None for m in endpoints}
    row_summary={f'{k[0]}x{k[1]}':{'per_class':v['metrics']['per_class'],'workpoints':v['workpoints'],
                                   'coverage':v['coverage'],'checkpoint_sha256':v['checkpoint_sha256']}
                 for k,v in cells.items()}
    policy = 'paired fixed smoke workpoints' if job['smoke'] else 'paired fixed full-dev workpoints'
    return {'cells':row_summary,'effects':effects, 'bootstrap':{
        'groups':len(groups),'repeats':2000,'seed':20261026,'policy':policy}}


def compare(job):
    run = Path(job['output'])
    baseline_path=ROOT/'runs/s3_17/scale.json'
    fixed=read_json(baseline_path)['arms']
    sizes=evaluation_sizes(job['identity'])
    baseline={s:{'ap':fixed[r]['classes'], 'pedestrian_point':fixed[r]['point'], 'coverage':fixed[r]['slices']}
              for s,r in ((640,'A640'),(960,'B960')) if s in sizes}
    result={'scope':'smoke' if job['smoke'] else 'single-seed fixed-budget source_dev mechanism comparison',
            'fixed_R':baseline,'fixed_R_result_sha256':sha256(baseline_path)}
    comparisons={epoch:compare_epoch(job, epoch) for epoch in run_observations(job)}
    if sizes == (960,):
        result['by_epoch']=comparisons
    else:
        result.update(next(iter(comparisons.values())))
    write_json(run/'comparison.json.tmp', result)
    (run/'comparison.json.tmp').replace(run/'comparison.json')
    print('Paired evaluation complete',flush=True)
