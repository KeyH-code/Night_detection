"""Local fixed-budget classification-head comparison for pedestrians and cyclists."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile

import numpy as np

import detection_metrics as dm
from evaluate import evidence_validator_class, original_gt, stage_subset, validator_args, verify_subset
from source_contract import ContractError, NAMES, check_equal, read_json, sha256, verified_file, write_json
from source_policy import load_rules, summarize
from train_scale import (READY, ROOT, SOURCE, environment_at, file_inventory, observation_checkpoint,
                         save_json, state_hash, timestamp)


SIZE, BATCH, ACCUMULATE, EPOCHS, SEED = 960, 4, 4, 4, 29
LAMBDA, MARGIN = .20, 1.
PLAN = ROOT / 'specs/s3/roadmap.md'
SOURCE_WEIGHT = ROOT / 'runs/scale_003/960'
NEGATIVES = ROOT / 'runs/s3_17/rider_negatives.json'
PRIOR = ROOT / 'runs/riders_001'


def input_spec(output):
    ready = read_json(SOURCE / 'READY.json')
    check_equal('frozen data', READY, sha256(SOURCE / 'READY.json'))
    entry = ready['manifests']['source_train']
    manifest = read_json(verified_file(SOURCE, entry))
    check_equal('train manifest', 'source_train', manifest['purpose'])
    records = {r['key']: r for r in manifest['records']}
    check_equal('train count', 4000, len(records))
    dev = verify_subset(SOURCE, READY, 'source_dev')
    dev_groups = {r['group_id'] for r in dev['records']}
    spec = read_json(output / 'data.json')
    check_equal('train manifest SHA', entry['sha256'], spec['input_sha256']['train_manifest'])
    selected = [records[k] for k in spec['train_keys']]
    check_equal('192 unique selected training images', 192, len({r['key'] for r in selected}))
    if len(selected)!=192 or len({r['group_id'] for r in selected})!=192 or not all(
        r['purpose']=='source_train' and r['group_id'] not in dev_groups for r in selected
    ):
        raise ContractError('Training and developer risk groups overlap')
    selected_by_key = {r['key']: r for r in selected}
    labels = {r['key']: original_gt(r, SOURCE) for r in selected}
    return spec, selected, selected_by_key, labels, dev


def inputs(output, negative=False):
    spec, selected, selected_by_key, labels, dev = input_spec(output)
    if negative:
        ledger = read_json(NEGATIVES)
        check_equal('audited negatives SHA',sha256(NEGATIVES),spec['input_sha256']['negative_ledger'])
        approved=[x for x in ledger['targets'] if x['review']=='include']
        targets={x['key']:x for x in approved}
        check_equal('approved target set',sorted(targets),ledger['accepted_keys'])
        check_equal('three approved groups',3,len({x['group'] for x in approved}))
        for key,t in targets.items():
            r=selected_by_key[key]
            for field in ('image_sha256','label_sha256'):
                check_equal(field,r[field],t[field])
            check_equal('audited scene group',r['group_id'],t['group'])
            check_equal('frozen cyclist GT',1,labels[key][t['gt']][0])
            np.testing.assert_allclose(labels[key][t['gt']][1],t['gt_original_box'],atol=1e-5,rtol=0)
        return selected,labels,targets,dev
    targets = {}
    for row in spec['rows']:
        if row['review'] != 'include':
            continue
        k, g, c = row['key'], row['gt'], row['gt_class']
        check_equal('audited scene group', selected_by_key[k]['group_id'], row['group'])
        check_equal('audited frozen class', c, labels[k][g][0])
        if c not in (0, 1) or (k, g) in targets:
            raise ContractError(f'Invalid supervision target: {k}/{g}')
        np.testing.assert_allclose(labels[k][g][1], row['box'], atol=1e-5, rtol=0)
        targets[k, g] = c
    counts = Counter(targets.values())
    if min(counts[0], counts[1]) < 4:
        raise ContractError('Need support in both confusion directions')
    return selected, labels, targets, dev


def pair_loss(logits, classes, balance):
    import torch.nn.functional as F
    z = logits.float()
    margins = z.gather(1, classes[:, None]).flatten() - z.gather(1, (1-classes)[:, None]).flatten()
    return (F.softplus(MARGIN-margins)*balance[classes]).mean()


def native_labels(dataset, records, labels):
    """Verify label indices, including the exact row order before letterboxing."""
    by_name = {Path(r['image']).name: r for r in records}
    check_equal('training loader keys', set(by_name), {Path(p).name for p in dataset.im_files})
    for item in dataset.labels:
        r = by_name[Path(item['im_file']).name]
        want = labels[r['key']]
        check_equal('native labels length', len(want), len(item['cls']))
        check_equal('label classes in row order', [c for c,b in want], item['cls'].astype(int).ravel().tolist())
        norm = np.array([b for c,b in want]).reshape(-1,4)
        norm /= [r['width'],r['height'],r['width'],r['height']]
        xywh = np.column_stack(((norm[:,:2]+norm[:,2:])/2,norm[:,2:]-norm[:,:2]))
        np.testing.assert_allclose(item['bboxes'],xywh,atol=2e-7,rtol=0)
    return by_name


def foreground_targets(batch, names, labels, targets, device):
    import torch
    from ultralytics.utils import ops
    pad = torch.full((len(batch['im_file']), max(len(x) for x in labels.values())+1), -1,
                     device=device, dtype=torch.long)
    for i, path in enumerate(batch['im_file']):
        key = names[Path(path).name]['key']
        wanted = labels[key]
        use = batch['batch_idx']==i
        check_equal('transformed class order', [c for c,b in wanted], batch['cls'][use].int().flatten().tolist())
        if wanted:
            h,w = batch['img'].shape[-2:]
            xyxy = ops.xywh2xyxy(batch['bboxes'][use]*batch['bboxes'].new_tensor([w,h,w,h]))
            restored = ops.scale_boxes((h,w),xyxy,batch['ori_shape'][i],batch['ratio_pad'][i])
            np.testing.assert_allclose(restored.numpy(),[b for c,b in wanted],atol=.04,rtol=0)
        for g in range(len(wanted)):
            if (key,g) in targets:
                pad[i,g]=targets[key,g]
    return pad


def checkpoints_ok(initial, model, names, active_rows=2):
    import torch
    state = model.state_dict()
    changed = 0
    for k,v in state.items():
        b = initial[k]
        a = v.detach().cpu()
        if k in names:
            check_equal(k+' other classes frozen', True, torch.equal(a[active_rows:],b[active_rows:]))
            changed += torch.count_nonzero(a[:active_rows] != b[:active_rows]).item()
        else:
            check_equal(k+' frozen', True, torch.equal(a,b))
    check_equal('classification layers changed', True, changed>0)
    return changed


def train_dataset(data, negative=False):
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import DEFAULT_CFG
    dataset = YOLODataset(img_path=str(data.parent/'images/source_train'), imgsz=SIZE, batch_size=BATCH,
                          augment=False, hyp=DEFAULT_CFG, rect=negative, cache=False, stride=32,
                          pad=.5 if negative else 0., task='detect',
                          data={'names':dict(enumerate(NAMES)), 'nc':len(NAMES)})
    if negative:
        check_equal('one common rectangular shape for random batches',1,len(np.unique(dataset.batch_shapes,axis=0)))
    return dataset


def frozen_decode(model, image):
    import torch
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        decoded,_ = model.predict(image[None].cuda().float()/255)
    return decoded[0]


def sha_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def training_image_map(dataset, names):
    return {names[Path(p).name]['key']:i for i,p in enumerate(dataset.im_files)}


def negative_loss(logits):
    import torch.nn.functional as F
    return F.softplus(logits.float()).mean()


def negative_anchors(checkpoint,data,records,labels,targets):
    """Relocate the audited output to the frozen model on actual training tensors."""
    import torch
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou
    threshold = read_json(NEGATIVES)['replay_thresholds']['shared_person']
    dataset=train_dataset(data,True)
    names=native_labels(dataset,records,labels)
    mapping=training_image_map(dataset,names)
    model,_=load_checkpoint(str(checkpoint),device='cuda:0')
    model.eval()
    ledger={}
    for key,target in sorted(targets.items()):
        item=dataset[mapping[key]]
        h,w=item['img'].shape[-2:]
        want=labels[key]
        check_equal('audit GT order',[c for c,b in want],item['cls'].int().flatten().tolist())
        gt=ops.xywh2xyxy(item['bboxes']*item['bboxes'].new_tensor([w,h,w,h]))
        restored=ops.scale_boxes((h,w),gt.clone(),item['ori_shape'],item['ratio_pad'])
        np.testing.assert_allclose(restored.numpy(),[b for c,b in want],atol=.04,rtol=0)
        raw=frozen_decode(model,item['img'])
        box=ops.xywh2xyxy(raw[:4].T).float()
        p=raw[4].float()
        reviewed=target['replay']['margin']['best']
        old_box=torch.tensor([reviewed['wrong_person']['original_box']],device=box.device,dtype=torch.float32)
        repl=ops.scale_boxes((h,w),box.clone(),item['ori_shape'],item['ratio_pad'])
        to_fp=box_iou(repl,old_box).flatten()
        to_gt=box_iou(box,gt.cuda())
        keep=(p>=threshold)&(to_fp>=.70)&(to_gt[:,target['gt']]>=.50)&((box[:,2:]>box[:,:2]).all(1))
        people=[i for i,(c,b) in enumerate(want) if c==0]
        if people:
            keep=keep&(to_gt[:,people].max(1)[0]<.10)
        ids=torch.nonzero(keep,as_tuple=False).flatten().tolist()
        if not ids:
            raise ContractError(f'{key}: no eligible error anchor on actual training tensor')
        ledger[key]={'gt':target['gt'],'shape':[h,w],'image_tensor_sha256':sha_tensor(item['img']),
                     'anchors':ids,'person_scores':p[ids].tolist(),'cyclist_scores':raw[5,ids].float().tolist(),
                     'rider_iou':to_gt[ids,target['gt']].tolist(),'fp_overlap':to_fp[ids].tolist(),
                     'reference_fp_box_original':reviewed['wrong_person']['original_box']}
    return ledger


def fit(arm, run, checkpoint, data, records, labels, targets, negative=None):
    import torch
    from torch.utils.data import DataLoader
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import ModelEMA
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda:0')
    dataset = train_dataset(data,negative is not None)
    names = native_labels(dataset, records, labels)
    model, _ = load_checkpoint(str(checkpoint), device=device)
    model.eval()
    model.requires_grad_(False)
    head = model.model[-1]
    check_equal('six-class training head', (6,3,False), (head.nc,len(head.cv3),head.end2end))
    head.training = True  # forward returns native logits while all child BN layers stay in eval.
    params = [x for tower in head.cv3 for x in (tower[-1].weight,tower[-1].bias)]
    local_names = {k for k,v in model.named_parameters() if any(v is z for z in params)}
    check_equal('all unmasked head channels', True, all(p.shape[0]==6 for p in params))
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    start_hash = state_hash(model)
    for param in params:
        param.requires_grad_(True)
    native_args = dict(model.args)
    model.args = SimpleNamespace(**native_args)
    loss_fn = v8DetectionLoss(model)
    optimizer = torch.optim.SGD(params, lr=.004, momentum=.937)
    ema = ModelEMA(model)
    ema.ema.args = native_args
    scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    active_rows=1 if negative is not None else 2
    if negative is None:
        count = Counter(targets.values())
        balance = torch.tensor([len(targets)/(2*count[j]) for j in (0,1)], device=device)
        test = torch.zeros((2,2),device=device,requires_grad=True)
        g, = torch.autograd.grad(pair_loss(test,torch.tensor([0,1],device=device),balance),test)
        check_equal('both rank gradient directions', True, bool((g.diag()<0).all() and (g.flip(1).diag()>0).all()))
    else:
        test=torch.zeros((2,2),device=device,requires_grad=True)
        g,=torch.autograd.grad(negative_loss(test[:,0]),test)
        check_equal('only false person channel is pushed down',True,bool((g[:,0]>0).all() and (g[:,1]==0).all()))
    history = []
    run.mkdir()
    for epoch in range(EPOCHS):
        lr = .004*(.1+.9*(1+math.cos(math.pi*epoch/EPOCHS))/2)
        optimizer.param_groups[0]['lr']=lr
        order = np.random.default_rng(SEED+epoch).permutation(len(dataset)).tolist()
        loader = DataLoader(dataset, batch_size=BATCH, sampler=order, num_workers=0,
                            collate_fn=YOLODataset.collate_fn)
        totals, seen, seen_targets, active_aux = [], [], set(), []
        evaluated_anchors = Counter()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        for step,batch in enumerate(loader,1):
            keys=[names[Path(p).name]['key'] for p in batch['im_file']]
            seen.extend(keys)
            hard = foreground_targets(batch,names,labels,targets if negative is None else {},device)
            picks=[]
            if negative is not None:
                for j,key in enumerate(keys):
                    if key in negative:
                        q=negative[key]
                        check_equal(key+' training tensor',q['image_tensor_sha256'],sha_tensor(batch['img'][j]))
                        picks.append((j,key,q['anchors']))
            for k in ('cls','bboxes','batch_idx'):
                batch[k]=batch[k].to(device)
            with torch.autocast('cuda',dtype=torch.float16):
                batch['img']=batch['img'].to(device).float()/255
                pred = model.predict(batch['img'])
                assign, native, _ = loss_fn.get_assigned_targets_and_loss(pred,batch)
                if negative is None:
                    fg, inds = assign[:2]
                    assigned = hard.gather(1,inds)
                    used = fg & (assigned>=0)
                    labels_at_anchor = assigned[used]
                    aux = pair_loss(pred['scores'].transpose(1,2)[used][:,:2],labels_at_anchor,balance) if bool(used.any()) else native[1]*0
                else:
                    aux=negative_loss(torch.cat([pred['scores'][j,0,ids] for j,key,ids in picks])) if picks else native[1]*0
                objective = native[1]+(LAMBDA*aux if arm in ('margin','negative') else 0)
            if not torch.isfinite(objective) or not torch.isfinite(aux):
                raise ContractError('Nonfinite classification objective')
            totals.append([float(native[1].detach()),float(aux.detach())])
            if negative is None:
                evaluated_anchors.update(labels_at_anchor.cpu().tolist())
                b,_ = torch.where(used)
                seen_targets.update((keys[j],i) for j,i in zip(b.tolist(),inds[used].tolist(),strict=True))
            else:
                if picks:
                    active_aux.append(float(aux.detach()))
                for j,key,ids in picks:
                    evaluated_anchors[key]+=len(ids)
                    seen_targets.add((key,negative[key]['gt']))
            scaler.scale(objective/ACCUMULATE).backward()
            if step % ACCUMULATE:
                continue
            scaler.unscale_(optimizer)
            for param in params:
                param.grad[active_rows:].zero_()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params,10.)):
                raise ContractError('Nonfinite gradient')
            before=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale()<before:
                raise ContractError('AMP skipped a head update')
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
        check_equal('exact training reads', [names[Path(dataset.im_files[i]).name]['key'] for i in order],seen)
        if negative is None:
            if min(evaluated_anchors[0],evaluated_anchors[1]) == 0:
                raise ContractError('A direction received no foreground supervision')
        else:
            check_equal('every audited negative anchor seen once per epoch',
                        {k:len(v['anchors']) for k,v in negative.items()},dict(evaluated_anchors))
        check_equal('fixed update count', (epoch+1)*12, ema.updates)
        record = {'epoch':epoch+1,'lr':lr,'updates':ema.updates,'images':seen,
                  'native_cls_mean':float(np.mean(totals,axis=0)[0]),
                  'assigned_anchors':dict(evaluated_anchors),'assigned_targets':sorted(seen_targets),
                  'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        aux_name='margin_mean' if negative is None else 'negative_active_batch_mean'
        record[aux_name]=float(np.mean(totals,axis=0)[1] if negative is None else np.mean(active_aux))
        history.append(record)
        print(arm,epoch+1,'updates',ema.updates,'classification',record['native_cls_mean'],
              aux_name,record[aux_name],'targets',len(seen_targets),flush=True)
        save_json(run/'train.json',{'history':history,'start_state_sha256':start_hash})
    changed = checkpoints_ok(initial,model,local_names,active_rows)
    ema.ema.eval()
    ema.ema.args = native_args
    # Do not EMA-average frozen tensors. Preserve their exact checkpoint bytes.
    state = ema.ema.state_dict()
    for k,v in state.items():
        if k in local_names:
            v[active_rows:].copy_(initial[k][active_rows:].to(v))
        else:
            v.copy_(initial[k].to(v))
    packed = deepcopy(ema.ema).half().cpu()
    path=run/'last.pt'
    torch.save({'ema':packed,'model':None,'epoch':EPOCHS-1,'updates':ema.updates,
                'optimizer':optimizer.state_dict(),'train_args':native_args,'scope':'rider head diagnostic'},path)
    reloaded,_=load_checkpoint(str(path))
    round_trip = checkpoints_ok(initial,reloaded,local_names,active_rows)
    save_json(run/'train.json',{'status':'complete','arm':arm,'history':history,'start_state_sha256':start_hash,
                               'changed_float_values':changed,'changed_fp16_values':round_trip,
                               'checkpoint_sha256':sha256(path),'trainable_rows':sorted(local_names)})


def raw_pairs(raw,row,_):
    import torch
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou
    refs=[i for i,c in enumerate(row['gt']['cls']) if c<2]
    if not refs:
        return []
    boxes=ops.xywh2xyxy(raw[:4].T).float()
    gt=torch.tensor([row['gt']['boxes'][i] for i in refs],device=raw.device)
    overlap=box_iou(gt,boxes)
    fit=overlap>=.5
    iou, anchor=overlap.max(1)
    p,c=raw[4].float(),raw[5].float()
    return [dict(gt=g,closest_iou=iv,closest_p=p0,closest_c=c0,best_p=ph if ph>=0 else None,
                 best_c=ch if ch>=0 else None)
            for g,iv,p0,c0,ph,ch in zip(refs,iou.tolist(),p[anchor].tolist(),c[anchor].tolist(),
              p.expand_as(overlap).masked_fill(~fit,-1).amax(1).tolist(),
              c.expand_as(overlap).masked_fill(~fit,-1).amax(1).tolist(),strict=True)]


def protected_decode(raw):
    import torch
    return sha_tensor(torch.cat((raw[:4],raw[5:]),dim=0))


def trace_negative(raw,row,context):
    return {'targets':raw_pairs(raw,row,context),'protected_decode_sha256':protected_decode(raw)}


def evaluate(path, destination, data, dev, negative=False):
    import torch
    import torchvision  # configure native NMS backend consistently with the sealed baseline
    from contextlib import redirect_stderr, redirect_stdout
    destination.mkdir()
    args=validator_args(data,imgsz=SIZE)
    args.update(cache=False,fraction=1.,single_cls=False,seed=17,deterministic=True)
    with (destination/'native.log').open('w',encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        validator=evidence_validator_class(trace_fn=trace_negative if negative else raw_pairs)(
            args=args,save_dir=destination/'validator',records=dev['records'],source_root=SOURCE,stream=None,
            trace_context=(True,))
        validator(model=str(path))
    check_equal('complete developer image set',{r['key'] for r in dev['records']},validator.keys)
    m,matches,curves=summarize(validator.rows,load_rules(ROOT/'RULES.md'))
    for cid, ap in zip(validator.metrics.box.ap_class_index,validator.metrics.box.all_ap,strict=True):
        np.testing.assert_allclose(m['metrics']['per_class'][NAMES[int(cid)]]['ap_iou'],ap,rtol=0,atol=1e-12)
    dm.write_rows(destination/'objects.jsonl.gz',validator.rows)
    dm.write_rows(destination/'matches.jsonl.gz',matches)
    overview={}
    target_rows=[]
    protected={}
    for row,match,trace in zip(validator.rows,matches,validator.traces,strict=True):
        scores=trace['targets'] if negative else trace
        if negative:
            protected[row['key']]=trace['protected_decode_sha256']
        gt,pc,conf,iou=dm.arrays(row)
        checks={c:{g for g,p in match['classes'][NAMES[c]]['pairs']} for c in (0,1)}
        for s in scores:
            g=s['gt']
            c=int(gt[g])
            rival=1-c
            eligible=np.flatnonzero((pc==rival)&(iou[g]>=.5)&(conf>=m['workpoints'][NAMES[rival]]['threshold']))
            other=np.flatnonzero(gt==rival)
            rival_only_wrong=any(not (iou[other,x]>=.5).any() for x in eligible)
            target_rows.append(dict(key=row['key'],group=row['group_id'],label=c,**s,
               correct_work=g in checks[c],rival_work_overlap=bool(eligible.size),
               rival_only_wrong=bool(rival_only_wrong and g not in checks[c])))
    for c in (0,1):
        selected=[x for x in target_rows if x['label']==c]
        overview[NAMES[c]]={'n':len(selected),'correct':sum(x['correct_work'] for x in selected),
                          'wrong_only':sum(x['rival_only_wrong'] for x in selected),
                          'rival_overlap':sum(x['rival_work_overlap'] for x in selected)}
    write_json(destination/'metrics.json',m)
    write_json(destination/'classes.json',{'summary':overview,'targets':target_rows})
    if negative:
        write_json(destination/'protected.json',protected)
    np.savez_compressed(destination/'pr.npz',**curves)
    torch.cuda.empty_cache()
    print('evaluated',destination.parent.name,overview,flush=True)


def output_summary(rows, p_threshold, c_threshold):
    summary=Counter()
    doubles=[]
    for row in rows:
        gt,pc,conf,iou=dm.arrays(row)
        point,p_pairs,p_selected=dm.image_point(row,0,p_threshold)
        c_point,c_pairs,_=dm.image_point(row,1,c_threshold)
        hit={int(p) for g,p in p_pairs}
        summary.update({'person_tp':point['tp'],'person_fp':point['fp'],'person_fn':point['fn'],
                        'cyclist_tp':c_point['tp'],'cyclist_fp':c_point['fp']})
        for j in p_selected:
            if int(j) not in hit:
                summary['fp_'+dm.error_type(row,int(j),p_pairs)['type']]+=1
        for g,c in c_pairs:
            for p in p_selected:
                if p in hit or iou[g,p]<.5 or (iou[gt==0,p]>=.1).any():
                    continue
                if dm.box_iou([row['pred']['boxes'][p]],[row['pred']['boxes'][c]])[0,0] >= .5:
                    doubles.append((row['key'],int(g)))
                    break
    return {'p_threshold':p_threshold,'c_threshold':c_threshold,'counts':dict(summary),
            'person_precision':summary['person_tp']/(summary['person_tp']+summary['person_fp']),
            'double_targets':sorted(doubles)}


def compare(output, negative=False):
    names=('control','negative' if negative else 'margin')
    metrics=[read_json(output/a/'dev960/metrics.json') for a in names]
    tables=[read_json(output/a/'dev960/classes.json') for a in names]
    pairs=[{(x['key'],x['gt']):x for x in t['targets']} for t in tables]
    check_equal('paired developer GT set',set(pairs[0]),set(pairs[1]))
    boot={}
    groups=sorted({x['group'] for x in pairs[0].values()})
    lookup={g:i for i,g in enumerate(groups)}
    counts=np.zeros((len(groups),2,3),int)
    for key,a in pairs[0].items():
        b=pairs[1][key]
        check_equal('fixed box geometry and GT',(a['group'],a['label'],a['closest_iou']),
                    (b['group'],b['label'],b['closest_iou']))
        counts[lookup[a['group']],a['label']]+=np.array([1,int(b['correct_work'])-int(a['correct_work']),
                                                       int(b['rival_only_wrong'])-int(a['rival_only_wrong'])])
    # Count all frozen scene groups, including images without these two classes.
    groups_all=sorted({x['group_id'] for x in dm.read_rows(output/names[0]/'dev960/objects.jsonl.gz')})
    group_matrix=np.array([counts[lookup[g]] if g in lookup else np.zeros((2,3),int) for g in groups_all])
    draws=np.array([m for _,m in dm.cluster_draws(groups_all,2000,20261026)])
    sample=np.einsum('rg,gct->rct',draws,group_matrix)
    for c in (0,1):
        k=sample[:,c,0]>0
        v=sample[k,c,1:]/sample[k,c,0,None]
        boot[NAMES[c]]={'recall95':np.quantile(v[:,0],[.025,.975]).tolist(),
                        'wrong_only_rate95':np.quantile(v[:,1],[.025,.975]).tolist()}
    prior=read_json(ROOT/'runs/s3_17/scale24_agreement.json')
    strata={'stable':set(),'deep_rival':set(),'deep_without_rival':set()}
    for row in prior['deep']['targets']:
        name='stable' if row['cohort']=='stable' else 'deep_rival' if row['higher_cyclist_outputs'] else 'deep_without_rival'
        strata[name].add((row['key'],row['gt']))
    mechanism={}
    for name,keys in strata.items():
        a,b=[[p[k] for k in sorted(keys)] for p in pairs]
        margin=lambda x:x['closest_p']-x['closest_c']
        diffs=[margin(y)-margin(x) for x,y in zip(a,b,strict=True)]
        mechanism[name]={'n':len(keys),'correct':[sum(x['correct_work'] for x in z) for z in (a,b)],
                         'wrong':[sum(x['rival_only_wrong'] for x in z) for z in (a,b)],
                         'same_box_margin_median_delta':float(np.median(diffs)),
                         'same_box_margin_delta_up':sum(x>1e-6 for x in diffs)}
    base_dir=PRIOR/'margin/dev960' if negative else SOURCE_WEIGHT/'epoch24/dev960'
    baseline=read_json(base_dir/'metrics.json')
    summary={'baseline':baseline,'control':metrics[0],names[1]:metrics[1],
              'confusion':dict(zip(names,[t['summary'] for t in tables],strict=True)),
              'mechanism':mechanism,'paired95':boot,'risk_groups':len(groups_all),
              'interpretation':'single training seed, fixed original geometry; exploratory source_dev only'}
    if negative:
        a,b=[read_json(output/k/'dev960/protected.json') for k in names]
        check_equal('frozen geometry, cyclist and other four raw scores on all dev images',a,b)
        summary['raw_protected_channels_exact']=len(a)
        p_shared=read_json(NEGATIVES)['replay_thresholds']['shared_person']
        c_shared=baseline['workpoints']['cyclist']['threshold']
        sources={'baseline':base_dir,**{name:output/name/'dev960' for name in names}}
        by_arm={}
        for name,path in sources.items():
            rows=dm.read_rows(path/'objects.jsonl.gz')
            by_arm[name]={'shared':output_summary(rows,p_shared,c_shared),'at_p75':
              output_summary(rows,summary[name]['workpoints']['pedestrian']['threshold'],c_shared)}
        summary['double_reporting']=by_arm
    write_json(output/'comparison.json',summary)


def train_probe(paths,data,records,labels,anchors):
    import torch
    import torchvision
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import box_iou
    dataset=train_dataset(data,True)
    names=native_labels(dataset,records,labels)
    mapping=training_image_map(dataset,names)
    output={}
    for name,path in paths.items():
        model,_=load_checkpoint(str(path),device='cuda:0')
        model.eval()
        output[name]={}
        for key,ledger in anchors.items():
            item=dataset[mapping[key]]
            check_equal(key+' probe tensor',ledger['image_tensor_sha256'],sha_tensor(item['img']))
            raw=frozen_decode(model,item['img'])
            ids=ledger['anchors']
            calc,_=nms.non_max_suppression(raw[None].clone(),.001,.7,nc=6,multi_label=True,max_det=300,return_idxs=True)
            h,w=item['img'].shape[-2:]
            from ultralytics.utils import ops
            rider=ops.xywh2xyxy(item['bboxes'][ledger['gt']:ledger['gt']+1]*item['bboxes'].new_tensor([w,h,w,h])).cuda()
            out=calc[0]
            overlap=box_iou(rider,out[:,:4]).flatten()
            picked=out[(overlap>=.5)&(out[:,5]==0),4]
            cyclist=out[(overlap>=.5)&(out[:,5]==1),4]
            output[name][key]={'anchors':ids,'person_anchor_scores':raw[4,ids].float().tolist(),
                               'protected_sha256':protected_decode(raw),
                               'post_person_max':float(picked.max()) if len(picked) else None,
                               'post_cyclist_max':float(cyclist.max()) if len(cyclist) else None}
        del model
        torch.cuda.empty_cache()
    for key in anchors:
        check_equal('training case protected decode',1,
                    len({x[key]['protected_sha256'] for x in output.values()}))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--negative',action='store_true',help='run the approved three-case rider negative comparison')
    parser.add_argument('--preflight-only',action='store_true',help='verify target relocation without starting updates')
    a=parser.parse_args()
    output=a.output.resolve()
    if output.parent!=ROOT/'runs' or not (output/'data.json').is_file() or (output/'run.json').exists():
        raise ContractError('Use a new run directory with the reviewed data.json')
    if a.scratch.resolve().is_relative_to(ROOT):
        raise ContractError('Use scratch outside the repository')
    if a.preflight_only and not a.negative:
        parser.error('--preflight-only applies to the negative profile')
    with tempfile.TemporaryDirectory(dir=a.scratch,prefix='riders-') as tmp:
        scratch=Path(tmp)
        environment_at(scratch)
        import gc, importlib.metadata, platform
        import torch
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        check_equal('CUDA device available',True,torch.cuda.is_available())
        from train_scale import LIBRARIES
        installed={k:importlib.metadata.version(k) for k in LIBRARIES}
        check_equal('locked environment',LIBRARIES,installed)
        if a.negative:
            ledger=read_json(NEGATIVES)
            check_equal('prior run manifest',ledger['identity']['riders_manifest'],sha256(PRIOR/'manifest.json'))
            start=PRIOR/'margin/last.pt'
            digest=sha256(start)
            check_equal('approved negative starting checkpoint',ledger['identity']['weight_sha256']['margin'],digest)
        else:
            start, digest=observation_checkpoint(SOURCE_WEIGHT,read_json(SOURCE_WEIGHT/'train.json'),24,125)
        train,labels,targets,dev=inputs(output,a.negative)
        train_data=stage_subset({'root':SOURCE,'purpose':'source_train','records':train},scratch/'train')
        anchors=negative_anchors(start,train_data,train,labels,targets) if a.negative else None
        if anchors:
            print('verified teacher anchors', {k:(v['shape'],v['anchors'],v['person_scores']) for k,v in anchors.items()},flush=True)
        if a.preflight_only:
            return
        if anchors:
            write_json(output/'anchors.json',anchors)
        dev_data=stage_subset(dev,scratch/'dev')
        (output/'code').mkdir()
        sources=(Path(__file__),PLAN,ROOT/'RULES.md',ROOT/'scripts/train_scale.py',ROOT/'scripts/evaluate.py',
                 ROOT/'scripts/detection_metrics.py',ROOT/'scripts/source_policy.py',ROOT/'scripts/source_contract.py',
                 ROOT/'scripts/source_sampling.py')
        for p in sources:
            shutil.copyfile(p,output/'code'/p.name)
        status={'status':'running','started':timestamp(),'checkpoint_sha256':digest,'ready':READY,
                'selected_data_sha256':sha256(output/'data.json'),'environment':installed,
                'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
                'size':SIZE,'batch':BATCH,'accumulate':ACCUMULATE,'epochs':EPOCHS,'seed':SEED,
                'lr0':.004,'margin_lambda':LAMBDA,'margin_target':None if a.negative else MARGIN,
                'training_scope':'rectangular fixed geometry, person classifier only' if a.negative else 'frozen class-head rows 0/1',
                'final_test_executed':False}
        save_json(output/'run.json',status)
        try:
            arms=('control','negative' if a.negative else 'margin')
            for arm in arms:
                run=output/arm
                fit(arm,run,start,train_data,train,labels,targets,anchors)
                gc.collect()
                torch.cuda.empty_cache()
            first,second=[read_json(output/arm/'train.json') for arm in arms]
            check_equal('same start weights',first['start_state_sha256'],second['start_state_sha256'])
            for k in ('images','lr','updates'):
                check_equal('identical paired '+k,[r[k] for r in first['history']],[r[k] for r in second['history']])
            if anchors:
                paths={'baseline':start,**{name:output/name/'last.pt' for name in arms}}
                write_json(output/'train_probe.json',train_probe(paths,train_data,train,labels,anchors))
            for arm in arms:
                run=output/arm
                evaluate(run/'last.pt',run/'dev960',dev_data,dev,a.negative)
                gc.collect()
            compare(output,a.negative)
            status.update(status='complete',finished=timestamp())
        except BaseException as exc:
            status.update(status='failed',failure=f'{type(exc).__name__}: {exc}',finished=timestamp())
            raise
        finally:
            save_json(output/'run.json',status)
        save_json(output/'manifest.json',{**status,'files':file_inventory(output,{'manifest.json'})})
        print('completed',output,flush=True)


if __name__=='__main__':
    main()
