"""Native 200-update adaptation with classification loss targeted at night tail classes."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import random
import time
from types import SimpleNamespace

import numpy as np

import night_rank
import night_train as train
from source_contract import ContractError, NAMES, canonical_hash, check_equal, sha256


def focused_bce(logits, target, positive_weight, negative_gain, hard_per_class):
    import torch
    import torch.nn.functional as f

    if logits.shape != target.shape or logits.ndim != 3 or (logits.shape[0],logits.shape[2]) != (2,6):
        raise ContractError('Expected [source, night] native scores in six-class order')
    if positive_weight < 1 or negative_gain < 0 or hard_per_class < 0:
        raise ContractError('Invalid focus multiplier')
    base = f.binary_cross_entropy_with_logits(logits,target,reduction='none')
    with torch.no_grad():
        s = logits[1,:,3:5].float()
        positive = target[1,:,3:5] > 0
        mask = torch.ones_like(base)
        active = mask[1,:,3:5]
        if hard_per_class and negative_gain:
            val, ids=s.masked_fill(positive,-torch.inf).topk(min(hard_per_class,len(s)),dim=0)
            hard = (1 + negative_gain*val.sigmoid().square()).to(active.dtype)
            active.scatter_(0,ids,hard)
        active.masked_fill_(positive,positive_weight)
    return base*mask


def validate(cfg):
    check_equal('registered focus treatment',(2000,200,2.,4.,24,40),
                tuple(cfg[k] for k in ('day_images','updates','positive_weight','negative_gain','hard_per_class','max_minutes')))


def fit_native(torch,cv2,payloads,plan,seed,original,cfg,until,phase,*,transform=None,rates=None,prefix_verify=None,target_only=False):
    from torch.nn.modules.batchnorm import _BatchNorm
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import ModelEMA

    started=time.monotonic()
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats()
    model,_=load_checkpoint(str(train.CHECKPOINT),device='cuda')
    model.train()
    args=dict(model.args)
    model.args=SimpleNamespace(**args)
    check_equal('frozen six classes', NAMES,list(model.names.values()))
    day_prefix=len(payloads['day'])
    if target_only:
        check_equal('no staged source or non-native intervention',(0,None,None),(day_prefix,transform,rates))
        check_equal('only night image per microbatch',{1},{len(m) for m in plan})
    else:
        check_equal('same ordered day slots',list(payloads['day']),[d[0] for d,_ in plan[:day_prefix]])
        check_equal('exact day whitelist',set(payloads['day']),{d[0] for d,_ in plan})
    check_equal('exact night whitelist',set(payloads['night']),{m[-1][0] for m in plan})
    check_equal('original treatment exposure and size',(200,10,640,2000),
                (original['updates'],original['accumulate'],original['size'],len(plan) if target_only else day_prefix))
    acc=original['accumulate']; count=cfg['updates']
    check_equal('fit has exactly all planned gradient microbatches',count*acc,len(plan))
    if rates is None:
        check_equal('unextended fit update budget',200,count)
    else:
        check_equal('extended original-native fit identity',(600,600,True,True),(count,len(rates),transform is None,prefix_verify is not None))
    for name,param in model.named_parameters():
        param.requires_grad_('.dfl.' not in name)
    bn={}
    for name,layer in model.named_modules():
        if isinstance(layer,_BatchNorm):
            layer.eval().requires_grad_(False)
            bn.update({f'{name}.{k}':v.detach().clone() for k,v in layer.state_dict().items()})
    wd,free=[],[]
    for layer in model.modules():
        for name,p in layer.named_parameters(recurse=False):
            if p.requires_grad:
                (wd if isinstance(layer,torch.nn.Conv2d) and name=='weight' else free).append(p)
    optimizer=torch.optim.SGD([{'params':wd,'weight_decay':original['weight_decay']},{'params':free,'weight_decay':0.}],
                              lr=original['lr_start'],momentum=original['momentum'])
    scaler=torch.amp.GradScaler('cuda',init_scale=1.)
    ema,loss_fn=ModelEMA(model),v8DetectionLoss(model)
    check_equal('historical BCE has no pre-existing class reweight',None,loss_fn.class_weights)
    positive_counts=torch.zeros(2,device='cuda',dtype=torch.long)

    def objective(scores,target):
        positive_counts.add_((target[1,:,3:5]>0).sum(dim=0))
        return focused_bce(scores,target,cfg['positive_weight'],cfg['negative_gain'],cfg['hard_per_class'])

    focused=transform is None and rates is None and not target_only
    if focused:
        loss_fn.bce=objective
    payload=payloads['day'] | payloads['night']
    trace=[]; seen=[]
    mode='target' if target_only else 'long' if rates is not None else 'focus' if focused else 'style'
    for step in range(count):
        if time.monotonic() >= until:
            raise ContractError('Local focus run exceeded its time budget')
        lr=(rates[step] if rates is not None else
            original['lr_end']+(original['lr_start']-original['lr_end'])*(1+math.cos(math.pi*step/(count-1)))/2)
        for group in optimizer.param_groups:
            group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        sums=torch.zeros(3,device='cuda')
        for i,micro in enumerate(plan[step*acc:(step+1)*acc],start=step*acc):
            batch=train._batch(torch,cv2,payload,micro,original['size'])
            seen.append([item[0] for item in micro])
            if transform is not None:
                batch['img']=transform(batch['img'],i)
            with torch.autocast('cuda',dtype=torch.float16):
                _,terms,_=loss_fn.get_assigned_targets_and_loss(model(batch['img']),batch)
                loss=terms.sum()/acc
            if not loss.isfinite():
                raise ContractError('Nonfinite focused training loss')
            scaler.scale(loss).backward(); sums+=terms.detach().float()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),original['clip'],error_if_nonfinite=True)
        scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()<scale:
            raise ContractError('AMP skipped focused update')
        ema.update(model)
        trace.append({'step':step+1,'lr':lr,'loss_terms':(sums/acc).cpu().tolist()})
        if prefix_verify is not None and step+1==200:
            prefix_verify(trace)
        if (step+1)%10==0:
            print(f'{mode} train seed={seed} updates={step+1}/{count}',flush=True)
    check_equal('exact observed microbatches',[[item[0] for item in micro] for micro in plan],seen)
    check_equal('exact focused updates',count,ema.updates)
    supports=positive_counts.cpu().tolist()
    if focused and not all(supports):
        raise ContractError('No focused native positive signal in a registered class')
    for k,v in bn.items():
        check_equal('frozen BN '+k,True,torch.equal(model.state_dict()[k],v))
        ema.ema.state_dict()[k].copy_(v)
    dest=phase/'last.pt'
    final=deepcopy(ema.ema).cpu().half().eval(); final.args=args
    if not all(torch.isfinite(v).all() for v in final.state_dict().values()):
        raise ContractError('Nonfinite focused checkpoint')
    scope=f'fixed last {mode} night adaptation'
    torch.save({'ema':final,'model':None,'epoch':count//10-1,'updates':count,'train_args':args,'scope':scope},dest)
    loaded,_=load_checkpoint(str(dest),device='cpu')
    check_equal('EMA exact reload',True,all(torch.equal(v.float(),loaded.state_dict()[k].float()) for k,v in final.state_dict().items()))
    return dest,{'updates':count,'skipped_updates':0,'bn_frozen_buffers':len(bn),'trace':trace,'microbatches':seen,
                 'augment_schedule_sha256':canonical_hash(plan),'initializer_sha256':sha256(train.CHECKPOINT),
                 'counts':{'source':{} if target_only else dict(Counter(row[0] for row in seen)),
                           'night':dict(Counter(row[-1] for row in seen))},
                 'focus_classes':[3,4] if focused else [],'focus_night_positive_anchors':supports,
                 'seconds':time.monotonic()-started,
                 'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(),
                 'last_sha256':sha256(dest),'checkpoint_roles':{'last':'last.pt','best':{'path':'last.pt','selected_on_dev':False}}}


fit=fit_native


if __name__ == '__main__':
    night_rank.main(kind='focus',fit=fit,validate=validate,description=__doc__)
