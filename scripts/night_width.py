"""Preserve the E1 prefix and replace only 20 images using label-blind medium-scale tail signals."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

import detection_metrics as dm
import night_adapt as adapt
import night_presence as presence
import night_rank
import night_strategy as strategy
import night_tail
from source_contract import ContractError,canonical_hash,check_equal

WIDTH=(12,64)


def width_score(original,variants,scale,ranges):
    a=strategy._boxes(original)
    scaled=(a[:,2]-a[:,0])*scale
    a=a[(scaled>=ranges[0])&(scaled<=ranges[1])]
    result=np.zeros(5,dtype=np.float64)
    for variant in variants:
        b=strategy._boxes(variant)
        for c in (3,4):
            x,y=a[a[:,5]==c],b[b[:,5]==c]
            if not len(x) or not len(y):
                continue
            iou=dm.box_iou(x[:,:4],y[:,:4]); confidence=np.minimum(x[:,4,None],y[None,:,4])
            score=(4*confidence*(1-confidence)*iou)[iou>=.5].max(initial=0.)
            result[c]=max(result[c],score)
    return result


def rank_width(keys,z,u,groups,parent,retained,per_tail,shortlist):
    if len(set(parent))!=len(parent) or not set(parent)<=set(keys) or not 0<retained<len(parent):
        raise ContractError('Expected a unique ordered parent image list with an unmodified prefix')
    if not 0<per_tail<=shortlist or set(keys)!=set(groups):
        raise ContractError('Unequal group inventory or invalid fixed tail quotas')
    order,ordered,cos,selected=strategy._geometry(keys,z,parent[:retained])
    scores=strategy._scores(keys,u)[order]
    merged=set(parent)
    used={groups[ordered[i]] for i in selected}
    candidates=[i for i,k in enumerate(ordered) if k not in merged and groups[k] not in used]
    ranking={}  # risk-group exclusion remains mandatory after building the shortlists
    for c in (3,4):
        ranking[c]=sorted((i for i in candidates if scores[i,c]>0),
                          key=lambda i:(-round(float(scores[i,c]),12),ordered[i]))[:shortlist]
    dist=1-cos[:,selected].max(axis=1)
    decisions=[]
    for _ in range(per_tail):
        for c in (3,4):
            valid=[i for i in ranking[c] if groups[ordered[i]] not in used]
            if not valid:
                raise ContractError(f'Insufficient independent width-filtered candidates for class {c}')
            def priority(i):
                score=float(scores[i,c]); weighted=score*(.5+.5*float(dist[i]))
                return -round(weighted,12),-round(score,12),ordered[i]
            i=min(valid,key=priority); key=ordered[i]
            decisions.append({'key':key,'class_id':c,'score':float(scores[i,c]),'distance':float(dist[i]),'group_id':groups[key]})
            selected.append(i); used.add(groups[key]); np.minimum(dist,1-cos[:,i],out=dist)
    return {'keys':[ordered[i] for i in selected],'steps':decisions,'shortlists':{str(c):len(v) for c,v in ranking.items()}}


def validate(cfg):
    check_equal('registered width selection',(2000,200,40,80,list(WIDTH),80,10),
                tuple(cfg[k] for k in ('day_images','updates','max_minutes','shortlist','range','retained','per_tail')))


def select(cfg,scratch):
    keys,z,scores,inventory,draws,memo=presence.blind_inputs(cfg,scratch,worker=__file__)
    check_equal('medium width capture mode','medium_width',memo['presence']['score_mode'])
    digest=memo['images_sha256']
    groups={k:r['group_id'] for k,r in inventory.items()}
    output={}
    for seed in cfg['seeds']:
        d,base=([r['key'] for r in adapt.draw_records(k,seed,draws[k],digest,inventory)] for k in ('d','e1'))
        def calculate():
            return rank_width(keys,z,scores,groups,base,cfg['retained'],cfg['per_tail'],cfg['shortlist'])
        result=calculate()
        check_equal('repeatable width selection',result,calculate())
        picked=result['keys']
        check_equal('100 unique images and unchanged E1 prefix',(100,100,base[:80],80),
                    (len(picked),len(set(picked)),picked[:80],len(set(base)&set(picked))))
        if set(picked)==set(d):
            raise ContractError('No distinct training treatment against D')
        records=[inventory[k] for k in picked]
        output[seed]={**result,'records':records,'records_sha256':canonical_hash(records),'original_e1_overlap':80,
                      'd_overlap':len(set(d)&set(picked)),'risk_groups':len({r['group_id'] for r in records})}
        print(f'Width seed {seed}: 100 blind images, unchanged parent prefix 80, shortlist sizes {result["shortlists"]}.',flush=True)
    return {**memo,'lists':output}


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='capture':
        presence.capture(Path(sys.argv[2]).absolute(),sys.argv[3],
                         restricted_score=lambda a,b,scale:width_score(a,b,scale,WIDTH))
    else:
        night_rank.main(kind='width',fit=night_tail.fit,validate=validate,select=select,description=__doc__)
