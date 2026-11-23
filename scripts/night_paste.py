"""Compose authorized source vehicles with paired authorized night scenes in training only."""
from __future__ import annotations

from collections import Counter
import math

import night_focus
import night_rank
from source_contract import ContractError,check_equal

SIDE=640
TOP,BOTTOM=140,500
GRID=tuple((x,y) for y in (307,358,410,461) for x in (109,211,320,429,531))


def composite(batch,attempt,widths):
    import torch
    import torch.nn.functional as f

    check_equal('paired native source/night tensor',(2,3,SIDE,SIDE),tuple(batch['img'].shape))
    if not widths or any(not 16<=w<=80 for w in widths):
        raise ContractError('Invalid fixed crop sizes')
    table=torch.cat((batch['batch_idx'][:,None],batch['cls'],batch['bboxes']),dim=1).detach().cpu().tolist()
    boxes=[]
    for source,c,x,y,w,h in table:
        if source not in (0,1) or not all(math.isfinite(i) for i in (x,y,w,h)):
            raise ContractError('Invalid crop source batch IDs or nonfinite boxes')
        boxes.append((int(source),int(c),((x-w/2)*SIDE,(y-h/2)*SIDE,(x+w/2)*SIDE,(y+h/2)*SIDE)))
    wanted=3+attempt%2
    width=widths[(attempt//2)%len(widths)]
    donors=[(c,coords) for source,c,coords in boxes if source==0 and c in (3,4) and
            0<=coords[0]<coords[2]<=SIDE and TOP<=coords[1]<coords[3]<=BOTTOM and
            coords[2]-coords[0]>=10 and coords[3]-coords[1]>=10]
    donors.sort(key=lambda x:(x[0]!=wanted,abs(x[1][2]-x[1][0]-width),x[1]))
    present=[q for source,_,q in boxes if source==1]
    choice=None
    for c,q in donors:
        crop=tuple(round(v) for v in q)
        h=round(width*(crop[3]-crop[1])/(crop[2]-crop[0]))
        if not 10<=h<=158:
            continue
        shift=attempt%len(GRID)
        for cx,cy in GRID[shift:]+GRID[:shift]:
            x,y=cx-width//2,cy-h//2
            dx,dy=x+width,y+h
            if not (0<=x<dx<=SIDE and TOP<=y<dy<=BOTTOM):
                continue
            if any(max(x,a)<min(dx,d) and max(y,b)<min(dy,e) for a,b,d,e in present):
                continue
            choice=(c,crop,(x,y,dx,dy),h)
            break
        if choice is not None:
            break
    if choice is None:
        return batch,None

    c,(a,b,d,e),(x,y,dx,dy),h=choice
    with torch.no_grad():
        source,night=batch['img'].unbind(0)
        donor=f.interpolate(source[:,b:e,a:d][None],size=(h,width),mode='bilinear',align_corners=False)[0]
        background=night[:,y:dy,x:dx]
        gain=((background.mean()+.06)/(donor.mean()+1e-5)).clamp(.32,.78)
        fade=[]
        for size in (h,width):
            pos=torch.arange(size,device=donor.device)
            fade.append((torch.minimum(pos,size-1-pos).float()/2).clamp(0,1))
        alpha=fade[0][:,None]*fade[1][None,:]
        out=batch['img'].clone()
        out[0]=night
        out[0,:,y:dy,x:dx]=(background*(1-alpha)+donor*gain*alpha).clamp(0,1)
        m=batch['batch_idx']==1
        existing=batch['bboxes'][m]
        xywh=existing.new_tensor([[(x+dx)/2/SIDE,(y+dy)/2/SIDE,width/SIDE,h/SIDE]])
        klass=batch['cls'][m]; v=batch['batch_idx'][m]
        output={**batch,'img':out,'bboxes':torch.cat((existing,xywh,existing)),
                'cls':torch.cat((klass,klass.new_tensor([[c]]),klass)),
                'batch_idx':torch.cat((torch.zeros_like(v),v.new_zeros(1),v))}
        event={'class_id':c,'donor_rect':[a,b,d,e],'dest_rect':[x,y,dx,dy],
               'preserved_night_labels':len(existing),'gain':float(gain)}
    return output,event


def validate(cfg):
    check_equal('registered paste geometry and exposures',(2000,200,40,[20,28,40,56],60),
                tuple(cfg[k] for k in ('day_images','updates','max_minutes','widths','minimum_exposures')))


def fit(torch,cv2,payloads,plan,seed,original,cfg,until,phase):
    check_equal('fixed input image geometry',{(1920,1080)},
                {tuple(p[1:3]) for d in payloads.values() for p in d.values()})
    events={}

    def transform(batch,i):
        if i%2==0:
            return batch
        out,event=composite(batch,i//2,cfg['widths'])
        if event is not None:
            if not events:
                check_equal('original night image and annotations untouched',True,
                            torch.equal(out['img'][1],batch['img'][1]) and all(torch.equal(
                                out[k][out['batch_idx']==1],batch[k][batch['batch_idx']==1]) for k in ('bboxes','cls')))
            events[str(i)]=event
        return out

    path,record=night_focus.fit_native(torch,cv2,payloads,plan,seed,original,cfg,until,phase,batch_transform=transform)
    counts=Counter(v['class_id'] for v in events.values())
    return path,dict(record,paste_events=events,paste_event_counts=dict(counts),
                     paste_support_met=all(counts[k]>=cfg['minimum_exposures'] for k in (3,4)),
                     source_untouched_exposures=2000-len(events),distinct_night_images=len(payloads['night']))


if __name__=='__main__':
    night_rank.main(kind='paste',fit=fit,validate=validate,description=__doc__)
