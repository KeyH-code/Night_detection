"""Test label-blind selection driven by class evidence shared across native image views."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

import detection_metrics as dm
import night_adapt as adapt
import night_rank
import night_strategy as strategy
import night_tail as tail
import night_train as train
from night_contract import POOL, POOL_MANIFEST, ROOT, _records, read_sealed_json, stage_selection
from source_contract import ContractError, canonical_hash, check_equal

VIEWS = ('original','bright','flip')


def score_presence(original, variants):
    x = strategy._boxes(original)
    scores = np.zeros(5,dtype=np.float64)
    for v in variants:
        y = strategy._boxes(v)
        if not len(x) or not len(y):
            continue
        for c in range(5):
            a,b = x[x[:,5]==c],y[y[:,5]==c]
            if not len(a) or not len(b):
                continue
            iou = dm.box_iou(a[:,:4], b[:,:4])
            q = np.minimum(a[:,4,None],b[None,:,4])*iou
            scores[c] = max(scores[c], q[iou >= .5].max(initial=0.))
    return scores


def capture(box, digest, *, restricted_score=None):
    import cv2
    import torch
    import torchvision
    import ultralytics
    import night_capture as view

    check_equal('frozen selection versions', view.VERSIONS, (torch.__version__,torchvision.__version__,ultralytics.__version__))
    model = view._open_model(torch)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    pool = strategy.BlindPool(box,digest,994)

    def predict(image,name):
        data,geom = view.rect_pixels(view.transform(image,name))
        x = torch.from_numpy(data).unsqueeze(0).to(device='cuda:0',dtype=torch.float16).div_(255)
        return view.restore(view.native_detections(model(x)),geom,name=='flip')

    sample = np.full((320,640,3),114,np.uint8)
    ok, pixels = cv2.imencode('.jpg',sample)
    check_equal('synthetic JPEG warmup',True,ok)
    cv2.imdecode(pixels,cv2.IMREAD_COLOR)
    with torch.inference_mode():
        toy = predict(sample,'original')
        score_presence(toy,[toy,toy])
    paths = [pool.root/r['image'] for r in pool.records.values()]
    read = strategy.worker_files(readable=paths)
    start=time.monotonic()
    result=[]
    with torch.inference_mode():
        for k in pool.records:
            im = cv2.imdecode(np.frombuffer(pool.read(k),dtype=np.uint8),cv2.IMREAD_COLOR)
            if im is None:
                raise ContractError(f'Invalid image under frozen key {k}')
            try:
                raw, bright, mirror = (predict(im,v) for v in VIEWS)
            except ContractError as exc:
                raise ContractError(f'Blind presence capture {k}: {exc}') from exc
            keep = (score_presence(raw,[bright,mirror]) if restricted_score is None else
                    restricted_score(raw,[bright,mirror],640/max(im.shape[:2])))
            result.append(keep.tolist())
    check_equal('only approved pool images opened in frozen order',paths,read)
    print(json.dumps({'keys':list(pool.records),'scores':result,'views':VIEWS,'seconds':time.monotonic()-start,
                      'images_read':len(read),'labels_read':False,'a_sha256':view.A_SHA,'capture':view.INFERENCE,
                      'score_mode':'presence' if restricted_score is None else 'medium_width'}),flush=True)


def validate(cfg):
    check_equal('registered presence treatment',(2000,200,[12,12,8,18,30],96,40),
                tuple(cfg[k] for k in ('day_images','updates','quotas','shortlist','max_minutes')))


def blind_inputs(cfg,scratch, *, worker=__file__):
    from seed_eval import probe_gpu

    _, pair, old = adapt._config(cfg['pair_sha256'])
    protocol=train.load_protocol()
    digest=protocol['data']['pool_images_sha256']
    rows=read_sealed_json(train.NIGHT,POOL_MANIFEST,digest,[])
    inv={r['key']:r for r in _records(rows,POOL,994,images_only=True)}
    folders={k:ROOT/'runs'/p for k,p in (('e1','pool_002'),('d','pool_001'))}
    seals={k:read_sealed_json(p,'manifest.json',pair['pool_manifest_sha256'][k],[]) for k,p in folders.items()}
    check_equal('sealed image-only parents',('sealed','sealed'),tuple(seals[k]['status'] for k in ('e1','d')))
    check_equal('fixed random anchor manifest',old['random_sha256'],seals['d']['files']['random.json']['sha256'])
    features=night_rank.accept._checked_file(folders['e1'],seals['e1'],'features.npz',[])
    keys,z=strategy.load_features(io.BytesIO(features))
    draws={k:adapt._json(folders[k],seals[k],name) for k,name in (('d','random.json'),('e1','e1.json'))}
    probe_gpu()
    with tempfile.TemporaryDirectory(dir=scratch,prefix='presence-') as parent:
        parent=Path(parent); box=parent/'blind'
        selection=stage_selection(train.NIGHT,digest,box)
        env={**os.environ, 'YOLO_AUTOINSTALL':'false','WANDB_MODE':'disabled','TEMP':str(parent),'TMP':str(parent),
             'YOLO_CONFIG_DIR':str(parent),'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1'}
        result=subprocess.run([sys.executable,'-B',str(Path(worker).resolve()),'capture',str(box),digest],
                              cwd=parent, env=env, capture_output=True,text=True,encoding='utf-8',check=True,timeout=900)
        captured=json.loads(result.stdout.strip().splitlines()[-1])
    check_equal('blind capture identity and complete keys',(keys,list(VIEWS),994,False,protocol['source_a']['checkpoint_sha256']),
                tuple(captured[k] for k in ('keys','views','images_read','labels_read','a_sha256')))
    scores=strategy._scores(keys,captured['scores'])
    memo={'selection_labels_read':False,'selection_development_read':False,'images_sha256':digest,
          'feature_sha256':seals['e1']['files']['features.npz']['sha256'],'image_staging_paths':len(selection['reads']),
          'presence':captured}
    return keys,z,scores,inv,draws,memo


def select(cfg,scratch):
    keys,z,scores,inv,draws,memo=blind_inputs(cfg,scratch)
    digest=memo['images_sha256']
    chosen={}
    for seed in cfg['seeds']:
        d,base=([r['key'] for r in adapt.draw_records(k,seed,draws[k],digest,inv)] for k in ('d','e1'))
        def rank():
            return tail.rank_quota(keys,z,scores,d[:20],cfg['shortlist'],cfg['quotas'],first_score=True)
        q=rank(); check_equal('replayed presence rank',q,rank())
        records=[inv[k] for k in q['keys']]
        check_equal('presence list is exact and distinct',(100,100,d[:20],cfg['quotas']),
                    (len(records),len(set(q['keys'])),q['keys'][:20],q['per_class']))
        if set(q['keys']) in (set(d),set(base)):
            raise ContractError('Presence selection has no distinct set to train')
        chosen[seed]={**q,'records':records,'records_sha256':canonical_hash(records),'original_e1_overlap':len(set(q['keys'])&set(base)),
                      'd_overlap':len(set(q['keys'])&set(d)),'risk_groups':len({r['group_id'] for r in records})}
        print(f'Presence seed {seed}: {len(records)} valid blind selections, {chosen[seed]["original_e1_overlap"]} images shared with E1.',flush=True)
    return {'lists':chosen,**memo}


if __name__ == '__main__':
    if len(sys.argv)==4 and sys.argv[1]=='capture':
        capture(Path(sys.argv[2]).absolute(),sys.argv[3])
    else:
        night_rank.main(kind='presence',fit=tail.fit,validate=validate,select=select,description=__doc__)
