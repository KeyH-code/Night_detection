"""Prepare the semantic-alias initialization, or train one gated full source-A seed."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import math
import csv
import hashlib

from seed_train import COCO, READY, ORIGINAL_FILES, ROOT, collect, inventory, replica_config
from source_contract import ContractError,NAMES,check_equal,read_json,sha256,stage_dataset,verify_source,write_json


def pack(path):
    import hashlib
    from train_source import code_identity
    if path.exists() or path.resolve().is_relative_to(ROOT):raise ContractError('Use a fresh external package path')
    files=(*code_identity(),*ORIGINAL_FILES,'scripts/source_resources.py','scripts/seed_train.py','scripts/source_alias.py','environment/requirements-linux.lock','specs/s3/scale.md')
    with tarfile.open(path,'x') as ar:
        for f in sorted(files):
            p=ROOT/f
            if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ContractError('Invalid package input '+f)
            if f in ORIGINAL_FILES:check_equal('original input '+f,ORIGINAL_FILES[f],sha256(p))
            ar.add(p,arcname=f,recursive=False)
    with tarfile.open(path,'r:') as ar:
        check_equal('exact tar members',sorted(files),ar.getnames())
        for item in ar:
            check_equal('regular tar entry',True,item.isfile());check_equal(item.name,sha256(ROOT/item.name),hashlib.file_digest(ar.extractfile(item),'sha256').hexdigest())
    print('ALIAS_BUNDLE_VERIFIED',len(files),'files',path.stat().st_size,'bytes',sha256(path),flush=True)


def initialization(weights,scratch,seed):
    import torch
    from ultralytics.nn.tasks import DetectionModel,load_checkpoint
    check_equal('official COCO initialization',COCO,sha256(weights));torch.manual_seed(seed);torch.set_num_threads(4)
    src,_=load_checkpoint(str(weights),device='cpu');names=dict(src.names);check_equal('COCO class zero','person',names[0]);check_equal('literal overlap',{'car','truck'},set(names.values())&set(NAMES))
    base=DetectionModel(deepcopy(src.yaml),nc=6,verbose=False);base.names=dict(enumerate(NAMES));alias=deepcopy(base);base.load(src,verbose=False)
    before={k:v.clone() for k,v in src.state_dict().items()};src.names=dict(names);src.names[0]='pedestrian';aliased=scratch/'source_alias_init.pt'
    torch.save({'model':src.half().eval(),'epoch':-1,'train_args':src.args,'scope':'COCO class zero renamed only for lookup, unchanged tensors'},aliased)
    restored,_=load_checkpoint(str(aliased),device='cpu');check_equal('one renamed COCO class',names|{0:'pedestrian'},restored.names)
    if not all(torch.equal(v,restored.state_dict()[k]) for k,v in before.items()):raise ContractError('COCO tensors changed across alias export')
    alias.load(restored,verbose=False);changed=[]
    for k,a in alias.state_dict().items():
        b=base.state_dict()[k]
        if torch.equal(a,b):continue
        if a.shape[0]!=6 or not torch.equal(a[1:],b[1:]):raise ContractError('Difference beyond pedestrian row '+k)
        changed.append(k)
    for a,c in zip(alias.model[-1].cv3,restored.model[-1].cv3,strict=True):
        for k in ('weight','bias'):check_equal('exact COCO person '+k,True,torch.equal(getattr(a[-1],k)[0],getattr(c[-1],k)[0]))
    check_equal('three scale weights+bias',6,len(changed))
    return aliased,{'original_sha256':COCO,'alias_sha256':sha256(aliased),'aliases':{'person':'pedestrian'},'same_coco_tensors':True,
                'changed_destination_tensors':changed,'destination_changed_row':0,'destination_other_rows_exact':True}


def reused_preflight(previous,expected,out,proof,ct):
    from train_source import code_identity
    folder=previous/'preflight';p=folder/'estimate.json';e=read_json(p);o=e['observation'];monitor=read_json(folder/'train_monitor.json');old=read_json(previous/'initialization.json');prior_env=read_json(previous/'environment/environment.json');env=read_json(out/'environment/environment.json')
    check_equal('same preflight seal',expected,sha256(p));check_equal('preflight artifacts',(2,o['timing_sha256'],o['train_metrics_sha256']),(o['epochs'],sha256(folder/'train_monitor.json'),sha256(folder/'train_metrics.csv')))
    check_equal('alias preflight previously passed without formal training',True,read_json(previous/'status.json')['no_formal_training'])
    for field in ('original_sha256','aliases','changed_destination_tensors','destination_other_rows_exact','same_coco_tensors'):
        check_equal('alias identity '+field,old[field],proof[field])
    for field in ('gpu','gpu_memory_bytes','cuda_runtime','torch','torchvision','ultralytics','python_full'):
        check_equal('preflight environment '+field,prior_env[field],env[field])
    ident=monitor['source_identity'];check_equal('preflight training core',(READY,ct['ready']['data_fingerprint'],code_identity(),16),(ident['ready_sha256'],ident['data_fingerprint'],ident['code'],ident['batch']))
    with (folder/'train_metrics.csv').open(encoding='utf-8',newline='') as f:check_equal('full preflight images',[(1,4000),(2,4000)],[(int(r['epoch']),int(r['images'])) for r in csv.DictReader(f)])
    required=float(e['conservative_required_seconds_at_100_epochs_with_transfer']);check_equal('conservative cost formula',float(e['real_preflight_seconds'])*50+720,required)
    if not math.isfinite(required) or required<=720:raise ContractError('Invalid preflight timing')
    return {'source':str(previous),'estimate_sha256':expected,'initialization':old,'observation':o,'required_seconds':required}


def observer_benchmark(dataset,selected,alias,tmp,seed):
    """Change only read workers, require ordered per-image input and output identity before use."""
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.nn.tasks import DetectionModel,load_checkpoint
    from evaluate import validator_args
    torch.manual_seed(seed);pre,_=load_checkpoint(str(alias),device='cpu');model=DetectionModel(deepcopy(pre.yaml),nc=6,verbose=False);model.names=dict(enumerate(NAMES));model.load(pre,verbose=False);model.args=pre.args;model.eval();weight=tmp/'comparison.pt';torch.save({'model':model.half(),'train_args':pre.args},weight)
    images=sorted((dataset.parent/r['image']).as_posix() for r in selected['source_train'])[:256];listing=tmp/'observer_images.txt';listing.write_text('\n'.join(images)+'\n',encoding='utf-8');config=tmp/'observer.yaml';write_json(config,{'path':tmp.as_posix(),'train':listing.as_posix(),'val':listing.as_posix(),'names':NAMES,'nc':6})
    class Observer(DetectionValidator):
        def update_metrics(self,preds,batch):
            for i,pred in enumerate(preds):
                self.order.append(Path(batch['im_file'][i]).name)
                x=batch['img'][i].detach().cpu().contiguous().numpy();self.input_hash.update(x.tobytes())
                for k in ('bboxes','cls','conf'):
                    x=pred[k].detach().cpu().contiguous().numpy();self.output_hash.update(str((x.shape,x.dtype)).encode());self.output_hash.update(x.tobytes())
            super().update_metrics(preds,batch)
    results={}
    for n in (0,4):
        val=Observer(args=validator_args(config,batch=16,device='0',workers=n),save_dir=tmp/f'bench{n}');val.order=[];val.input_hash=hashlib.sha256();val.output_hash=hashlib.sha256();clock=time.monotonic();val(model=str(weight));elapsed=time.monotonic()-clock
        check_equal('observer covers sample',sorted(Path(i).name for i in images),sorted(val.order));check_equal('actual observer workers',n,val.dataloader.num_workers)
        check_equal('deterministic validation transforms',False,val.dataloader.dataset.augment);results[n]={'images':val.seen,'batches':len(val.dataloader),'order_sha256':hashlib.sha256('\n'.join(val.order).encode()).hexdigest(),'input_sha256':val.input_hash.hexdigest(),'output_sha256':val.output_hash.hexdigest(),'seconds':elapsed};del val
    same=all(results[0][k]==results[4][k] for k in ('images','batches','order_sha256','input_sha256','output_sha256'))
    return {'workers':4 if same and results[4]['seconds']<results[0]['seconds'] else 0,'bitwise_equal':same,'results':results}


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--bundle',type=Path);p.add_argument('--source',type=Path);p.add_argument('--weights',type=Path,default=ROOT/'weights/yolo11m.pt');p.add_argument('--scratch',type=Path);p.add_argument('--seed',type=int,choices=(17,29,43),default=29);p.add_argument('--output',type=Path);p.add_argument('--deadline',type=float,help='Verified platform shutdown Unix time')
    p.add_argument('--preflight-run',type=Path,help='Validated persistent run with measured, matching preflight');p.add_argument('--expected-preflight-sha')
    a=p.parse_args()
    if a.bundle:pack(a.bundle);return
    if not a.scratch:p.error('--scratch required for verification or training')
    scratch=a.scratch.resolve()
    if scratch.is_relative_to(ROOT) or not scratch.is_dir():p.error('Existing external scratch directory required')
    if not a.check and (a.source is None or a.output is None or a.output.exists() or sys.platform!='linux'):p.error('Training requires Linux, a verified source and a fresh output')
    if not a.check and (not a.deadline or not math.isfinite(a.deadline) or time.time()+20*60>=a.deadline):p.error('Require an active verified shutdown deadline with preflight and transfer reserve')
    from train_source import configure_paths,environment,run_segment,code_identity
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        tmp=Path(d);os.environ['NIGHT_SOURCE_SCRATCH']=str(tmp);configure_paths();alias,proof=initialization(a.weights.resolve(),tmp,a.seed)
        if a.check:print('ALIAS_VERIFIED',proof,flush=True);return
        if a.output.resolve().is_relative_to(a.source.resolve()) or a.output.resolve().is_relative_to(scratch):
            raise ContractError('Training evidence must be persistent and outside the frozen dataset and transient scratch')
        import psutil
        from source_resources import snapshot,shared_memory_risk
        quota=snapshot(os.cpu_count(),psutil.virtual_memory().total)
        cfg,original=replica_config(29);cfg['training']['seed']=a.seed;cfg['model']['path']=str(alias);cfg['model']['initialization_aliases']=proof['aliases'];cfg['monitor']={'enabled':True,'every':1,'batch':16,'workers':0}
        if quota['cpu_count']<cfg['cloud']['min_cpus'] or quota['ram_bytes']/2**30<cfg['cloud']['min_ram_gib'] or shared_memory_risk(quota):
            raise ContractError('Effective Linux CPU, RAM or /dev/shm below original source-A requirement')
        out=a.output.resolve();out.mkdir();environment(cfg,out/'environment',cloud=True);ct=verify_source(a.source.resolve(),READY)
        check_equal('gradient and dev counts',(4000,500),tuple(len(ct['manifests'][k]) for k in ('source_train','source_dev')))
        work=out/'work';write_json(out/'initialization.json',proof)
        code=out/'code';files=(*code_identity(),*ORIGINAL_FILES,'scripts/source_alias.py','scripts/seed_train.py','specs/s3/scale.md','environment/requirements-linux.lock')
        for n in files:
            dest=code/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
        from seed_train import clean_promoted
        from train_monitor import audit_observation
        with tempfile.TemporaryDirectory(prefix='sandbox-',dir=out) as data_dir:
            dataset,selected=stage_dataset(ct,Path(data_dir)/'data');receipt=out/'preflight';receipt.mkdir()
            if a.preflight_run:
                reuse=reused_preflight(a.preflight_run,a.expected_preflight_sha,out,proof,ct);required=reuse['required_seconds'];observation=reuse['observation'];write_json(receipt/'reuse.json',reuse)
                write_json(out/'status.json',{'status':'observer_check','seed':a.seed,'recipe':'coco_person_alias'})
                bench=observer_benchmark(dataset,selected,alias,tmp,a.seed);write_json(receipt/'workers.json',bench);cfg['monitor']['workers']=bench['workers']
            else:
                probe=out/'probe';clock=time.monotonic();write_json(out/'status.json',{'status':'preflight','seed':a.seed,'recipe':'coco_person_alias','ready':READY})
                run_segment(cfg,ct,{'sha256':sha256(alias)},probe,dataset,16,'preflight');observation=audit_observation(probe);check_equal('two complete observed epochs',2,observation['epochs'])
                clock=time.monotonic()-clock
                for f in ('train_metrics.csv','train_monitor.json','results.csv','source_run.json','audit_initial/runtime.json'):
                    dest=receipt/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(probe/f,dest)
                required=clock/2*100+12*60;write_json(receipt/'estimate.json',{'real_preflight_seconds':clock,'observation':observation,'conservative_required_seconds_at_100_epochs_with_transfer':required,'deadline':a.deadline});clean_promoted(probe,out)
            if time.time()+required>=a.deadline:
                write_json(out/'status.json',{'status':'window_insufficient','no_formal_training':True,'preflight':observation});raise ContractError('Measured full train/dev observation exceeds remaining platform window; keep receipt, do not launch formal training')
            write_json(out/'status.json',{'status':'training','seed':a.seed,'recipe':'coco_person_alias','ready':READY})
            run_segment(cfg,ct,{'sha256':sha256(alias)},work,dataset,16,'formal');saved=collect(work,out/str(a.seed),original,a.seed);audit=audit_observation(work);check_equal('full observation coverage',saved['epochs'],audit['epochs'])
            for f in ('train_metrics.csv','train_monitor.json'):shutil.copyfile(work/f,out/str(a.seed)/f)
        clean_promoted(work,out)
        write_json(out/'status.json',{'status':'complete','seed':a.seed,'run':saved});write_json(out/'manifest.json',{'status':'complete','scope':'single original-A training with explicit person alias, pending local dev replay','seed':a.seed,'run':saved,'ready':READY,'initialization':proof,'final_test_executed':False,'files':inventory(out)})


if __name__=='__main__':main()
