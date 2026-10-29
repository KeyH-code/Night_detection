import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from source_contract import ContractError, read_json, sha256, write_json
from source_release import publish_ready


def fixture(tmp_path, status='passed', wrong_hash=False):
    archive = tmp_path/'kit.tar.gz'
    archive.write_bytes(b'fixture archive bytes, not real data')
    sidecar = tmp_path/'kit.tar.gz.json'
    qa = tmp_path/'qa.json'
    write_json(sidecar, {'status': 'PACKAGED_PENDING_VALIDATION', 'sha256': sha256(archive),
                        'ready_sha256': 'known frozen READY'})
    write_json(qa, {'status': status, 'archive_sha256': '0'*64 if wrong_hash else sha256(archive),
                   'ready_sha256': 'known frozen READY', 'final_tests_run': False})
    return archive, sidecar, qa


def test_failed_relocation_never_publishes_ready(tmp_path):
    archive, sidecar, qa = fixture(tmp_path, status='failed')
    before = sidecar.read_bytes()
    with pytest.raises(ContractError, match='must pass'):
        publish_ready(archive, sidecar, qa)
    assert sidecar.read_bytes() == before
    assert read_json(sidecar)['status'] == 'PACKAGED_PENDING_VALIDATION'


def test_wrong_archive_hash_never_publishes_ready(tmp_path):
    archive, sidecar, qa = fixture(tmp_path, wrong_hash=True)
    before = sidecar.read_bytes()
    with pytest.raises(ContractError, match='SHA'):
        publish_ready(archive, sidecar, qa)
    assert sidecar.read_bytes() == before


def test_success_atomically_publishes_actual_archive_hash(tmp_path):
    archive, sidecar, qa = fixture(tmp_path)
    result = publish_ready(archive, sidecar, qa)
    assert result['status'] == 'READY_FOR_CLOUD'
    assert result['sha256'] == sha256(archive) == read_json(sidecar)['sha256']
    assert not sidecar.with_suffix('.json.part').exists()


def negative_run(tmp_path):
    ident = {'s3_policy': 'synthetic identity'}
    write_json(tmp_path/'source_run.json', {'identity': ident})
    weight = tmp_path/'weights/last.pt'; weight.parent.mkdir(); weight.write_bytes(b'synthetic checkpoint')
    check = sha256(weight)
    write_json(weight.with_suffix('.pt.sha256.json'), {'sha256': check, 'epoch_0based': 1, 'identity': ident})
    roles = {role: {'path': 'weights/last.pt', 'epoch': 2, 'sha256': check, 'status': status}
             for role, status in [('last', 'complete_epoch_resumable'), ('diagnostic', 'diagnostic_not_qualification')]}
    write_json(tmp_path/'checkpoint_roles.json', roles)
    write_json(tmp_path/'final_eval/run.json', {'selected_role': 'diagnostic', 'roles': roles,
        'qualification': 'no_eligible_checkpoint', 'final_tests_run': False,
        'last_dev_checkpoint_sha256': check,
        'results': {p: {'images': n, 'checkpoint_sha256': check} for p,n in [('source_train',4000),('source_dev',500)]}})
    for purpose in ('source_train','source_dev'):
        for name in ('metrics.json','objects.jsonl.gz','matches.jsonl.gz','pr.npz'):
            (tmp_path/'final_eval'/purpose).mkdir(exist_ok=True)
            (tmp_path/'final_eval'/purpose/name).write_bytes(b'synthetic evidence fixture')
    return roles


def test_zero_eligible_run_is_deliverable_without_fabricating_best(tmp_path):
    from source_release import audit_training_roles
    negative_run(tmp_path)
    result = audit_training_roles(tmp_path)
    assert not result['eligible_checkpoint_exists'] and result['selected_role'] == 'diagnostic'
    assert not (tmp_path/'weights/best.pt').exists()
    (tmp_path/'weights/last.pt').write_bytes(b'changed')
    with pytest.raises(ContractError): audit_training_roles(tmp_path)


def test_diagnostic_weight_cannot_be_renamed_eligible_best_by_metadata(tmp_path):
    from source_release import audit_training_roles
    roles = negative_run(tmp_path)
    roles['best'] = dict(roles['diagnostic'])
    write_json(tmp_path/'checkpoint_roles.json', roles)
    with pytest.raises(ContractError, match='qualification'): audit_training_roles(tmp_path)


def test_cloud_handoff_reloads_roles_without_running_final_inference(tmp_path):
    import torch
    from types import SimpleNamespace
    from source_contract import NAMES, S3_EVALUATION
    from source_trainer import S3Trainer
    from source_release import audit_training_roles
    identity={'s3_policy':'synthetic','evaluation_policy':S3_EVALUATION}
    write_json(tmp_path/'source_run.json', {'identity':identity})
    model=torch.nn.Linear(2,2);model.names=dict(enumerate(NAMES))
    path=tmp_path/'weights/last.pt';path.parent.mkdir()
    dev={'metrics':{'images':500}}
    torch.save({'source_identity':identity,'epoch':1,'ema':model,'source_observation':{'dev':dev}},path)
    digest=sha256(path)
    write_json(path.with_suffix('.pt.sha256.json'),{'sha256':digest,'identity':identity,'epoch_0based':1})
    roles={r:{'path':'weights/last.pt','epoch':2,'sha256':digest,'status':'complete_epoch_resumable' if r=='last' else 'diagnostic_not_qualification'} for r in ('last','diagnostic')}
    write_json(tmp_path/'checkpoint_roles.json',roles)
    write_json(tmp_path/'observations/epoch002.json',{'dev':dev})
    def forbidden(*args,**kwargs):raise AssertionError('cloud finalization must not infer')
    trainer=SimpleNamespace(source_identity=identity,save_dir=tmp_path,last=path,s3_observer=SimpleNamespace(evaluate=forbidden))
    S3Trainer.final_eval(trainer)
    result=audit_training_roles(tmp_path)
    assert result['evaluation_status']=='pending_local_evaluation'
    final=read_json(tmp_path/'final_eval/run.json')
    assert final['results']=={} and final['weight_reload_checked']
    identity.pop('evaluation_policy')
    write_json(tmp_path/'source_run.json',{'identity':identity})
    write_json(path.with_suffix('.pt.sha256.json'),{'sha256':digest,'identity':identity,'epoch_0based':1})
    with pytest.raises(ContractError,match='deferred evaluation identity'):
        audit_training_roles(tmp_path)


def test_local_export_retains_cloud_roles_and_thresholds_after_device_drift(tmp_path):
    import copy
    from types import SimpleNamespace
    import detection_metrics as dm
    from source_contract import S3_EVALUATION
    from source_policy import load_rules,summarize
    from source_trainer import S3Trainer
    rules=load_rules(Path(__file__).resolve().parents[2]/'RULES.md')
    boxes=[[i*20.,0.,i*20.+5.,10.] for i in range(6)]
    row={'key':'one','purpose':'source_dev','group_id':'g','width':640,'height':640,
         'gt':{'cls':list(range(6)),'boxes':boxes,'original_boxes':boxes},
         'pred':{'cls':list(range(6)),'conf':[.9]*6,'boxes':boxes}}
    flags,pairs=dm.ap_flags(row);row.update(ap_tp=flags.tolist(),ap_pairs=pairs)
    cloud_record=summarize([row],rules)[0]
    for p in cloud_record['workpoints'].values():p.update(threshold=.95,t_rule=.95)
    roles=negative_run(tmp_path)
    identity={'s3_policy':'synthetic','evaluation_policy':S3_EVALUATION}
    write_json(tmp_path/'source_run.json',{'identity':identity})
    write_json(tmp_path/'observations/epoch002.json',{'dev':cloud_record})
    calls=[]
    class Observer:
        def evaluate(self,purpose,model,points=None):
            calls.append((purpose,copy.deepcopy(points)))
            current=copy.deepcopy(row);current['purpose']=purpose
            result,matches,curves=summarize([current],rules,points)
            result['native_f1']={}
            return result,matches,curves,[current]
    observer=Observer();observer.rules=rules
    t=SimpleNamespace(source_identity=identity,save_dir=tmp_path,last=tmp_path/'weights/last.pt',
                      s3_local_export=True,s3_observer=observer,s3_observation={'dev':cloud_record})
    before={p:sha256(tmp_path/p) for p in {r['path'] for r in roles.values()}}
    S3Trainer.final_eval(t)
    assert [p for p,_ in calls]==['source_dev','source_train','source_dev']
    assert calls[1][1]['pedestrian']['threshold']==.95
    assert calls[2][1]['pedestrian']['threshold']==.95
    dev=read_json(tmp_path/'final_eval/source_dev/metrics.json')
    assert dev['workpoints']['pedestrian']['tp']==0
    assert dev['local_reselected_workpoints']['pedestrian']['threshold']<.95
    assert read_json(tmp_path/'final_eval/run.json')['cross_device']['selected_workpoint_counts_equal'] is False
    assert read_json(tmp_path/'checkpoint_roles.json')==roles
    assert {p:sha256(tmp_path/p) for p in before}==before


@pytest.mark.parametrize('broken', ['none','dev','train','worker','pr'])
def test_dev_only_cloud_seal_requires_complete_dev_and_consumption(tmp_path,broken):
    from source_contract import S3_EVALUATION
    from source_release import seal_training
    from train_monitor import append_epoch
    identity={'s3_policy':'synthetic','evaluation_policy':S3_EVALUATION}
    write_json(tmp_path/'source_run.json',{'identity':identity})
    weight=tmp_path/'weights/last.pt';weight.parent.mkdir();weight.write_bytes(b'synthetic checkpoint')
    digest=sha256(weight)
    write_json(weight.with_suffix('.pt.sha256.json'),{'sha256':digest,'identity':identity,'epoch_0based':1})
    roles={r:{'path':'weights/last.pt','epoch':2,'sha256':digest,'status':'complete_epoch_resumable' if r=='last' else 'diagnostic_not_qualification'} for r in ('last','diagnostic')}
    write_json(tmp_path/'checkpoint_roles.json',roles)
    write_json(tmp_path/'final_eval/run.json',{'status':'pending_local_evaluation','selected_role':'diagnostic',
        'roles':roles,'results':{},'last_dev_checkpoint_sha256':digest,'weight_reload_checked':True,
        'final_tests_run':False,'qualification':'no_eligible_checkpoint'})
    for epoch in (1,2):
        pr=tmp_path/f'observations/epoch{epoch:03d}.npz';pr.parent.mkdir(exist_ok=True);pr.write_bytes(b'fixture PR')
        consumed=tmp_path/f'sampling/epoch{epoch:03d}.npz';consumed.parent.mkdir(exist_ok=True);consumed.write_bytes(b'fixture sequence')
        write_json(pr.with_suffix('.json'),{'epoch':epoch,'dev':{'metrics':{'images':499 if broken=='dev' else 500}},
            'train':{'status':'deferred_local','images_evaluated':0},'pr_sha256':sha256(pr)})
        write_json(consumed.with_suffix('.json'),{'anchor':{'slots':4000},'consumed_sha256':sha256(consumed),
            'worker_exit':[{'alive':False,'exitcode':1 if broken=='worker' else 0}]*4})
        append_epoch(tmp_path/'dev_metrics.csv',{'epoch':epoch,'images':500})
    write_json(tmp_path/'train_monitor.json',{'source_identity':identity,'schema_version':3,'epoch':2,
        'batch':1,'amp_skipped_steps':0,'gradient_batches':500,'full_dev_images':500,
        'full_train_images':4000 if broken=='train' else 0,'overhead_ratio':1.})
    (tmp_path/'timings.jsonl').write_text(''.join(__import__('json').dumps(
        {'epoch':e,'epoch_seconds':10.,'evaluation_seconds':5.})+'\n' for e in (1,2)),encoding='utf-8')
    (tmp_path/'source_metrics.jsonl').write_text('{"epoch":1}\n{"epoch":2}\n',encoding='utf-8')
    if broken=='pr':(tmp_path/'observations/epoch002.npz').write_bytes(b'tampered')
    if broken!='none':
        with pytest.raises(ContractError):seal_training(tmp_path,'preflight')
        assert not (tmp_path/'manifest.json').exists()
    else:
        sealed=seal_training(tmp_path,'preflight')
        assert sealed['status']=='completed_preflight'
        assert sealed['evaluation_status']=='pending_local_evaluation'
        assert not (tmp_path/'train_metrics.csv').exists()


def test_local_finalizer_verifies_handoff_preserves_cloud_record_and_reseals(tmp_path,monkeypatch):
    import sys,importlib.metadata
    import torch
    import source_contract,train_monitor
    from source_contract import S3_EVALUATION
    from source_policy import load_rules,reference_at
    from source_trainer import S3Trainer
    from source_release import audit_training_roles
    from train_source import code_identity
    root=Path(__file__).resolve().parents[2]
    ready=tmp_path/'data/READY.json';write_json(ready,{'synthetic':True})
    rules=load_rules(root/'RULES.md')
    identity={'scope':'formal','evaluation_policy':S3_EVALUATION,'code':code_identity(),
              'ready_sha256':sha256(ready),'s3_policy':{'rules_sha256':rules['sha256']}}
    write_json(tmp_path/'source_run.json',{'identity':identity,'scope':'formal'})
    weight=tmp_path/'weights/last.pt';weight.parent.mkdir();weight.write_bytes(b'synthetic role')
    digest=sha256(weight)
    write_json(weight.with_suffix('.pt.sha256.json'),{'sha256':digest,'identity':identity,'epoch_0based':99})
    roles={r:{'path':'weights/last.pt','epoch':100,'sha256':digest,
              'status':'complete_epoch_resumable' if r=='last' else 'diagnostic_not_qualification'} for r in ('last','diagnostic')}
    write_json(tmp_path/'checkpoint_roles.json',roles)
    write_json(tmp_path/'observations/epoch100.json',{'dev':{'workpoints':{},'metrics':{}}})
    environment={'python':'.'.join(sys.version.split()[0].split('.')[:2]),
        **{n:importlib.metadata.version(n) for n in ('torch','torchvision','ultralytics')}}
    write_json(tmp_path/'audit_initial/config_resolved.json',{'environment':environment})
    write_json(tmp_path/'policy_inputs.json',{'rules':rules,'reference':reference_at(root)})
    handoff={'status':'pending_local_evaluation','selected_role':'diagnostic','roles':roles,'results':{},
             'last_dev_checkpoint_sha256':digest,'weight_reload_checked':True,'final_tests_run':False,
             'qualification':'no_eligible_checkpoint'}
    write_json(tmp_path/'final_eval/run.json',handoff);original=sha256(tmp_path/'final_eval/run.json')
    write_json(tmp_path/'manifest.json',{'status':'completed_formal','identity':identity,
        'files':{p.relative_to(tmp_path).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
                 for p in tmp_path.rglob('*') if p.is_file()}})
    monkeypatch.setattr(source_contract,'verify_source',lambda *_:{'synthetic':True})
    def stage(_,folder):
        folder.mkdir(parents=True);write_json(folder/'dataset.yaml',{})
        return folder/'dataset.yaml',{}
    monkeypatch.setattr(source_contract,'stage_dataset',stage)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'synthetic GPU')
    def export(t):
        record={'metrics':{'per_class':{n:{'instances':100,'groups':80,'ap50':.96,'ap75':.8,'ap50_95':.8}
                        for n in source_contract.NAMES}},
                'workpoints':{n:{'origin':'rule','t_rule':.5,'precision':.95,'recall':.9,'tp':90,'fp':4}
                              for n in source_contract.NAMES},
                'slices':[{'class':'pedestrian','slice':s,'recall50':.8} for s in ('area_S','width_le8')]}
        for purpose in ('source_train','source_dev'):
            folder=t.save_dir/'final_eval'/purpose;folder.mkdir()
            write_json(folder/'metrics.json',record)
            for name in ('objects.jsonl.gz','matches.jsonl.gz','pr.npz'):(folder/name).write_bytes(b'synthetic export mock')
        write_json(t.save_dir/'final_eval/run.json',{**handoff,'status':'complete_local',
            'results':{p:{'images':n,'checkpoint_sha256':digest} for p,n in [('source_train',4000),('source_dev',500)]}})
    monkeypatch.setattr(S3Trainer,'final_eval',export)
    monkeypatch.setattr(train_monitor,'audit_observation',lambda path:{'epochs':100,'roles':audit_training_roles(path)})
    result=train_monitor.finalize_s3_local(tmp_path,ready.parent,sha256(ready))
    assert result['status']=='complete_local'
    assert sha256(tmp_path/'final_eval/cloud.json')==original
    assert sha256(weight)==digest and read_json(tmp_path/'checkpoint_roles.json')==roles
    assert read_json(tmp_path/'manifest.json')['evaluation_status']=='complete_local'
