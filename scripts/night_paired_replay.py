"""Independently recompute S7 340-image training consumption, paired effects and protection."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path

import detection_metrics as dm
import night_paired as fit
from night_contract import read_sealed_json
from source_contract import canonical_hash, check_equal, read_json, sha256, write_json


def verify_training(path,s,arm,ctx):
    state,_,_,_,chosen,plans,day,night=ctx;plan=plans[arm];get=lambda n:fit.blind.sealed(path,s,n)
    receipt=get('run.json');t=get('43/train.json');inp=get('43/inputs.json');p=get('43/preflight.json');nr=inp['night']
    nightkeys=list(dict.fromkeys(n[0] for _,n in plan));daykeys=[d[0] for d,_ in plan]
    check_equal('complete local 43 experiment',('complete','complete',arm,43,False,False,sha256(fit.PAIR/'protocol.json')),
                (s['status'],receipt['status'],receipt['arm'],receipt['seed'],receipt['cloud_used'],receipt['final_test_executed'],receipt['protocol_sha256']))
    check_equal('full training plan and input audit',(nightkeys,daykeys,canonical_hash(plan),canonical_hash(plan),state['source_train'],
                [night[k][typ] for k in nightkeys for typ in ('image','label')],canonical_hash([day[k][typ] for k in daykeys for typ in ('image','label')])),
                ([r['key'] for r in nr['records']],inp['source_keys'],inp['schedule_sha256'],t['augment_schedule_sha256'],inp['source_manifest'],nr['source_reads'],inp['source_reads_sha256']))
    for r in nr['records']:
        k=r['key'];check_equal('staged hashes, target purpose and group',tuple(night[k][i] for i in ('image_sha256','label_sha256','purpose','group_id')),tuple(r[i] for i in ('image_sha256','label_sha256','purpose','group_id')))
    check_equal('actual 2000 pairs, 200 updates, fixed-last and A',([[d[0],n[0]] for d,n in plan],200,0,state['a_sha256'],s['files']['43/last.pt']['sha256'],False,2000),
            (t['microbatches'],t['updates'],t['skipped_updates'],t['initializer_sha256'],t['last_sha256'],t['checkpoint_roles']['best']['selected_on_dev'],t['floor_signal']['calls']))
    check_equal('every intended exposure and no extras',({'source':dict.fromkeys(daykeys,1),'night':dict(Counter(n[0] for _,n in plan))},list(range(1,201))),
                (t['counts'],[r['step'] for r in t['trace']]))
    new=set(chosen[arm]);check_equal('real no-update data coverage',(0,True,40,new),(p['updates'],p['weights_unchanged'],p['new_images'],{plan[i][1][0] for i in p['microbatch_indices']}&new))
    fit.blind.require(p['gradient_checked_batches']>0 and p['active_gt']>0 and t['floor_signal']['active_gt']>0,'No verified floor supervision')
    return {'updates':t['updates'],'skips':t['skipped_updates'],'night_unique':len(nightkeys),'protocol_sha256':receipt['protocol_sha256'],
            'wall_seconds':receipt['seconds'],'training_seconds':t['seconds'],'weight_sha256':t['last_sha256']}


def dev_replay(path,seal,approved,policy,ref):
    get=lambda n:fit.blind.sealed(path,seal,'43/dev/'+n)
    ev=get('evaluation.json');rows={};metrics={}
    for domain,purpose in [('day','source_dev'),('night','night_dev')]:
        rr=fit.ac.sealed_objects(path,seal,'43/dev/'+domain+'/objects.jsonl.gz',purpose,approved[purpose],[])
        m=fit.ac.evaluate_dev(rr,policy);m['capture']=fit.ac._audit_views(ev[domain+'_views'],ev[domain+'_fusion_caps']);check_equal('recomputed day/night metrics and truncation',m,get(domain+'/metrics.json'))
        rows[domain]=rr;metrics[domain]=m
    assessment=fit.ac._assessment(ref,metrics['day'],metrics['night'],policy);check_equal('recomputed day and night gates',assessment,ev['assessment'])
    result={'assessment':assessment,'night':{},'day_ap50_95':{n:v['ap50_95'] for n,v in metrics['day']['per_class'].items()},'day_macro_ap50_95':metrics['day']['macro_ap50_95']};hits={};pts=metrics['night']['workpoints']
    for cid,name in enumerate(fit.ac.CLASSES):
        p=pts[name];cur=dm.work_curve(rows['night'],cid);needed=fit.ac._recall(ref['workpoints'][name])*Fraction(policy['recovery_ratio'])*cur['gt'];floor=-(-needed.numerator//needed.denominator)
        hits[name]={(r['key'],r['gt']['ids'][g]) for r in rows['night'] for g,_ in dm.image_point(r,cid,p['threshold'])[1]};check_equal('exact TP identities',p['tp'],len(hits[name]));fps=cur['fp'][cur['tp']>=floor]
        result['night'][name]={'tp':p['tp'],'fp':p['fp'],'gt':p['tp']+p['fn'],'target_tp':floor,'min_fp_to_target':int(fps.min()) if len(fps) else None,'recall':p['recall'],'threshold':p['threshold']}
    return result,rows,pts,hits


def replay():
    ctx=fit.load_inputs();state,_,pair,*_=ctx;check_equal('source registrations',fit.lock(ctx),read_json(fit.PAIR/'protocol.json'))
    af=fit.RUN/'accept_001';a=read_sealed_json(af,'manifest.json',pair['accept_manifest_sha256'],[]);policy,ref=(fit.blind.sealed(af,a,n+'.json') for n in ('protocol','reference'))
    approved={}
    for name,root in [('night_dev',fit.tr.NIGHT),('source_dev',fit.tr.SOURCE)]:
        ds=state[name];check_equal('development identity',policy['development'][name],ds['sha256']);doc=read_sealed_json(root,ds['path'],ds['sha256'],[]);approved[name]=fit.ac._index(doc['records'],name)
    rr={};metrics={};checks={};seals={}
    for arm,name in {'control':'expand_010',**fit.ARMS}.items():
        path=fit.RUN/name;digest=sha256(path/'manifest.json');s=read_sealed_json(path,'manifest.json',digest,[]);seals[name]=digest
        if arm=='control':check_equal('fixed 300 control',fit.mat.PARENTS[name],digest)
        else:
            check_equal('complete local evidence inventory',s['files'],fit.tr._receipt(path));checks[arm]=verify_training(path,s,arm,ctx)
        m,rows,points,hits=dev_replay(path,s,approved,policy,ref);metrics[arm]=m;rr[arm]=(rows,points,hits)
    check_equal('original 33 tram targets',33,len(rr['control'][2]['tram']));effects={}
    for before,after in [('control','targeted'),('control','random'),('random','targeted')]:
        l,x,h=rr[before];r,y,j=rr[after]
        check_equal('identical day evaluation population',fit.ac._pair_identity(l['day']),fit.ac._pair_identity(r['day']))
        delta=fit.exp.effects([(l['night'],x),(r['night'],y)],[-1,1],policy)
        effects[after+'_minus_'+before]={'paired_night_recall':delta,'hit_changes':{n:{'retained':len(h[n]&j[n]),'lost':len(h[n]-j[n]),'new':len(j[n]-h[n])} for n in fit.ac.CLASSES},
              'original_tram_lost_ids':sorted([list(z) for z in h['tram']-j['tram']]) if before=='control' else None,
              'day_macro_delta':metrics[after]['day_macro_ap50_95']-metrics[before]['day_macro_ap50_95']}
    joint={a:metrics[a]['assessment']['point_met'] and effects[a+'_minus_control']['hit_changes']['tram']['lost']==0 for a in fit.ARMS}
    return {'seals':seals,'paired_protocol_sha256':sha256(fit.PAIR/'protocol.json'),'training':checks,'development':metrics,'contrasts':effects,'joint_numeric_and_retention':joint}


if __name__=='__main__':
    m=replay();file=fit.PAIR/'comparison.json';manifest=fit.PAIR/'manifest.json'
    if file.exists():check_equal('replay matches recorded comparison',m,read_json(file))
    else:write_json(file,m)
    seal=dict(status='sealed',source_sha256=sha256(Path(__file__)),files=fit.tr._receipt(fit.PAIR))
    if manifest.exists():check_equal('paired comparison receipt',seal,read_json(manifest))
    else:write_json(manifest,seal)
    print('Verified 331 night, 500 day; per-arm 200 updates, common A and 33 original tram IDs. Seal:',sha256(manifest))
    for name,row in m['development'].items():
        a=row['assessment'];print(name,'night',[(n,v['tp'],v['fp']) for n,v in row['night'].items()],'classes',a['classes_met'],'day delta pp',a['day_protection']['macro_delta']*100,'day passed',a['day_protection']['met'])
    for n,e in m['contrasts'].items():print(n,'net and original hits',e['hit_changes'],'night macro delta',e['paired_night_recall']['effects']['macro'])
