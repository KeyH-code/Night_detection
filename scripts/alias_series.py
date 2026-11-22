"""Run price-capped alias jobs in series; finish local qualification before paying for another seed."""
import argparse
from pathlib import Path

import alias_queue as queue
import alias_eval as evaluate
from source_contract import ContractError,read_json,sha256
from seed_train import ROOT

ORDER=((29,'alias_002'),(17,'alias_003'),(43,'alias_004'))


def run(scratch):
    verdicts={}
    for seed,name in ORDER:
        base=ROOT/'runs'/name;receipt=base.with_suffix('.queue.json')
        if receipt.exists():
            r=read_json(receipt)
            if (r.get('status'),r.get('seed'),r.get('platform_state'))!=('download_verified',seed,'Stopped'):
                raise ContractError(f'{name}: existing job needs reconciliation; no re-submission')
        else:
            queue.run(seed,name,scratch,'alias_001',None if seed==29 else 'alias_002')
        dev=base/'dev'
        if not dev.exists():evaluate.run(base,scratch)
        seal=read_json(dev/'manifest.json');queue.verify_inventory(dev,seal);v=read_json(dev/'comparison.json')
        if (seal['seed'],seal['source_training_seal'],seal['checkpoint']['sha256'])!=(seed,sha256(base/'run/manifest.json'),sha256(base/'run'/str(seed)/'weights/best.pt')):raise ContractError('Local comparison not bound to training')
        verdicts[seed]=v
        print('SERIES_VERDICT',seed,v,flush=True)
        if not v['five_pass']:break
    repeated={t for v in verdicts.values() for t in v['relative_warnings'] if sum(t in w['relative_warnings'] for w in verdicts.values())>=2}
    print('SERIES_RESULT',{'evaluated':list(verdicts),'numeric_five_all_three':len(verdicts)==3 and all(v['five_pass'] for v in verdicts.values()),'repeated_relative_warnings':sorted(repeated)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--run',action='store_true',required=True);a=p.parse_args();run(a.scratch.resolve())
