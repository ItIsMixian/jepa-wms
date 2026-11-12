"""Exploratory fixed-scale followups on the already-used primary test cohort."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    names={'vanilla':('online','vanilla_stepmatched'),
           'vanilla125':('online','vanilla_calibrated'),
           'pi':('online','learned'),
           'pi_identity':('followup_20261112','identity'),
           'pi125':('followup_20261112','fixed125'),
           'vanilla08':('followup_20261112','vanilla_fixed08')}
    arrays={k:[] for k in names};per_seed=[];scales={};code_hashes={}
    for seed in [43,44]:
        refs={k:json.loads((a.root/folder/'cube'/f'seed{seed}'/arm/'arm_audit.json').read_text())
              for k,(folder,arm) in names.items()}
        baseline=refs['vanilla'];expected=baseline['results']
        for name,r in refs.items():
            assert r['status']=='complete' and len(r['results']['episode_successes'])==50
            assert r['evaluation']==baseline['evaluation']
            assert r['candidate_rng_audit']==baseline['candidate_rng_audit']
            for key in ['context_rows','goal_rows','paired_start_sha256','environment_state_sha256']:
                assert r['results'][key]==expected[key],(name,key)
            for key in ['frozen_manifest_sha256','calibration_sha256']:
                assert r[key]==baseline[key]
            target=refs['pi'] if name.startswith('pi') else baseline
            assert r['checkpoint']['sha256']==target['checkpoint']['sha256']
            assert r['checkpoint']['total_optimizer_steps']==25592
            folder=names[name][0]
            if folder in code_hashes:assert code_hashes[folder]==r['evaluation_overlay_sha256']
            code_hashes[folder]=r['evaluation_overlay_sha256']
            arrays[name].extend(r['results']['episode_successes'])
            scales[name]=r['effective_scale']
        assert refs['pi_identity']['effective_scale']==1.
        assert refs['pi125']['effective_scale']==1.25
        assert refs['vanilla08']['effective_scale']==.8
        per_seed.append({'seed':seed,'success_rate':{k:v['results']['success_rate'] for k,v in refs.items()}})
    rng=np.random.default_rng(20261112);indices=rng.integers(0,100,size=(10000,100))
    comparisons=[]
    for left,right in [('pi','pi_identity'),('pi125','vanilla125'),('pi_identity','vanilla'),
                       ('pi125','pi'),('vanilla08','vanilla')]:
        x=np.asarray(arrays[left],bool);y=np.asarray(arrays[right],bool)
        d=x.astype(float)-y.astype(float);win=int((x&~y).sum());lose=int((y&~x).sum());n=win+lose
        pv=min(1.,2*sum(math.comb(n,i) for i in range(min(win,lose)+1))/2**n) if n else 1.
        comparisons.append({'left':left,'right':right,'difference_pp':float(d.mean()*100),
          'paired_bootstrap_ci95_pp':np.percentile(d[indices].mean(1)*100,[2.5,97.5]).tolist(),
          'left_only_successes':win,'right_only_successes':lose,'mcnemar_p_raw':pv})
    running=0.
    for rank,i in enumerate(sorted(range(len(comparisons)),key=lambda i:comparisons[i]['mcnemar_p_raw'])):
        running=max(running,min(1.,comparisons[i]['mcnemar_p_raw']*(len(comparisons)-rank)))
        comparisons[i]['mcnemar_p_holm']=running
    result={'status':'FOLLOWUP_DIAGNOSTICS_COMPLETE','paired_episodes_per_contrast':100,
            'training_seeds':1,'fresh_confirmatory_test':False,'per_seed':per_seed,
            'success_rates':{k:float(np.mean(v)*100) for k,v in arrays.items()},
            'scales':scales,'comparisons':comparisons,'evaluation_code_hashes':code_hashes,
            'scope':'Fixed diagnostic controls chosen after the primary study; same cohort reused. Confidence intervals are unadjusted; Holm p-values cover these five contrasts. No new parameter optimization.'}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
