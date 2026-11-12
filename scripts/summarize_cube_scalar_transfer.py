"""Descriptive 2x2 weight-by-input-scale interaction on a previously used cohort."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    sources={'V_1':('online','vanilla_stepmatched'),'V_sPI':('scalar_transfer_20261112','vanilla_pi_scale'),
             'PI_1':('followup_20261112','identity'),'PI_sPI':('online','learned')}
    vectors={k:[] for k in sources};rates=[]
    for seed in (43,44):
        audits={k:json.loads((a.root/folder/'cube'/f'seed{seed}'/arm/'arm_audit.json').read_text()) for k,(folder,arm) in sources.items()}
        base=audits['V_1']
        for name,r in audits.items():
            assert r['status']=='complete' and len(r['results']['episode_successes'])==50
            assert r['evaluation']==base['evaluation'] and r['candidate_rng_audit']==base['candidate_rng_audit']
            for key in ['context_rows','goal_rows','paired_start_sha256','environment_state_sha256']:
                assert r['results'][key]==base['results'][key]
            for key in ['frozen_manifest_sha256','calibration_sha256']:
                assert r[key]==base[key]
            assert r['checkpoint']['total_optimizer_steps']==25592
            vectors[name].extend(r['results']['episode_successes'])
        assert audits['V_sPI']['effective_scale']==audits['PI_sPI']['effective_scale']
        assert audits['V_1']['checkpoint']['sha256']==audits['V_sPI']['checkpoint']['sha256']
        assert audits['PI_1']['checkpoint']['sha256']==audits['PI_sPI']['checkpoint']['sha256']
        rates.append({'seed':seed,'success_rates':{k:v['results']['success_rate'] for k,v in audits.items()}})
    x={k:np.asarray(v,dtype=float) for k,v in vectors.items()}
    rng=np.random.default_rng(20261112);ix=rng.integers(0,100,size=(10000,100))
    contrasts=[]
    for left,right in [('V_sPI','V_1'),('PI_sPI','V_sPI')]:
        d=x[left]-x[right];wins=int((d>0).sum());losses=int((d<0).sum());n=wins+losses
        pv=min(1.,2*sum(math.comb(n,i) for i in range(min(wins,losses)+1))/2**n) if n else 1.
        contrasts.append({'left':left,'right':right,'delta_pp':float(d.mean()*100),
                         'paired_ci95_pp':np.percentile(d[ix].mean(1)*100,[2.5,97.5]).tolist(),
                         'left_only':wins,'right_only':losses,'mcnemar_p_raw':pv})
    interaction=(x['PI_sPI']-x['PI_1'])-(x['V_sPI']-x['V_1'])
    result={'status':'SCALAR_TRANSFER_COMPLETE','paired_episodes':100,'training_seeds':1,
            'fresh_confirmatory_test':False,'per_seed':rates,
            'success_rates':{k:float(v.mean()*100) for k,v in x.items()},'contrasts':contrasts,
            'interaction':{'definition':'(PI_sPI - PI_1) - (V_sPI - V_1)',
                           'delta_pp':float(interaction.mean()*100),
                           'paired_ci95_pp':np.percentile(interaction[ix].mean(1)*100,[2.5,97.5]).tolist()},
            'scope':'Exploratory fixed-checkpoint weight-by-scale interaction; intervals and p-values unadjusted; no new training or online scale selection.'}
    with (a.root/'scalar_transfer_20261112/summary.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
