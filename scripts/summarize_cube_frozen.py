"""Paired inference for the predeclared three-arm, 100-episode screen."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.summarize_stablewm_task_seed_pair import mcnemar_exact

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    arms=('vanilla_stepmatched','vanilla_calibrated','learned')
    outcomes={arm:[] for arm in arms}
    seeds=[]
    common=None
    for seed in (43,44):
        records={arm:json.loads((a.root/f'seed{seed}'/arm/'arm_audit.json').read_text()) for arm in arms}
        ref=records[arms[0]]
        for arm,r in records.items():
            assert r['status']=='complete' and len(r['results']['episode_successes'])==50
            assert r['eval_seed']==seed and r['arm']==arm
            assert r['checkpoint']['total_optimizer_steps']==25592
            assert r['checkpoint']['common_trainable_initialization_sha256']==ref['checkpoint']['common_trainable_initialization_sha256']
            for key in ('paired_start_sha256','environment_state_sha256','context_rows','goal_rows'):
                assert r['results'][key]==ref['results'][key],key
            assert r['candidate_rng_audit']==ref['candidate_rng_audit']
            signature=tuple(r[k] for k in ('frozen_manifest_sha256','calibration_sha256','evaluation_overlay_sha256'))
            if common is None: common=signature
            assert signature==common
            outcomes[arm].extend(r['results']['episode_successes'])
        assert records['vanilla_calibrated']['checkpoint']['sha256']==ref['checkpoint']['sha256']
        assert records['vanilla_calibrated']['scale_audit']['parameter_unchanged']
        seeds.append({'seed':seed, 'success_rates':{k:v['results']['success_rate'] for k,v in records.items()}})
    rng=np.random.default_rng(20261110)
    indices=rng.integers(0,100,size=(10000,100))
    comparisons={}
    for left,right in (('learned','vanilla_calibrated'),('learned','vanilla_stepmatched'),('vanilla_calibrated','vanilla_stepmatched')):
        x=np.asarray(outcomes[left],dtype=bool); y=np.asarray(outcomes[right],dtype=bool)
        delta=x.astype(float)-y.astype(float)
        comparisons[f'{left}_minus_{right}']={
            'difference_pp':float(delta.mean()*100),
            'paired_episode_ci95_pp':np.percentile(delta[indices].mean(1)*100,[2.5,97.5]).tolist(),
            'mcnemar':mcnemar_exact(x,y)}
    result={'status':'complete','training_seeds':1,'paired_episodes':100,
            'practical_margin_pp':5,'per_seed':seeds,
            'success_rates':{k:float(np.mean(v)*100) for k,v in outcomes.items()},
            'comparisons':comparisons,'provenance_sha256':common,
            'scope':'conditional on these fixed checkpoints; not training-seed robustness'}
    with a.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
