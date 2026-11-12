"""Descriptive paired profiles; does not select an online-test scale."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.out/'manifest.json').read_text());groups=manifest['groups']
    records={}
    lewm_val=np.array([g['lewm_source_clip_validation'] for g in groups])
    jepa_val=np.array([g['jepa_episode_validation'] for g in groups])
    cohorts={'all_descriptive':np.ones(len(groups),bool),
             'jepa_validation':jepa_val,'common_validation':lewm_val&jepa_val,
             'both_training':~lewm_val&~jepa_val}
    pair=np.triu_indices(4,1)
    for arm in manifest['checkpoints']:
        folder=a.out/'profiles'/arm
        complete=json.loads((folder/'complete.json').read_text())
        assert complete['status']=='PROFILE_COMPLETE' and complete['model_state_unchanged']
        provenance=complete['provenance']
        costs=[];mses=[];real=[];physical=[];ids=[];valid=[]
        for start in range(0,len(groups),4):
            with np.load(folder/f'chunk_{start:05d}.npz') as z:
                costs.append(z['predicted_cost']);mses.append(z['endpoint_mse'])
                real.append(z['real_latent_cost']);physical.append(z['physical_cost']);ids.extend(z['group_ids'].tolist())
                d=z['real_latent_cost'][:,pair[0]]-z['real_latent_cost'][:,pair[1]]
                e=(d*d).mean(1);median=np.sort(e)[(len(e)-1)//2]
                valid.extend((e>max(1e-8,1e-4*median)).tolist())
        assert ids==[g['group_id'] for g in groups]
        costs=np.concatenate(costs,axis=2);mses=np.concatenate(mses,axis=2)
        real=np.concatenate(real);physical=np.concatenate(physical);valid=np.array(valid)
        rd=real[:,pair[0]]-real[:,pair[1]];energy=(rd*rd).mean(1)
        rows=[]
        for hi,history in enumerate(provenance['histories']):
          for si,scale in enumerate(provenance['scales']):
            pred=costs[hi,si];pd=pred[:,pair[0]]-pred[:,pair[1]]
            chosen=pred.argmin(1); idx=np.arange(len(real))
            vectors={'landscape_error':((pd-rd)**2).mean(1)/np.maximum(energy,1e-30),
                     'pairwise_accuracy':(pd*rd>0).mean(1),'endpoint_mse':mses[hi,si],
                     'latent_top1_regret':real[idx,chosen]-real.min(1),
                     'physical_top1_regret_m':physical[idx,chosen]-physical.min(1),
                     'physical_selected_distance_m':physical[idx,chosen]}
            for cohort,mask in cohorts.items():
                selected=mask&valid
                if not selected.any():continue
                rows.append({'history':history,'scale':scale,'cohort':cohort,
                    'groups':int(selected.sum()),
                    **{key:float(value[selected].mean()) for key,value in vectors.items()}})
        records[arm]={'provenance':provenance,'profile':rows,'diagnostic_minima':[]}
        for history in provenance['histories']:
          for cohort in cohorts:
            subset=[r for r in rows if r['history']==history and r['cohort']==cohort]
            if not subset:continue
            minima={'history':history,'cohort':cohort}
            for metric in ('landscape_error','endpoint_mse','physical_top1_regret_m','latent_top1_regret'):
                best=min(subset,key=lambda r:r[metric]);minima[metric]={'scale':best['scale'],'value':best[metric]}
            best=max(subset,key=lambda r:r['pairwise_accuracy'])
            minima['pairwise_accuracy']={'scale':best['scale'],'value':best['pairwise_accuracy']}
            records[arm]['diagnostic_minima'].append(minima)
    summary={'status':'CROSS_MODEL_PROFILES_COMPLETE','groups':len(groups),
             'scope':manifest['scope'],'models':records,
             'notes':['No online control outcomes collected; minima are descriptive and must not be deployed based on these diagnostics alone.',
                      'Scalar values and raw latent errors are model-dependent; compare physical regret or within-model trends.',
                      'Common validation means LeWM source-clip validation intersected with JEPA episode validation; it is not wholly unseen physical episodes for LeWM.',
                      'Only one training seed; no training-seed robustness claim.'],
             'optimizer_steps':0}
    path=a.out/'cross_model_summary.json'
    with path.open('x') as f:json.dump(summary,f,indent=2)
    print(f'[CROSS MODEL COMPLETE] {path}',flush=True)

if __name__=='__main__':main()
