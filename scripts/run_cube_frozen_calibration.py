"""Frozen vanilla grid search on the locked original planner-training groups."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset
    from frozen_vanilla_input_scale import vanilla_input_scale
    from prepare_cube_frozen_calibration import digest, require
    from app.vjepa_wm.modelcustom.stablewm_pointmaze_cost import build_stablewm_pointmaze_cost
    from app.plan_common.datasets.planner_landscape_h5 import PlannerLandscapeH5Dataset
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.planner_landscape import normalized_pairwise_landscape_error, pairwise_sign_accuracy
    require(torch.cuda.is_available(), 'GPU unavailable; no calibration started')
    require(not args.output.exists(), 'output exists; preserve previous calibration')
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads((args.manifest.parent / 'identity_cpu_audit.json').read_text())
    require(audit['status'] == 'CPU_PREDICTOR_EQUIVALENCE_PASS', 'CPU identity gate missing')
    ck = manifest['checkpoints']['vanilla']
    require(digest(ck['path']) == ck['sha256'], 'checkpoint changed')
    require(digest(ck['config']) == ck['config_sha256'], 'config changed')
    config = yaml.safe_load(Path(ck['config']).read_text())
    pi_config = yaml.safe_load(Path(manifest['checkpoints']['pi']['config']).read_text())
    adapter = build_stablewm_pointmaze_cost(
        checkpoint_dir=Path(ck['path']).parent, checkpoint=Path(ck['path']).name,
        training_config=config, source_h5=manifest['source']['path'],
        arm='vanilla_stepmatched', owner='vanilla', device='cuda')
    wm = adapter.model.model
    wm.eval().requires_grad_(False)
    metadata = adapter.source_metadata
    planner = pi_config['planner_identified']
    dataset = PlannerLandscapeH5Dataset(
        source_h5=manifest['source']['path'], sidecar_h5=manifest['sidecar']['path'],
        transform=make_transforms(img_size=config['data']['img_size'], **config['data_aug']),
        action_mean=metadata.action_mean, action_std=metadata.action_std,
        proprio_mean=metadata.proprio_mean, proprio_std=metadata.proprio_std,
        source_metadata=metadata, expected_protocol='cube_planner_counterfactual_v1',
        expected_config_sha256=planner['expected_config_sha256'], history_size=3,
        frameskip=5, action_skip=1, goal_offset_steps=25)
    # Dataset filters invalid history/goal contexts internally. Map original IDs.
    import h5py
    with h5py.File(manifest['sidecar']['path'], 'r') as f:
        original_rows = np.asarray(f['selected_context_rows'][:])
    lookup = {int(row): i for i,row in enumerate(dataset.context_rows)}
    require(len(lookup) == len(dataset), 'duplicate context rows require group-ID mapping')
    ids = [lookup[int(original_rows[i])] for i in manifest['calibration_group_ids']]
    loader = DataLoader(Subset(dataset, ids), batch_size=4, shuffle=False, num_workers=0)
    grid = manifest['scale_grid']
    sums = {s: np.zeros(6, dtype=np.float64) for s in grid}
    identity_checked = False
    def repeat(x, k):
        return x[:,None].expand(x.shape[0],k,*x.shape[1:]).flatten(0,1)
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            batch = {k:v.cuda() for k,v in batch.items()}
            n,k = batch['next_visual'].shape[:2]
            h = batch['context_visual'].shape[1]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                encoded = wm.encode_obs({'visual':torch.cat((batch['context_visual'],batch['goal_visual'],batch['next_visual']),1), 'proprio':None})['visual']
                context, goal, targets = encoded[:,:h], encoded[:,h:h+1], encoded[:,h+1:]
                acts = wm.encode_act(batch['action'].flatten(0,1))
                prop = repeat(wm.encode_proprio(batch['context_proprio']), k)
                context = repeat(context,k)
                if not identity_checked:
                    before = wm.forward_pred(context,acts,prop)[0]
                    with vanilla_input_scale(wm.predictor,1.):
                        after = wm.forward_pred(context,acts,prop)[0]
                    require(torch.equal(before,after), 'real-image identity equivalence failed')
                    identity_checked = True
                    print('REAL_IMAGE_IDENTITY_EQUIVALENCE_PASS', flush=True)
                real = (targets.float()-goal.float()).square().flatten(2).mean(2)
                for scale in grid:
                    with vanilla_input_scale(wm.predictor,scale):
                        predicted = wm.forward_pred(context,acts,prop)[0][:,-1:]
                    costs = (predicted.float()-repeat(goal,k).float()).square().flatten(1).mean(1).reshape(n,k)
                    loss, diag = normalized_pairwise_landscape_error(costs,real,
                        eps=1e-8,relative_energy_floor=1e-4,return_diagnostics=True)
                    valid = diag['valid_group']; count = int(valid.sum())
                    sign = float(pairwise_sign_accuracy(costs[valid],real[valid])) if count else 0.
                    selected = real.gather(1,costs.argmin(1,keepdim=True)).squeeze(1)
                    regret = selected-real.min(1).values
                    mse = (predicted.float().reshape(n,k,*predicted.shape[2:])-targets.float()).square().flatten(2).mean(2).mean(1)
                    sums[scale] += [float(loss)*count,sign*count,count,float(regret.sum()),float(mse.sum()),n]
            if step % 100 == 0:
                print(f'calibration groups {min((step+1)*4,len(ids))}/{len(ids)}',flush=True)
    records=[]
    for scale,v in sums.items():
        require(v[2]>0, 'no eligible groups')
        records.append({'scale':scale,'loss':v[0]/v[2],'sign_accuracy':v[1]/v[2],
                        'valid_groups':int(v[2]),'top1_latent_regret':v[3]/v[5],
                        'endpoint_mse':v[4]/v[5],'groups':int(v[5])})
    best=min(records,key=lambda r:(r['loss'],abs(np.log(r['scale']))))
    result={'status':'FROZEN_VANILLA_CALIBRATION_COMPLETE','records':records,
            'selected_scale':best['scale'],'boundary_selected':best['scale'] in (min(grid),max(grid)),
            'optimizer_steps':0,'real_image_identity_bit_exact':identity_checked,
            'source_checkpoint_sha256':ck['sha256'],
            'preparation_sha256':digest(args.manifest),'online_test_started':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
