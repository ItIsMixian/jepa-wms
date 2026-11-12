"""Bounded descriptive profiles on shared Cube branches. No training or online tuning.

Each backend runs in a separate process using its original repository. Latent
metrics are compared within a checkpoint; physical regret uses a shared goal.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import hdf5plugin  # register source compression filters
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/lewm_data')
SOURCE=ROOT/'ogbench/cube_single_expert.h5'
SIDECAR=ROOT/'ogbench/cube_counterfactual_cem0_seed3072.h5'
JEPA=Path('/root/autodl-tmp/jepa-wms')
LEWM=Path('/root/autodl-tmp/lewm_figure6_crossmodel_clean')
PREP=Path('/root/autodl-tmp/pi_ltc_frozen_calibration_v1/preparation_manifest.json')
SCALES=[.25,.4,.5,.6,.75,1.,1.25,1.5,2.,3.,4.]
MODELS={
 'lewm_baseline':ROOT/'cube_baseline_seed3072/cube_baseline_object.ckpt',
 'lewm_residual':ROOT/'cube_residual_seed3072/cube_residual_object.ckpt',
 'lewm_pi':ROOT/'cube_ltc_planner_identified_global_seed3072/cube_ltc_planner_identified_global_epoch_3_object.ckpt',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    payload=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text()!=payload:raise RuntimeError(f'Preserve conflicting output: {path}')
        return
    temp=path.with_suffix('.tmp');temp.write_text(payload);temp.replace(path)

def prepare(out):
    prior=json.loads(PREP.read_text())
    with h5py.File(SOURCE,'r') as f,h5py.File(SIDECAR,'r') as s:
        offsets=f['ep_offset'][:];lengths=f['ep_len'][:]
        rows=s['selected_context_rows'][:];eps=s['selected_episode_id'][:]
        clips=s['selected_source_clip_index'][:]
        valid=(rows>=offsets[eps]+10)&(rows+25<offsets[eps]+lengths[eps])
        nclips=int(np.maximum(lengths-20+1,0).sum())
        order=torch.randperm(nclips,generator=torch.Generator().manual_seed(3072)).numpy()
        membership=np.zeros(nclips,dtype=bool);membership[order[:int(.9*nclips)]]=True
        lewm_val=~membership[clips]
        train=set(prior['calibration_group_ids'])
        jepa_val=np.array([i not in train for i in range(len(rows))])&valid
        # All previously used JEPA validation groups + a bounded training sample.
        rng=np.random.default_rng(20261112)
        chosen=np.sort(np.concatenate((np.flatnonzero(jepa_val),rng.choice(sorted(train),2048,replace=False))))
        group_records=[{'group_id':int(i),'context_row':int(rows[i]),'episode_id':int(eps[i]),
                        'lewm_source_clip_validation':bool(lewm_val[i]),
                        'jepa_episode_validation':bool(jepa_val[i])} for i in chosen]
    checkpoints={k:{'path':str(v),'sha256':sha(v)} for k,v in MODELS.items()}
    for arm in ('vanilla','pi'):
        checkpoints['jepa_'+arm]=prior['checkpoints'][arm]
    result={'schema_version':1,'study':'descriptive_shared_candidate_profiles',
            'source':str(SOURCE),'sidecar':str(SIDECAR),'scales':SCALES,'histories':[1,3],
            'groups':group_records,'checkpoints':checkpoints,
            'common_validation_groups':sum(r['lewm_source_clip_validation'] and r['jepa_episode_validation'] for r in group_records),
            'scope':'Previously used source data; not a fresh confirmatory test. LeWM split is by clip, JEPA by episode. Training budgets differ across architectures.',
            'no_online_scale_selection':True,'optimizer_steps':0}
    write_json(out/'manifest.json',result)
    print(json.dumps({'groups':len(chosen),'common_validation_groups':result['common_validation_groups']},indent=2),flush=True)

def read_rows(ds,rows):
    uniq,inv=np.unique(rows,return_inverse=True)
    return np.asarray(ds[uniq.tolist()])[inv]

def repeat(x,k):return x[:,None].expand(x.shape[0],k,*x.shape[1:]).flatten(0,1)

@contextmanager
def intervention(model,backend,arm,scale):
    pred=model.predictor
    if arm.endswith('_pi'):
        if backend=='jepa':
            mod=pred.planner_input_scale
            mod.set_runtime_override(scale)
            try:yield
            finally:mod.clear_runtime_override()
        else:
            pred.set_runtime_scale_override(scale)
            try:yield
            finally:pred.clear_runtime_scale_override()
    else:
        target=pred.predictor_embed if backend=='jepa' else getattr(pred,'delta_predictor',pred)
        h=target.register_forward_pre_hook(lambda module,args:(args[0]*scale,*args[1:]))
        try:yield
        finally:h.remove()

def run(out,arm,pilot):
    manifest=json.loads((out/'manifest.json').read_text())
    backend=arm.split('_')[0]
    sys.path.insert(0,str(JEPA if backend=='jepa' else LEWM))
    torch.set_num_threads(4);torch.manual_seed(20261112)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    cp=manifest['checkpoints'][arm]
    if sha(cp['path'])!=cp['sha256']:raise RuntimeError('checkpoint changed')
    if backend=='jepa':
        import yaml
        from app.vjepa_wm.modelcustom.stablewm_pointmaze_cost import build_stablewm_pointmaze_cost
        from app.plan_common.datasets.transforms import make_transforms
        cfg=yaml.safe_load(Path(cp['config']).read_text())
        adapter=build_stablewm_pointmaze_cost(checkpoint_dir=Path(cp['path']).parent,
            checkpoint=Path(cp['path']).name,training_config=cfg,source_h5=SOURCE,
            arm='learned' if arm.endswith('_pi') else 'vanilla_stepmatched',
            owner='pi' if arm.endswith('_pi') else 'vanilla',device='cuda')
        model=adapter.model.model;metadata=adapter.source_metadata
        transform=make_transforms(img_size=224,**cfg['data_aug'])
        amean=metadata.action_mean;astd=metadata.action_std
    else:
        import _compat
        from scripts.cube_ltc_landscape_audit import _find_jepa
        from scripts.cube_dca2_scale_case_mechanism import build_pixel_transform
        model=_find_jepa(torch.load(cp['path'],map_location='cpu',weights_only=False)).cuda()
        transform=build_pixel_transform(224)
        with h5py.File(SIDECAR,'r') as s:
            cfg=json.loads(s.attrs['config_json'])
            amean=np.array(cfg['raw_action_mean'],dtype=np.float32)
            astd=np.array(cfg['raw_action_std'],dtype=np.float32)
    model.eval().requires_grad_(False)
    original_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    scale_original=(float(model.predictor.planner_input_scale.input_scale) if backend=='jepa'
                    else float(model.predictor.input_scale)) if arm.endswith('_pi') else 1.
    scales=list(manifest['scales'])
    if scale_original not in scales:scales.append(scale_original)
    groups=manifest['groups'][:32] if pilot else manifest['groups']
    directory=out/('pilot' if pilot else 'profiles')/arm
    directory.mkdir(parents=True,exist_ok=True)
    provenance={'manifest_sha256':sha(out/'manifest.json'),'script_sha256':sha(__file__),
                'checkpoint_sha256':cp['sha256'],'arm':arm,'groups':len(groups),
                'scales':scales,'histories':[1,3],'native_scale':scale_original,
                'precision':'bfloat16 autocast' if backend=='jepa' else 'float32',
                'raw_latent_cost_reduction':'mean' if backend=='jepa' else 'sum'}
    write_json(directory/'provenance.json',provenance)
    started=time.monotonic(); identity_histories=set()
    with h5py.File(SOURCE,'r') as source,h5py.File(SIDECAR,'r') as side,torch.inference_mode():
      for start in range(0,len(groups),4):
        chunk=directory/f'chunk_{start:05d}.npz'
        selected=groups[start:start+4]
        gids=np.array([g['group_id'] for g in selected]);rows=np.array([g['context_row'] for g in selected]);n=len(rows)
        if chunk.exists():
            with np.load(chunk) as saved:
                if not np.array_equal(saved['group_ids'],gids):raise RuntimeError('chunk group mismatch')
            continue
        history_rows=rows[:,None]+np.array([-10,-5,0])[None,:]
        branches=(gids[:,None]*4+np.arange(4)[None,:]).reshape(-1)
        context=read_rows(source['pixels'],history_rows.reshape(-1)).reshape(n,3,224,224,3)
        goal=read_rows(source['pixels'],rows+25)[:,None]
        target=np.asarray(side['next_pixels'][branches]).reshape(n,4,224,224,3)
        images=torch.from_numpy(np.concatenate((context,goal,target),1)).cuda().flatten(0,1).permute(0,3,1,2)
        if backend=='jepa':images=images.float()/255.
        images=transform(images).reshape(n,8,3,224,224)
        prefix=read_rows(source['action'],(rows[:,None]+np.arange(-10,0)).reshape(-1)).reshape(n,2,5,5)
        branch_actions=np.asarray(side['raw_action'][branches],dtype=np.float32).reshape(n,4,1,5,5)
        actions=np.concatenate((np.broadcast_to(prefix[:,None],(n,4,2,5,5)),branch_actions),axis=2)
        actions=(torch.from_numpy(actions)-torch.as_tensor(amean))/(torch.as_tensor(astd))
        actions=actions.reshape(n,4,3,25).cuda()
        if not torch.isfinite(actions).all():raise RuntimeError('nonfinite action')
        physical=np.linalg.norm(np.asarray(side['next_cube_xyz'][branches]).reshape(n,4,3)-read_rows(source['privileged_block_0_pos'],rows+25)[:,None,:],axis=-1)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=backend=='jepa'):
          if backend=='jepa':
            encoded=model.encode_obs({'visual':images,'proprio':None})['visual']
            unique_rows,inverse_rows=np.unique(history_rows.reshape(-1),return_inverse=True)
            prop=torch.from_numpy(metadata.load_proprio(source,unique_rows)[inverse_rows]).reshape(n,3,-1)
            prop=(prop-torch.as_tensor(metadata.proprio_mean))/torch.as_tensor(metadata.proprio_std)
            prop=model.encode_proprio(prop.cuda())
          else:encoded=model.encode({'pixels':images})['emb']
          context_e,goal_e,targets_e=encoded[:,:3],encoded[:,3:4],encoded[:,4:]
          real=(targets_e.float()-goal_e.float()).square().flatten(2)
          real=real.mean(2) if backend=='jepa' else real.sum(2)
          costs=[];mses=[]
          for history in [1,3]:
            x=repeat(context_e[:,-history:],4)
            a=actions[:,:,-history:].flatten(0,1)
            if backend=='jepa':
                ae=model.encode_act(a);pe=repeat(prop[:,-history:],4)
                forward=lambda:model.forward_pred(x,ae,pe)[0][:,-1:]
            else:
                ae=model.action_encoder(a)
                forward=lambda:model.predict(x,ae)[:,-1:]
            if history not in identity_histories:
                native=forward()
                with intervention(model,backend,arm,scale_original):same=forward()
                if not torch.equal(native,same):raise RuntimeError('native-scale identity check failed')
                identity_histories.add(history);print(f'[identity PASS] {arm} history={history}',flush=True)
            c=[];m=[]
            for scale in scales:
                with intervention(model,backend,arm,scale):prediction=forward()
                delta=(prediction.float()-repeat(goal_e,4).float()).square().flatten(1)
                c.append((delta.mean(1) if backend=='jepa' else delta.sum(1)).reshape(n,4).cpu().numpy())
                m.append((prediction.float().reshape(n,4,*prediction.shape[2:])-targets_e.float()).square().flatten(2).mean(2).mean(1).cpu().numpy())
            costs.append(np.stack(c));mses.append(np.stack(m))
        with chunk.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,group_ids=gids,predicted_cost=np.stack(costs),endpoint_mse=np.stack(mses),
                                real_latent_cost=real.cpu().numpy(),physical_cost=physical)
        chunk.with_suffix('.tmp').replace(chunk)
        if start%100==0:print(f'[progress] {arm} {start+n}/{len(groups)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    if not all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in original_state.items()):
        raise RuntimeError('model state changed')
    write_json(directory/'complete.json',{'status':'PROFILE_COMPLETE','provenance':provenance,
        'model_state_unchanged':True,'optimizer_steps':0,'identity_histories_checked':sorted(identity_histories)})
    print(f'[complete] {arm} {len(groups)} groups elapsed={time.monotonic()-started:.1f}s',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=['jepa_vanilla','jepa_pi',*MODELS])
    p.add_argument('--pilot',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.prepare:prepare(a.out)
    else:run(a.out,a.arm,a.pilot)

if __name__=='__main__':main()
