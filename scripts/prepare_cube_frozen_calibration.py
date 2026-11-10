"""CPU-only preparation; reads HDF5 metadata, never pixels or action arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

COMMIT = '49b87f6f4a8c620a509e51069676be24eb110338'
GRID = [0.25, 0.5, 0.75, 1., 1.25, 1.5, 2., 3., 4.]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--sidecar', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    folders = {
        'pi': args.root / 'cube_jepa_wm_pi_ltc_step51184_v7_seed3072',
        'vanilla': args.root / 'cube_jepa_wm_vanilla_step51184_v7_seed3072',
    }
    audits, configs, initials = {}, {}, []
    for owner, folder in folders.items():
        name = 'jepa-planner-best.pth.tar' if owner == 'pi' else 'jepa-e1.pth.tar'
        path = folder / name
        cfg_path = folder / (owner + '_training_config.yaml')
        cfg = yaml.safe_load(cfg_path.read_text())
        ck = torch.load(path, map_location='cpu', weights_only=False)
        for key, value in {'epoch': 2, 'total_optimizer_steps': 25592,
                           'optimizer_steps_per_epoch': 12796,
                           'optimizer_step_in_epoch': 0,
                           'gradient_accumulation_steps': 8,
                           'checkpoint_role': 'training'}.items():
            require(ck.get(key) == value, f'{owner}: {key} differs')
        require(ck['training_provenance']['repository_commit'] == COMMIT,
                f'{owner}: training commit differs')
        keys = [k for k in ck['predictor'] if k.endswith('planner_input_scale.log_scale')]
        require(len(keys) == (1 if owner == 'pi' else 0), 'unexpected scale keys')
        init = ck.get('common_trainable_initialization_sha256')
        require(isinstance(init, str) and len(init) == 64, 'missing initialization digest')
        initials.append(init)
        audits[owner] = {'path': str(path), 'sha256': digest(path),
                         'config': str(cfg_path), 'config_sha256': digest(cfg_path),
                         'provenance': ck['training_provenance'],
                         'scale': float(ck['predictor'][keys[0]].exp()) if keys else 1.0}
        configs[owner] = cfg
        del ck
    require(initials[0] == initials[1], 'common initializations differ')
    require(configs['pi']['data'] == configs['vanilla']['data'], 'data configs differ')
    require(configs['pi']['data_aug'] == configs['vanilla']['data_aug'], 'transforms differ')
    for key in ('source_h5', 'content_hash_mode'):
        require(audits['pi']['provenance'].get(key) == audits['vanilla']['provenance'].get(key),
                f'provenance differs: {key}')
    pi_sidecar = audits['pi']['provenance']['sidecar_h5']
    require(Path(pi_sidecar['path']).resolve() == args.sidecar.resolve(), 'PI sidecar path differs')
    require(pi_sidecar['bytes'] == args.sidecar.stat().st_size, 'PI sidecar size differs')
    require(audits['vanilla']['provenance']['sidecar_h5']['path'] is None,
            'vanilla unexpectedly trained with a sidecar')
    with h5py.File(args.source, 'r') as f:
        offsets = np.asarray(f['ep_offset'][:], dtype=np.int64)
        lengths = np.asarray(f['ep_len'][:], dtype=np.int64)
        require(len(offsets) == 10000 and len(lengths) == 10000, 'unexpected episode layout')
        physical_ids = np.asarray(f['ep_idx'][offsets], dtype=np.int64)
    cfg = configs['pi']['data']
    order = torch.randperm(len(offsets), generator=torch.Generator().manual_seed(cfg['seed'])).numpy()
    train = set(order[:int(cfg['custom']['split_ratio'] * len(order))].tolist())
    with h5py.File(args.sidecar, 'r') as f:
        require(bool(f.attrs.get('complete')), 'incomplete sidecar')
        require(f.attrs.get('config_sha256') == configs['pi']['planner_identified']['expected_config_sha256'],
                'sidecar config differs')
        rows = np.asarray(f['source_context_row'][:], dtype=np.int64)
        # branch rows repeat each source context; selected rows preserve group order.
        selected_rows = np.asarray(f['selected_context_rows'][:], dtype=np.int64)
    positions = np.searchsorted(offsets, selected_rows, side='right') - 1
    require(np.all(positions >= 0) and np.all(selected_rows < offsets[positions] + lengths[positions]),
            'sidecar contexts outside episodes')
    history = cfg['custom']['num_hist']
    stride = cfg['custom']['frameskip']
    valid = (selected_rows >= offsets[positions] + (history - 1) * stride) & (
        selected_rows + 25 < offsets[positions] + lengths[positions])
    calibration = [int(i) for i in range(len(selected_rows)) if valid[i] and int(positions[i]) in train]
    used_sidecar = set(np.searchsorted(offsets, rows, side='right').astype(int).tolist())
    previous = args.root / 'cube_twopass_seed42_clean_v1' / 'cube'
    prior_files = sorted(previous.glob('seed*/**/arm_audit.json'))
    require(bool(prior_files), 'prior evaluation audits missing; cannot certify exclusions')
    prior_positions = set()
    prior_audits = []
    for path in prior_files:
        old = json.loads(path.read_text())
        require(old.get('status') == 'complete', f'incomplete prior audit: {path}')
        prior_rows = old['results']['context_rows']
        prior_positions.update(np.searchsorted(offsets, prior_rows, side='right').astype(int).tolist())
        prior_audits.append({'path': str(path), 'sha256': digest(path)})
    forbidden = train | used_sidecar | prior_positions
    eligible = sorted(i for i in range(len(offsets)) if i not in forbidden and lengths[i] >= 76)
    require(len(eligible) >= 100, f'only {len(eligible)} eligible physical episodes; need 100')
    rng = np.random.default_rng(20261109)
    chosen = rng.choice(eligible, 100, replace=False)
    starts = []
    for index, position in enumerate(chosen):
        # Preserve 25-step goal offset and enough room for a 50-step rollout.
        local = int(rng.integers(0, int(lengths[position]) - 50))
        row = int(offsets[position]) + local
        starts.append({'eval_seed': 43 if index < 50 else 44,
                       'physical_episode_id': int(physical_ids[position]),
                       'episode_position': int(position), 'start_step': local,
                       'context_row': row, 'goal_row': row + 25})
    result = {'status': 'CPU_METADATA_PREPARATION_COMPLETE', 'schema_version': 1,
              'training_commit': COMMIT, 'checkpoints': audits,
              'common_initialization_sha256': initials[0],
              'source': {'path': str(args.source), 'bytes': args.source.stat().st_size},
              'sidecar': {'path': str(args.sidecar), 'bytes': args.sidecar.stat().st_size},
              'dataset_content_hashed': False, 'pixels_read': False,
              'calibration_group_ids': calibration, 'scale_grid': GRID,
              'selection': 'minimum training-group normalized landscape loss; exact tie nearest identity',
              'practical_margin_pp': 5, 'old_validation_is_development_only': True,
              'prior_evaluation_audits': prior_audits,
              'eligible_unseen_episode_count': len(eligible), 'starts': starts,
              'cuda_available': torch.cuda.is_available(),
              'identity_equivalence_verified': False, 'gpu_evaluation_started': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output.exists():
        require(args.output.read_text() == payload, 'existing manifest differs; preserve it')
    else:
        with args.output.open('x') as f:
            f.write(payload)
    print(json.dumps({'status': result['status'], 'calibration_groups': len(calibration),
                      'eligible_unseen_episodes': len(eligible), 'locked_starts': len(starts),
                      'cuda_available': result['cuda_available'], 'output': str(args.output)}, indent=2))

if __name__ == '__main__':
    main()
