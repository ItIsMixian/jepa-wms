"""Evaluation-only AdaLN input multiplication, without new parameters or keys."""
from contextlib import contextmanager
import math

@contextmanager
def vanilla_input_scale(predictor, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('scale must be finite and positive')
    if getattr(predictor, 'planner_input_scale', None) is not None:
        raise ValueError('this adapter is only for vanilla predictors')
    # AdaLN applies the PI scalar directly before this exact input projection.
    projection = predictor.predictor_embed
    def multiply(module, inputs):
        return (inputs[0] * value, *inputs[1:])
    handle = projection.register_forward_pre_hook(multiply)
    try:
        yield
    finally:
        handle.remove()

def main():
    import argparse
    import json
    import sys
    from pathlib import Path
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
    torch.set_num_threads(4)
    torch.manual_seed(20261109)
    model = vit_predictor_AdaLN(
        img_size=224, patch_size=14, num_frames=4, tubelet_size=1,
        embed_dim=384, predictor_embed_dim=384, depth=6, num_heads=16,
        use_silu=False, use_rope=True, local_window=(3,-1,-1),
        action_dim=25, proprio_dim=19, proprio_emb_dim=16,
        proprio_encoding='feature', proprio_encoder_inpred=False,
        action_encoder_inpred=True, planner_identified_input_scale=False)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = {k.removeprefix('module.'): v for k,v in ck['predictor'].items()}
    model.load_state_dict(state, strict=True)
    del ck
    model.eval().requires_grad_(False)
    x = torch.randn(1,3,1,16,16,384)
    actions = torch.randn(1,3,25)
    proprio = torch.randn(1,3,1,16).expand(1,3,256,16)
    def check(a,b):
        if isinstance(a, (list,tuple)):
            return all(check(u,v) for u,v in zip(a,b))
        return (a is b) if a is None else torch.equal(a,b)
    with torch.inference_mode():
        baseline = model(x,actions,proprio)
        with vanilla_input_scale(model,1.):
            identity = model(x,actions,proprio)
        require_identity = check(baseline,identity)
        with vanilla_input_scale(model,1.25):
            scaled = model(x,actions,proprio)
        manual = model(x*1.25,actions,proprio)
        restored = model(x,actions,proprio)
    assert require_identity, 'identity changed prediction'
    assert check(scaled,manual), 'hook does not match intended PI insertion point'
    assert check(baseline,restored), 'hook not removed'
    assert set(state) == set(model.state_dict()), 'checkpoint keys changed'
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()), 'weights changed'
    assert not any(p.requires_grad for p in model.parameters())
    result = {'status':'CPU_PREDICTOR_EQUIVALENCE_PASS',
              'actual_checkpoint':args.checkpoint, 'strict_load':True,
              'identity_bit_exact':True, 'scale_matches_manual_input':True,
              'hook_removed':True, 'parameters_unchanged':True,
              'synthetic_latent_inputs':True, 'full_image_cost_gpu_check_pending':True}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
