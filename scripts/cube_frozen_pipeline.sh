#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/pi_ltc_frozen_calibration_v1
export JEPA_REPO=/root/autodl-tmp/jepa-wms
export PYTHONPATH="$JEPA_REPO:$PWD"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JEPAWM_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/lewm_data/jepa_wms_noscan/pi_ltc_cross_model
COMMIT=49b87f6f4a8c620a509e51069676be24eb110338
MARKER=f259ec4020fbbe9ab7765c4ca9dc2d67c6c21b7c5b11b8b6bab564fb77c5a5ba
run_arm() {
  local seed="$1" owner="$2" arm="$3" mode="$4"
  local directory checkpoint config episodes=50
  local extra=()
  if [ "$owner" = pi ]; then
    directory="$ROOT/cube_jepa_wm_pi_ltc_step51184_v7_seed3072"
    checkpoint=jepa-planner-best.pth.tar
    extra+=(--require-heldout-selection)
  else
    directory="$ROOT/cube_jepa_wm_vanilla_step51184_v7_seed3072"
    checkpoint=jepa-e1.pth.tar
  fi
  config="$directory/${owner}_training_config.yaml"
  if [ "$mode" = smoke ]; then episodes=1; extra+=(--smoke); fi
  echo "[launch] seed=$seed arm=$arm mode=$mode"
  "$PY" eval_cube_frozen_arm.py --task cube \
    --lewm-repo /root/autodl-tmp/lewm_figure6_crossmodel_clean \
    --source-h5 /root/autodl-tmp/lewm_data/ogbench/cube_single_expert.h5 \
    --source-sha256 "$MARKER" --sidecar-sha256 "$MARKER" \
    --preflight-audit "$ROOT/preflight/cube/$COMMIT/data_training_preflight.json" \
    --training-config "$config" --checkpoint-dir "$directory" --checkpoint "$checkpoint" \
    --owner "$owner" --arm "$arm" --output-root "$PWD/online" \
    --eval-seed "$seed" --episodes "$episodes" --candidate-chunk-size 32 \
    --expected-commit "$COMMIT" --expected-training-commit "$COMMIT" \
    --expected-lewm-commit 2f4b66934b844a2cacc9454106b9ab163e20c4be \
    --expected-completed-passes 2 --device cuda --reuse-complete \
    --frozen-manifest "$PWD/preparation_manifest.json" \
    --calibration "$PWD/calibration.json" "${extra[@]}"
}
run_arm 43 vanilla vanilla_calibrated smoke
for seed in 43 44; do
  run_arm "$seed" vanilla vanilla_stepmatched full
  run_arm "$seed" vanilla vanilla_calibrated full
  run_arm "$seed" pi learned full
done
"$PY" summarize_cube_frozen.py --root "$PWD/online/cube" --output "$PWD/three_arm_summary.json"
echo '[FROZEN THREE-ARM COMPLETE]'
