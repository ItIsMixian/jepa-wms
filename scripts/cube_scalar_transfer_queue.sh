#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/pi_ltc_frozen_calibration_v1
export PYTHONPATH=/root/autodl-tmp/jepa-wms:$PWD
export JEPA_REPO=/root/autodl-tmp/jepa-wms
export JEPAWM_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/lewm_data/jepa_wms_noscan/pi_ltc_cross_model
COMMIT=49b87f6f4a8c620a509e51069676be24eb110338
MARKER=f259ec4020fbbe9ab7765c4ca9dc2d67c6c21b7c5b11b8b6bab564fb77c5a5ba
VANILLA="$ROOT/cube_jepa_wm_vanilla_step51184_v7_seed3072"
run_arm() {
  local seed="$1" mode="$2" episodes=50
  local extra=()
  if [ "$mode" = smoke ]; then episodes=1; extra+=(--smoke); fi
  echo "[TRANSFER START] seed=$seed mode=$mode $(date -u +%FT%TZ)"
  "$PY" eval_cube_scalar_transfer.py --task cube \
    --lewm-repo /root/autodl-tmp/lewm_figure6_crossmodel_clean \
    --source-h5 /root/autodl-tmp/lewm_data/ogbench/cube_single_expert.h5 \
    --source-sha256 "$MARKER" --sidecar-sha256 "$MARKER" \
    --preflight-audit "$ROOT/preflight/cube/$COMMIT/data_training_preflight.json" \
    --training-config "$VANILLA/vanilla_training_config.yaml" \
    --checkpoint-dir "$VANILLA" --checkpoint jepa-e1.pth.tar \
    --owner vanilla --arm vanilla_pi_scale --output-root "$PWD/scalar_transfer_20261112" \
    --eval-seed "$seed" --episodes "$episodes" --candidate-chunk-size 32 \
    --expected-commit "$COMMIT" --expected-training-commit "$COMMIT" \
    --expected-lewm-commit 2f4b66934b844a2cacc9454106b9ab163e20c4be \
    --expected-completed-passes 2 --device cuda --reuse-complete \
    --frozen-manifest "$PWD/preparation_manifest.json" --calibration "$PWD/calibration.json" "${extra[@]}"
}
run_arm 43 smoke
run_arm 43 full
run_arm 44 full
"$PY" summarize_cube_scalar_transfer.py --root "$PWD"
echo '[SCALAR TRANSFER COMPLETE] Instance remains on at user request.'
