#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/pi_ltc_cross_model_profiles_20261112
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
export JEPAWM_DINOV2_REPO=/root/.cache/torch/hub/facebookresearch_dinov2_main
PY=/root/miniconda3/bin/python
for phase in pilot profiles; do
  for arm in lewm_baseline lewm_residual lewm_pi jepa_vanilla jepa_pi; do
    if test -f "$phase/$arm/complete.json"; then
      "$PY" -c 'import json,hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); j=json.loads(p.read_text()); assert j["status"]=="PROFILE_COMPLETE" and j["model_state_unchanged"]; assert j["provenance"]["script_sha256"]==hashlib.sha256(Path("cube_cross_model_profiles.py").read_bytes()).hexdigest(); assert j["provenance"]["manifest_sha256"]==hashlib.sha256(Path("manifest.json").read_bytes()).hexdigest(); print("[reuse]",p)' "$phase/$arm/complete.json"
      continue
    fi
    extra=()
    if [ "$phase" = pilot ]; then extra+=(--pilot); fi
    echo "[START] $phase $arm $(date -u +%FT%TZ)"
    "$PY" -u cube_cross_model_profiles.py --out . --arm "$arm" "${extra[@]}"
  done
done
if ! test -f cross_model_summary.json; then
  "$PY" summarize_cube_cross_profiles.py --out .
fi
echo '[QUEUE COMPLETE] GPU instance intentionally left powered on at user request.'
