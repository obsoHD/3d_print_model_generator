#!/usr/bin/env bash
# gen3d container entrypoint: prep data dirs, headless display for slicer/CAD,
# then launch the dashboard (which spawns the python pipeline on demand).
set -euo pipefail

DATA=/data
mkdir -p "$DATA/hf" "$DATA/models" "$DATA/outputs"

# All pipeline subprocesses share the one container interpreter.
export GEN3D_PY="${GEN3D_PY:-$(command -v python)}"

# The pipeline + dashboard resolve models/outputs relative to the repo
# (TABLETOP/models, TABLETOP/outputs). Symlink those onto the NVMe data volume
# so heavy weights + results live on /data, not inside the image layer.
APP=/app/tabletop
for d in models outputs; do
  if [ ! -L "$APP/$d" ]; then
    rm -rf "$APP/$d" 2>/dev/null || true
    ln -s "$DATA/$d" "$APP/$d"
  fi
done
export GEN3D_MODELS_DIR="${GEN3D_MODELS_DIR:-$DATA/models}"
export GEN3D_OUTPUTS_DIR="${GEN3D_OUTPUTS_DIR:-$DATA/outputs}"
export HF_HOME="${HF_HOME:-$DATA/hf}"

echo "[entrypoint] GPUs: ${NVIDIA_VISIBLE_DEVICES:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print(f"[entrypoint] torch {torch.__version__} cuda_avail={torch.cuda.is_available()} "
      f"devs={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"           gpu{i}: {torch.cuda.get_device_name(i)} "
              f"cap={torch.cuda.get_device_capability(i)}")
PY

# Headless X for PrusaSlicer/Bambu CLI + any GL-dependent CAD render.
export DISPLAY=:99
( Xvfb :99 -screen 0 1280x1024x24 >/dev/null 2>&1 & ) || true

cd /app/tabletop/dashboard
export DASHBOARD_PORT="${DASHBOARD_PORT:-7800}"
echo "[entrypoint] starting dashboard on :${DASHBOARD_PORT}"
exec node server.js
