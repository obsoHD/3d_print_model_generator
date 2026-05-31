#!/usr/bin/env bash
# gen3d container entrypoint: prep data dirs, headless display for slicer/CAD,
# then launch the dashboard (which spawns the python pipeline on demand).
set -euo pipefail

DATA=/data
mkdir -p "$DATA/hf" "$DATA/models" "$DATA/outputs" "$DATA/outputs/logs"

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

# Some libs cache models under $HOME, NOT HF_HOME — which means they re-download
# every rebuild. Symlink those onto /data so they persist:
#   - hy3dgen (Hunyuan 2.1/2mv) → ~/.cache/hy3dgen  (the multi-GB .ckpt weights!)
#   - rembg                     → ~/.u2net
mkdir -p "$DATA/cache/hy3dgen" "$DATA/u2net" /root/.cache
[ -L /root/.cache/hy3dgen ] || { rm -rf /root/.cache/hy3dgen; ln -s "$DATA/cache/hy3dgen" /root/.cache/hy3dgen; }
[ -L /root/.u2net ]         || { rm -rf /root/.u2net;         ln -s "$DATA/u2net"         /root/.u2net; }
export U2NET_HOME="$DATA/u2net"

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

# Startup diagnostics: report deps / CUDA / models / engine readiness in one
# place. Informational only (never blocks). Shown in `docker logs` + persisted.
echo "[entrypoint] running startup diagnostics..."
python "$APP/pipeline/diagnostics.py" 2>&1 | tee "$DATA/outputs/logs/startup_diagnostics.log" || true

cd /app/tabletop/dashboard
export DASHBOARD_PORT="${DASHBOARD_PORT:-7800}"
echo "[entrypoint] starting dashboard on :${DASHBOARD_PORT}"
exec node server.js
