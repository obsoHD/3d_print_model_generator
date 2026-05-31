"""diagnostics - startup health report for the gen3d container.

Runs once at container boot (from entrypoint.sh) and prints an OK/WARN/FAIL
report for: interpreter + env, torch/CUDA, key python deps (incl. pymeshlab
*plugin* health, not just import), native binaries, data dirs + provisioned
models, the Hunyuan libs, and per-engine readiness.

Always exits 0 — it is informational and must never block startup.
Output goes to stdout (so `docker compose logs` shows it) and is also tee'd to
/data/outputs/logs/startup_diagnostics.log by the entrypoint.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP = HERE.parent
sys.path.insert(0, str(HERE))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
_counts = {"ok": 0, "warn": 0, "fail": 0}


def line(status: str, label: str, detail: str = "") -> None:
    if status == OK:
        _counts["ok"] += 1
    elif status == WARN:
        _counts["warn"] += 1
    else:
        _counts["fail"] += 1
    print(f"  {status} {label}" + (f"  —  {detail}" if detail else ""), flush=True)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check_import(mod: str, attr: str = "__version__"):
    try:
        m = importlib.import_module(mod)
        line(OK, mod, str(getattr(m, attr, "imported")))
        return m
    except Exception as e:  # noqa: BLE001
        line(FAIL, mod, f"import failed: {e}")
        return None


def main() -> int:
    print("=" * 64)
    print(" gen3d startup diagnostics")
    print("=" * 64, flush=True)

    # ---- interpreter / env ----
    section("interpreter / env")
    line(OK, "python", sys.version.split()[0])
    line(OK, "executable", sys.executable)
    for k in ("GEN3D_PY", "HY_OCTREE", "HY_STEPS", "HY_NUM_CHUNKS",
              "HY_CPU_OFFLOAD", "CONCEPT_RES", "DASHBOARD_PORT",
              "GEN3D_MODELS_DIR", "HF_HOME"):
        line(OK, k, os.environ.get(k, "(unset)"))
    line(OK if os.environ.get("HF_TOKEN") else WARN, "HF_TOKEN",
         "set" if os.environ.get("HF_TOKEN")
         else "MISSING — gated Hunyuan downloads will 401")

    # ---- torch / CUDA ----
    section("torch / CUDA")
    torch = check_import("torch")
    if torch is not None:
        try:
            avail = bool(torch.cuda.is_available())
            line(OK if avail else FAIL, "cuda available", str(avail))
            if avail:
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    line(OK, f"gpu{i}",
                         f"{p.name} {round(p.total_memory/1e9)}GB "
                         f"cap={p.major}.{p.minor}")
        except Exception as e:  # noqa: BLE001
            line(FAIL, "cuda probe", str(e))

    # ---- python deps ----
    section("python deps")
    for mod in ("numpy", "trimesh", "pymeshfix", "diffusers", "transformers",
                "accelerate", "huggingface_hub", "rembg", "onnxruntime",
                "pygltflib", "skimage", "build123d", "PIL", "cv2"):
        check_import(mod)
    try:
        importlib.import_module("diso")
        line(OK, "diso", "imported")
    except Exception as e:  # noqa: BLE001
        line(FAIL, "diso", str(e))

    # pymeshlab: import alone isn't enough — its Qt/GL plugins must load or every
    # mesh filter is dead. Probe the filter count as a health signal.
    try:
        import pymeshlab  # noqa: WPS433
        ms = pymeshlab.MeshSet()
        nf = -1
        for getter in (lambda: len(ms.filter_list()),
                       lambda: len(pymeshlab.filter_list())):
            try:
                nf = getter()
                break
            except Exception:  # noqa: BLE001
                continue
        if nf > 50:
            line(OK, "pymeshlab", f"{nf} filters loaded (plugins OK)")
        elif nf >= 0:
            line(WARN, "pymeshlab",
                 f"only {nf} filters — GL/Qt plugins likely missing "
                 f"(check libOpenGL.so.0)")
        else:
            line(WARN, "pymeshlab", "imported, filter count unknown")
    except Exception as e:  # noqa: BLE001
        line(FAIL, "pymeshlab", str(e))

    # ---- native binaries ----
    section("binaries")
    for b in ("blender", "nvcc", "prusa-slicer", "PrusaSlicer", "node"):
        p = shutil.which(b)
        line(OK if p else WARN, b, p or "not found")

    # ---- data dirs + models ----
    section("data dirs + models")
    for d in ("/data/models", "/data/hf", "/data/outputs",
              "/data/outputs/logs"):
        pth = Path(d)
        if pth.exists():
            line(OK if os.access(d, os.W_OK) else WARN, d,
                 "writable" if os.access(d, os.W_OK) else "NOT writable")
        else:
            line(WARN, d, "missing")
    md = Path(os.environ.get("GEN3D_MODELS_DIR", "/data/models"))
    if md.exists():
        subs = sorted(p.name for p in md.iterdir() if p.is_dir())
        line(OK if subs else WARN, "concept models",
             ", ".join(subs) if subs
             else "none (Hunyuan auto-downloads; concept models need copying)")

    # ---- Hunyuan libs ----
    section("hunyuan libs")
    for lib, sub, inner in (("Hunyuan3D-2.1", "hy3dshape", "pipelines.py"),
                            ("Hunyuan3D-2", "hy3dgen", "shapegen")):
        d = TABLETOP / lib / sub
        line(OK if d.exists() else FAIL, f"{lib}/{sub}",
             "present" if d.exists() else "MISSING")

    # ---- engine readiness (file-level, cheap) ----
    section("engine readiness")
    try:
        import orchestrate  # noqa: WPS433
        rep = orchestrate.check_models()
        for name, info in rep.get("engines", {}).items():
            ready = bool(info.get("ready"))
            line(OK if ready else WARN, f"engine:{name}",
                 "ready" if ready else "not ready")
    except Exception as e:  # noqa: BLE001
        line(WARN, "check_models", str(e))

    # ---- summary ----
    print("\n" + "=" * 64)
    verdict = ("FAIL" if _counts["fail"] else
               "WARN" if _counts["warn"] else "ALL OK")
    print(f" summary [{verdict}]: {_counts['ok']} ok · "
          f"{_counts['warn']} warn · {_counts['fail']} fail")
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — diagnostics must never abort startup
        print(f"[diagnostics] crashed (ignored): {e}", flush=True)
        sys.exit(0)
