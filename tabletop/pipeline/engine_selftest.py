"""engine_selftest - real "will it run?" health probe for every engine.

For each engine we do TWO checks:
  1. files   — is the engine's library actually on disk (cheap, instant)
  2. import  — can its core module be IMPORTED in a subprocess (the real test;
               surfaces missing CUDA exts / broken deps BEFORE a user hits them)

The import probe runs each engine in its own subprocess (so one engine's heavy
torch/CUDA import can't poison the others, and a hard crash is contained) with a
timeout, a few at a time. Emits a single JSON blob on stdout for the dashboard
`/api/health` endpoint.

Status per engine:
  ok      — files present AND import succeeds            (green — ready to run)
  deps    — files present but import FAILS               (amber — needs deps/exts)
  absent  — library not cloned/installed                 (grey — not installed)

Run:  python engine_selftest.py            -> JSON to stdout
      python engine_selftest.py --pretty   -> human-readable table
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP = HERE.parent
PY = os.environ.get("GEN3D_PY") or sys.executable

# Each engine: (id, label, lib_subdir|None, extra_pythonpath, file_marker, import_code)
#   lib_subdir       — repo dir under tabletop/ (None = no repo, pure dep)
#   extra_pythonpath — path (relative to lib dir) to also add to sys.path ("" = lib root)
#   file_marker      — file whose existence proves the lib is installed
#   import_code      — the minimal import that proves the engine can actually load
ENGINES = [
    ("hunyuan21", "Hunyuan 2.1 · single", "Hunyuan3D-2.1", "hy3dshape",
     "hy3dshape/hy3dshape/pipelines.py", "import hy3dshape"),
    ("hunyuan2mv", "Hunyuan · multi-view", "Hunyuan3D-2", "",
     "hy3dgen/shapegen/pipelines.py", "import hy3dgen"),
    ("triposg", "TripoSG", "TripoSG", "",
     "triposg/inference_utils.py", "import triposg, diso"),
    ("craftsman", "CraftsMan3D", "CraftsMan3D", "",
     "craftsman/__init__.py", "import craftsman"),
    # Deeper probes than just the pipeline class: these pull the model/attention/
    # sparse submodules that actually fail at RUN time (xformers, spconv, o_voxel,
    # cumesh) — so the health screen catches them instead of a live run.
    ("hi3dgen", "Hi3DGen", "Stable3DGen", "",
     "hi3dgen/__init__.py",
     "from hi3dgen.pipelines import Hi3DGenPipeline; from hi3dgen.models import sparse_structure_flow"),
    ("trellis", "TRELLIS.2", "TRELLIS", "",
     "trellis2/__init__.py",
     "import o_voxel; from trellis2.pipelines import Trellis2ImageTo3DPipeline"),
    ("parametric", "Parametric CAD", None, "",
     None, "import build123d"),
]

PROBE_TIMEOUT = int(os.environ.get("SELFTEST_TIMEOUT", "150"))


def _last_error(text: str) -> str:
    """Pull the most useful one-liner out of a traceback."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    # Prefer the final exception line (e.g. "ModuleNotFoundError: No module ...")
    for ln in reversed(lines):
        if any(ln.startswith(p) for p in (
                "ModuleNotFoundError", "ImportError", "RuntimeError",
                "OSError", "AttributeError", "ValueError", "Exception",
                "TypeError", "NameError", "FileNotFoundError")):
            return ln[:240]
    return lines[-1][:240]


def probe_engine(spec) -> dict:
    eid, label, lib, extra, marker, code = spec
    lib_dir = (TABLETOP / lib) if lib else None
    files_ok = True if marker is None else (
        lib_dir is not None and (lib_dir / marker).exists())

    res = {"id": eid, "label": label, "files": bool(files_ok),
           "import_ok": None, "status": "absent", "detail": "", "ms": 0}

    if not files_ok:
        res["detail"] = f"not installed ({lib}/ missing)" if lib else "missing"
        res["status"] = "absent"
        return res

    # Build the probe snippet: set the safe attention/spconv env, extend sys.path,
    # then run the import. Print a sentinel on success.
    paths = []
    if lib_dir is not None:
        paths.append(str(lib_dir))
        if extra:
            paths.append(str(lib_dir / extra))
    path_lines = "".join(f"sys.path.insert(0, r{p!r})\n" for p in paths)
    snippet = (
        "import sys, os\n"
        "os.environ['ATTN_BACKEND']='sdpa'\n"   # match runtime: pure-torch attn
        "os.environ.setdefault('SPCONV_ALGO','native')\n"
        + path_lines
        + f"{code}\n"
        "print('SELFTEST_OK')\n"
    )
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}  # don't hit network
    import time
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            [PY, "-c", snippet],
            cwd=str(lib_dir) if lib_dir else str(TABLETOP),
            env=env, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        if p.returncode == 0 and "SELFTEST_OK" in (p.stdout or ""):
            res["import_ok"] = True
            res["status"] = "ok"
            res["detail"] = "imports cleanly"
        else:
            res["import_ok"] = False
            res["status"] = "deps"
            res["detail"] = _last_error(out) or f"exit {p.returncode}"
    except subprocess.TimeoutExpired:
        res["import_ok"] = False
        res["status"] = "deps"
        res["detail"] = f"import timed out (> {PROBE_TIMEOUT}s)"
    except Exception as e:  # noqa: BLE001
        res["import_ok"] = False
        res["status"] = "deps"
        res["detail"] = str(e)[:240]
    res["ms"] = int((time.monotonic() - t0) * 1000)
    return res


def core_info() -> dict:
    info = {"python": sys.version.split()[0], "interp": PY,
            "torch": None, "cuda": False, "gpus": []}
    try:
        import torch  # noqa: WPS433
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            for i in range(torch.cuda.device_count()):
                pr = torch.cuda.get_device_properties(i)
                info["gpus"].append(
                    f"{pr.name} · {round(pr.total_memory/1e9)}GB · sm_{pr.major}{pr.minor}")
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)[:200]
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    core = core_info()
    # Probe engines a few at a time (each loads torch — limit RAM spikes).
    with ThreadPoolExecutor(max_workers=4) as ex:
        engines = list(ex.map(probe_engine, ENGINES))

    summary = {"ok": 0, "deps": 0, "absent": 0}
    for e in engines:
        summary[e["status"]] = summary.get(e["status"], 0) + 1

    report = {"core": core, "engines": engines, "summary": summary}

    if args.pretty:
        print(f"python {core['python']}  torch {core['torch']}  "
              f"cuda={core['cuda']}")
        for g in core["gpus"]:
            print(f"  gpu: {g}")
        print("-" * 60)
        glyph = {"ok": "[ OK ]", "deps": "[DEPS]", "absent": "[ -- ]"}
        for e in engines:
            print(f"  {glyph[e['status']]} {e['label']:<22} "
                  f"{e['detail']}  ({e['ms']}ms)")
        print("-" * 60)
        print(f"  {summary['ok']} ready · {summary['deps']} need-deps · "
              f"{summary['absent']} absent")
    else:
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — never crash; emit an error blob
        print(json.dumps({"error": str(e), "engines": [], "core": {}}))
        sys.exit(0)
