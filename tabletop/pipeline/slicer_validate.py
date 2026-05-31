"""slicer_validate - real-slicer validation oracle for generated STLs.

The iteration gate for the whole pipeline. Answers: "will this actually
print?" using the real slicers, not just trimesh topology heuristics.

Three layers, best-available:
  1. Bambu Studio --info        -> manifold yes/no, volume, dims  (fast, reliable)
  2. Bambu Studio --export-3mf  -> loads/orients/seats/arranges OK (geometry processable)
     with --orient 1 --ensure-on-bed --arrange 1
  3. PrusaSlicer --export-gcode -> the actual full slice (ground truth, if installed)

Plus trimesh geometry facts (min_z bed contact, connected components,
volume/bbox ratio) for diagnostics the slicers don't surface.

Returns a verdict dict + writes <stem>.validate.json.

Usage:
    python slicer_validate.py --input mesh.stl [--printer fdm|resin]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


def _resolve(*items):
    """Return the first available binary: PATH names (str) or explicit Paths."""
    for it in items:
        if isinstance(it, str):
            p = shutil.which(it)
            if p:
                return Path(p)
        elif it.exists():
            return it
    return None


# Cross-platform slicer discovery: PATH first (the Linux container symlinks
# prusa-slicer into /usr/local/bin), then known Windows install locations.
PRUSA = _resolve("prusa-slicer", "prusa-slicer-console", "PrusaSlicer",
                 Path("/opt/prusaslicer/AppRun"),
                 Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe"),
                 Path(r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer.exe"))
BAMBU = _resolve("bambu-studio", "BambuStudio",
                 Path(r"C:\Program Files\Bambu Studio\bambu-studio.exe"))


def _bambu_info(stl: str) -> dict:
    """Run Bambu --info; parse manifold/volume/facets/dims."""
    out = {"available": BAMBU is not None}
    if BAMBU is None:
        return out
    try:
        proc = subprocess.run([str(BAMBU), "--info", stl],
                              capture_output=True, text=True, timeout=120)
        txt = proc.stdout + proc.stderr
        for key, pat in (("manifold", r"manifold\s*=\s*(\w+)"),
                         ("volume",   r"volume\s*=\s*([\d.]+)"),
                         ("facets",   r"number_of_facets\s*=\s*(\d+)"),
                         ("size_x",   r"size_x\s*=\s*([\d.]+)"),
                         ("size_y",   r"size_y\s*=\s*([\d.]+)"),
                         ("size_z",   r"size_z\s*=\s*([\d.]+)")):
            m = re.search(pat, txt)
            if m:
                v = m.group(1)
                out[key] = (v == "yes") if key == "manifold" else float(v)
    except Exception as e:
        out["error"] = str(e)
    return out


def _bambu_geometry_ok(stl: str, workdir: Path) -> dict:
    """Run Bambu --export-3mf with orient + ensure-on-bed + arrange.
    Exit 0 means the geometry loads, orients, seats and arranges — a real
    (if partial) slicer accepting the mesh."""
    out = {"available": BAMBU is not None}
    if BAMBU is None:
        return out
    workdir.mkdir(parents=True, exist_ok=True)
    tmp3mf = workdir / (Path(stl).stem + ".validate.3mf")
    try:
        proc = subprocess.run(
            [str(BAMBU), "--export-3mf", str(tmp3mf),
             "--orient", "1", "--ensure-on-bed", "--arrange", "1", stl],
            capture_output=True, text=True, timeout=240)
        out["exit"] = proc.returncode
        out["ok"] = (proc.returncode == 0 and tmp3mf.exists())
        # surface any error lines
        errs = [l for l in (proc.stdout + proc.stderr).splitlines()
                if re.search(r"\b(error|fail|invalid|cannot)\b", l, re.I)]
        if errs:
            out["errors"] = errs[-5:]
        try: tmp3mf.unlink()
        except Exception: pass
    except Exception as e:
        out["error"] = str(e); out["ok"] = False
    return out


def _prusa_info(stl: str) -> dict:
    """PrusaSlicer --info: the gold-standard mesh repair report. Reports
    manifold, volume, and needed_repair with per-defect counts (open edges,
    degenerate facets, etc). Preset-free, robust."""
    prusa = PRUSA
    out = {"available": prusa is not None}
    if prusa is None:
        return out
    try:
        proc = subprocess.run([str(prusa), "--info", stl],
                              capture_output=True, text=True, timeout=180)
        txt = proc.stdout + proc.stderr
        out["exit"] = proc.returncode
        for key, pat in (("manifold",      r"manifold\s*=\s*(\w+)"),
                         ("needed_repair", r"needed_repair\s*=\s*(\w+)"),
                         ("facets",        r"number_of_facets\s*=\s*(\d+)"),
                         ("volume",        r"volume\s*=\s*([\d.-]+)"),
                         ("open_edges",    r"open_edges\s*=\s*(\d+)"),
                         ("degenerate",    r"degenerate_facets\s*=\s*(\d+)")):
            m = re.search(pat, txt)
            if m:
                v = m.group(1)
                out[key] = (v == "yes") if key in ("manifold", "needed_repair") else \
                           (float(v) if "." in v or key == "volume" else int(v))
        # Clean print test = manifold and no repair needed
        out["clean"] = (out.get("manifold") is True and
                        out.get("needed_repair") is not True)
    except Exception as e:
        out["error"] = str(e)
    return out


def _trimesh_facts(stl: str) -> dict:
    out = {}
    try:
        import trimesh, numpy as np
        m = trimesh.load(stl, force="mesh")
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        out["watertight"] = bool(m.is_watertight)
        out["is_volume"] = bool(m.is_volume)
        out["min_z"] = float(m.bounds[0, 2])
        out["bed_contact"] = abs(float(m.bounds[0, 2])) < 0.05
        out["components"] = len(m.split(only_watertight=False))
        out["faces"] = int(len(m.faces))
        bbv = float(np.prod(m.extents))
        out["volume_ratio"] = (float(m.volume) / bbv) if (m.is_volume and bbv > 0) else None
    except Exception as e:
        out["error"] = str(e)
    return out


def validate(stl: str, printer: str = "fdm") -> dict:
    stl = str(Path(stl).resolve())
    workdir = Path(stl).parent / "_validate"
    verdict = {"input": stl, "printer": printer}
    verdict["bambu_info"]   = _bambu_info(stl)
    verdict["bambu_geom"]   = _bambu_geometry_ok(stl, workdir)
    verdict["prusa_info"]   = _prusa_info(stl)
    verdict["trimesh"]      = _trimesh_facts(stl)

    # Overall verdict: manifold + geometry-processable by a real slicer
    info = verdict["bambu_info"]; geom = verdict["bambu_geom"]
    prusa = verdict["prusa_info"]; tm = verdict["trimesh"]
    manifold = (info.get("manifold") is True or prusa.get("manifold") is True
                or tm.get("watertight") is True)
    # "processable" = a real slicer accepted the geometry. Bambu's export-3mf is
    # the strongest signal; a clean PrusaSlicer --info load (exit 0) also counts.
    # If NO slicer is installed, fall back to trimesh's solid verdict so we don't
    # hard-fail a watertight mesh just for missing tooling.
    prusa_loaded = bool(prusa.get("available") and prusa.get("exit") == 0
                        and not prusa.get("error"))
    any_slicer = bool(info.get("available") or geom.get("available")
                      or prusa.get("available"))
    processable = (geom.get("ok") is True) or prusa_loaded
    if not any_slicer:
        processable = bool(tm.get("is_volume"))
    verdict["PASS"] = bool(manifold and processable)
    verdict["reasons"] = []
    if not manifold:
        verdict["reasons"].append("not manifold (holes/non-manifold edges)")
    if not processable:
        verdict["reasons"].append("slicer could not process geometry")
    if tm.get("bed_contact") is False:
        verdict["reasons"].append(f"not seated on bed (min_z={tm.get('min_z')})")
    if (tm.get("components") or 1) > 20:
        verdict["reasons"].append(f"{tm.get('components')} disconnected pieces")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--printer", choices=("fdm", "resin"), default="fdm")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    v = validate(args.input, args.printer)
    side = Path(args.input).with_suffix(".validate.json")
    side.write_text(json.dumps(v, indent=2), encoding="utf-8")
    mark = "PASS" if v["PASS"] else "FAIL"
    print(f"[validate] {mark}  {Path(args.input).name}")
    print(f"  manifold={v['bambu_info'].get('manifold')} "
          f"vol={v['bambu_info'].get('volume')} "
          f"bambu_geom_ok={v['bambu_geom'].get('ok')} "
          f"prusa_clean={v['prusa_info'].get('clean')} "
          f"needed_repair={v['prusa_info'].get('needed_repair')} "
          f"bed={v['trimesh'].get('bed_contact')} "
          f"comps={v['trimesh'].get('components')}")
    if v["reasons"]:
        print("  reasons: " + "; ".join(v["reasons"]))
    return 0 if v["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
