"""dashboard/server.py — Pipeline status API + static server."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import base64
import urllib.parse

TABLETOP = Path(__file__).resolve().parent.parent
CONCEPTS  = TABLETOP / "outputs" / "concepts"
MESHES    = TABLETOP / "outputs" / "meshes"
STL_DIR   = TABLETOP / "outputs" / "stl"
PREVIEWS  = TABLETOP / "outputs" / "previews"
DASHBOARD = Path(__file__).resolve().parent

PORT = int(os.environ.get("DASHBOARD_PORT", 7842))


# ── GPU ──────────────────────────────────────────────────────────────────────

def _gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            p = [x.strip() for x in r.stdout.strip().split(",")]
            return {
                "util": int(p[0]), "mem_used": int(p[1]),
                "mem_total": int(p[2]), "temp": int(p[3]),
            }
    except Exception:
        pass
    return {"util": 0, "mem_used": 0, "mem_total": 16303, "temp": 0}


# ── PROCESS DETECTION ────────────────────────────────────────────────────────

def _active_stage():
    """Return dict describing what the pipeline is doing right now."""
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "commandline,processid", "/format:csv"],
            capture_output=True, text=True, timeout=5,
        )
        lines = r.stdout.strip().splitlines()
        for line in lines:
            low = line.lower()
            if "orchestrate.py" in low:
                return {"running": True, "script": "orchestrate.py", "stage": None}
            if "hy3d_infer.py" in low:
                return {"running": True, "script": "hy3d_infer.py", "stage": "mesh"}
    except Exception:
        pass
    return {"running": False, "script": None, "stage": None}


# ── RUNS ─────────────────────────────────────────────────────────────────────

def _parse_run_id(stem: str):
    m = re.match(r"(\d{8})_(\d{6})_(.+)_([0-9a-f]{6})$", stem)
    if not m:
        return None, stem
    ts = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
    prompt = m.group(3).replace("_", " ")
    return ts.isoformat(sep=" "), prompt


def _runs():
    runs = []
    if not CONCEPTS.exists():
        return runs
    pngs = sorted(CONCEPTS.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    for png in pngs:
        ts, prompt = _parse_run_id(png.stem)
        glb  = MESHES / f"{png.stem}.glb"
        stls = list(STL_DIR.glob("*.stl")) if STL_DIR.exists() else []

        step, step_label = 1, "concept"
        if glb.exists():
            step, step_label = 3, "blender"
            if stls:
                newest_stl = max(stls, key=lambda p: p.stat().st_mtime)
                if newest_stl.stat().st_mtime >= glb.stat().st_mtime - 1:
                    step, step_label = 4, "done"
        runs.append({
            "run_id":    png.stem,
            "timestamp": ts,
            "prompt":    prompt,
            "step":      step,
            "step_label": step_label,
            "concept_file": png.name,
            "concept_size": round(png.stat().st_size / 1024),
            "glb_file":  glb.name if glb.exists() else None,
            "glb_mb":    round(glb.stat().st_size / 1024 / 1024, 1) if glb.exists() else None,
        })
    return runs


def _stl_info():
    if not STL_DIR.exists():
        return []
    out = []
    for stl in sorted(STL_DIR.glob("*.stl"), key=lambda p: p.stat().st_mtime, reverse=True)[:4]:
        cj = stl.with_suffix(".cleanup.json")
        info = {"name": stl.name, "mb": round(stl.stat().st_size / 1024 / 1024, 1)}
        if cj.exists():
            try:
                d = json.loads(cj.read_text())
                info.update({
                    "tris": d.get("triangles"),
                    "dims": d.get("dims_mm"),
                    "manifold": d.get("non_manifold_after", -1) == 0,
                })
            except Exception:
                pass
        out.append(info)
    return out


def _status():
    gpu   = _gpu()
    proc  = _active_stage()
    runs  = _runs()
    stls  = _stl_info()

    current = None
    if runs:
        r = runs[0]
        if proc["running"]:
            stage = proc["stage"] or ("mesh" if r["step"] < 3 else "blender")
            current = {**r, "active": True, "active_stage": stage}
        elif r["step"] == 4:
            current = {**r, "active": False, "active_stage": "done"}
        else:
            current = {**r, "active": False, "active_stage": "idle"}

    return {
        "ts": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "gpu": gpu,
        "proc": proc,
        "current": current,
        "history": runs,
        "stls": stls,
    }


# ── HTTP ─────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # silence access log

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/status":
            body = json.dumps(_status()).encode()
            self._send(200, "application/json", body)

        elif path.startswith("/api/image/"):
            fname = urllib.parse.unquote(path[len("/api/image/"):])
            img_path = CONCEPTS / fname
            if img_path.exists() and img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                self._send(200, "image/png", img_path.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")

        elif path == "/" or path == "/index.html":
            html = (DASHBOARD / "index.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", html)

        else:
            self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{PORT}", flush=True)
    HTTPServer(("", PORT), Handler).serve_forever()
