# Pipeline Studio — Project Overview (for dashboard integrators)

> **Who this is for:** another agent/session building a **unified admin dashboard
> across several apps**. This document explains what this app is, how it runs, and
> how to monitor/control it. The companion file **`ADMIN_API.md`** (in
> `tabletop/dashboard/`) is the exact HTTP endpoint reference — read both. This
> file = the "what & how to integrate"; `ADMIN_API.md` = the "endpoint contract".

---

## 1. What this app is

**Pipeline Studio** is a fully-local **image/text → 3D-printable model** pipeline.
You give it a prompt or an image; it produces a **watertight, slicer-validated STL**
(plus a GLB for preview). It's aimed at tabletop 3D printing — character **minis**
and **terrain**.

- Runs **100% locally** on a dual **RTX 5090** (Blackwell, sm_120) Linux box.
- Ships a **web dashboard** (the thing you'll integrate) + a **Python pipeline**.
- One job at a time, long-running (minutes), GPU + CPU heavy.

---

## 2. Architecture (what's running)

```
Host: Linux workstation (obso@ai-pc), repo at /opt/gen3d  ──►  Docker container "gen3d-app"
                                                                  (/opt/gen3d → /app)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  container gen3d-app                                                        │
 │                                                                            │
 │  Node dashboard server   tabletop/dashboard/server.js                      │
 │    • binds 0.0.0.0:7842  (DASHBOARD_PORT)   ← THE API + UI you integrate    │
 │    • serves index.html (the browser UI)                                     │
 │    • spawns the pipeline as a DETACHED child (survives server restarts)     │
 │                                                                            │
 │  Python pipeline         tabletop/pipeline/orchestrate.py  (+ engines)      │
 │    • concept gen → mesh engine → mesh gauntlet → STL → slicer validate      │
 │    • interpreter = $GEN3D_PY                                                 │
 │                                                                            │
 │  Outputs (bind-mounted to host /opt/gen3d/data/outputs)                     │
 │    /app/tabletop/outputs/{concepts,meshes,stl,logs}/                        │
 └──────────────────────────────────────────────────────────────────────────┘
```

- **Control plane:** the Node server is the *only* thing a dashboard talks to —
  plain HTTP/JSON on **port 7842** (override via `DASHBOARD_PORT`). Binds all
  interfaces, so it's reachable across the LAN.
- **Single source of truth for the API:** `tabletop/dashboard/server.js`.
- **Code lives in git** (`github.com/obsoHD/3d_print_model_generator`); model
  weights / venvs / outputs do **not** (they're gitignored, re-created on deploy).

---

## 3. How to reach it

| | |
|---|---|
| Base URL | `http://<workstation-ip>:7842` (e.g. `http://ai-pc.local:7842`) |
| Find the port | `docker exec gen3d-app printenv DASHBOARD_PORT` or `docker port gen3d-app` |
| Format | JSON in/out (POST bodies JSON); logs/images/meshes are raw bytes |
| CORS | `*` (cross-origin OK) |
| Auth | **None** — see §8 Security |
| Health probe | `GET /api/health` (engine self-test) · liveness = server answers `/api/status` |

**First call any integrator should make:** `GET /api/capabilities` — it's
self-describing (valid engines, kinds, printers, detail presets, every `/api/run`
parameter, env knobs, and the endpoint list). You should never hard-code enums;
read them from there.

---

## 4. The control model (important for a dashboard)

- **One run at a time.** `POST /api/run` returns `{ok:false, error:"pipeline
  already running — kill it first"}` if busy. The dashboard should disable "start"
  while `proc.running` is true.
- **Fire-and-poll.** `/api/run` returns immediately; the job runs detached. Poll
  **`GET /api/status`** (~every 2 s) for progress. There is **no websocket** — it's
  poll-based.
- **Detached jobs survive a dashboard restart.** Restarting `gen3d-app` does not
  kill an in-flight pipeline; status re-attaches via the run's files/logs.
- **Stop:** `POST /api/kill` (SIGKILL the pipeline) or `POST /api/forcekill`
  (kill + delete that run's data).
- **Long jobs:** a mini is ~5–15 min depending on engine + detail preset. Alpha-wrap
  (TRELLIS print finishing) and decimation are **CPU-bound** and can each take
  minutes — show a spinner, don't time out aggressively.

### Job lifecycle (stages you can surface)
`concept (gen or upload)` → `multi-view (only mv engines)` → `mesh (engine)` →
`mesh gauntlet (repair / scale / seat / flatten base)` → `STL export` →
`slicer validation (Bambu/PrusaSlicer)` → `done`.
`/api/status.proc.stage` and `current.active_stage` tell you where it is;
`current.timing` gives progress/ETA.

---

## 5. What to render in a unified dashboard

Everything below comes from **`GET /api/status`** unless noted. (Field details in
`ADMIN_API.md`.)

| Tile | Source | Notes |
|---|---|---|
| **App status / liveness** | server answers `/api/status` | up/down |
| **GPU** | `status.gpu` | per-GPU util/mem/temp (2× 5090) |
| **System** | `status.sys` | CPU/RAM/disk/network |
| **Current job** | `status.current` + `status.proc` | stage, step, `timing` (ETA/%), `active` |
| **Start-job form** | build from `GET /api/capabilities` | engine/kind/printer/detail selects + prompt + image upload |
| **Outputs gallery** | `status.stls` | name, size, tris, dims, `manifold` (validated ✓). Download via `/api/download/<name>` |
| **3D preview** | `GET /api/mesh/<file.glb>` | feed to a three.js/`<model-viewer>` |
| **Concept image** | `GET /api/image/<file.png>` | |
| **Engine health** | `GET /api/health[?run=1]` | which engines import + CUDA/deps OK |
| **Logs** | `GET /api/logs` then `GET /api/log/<file>[?full=1]` | live tail (8 KB) or full for debugging |
| **Controls** | POSTs | Start (`/api/run`), Kill (`/api/kill`), Force-kill, Delete (`/api/delete`), **Lighten** (`/api/decimate`) |

### Minimal driver loop (works for an LLM or a UI)
1. `GET /api/capabilities` → know valid params.
2. `GET /api/status` → ensure `proc.running == false`.
3. `POST /api/run {engine, kind, prompt, detail, …}`.
4. Poll `GET /api/status` until `proc.running == false` / `current.active_stage == "done"`.
5. Success → `status.stls[]`; `GET /api/download/<name>` (optionally `POST /api/decimate` first to speed slicing).
6. Failure → `GET /api/logs` → `GET /api/log/<newest>?full=1` to diagnose.

---

## 6. Engines & key knobs (summary; live values in `/api/capabilities`)

- **Engines:** `trellis` (TRELLIS.2 4B, default, best detail) · `trellis_mv`
  (multi-view, experimental) · `hunyuan21` · `hunyuan2mv` (trained multi-view) ·
  `triposg` · `craftsman` · `hi3dgen` · `parametric` (build123d terrain, watertight
  by construction).
- **Kinds:** `mini` (~160 mm, gets a flat closed base) · `terrain` · `prop`.
- **Detail presets** (TRELLIS print finishing, detail vs time): `fast` · `balanced`
  (default) · `fine` · `ultra` — map to alpha-wrap fineness (`TRELLIS_ALPHA`).
- **Notable env knobs** (set in the container env; surfaced in capabilities):
  `TRELLIS_ALPHA`, `TRELLIS_ALPHAWRAP`, `TRELLIS_TEXTURE`, `TRELLIS_MAX_FACES`,
  `TRELLIS_PRINTSAFE`, `FLATTEN_BASE`, `GEN3D_PY`, `DASHBOARD_PORT`.

**Why printable output is reliable:** TRELLIS dual-contouring output is triangle
soup; the finisher runs **CGAL alpha-wrapping** (pymeshlab `generate_alpha_wrap`)
to produce one watertight 2-manifold that encloses the figure, then the **mesh
gauntlet** scales/seats/flattens the base, and **PrusaSlicer/Bambu** CLI validates
it (`status.stls[].manifold == true` means it passed).

---

## 7. Deploy / operate (how changes land)

The dashboard + pipeline are updated by pulling git and copying files into the
running container (no rebuild needed for code changes):

```bash
cd /opt/gen3d && git pull
docker cp tabletop/pipeline/.  gen3d-app:/app/tabletop/pipeline/
docker cp tabletop/dashboard/. gen3d-app:/app/tabletop/dashboard/
docker restart gen3d-app          # needed only for server.js changes
```
- `index.html` is served live (just hard-refresh the browser).
- `server.js` changes require `docker restart gen3d-app`.
- Python pipeline files are loaded fresh per run (no restart needed).
- Logs of every run: `/app/tabletop/outputs/logs/<ts>_<slug>.log` (host:
  `/opt/gen3d/data/outputs/logs/`).

---

## 8. Security (read before exposing it)

- **The API is unauthenticated and binds `0.0.0.0`.** Anyone who can reach
  port 7842 can start/kill/delete. Treat the port as privileged.
- Keep it on a **trusted LAN**; do not port-forward it raw.
- For remote/admin-dashboard access, front it with a **reverse proxy** adding
  Basic-Auth or a bearer token, or reach it over **VPN/Tailscale**.
- An **optional `ADMIN_TOKEN` bearer check** can be added server-side if you want
  per-request auth without a proxy (not enabled by default — ask the maintainer).
- Destructive routes: `/api/run`, `/api/kill`, `/api/forcekill`, `/api/delete`,
  `/api/decimate`.

---

## 9. Suggested "app contract" for the multi-app dashboard

If the other apps follow the same shape, the unified dashboard can treat each app
uniformly. This app already conforms to:

- `GET /api/capabilities` → self-description (enums + param schema + endpoint map)
- `GET /api/status` → live heartbeat (running flag, current job, resources, outputs)
- `GET /api/health` → component/engine self-test
- `POST /api/run` → start work (params from capabilities)
- `POST /api/kill` → stop work
- `GET /api/logs` + `GET /api/log/<f>?full=1` → diagnostics
- `GET /api/download/<f>` → fetch artifacts

Register this app in the dashboard as:
```json
{
  "id": "pipeline-studio",
  "name": "Pipeline Studio (3D print gen)",
  "base_url": "http://ai-pc.local:7842",
  "capabilities": "/api/capabilities",
  "status": "/api/status",
  "health": "/api/health",
  "run": "/api/run",
  "kill": "/api/kill",
  "logs": "/api/logs",
  "single_concurrency": true,
  "auth": "none (LAN only)"
}
```

→ **Full endpoint reference: `tabletop/dashboard/ADMIN_API.md`.**
