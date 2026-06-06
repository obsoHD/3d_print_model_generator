# Pipeline Studio — Admin / Control API

The dashboard server (`tabletop/dashboard/server.js`) exposes a plain HTTP/JSON
API that controls the **entire** image/prompt → printable-3D pipeline. It's the
same API the browser UI uses, so anything you can do in the dashboard you can do
programmatically — which means an **admin LLM** (or any script) can drive it end
to end: start jobs, monitor progress, read logs, fetch/clean/download results,
and run engine health checks.

- **Base URL:** `http://<workstation-ip>:<PORT>` (server binds `0.0.0.0`, so it's
  reachable from any machine on the LAN). `PORT` is the `PORT` env (check the
  container/launch). Example: `http://ai-pc.local:8787`.
- **Format:** all POST bodies are JSON (`Content-Type: application/json`); all
  responses are JSON unless noted (logs/images/meshes are raw bytes).
- **CORS:** `Access-Control-Allow-Origin: *` — callable cross-origin.
- **Errors:** handlers never crash the server; failures return
  `{ "ok": false, "error": "<message>" }` with HTTP 200. Unknown routes → 404.

> ⚠️ **No authentication.** Anyone who can reach the port can control the app and
> read/delete outputs. Keep it on a trusted LAN, or put it behind a reverse proxy
> / firewall / VPN. See [Security](#security).

---

## Start here: `/api/capabilities`

`GET /api/capabilities` returns a **self-describing** snapshot — valid engine
names, kinds, printers, detail presets, every `/api/run` parameter, the tunable
env knobs, and the full endpoint list. An admin LLM should fetch this first so it
never has to guess valid values.

```bash
curl -s http://HOST:PORT/api/capabilities | jq
```

---

## Control endpoints (POST)

### `POST /api/run` — start a generation
Kicks off the full pipeline (concept → mesh → gauntlet → STL → slicer-validate),
detached so it survives a dashboard restart. Returns immediately; poll
`/api/status` for progress. **Rejects if a run is already in progress** — kill it
first.

Body (all optional unless noted; see `/api/capabilities` for live enums):

| field | type | notes |
|---|---|---|
| `engine` | string | `trellis` (default UI) `trellis_mv` `hunyuan21` `hunyuan2mv` `triposg` `craftsman` `hi3dgen` `parametric` |
| `kind` | string | `mini` \| `terrain` \| `prop` (default `terrain`). `mini` → ~160 mm + **flat closed base** |
| `printer` | string | `resin` (default) \| `fdm` |
| `prompt` | string | names the model; if no image is supplied it's the **generation** prompt |
| `detail` | string | `fast` \| `balanced` \| `fine` \| `ultra` — TRELLIS alpha-wrap fineness (detail vs time) |
| `with_color` | bool | bake TRELLIS texture (preview only, slower) |
| `seed` | int | reproducibility |
| `concept_image_data_url` | string | `data:image/png;base64,…` single-view upload |
| `input_image_path` | string | server-side path to an existing image (instead of upload) |
| `mv_front_data` / `mv_left_data` / `mv_back_data` | string | data URLs for multi-view engines |
| `mv_gif_data` | string | data URL of a turntable GIF/MP4 (auto-split into 3 views) |
| `mv_reverse` | bool | reverse turntable spin |
| `mv_engine` | string | multi-view backend (`hunyuan2mv` \| `trellis_mv`) |

Response: `{ ok, pid, out_stl, log, mv_engine }` (or `{ ok:false, error }`).

```bash
# text-prompt terrain on TRELLIS, fine detail
curl -s -XPOST http://HOST:PORT/api/run -H 'Content-Type: application/json' -d '{
  "engine":"trellis","kind":"mini","printer":"resin",
  "prompt":"dwarf paladin with warhammer","detail":"fine"
}' | jq

# from an existing server-side image
curl -s -XPOST http://HOST:PORT/api/run -H 'Content-Type: application/json' -d '{
  "engine":"trellis","kind":"mini","prompt":"ranger",
  "input_image_path":"/app/tabletop/outputs/concepts/foo.png"
}' | jq
```

### `POST /api/concept` — generate one concept image
`{ "prompt": "...", "seed": 123 }` → renders a single concept PNG (for the
prompt-first flow / retries). Returns the saved image path. Synchronous (waits).

### `POST /api/kill` — stop the running pipeline
No body. SIGKILLs all pipeline processes (orchestrate + children:
gauntlet/blender/pymeshlab/etc.). Use when a run is stuck.

### `POST /api/forcekill` — stop **and delete** a run
`{ "run_id": "<id>" }` (run_id optional → current). Kills the pipeline **and
removes that run's artifacts/data**. Harder reset than `/api/kill`.

### `POST /api/delete` — delete an output
`{ "name": "model.stl" }` (or `.glb`). Removes the result file + its slicer/
cleanup sidecar reports. Allowlisted to the outputs dirs by extension.

### `POST /api/decimate` — lighten an STL for fast slicing
`{ "name": "model.stl", "target": 250000 }` (target optional, default 250k).
Quadric edge-collapse **in place**, topology-preserved so it stays watertight.
Turns a ~1.5 M-face mesh into a fast-slicing one in ~10–60 s. Synchronous.

### `POST /api/reveal` — open in host file manager
`{ "path": "/app/tabletop/outputs/stl/model.stl" }`. **Host-side only** (opens a
GUI file manager on the machine running the server) — not useful for a remote LLM,
listed for completeness. Allowlisted to the outputs dir.

---

## Monitor & read endpoints (GET)

### `GET /api/status` — the live heartbeat (poll this)
The primary monitoring endpoint. Returns:

| key | meaning |
|---|---|
| `ts` | server timestamp |
| `gpu` | per-GPU util/mem/temp (dual 5090) |
| `sys` | CPU/RAM/disk + network throughput |
| `proc` | `{ running, stage, … }` — is a pipeline running and what stage |
| `current` | the active/last run: `step`, `active`, `active_stage`, `timing` (ETA/progress) |
| `history` | recent runs (id, prompt, engine, step, result manifest) |
| `stls` | finished STLs: `name`, `mb`, `tris`, `dims`, `manifold` (validated) |
| `log` | filename of the newest run log (feed to `/api/log/<file>`) |

```bash
watch -n2 'curl -s http://HOST:PORT/api/status | jq "{stage:.proc.stage, step:.current.step, eta:.current.timing}"'
```

### `GET /api/health[?run=1]` — engine self-test
Returns the cached engine probe report (`engine_selftest.py`): which engines
import + their CUDA/deps status. `?run=1` forces a fresh probe (can take minutes;
response is immediate with `running:true`, poll again for the report).

### `GET /api/logs` — list run logs
`{ ok, logs: [{ name, bytes, mtime }] }`, newest first. Pick a `name`, then fetch
its text via `/api/log/<name>`.

### `GET /api/log/<file>[?full=1]` — run log text
Plain text. Default returns the **last 8 KB** (live tail); add **`?full=1`** for
the entire log (use for debugging a failed run). The full log captures everything
the live dashboard truncates.

```bash
curl -s "http://HOST:PORT/api/log/$(curl -s http://HOST:PORT/api/logs | jq -r '.logs[0].name')?full=1"
```

### `GET /api/mesh/<file.glb>` — GLB bytes (for the 3D viewer)
### `GET /api/image/<file.png>` — concept image bytes
### `GET /api/download/<file>` — force-download a result
STL/GLB/3MF/OBJ/PLY/GCODE with `Content-Disposition: attachment`. Works from any
machine on the network. Allowlisted to the outputs result dirs by extension.

---

## Typical admin-LLM loop

1. `GET /api/capabilities` — learn valid engines/kinds/params.
2. `GET /api/status` — confirm nothing is running (`proc.running == false`).
3. `POST /api/run` — start a job with the chosen params.
4. Poll `GET /api/status` until `current.active == false` and `current.active_stage == "done"`
   (or `proc.running == false`).
5. On failure: `GET /api/logs` → `GET /api/log/<newest>?full=1` to diagnose.
6. On success: read `stls[]` for the result; `GET /api/download/<name>` to fetch,
   or `POST /api/decimate` first if it needs to be lighter for slicing.
7. Reset if stuck: `POST /api/kill` (or `POST /api/forcekill`).

---

## Output locations (inside the container)
- STLs: `/app/tabletop/outputs/stl/`
- GLB meshes (+ `.gauntlet.glb`, `.gauntlet.json` reports): `/app/tabletop/outputs/meshes/`
- Concept images: `/app/tabletop/outputs/concepts/`
- Full per-run logs: `/app/tabletop/outputs/logs/<ts>_<slug>.log`
- Print manifests: `/app/tabletop/outputs/stl/<name>.print.json`

(Host bind-mount: `/opt/gen3d/data/outputs/…`.)

## Security
The API is **unauthenticated and binds all interfaces.** For an admin-LLM setup:
- Keep it on a trusted LAN / behind the firewall; do **not** port-forward it.
- If you need remote access, front it with a reverse proxy (nginx/Caddy) that
  adds **Basic-Auth or a bearer token**, or reach it over a **VPN/Tailscale**.
- Destructive routes (`/api/delete`, `/api/forcekill`, `/api/decimate`,
  `/api/run`) are reachable by anyone who can hit the port — treat the port as
  privileged.
