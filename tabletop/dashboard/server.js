// server.js — Pipeline Dashboard (Node.js, zero deps)
const http = require('http');
const fs   = require('fs');
const path = require('path');
const cp   = require('child_process');

const PORT      = parseInt(process.env.DASHBOARD_PORT || '7842');
const DASH_DIR  = __dirname;
const TABLETOP  = path.resolve(__dirname, '..');
const CONCEPTS  = path.join(TABLETOP, 'outputs', 'concepts');
const MESHES    = path.join(TABLETOP, 'outputs', 'meshes');
const STL_DIR   = path.join(TABLETOP, 'outputs', 'stl');
const LOG_DIR   = path.join(TABLETOP, 'outputs', 'logs');
// Ensure all output dirs exist (fresh /data/outputs only has what we make here).
// Missing concepts/ was crashing saveUserImage on upload.
for (const d of [CONCEPTS, MESHES, STL_DIR, LOG_DIR]) {
  try { if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true }); } catch (e) {}
}
let _activeLog = null;   // path to current run's log file

// ── CPU + RAM ────────────────────────────────────────────────────────────────
const os = require('os');
let _lastCpu = null;
function getSystem() {
  // CPU: snapshot, then diff against previous call (gives % util since last poll)
  const cpus = os.cpus();
  const agg = cpus.reduce((a, c) => ({
    user: a.user + c.times.user, nice: a.nice + c.times.nice,
    sys: a.sys + c.times.sys, idle: a.idle + c.times.idle,
    irq: a.irq + c.times.irq,
  }), { user:0, nice:0, sys:0, idle:0, irq:0 });
  agg.total = agg.user + agg.nice + agg.sys + agg.idle + agg.irq;
  let cpuPct = 0;
  if (_lastCpu) {
    const dTotal = agg.total - _lastCpu.total;
    const dIdle  = agg.idle  - _lastCpu.idle;
    cpuPct = dTotal > 0 ? Math.round(100 * (1 - dIdle/dTotal)) : 0;
  }
  _lastCpu = agg;
  const totalMem = os.totalmem();
  const freeMem  = os.freemem();
  return {
    cpu_util:   cpuPct,
    cpu_cores:  cpus.length,
    cpu_model:  cpus[0] ? cpus[0].model.split(' ').slice(0, 4).join(' ') : 'CPU',
    ram_used_mb:  Math.round((totalMem - freeMem) / 1048576),
    ram_total_mb: Math.round(totalMem / 1048576),
    loadavg:    os.loadavg(),
    uptime_s:   Math.round(os.uptime()),
  };
}

// ── GPU ──────────────────────────────────────────────────────────────────────
function getGPU() {
  try {
    const out = cp.execSync(
      'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits',
      { timeout: 3000 }
    ).toString().trim();
    const [util, used, total, temp] = out.split(',').map(s => parseInt(s.trim()));
    return { util, mem_used: used, mem_total: total, temp };
  } catch { return { util: 0, mem_used: 0, mem_total: 16303, temp: 0 }; }
}

// ── PROCESS DETECTION + log-based substage tracking ─────────────────────────
// Stage map (6 substages, in pipeline order):
//   concept    — RealVisXL/DreamShaper/FLUX inferring concept image
//   bg_removal — rembg cutting background to white
//   multiview  — Zero123++ generating 4 views from 1 concept
//   mesh       — Hunyuan3D building 3D mesh from views
//   blender    — Blender cleanup (loose parts, voxel remesh, baseplate trim)
//   done       — finished
let _stageCache = { running: false, stage: null, substep: '' };

function _readLogTail(maxBytes = 4096) {
  if (!_activeLog || !fs.existsSync(_activeLog)) return '';
  try {
    const stat = fs.statSync(_activeLog);
    const start = Math.max(0, stat.size - maxBytes);
    const fd = fs.openSync(_activeLog, 'r');
    const buf = Buffer.alloc(stat.size - start);
    fs.readSync(fd, buf, 0, buf.length, start);
    fs.closeSync(fd);
    return buf.toString('utf8');
  } catch { return ''; }
}

function _deriveSubstage(procRunning, cmdLines, logTail) {
  // Process gives the gross signal; log tail gives the fine-grained substage.
  const hasHy3d   = cmdLines.includes('hy3d_infer.py');
  const hasZ123   = cmdLines.includes('_zero123_infer.py');
  const hasConcept= cmdLines.includes('_concept_infer.py');
  const hasOrch   = cmdLines.includes('orchestrate.py');
  if (!hasOrch && !hasHy3d && !hasZ123 && !hasConcept) {
    return { running: false, stage: null, substep: '' };
  }
  // Last few log lines reveal the active substep
  const lines = logTail.split(/\r?\n/).filter(Boolean);
  const last = lines.slice(-10).join('\n').toLowerCase();
  // Order matters — later stages override earlier markers in the same log
  if (last.includes('[4/4]') || last.includes('=== done')) {
    return { running: false, stage: 'done', substep: 'pipeline complete' };
  }
  if (hasHy3d || /\[2\/4\].*mesh|\[hy3d_infer\]/i.test(logTail)) {
    let sub = 'HY3D inference';
    if (/loading.*pipeline|loading.*weights/i.test(last)) sub = 'loading HY3D model weights';
    else if (/preprocessing image/i.test(last)) sub = 'preprocessing input image(s)';
    else if (/shape generation/i.test(last)) sub = 'running shape diffusion (50 steps)';
    else if (/mesh saved|export/i.test(last)) sub = 'saving mesh GLB';
    return { running: true, stage: 'mesh', substep: sub };
  }
  if (hasZ123 || last.includes('zero123')) {
    let sub = 'generating multi-view';
    if (last.includes('loading pipeline'))    sub = 'loading Zero123++ model';
    else if (last.includes('generating 6'))   sub = 'diffusing 6 views (50 steps)';
    else if (last.includes('saved'))          sub = 'splitting 2×3 grid into views';
    return { running: true, stage: 'multiview', substep: sub };
  }
  // View-fix phase — high-strength img2img regen on broken side views
  if (/fixing (front|right|back|left|top) view/i.test(last) ||
      /view-fix pass regenerated/i.test(last)) {
    const m = /fixing (front|right|back|left|top) view\s*\[(broken[^\]]*|force)/i.exec(last);
    const what = m ? `fixing ${m[1]} view (${m[2]})` : 'view-fix img2img (str 0.55)';
    return { running: true, stage: 'multiview', substep: what };
  }
  // MV upscale phase — SDXL Refiner re-synthesizing each side view at 1024
  if (last.includes('upscale') || last.includes('upscaling') || /upscaled \d+ side/i.test(last)) {
    return { running: true, stage: 'multiview',
             substep: 'upscaling views (SDXL Refiner img2img → 1024×1024)' };
  }
  if (last.includes('background removed')) {
    return { running: true, stage: 'multiview', substep: 'about to launch multi-view' };
  }
  // Concept refiner phase — SDXL Refiner img2img on the rembg'd concept
  if (/\[image_refine\]/i.test(last) && /concept|refine|strength/i.test(last)) {
    let sub = 'refining concept (SDXL Refiner img2img str 0.22)';
    if (last.includes('sdxl-refiner')) sub = 'SDXL Refiner sharpening concept detail';
    return { running: true, stage: 'concept', substep: sub };
  }
  if (last.includes('rembg') || last.includes('background')) {
    return { running: true, stage: 'bg_removal', substep: 'running rembg/u2net segmentation' };
  }
  // Detail enhance phase — PyMeshLab after Blender
  if (/\[detail_enhance\]/i.test(last) || /\[3b\/4\]/i.test(last)) {
    let sub = 'PyMeshLab detail enhancement';
    if (last.includes('sharpened')) sub = 'PyMeshLab normal-unsharp mask applied';
    else if (last.includes('smoothed')) sub = 'PyMeshLab edge-preserving smooth applied';
    return { running: true, stage: 'blender', substep: sub };
  }
  if (hasConcept || last.includes('concept_gen') || last.includes('concept_infer')) {
    let sub = 'generating concept image';
    if (last.includes('flux_schnell'))     sub = 'FLUX.1-schnell · 4 steps (CPU offload)';
    else if (last.includes('realvis_v5'))  sub = 'RealVisXL v5 · DPM++ 2M Karras · 35 steps';
    else if (last.includes('dreamshaper')) sub = 'DreamShaper XL Turbo · 6 steps';
    else if (last.includes('user-provided'))sub = 'using user-provided image as concept';
    return { running: true, stage: 'concept', substep: sub };
  }
  if (hasOrch && /\[3\/4\].*blender|blender_cleanup/i.test(logTail)) {
    let sub = 'Blender cleanup';
    if (last.includes('loose'))         sub = 'removing loose / flat artifacts';
    else if (last.includes('voxel'))    sub = 'voxel remesh + Taubin smooth';
    else if (last.includes('baseplate'))sub = 'trimming oversized baseplate';
    else if (last.includes('manifold')) sub = 'manifold check + hole fill';
    else if (last.includes('stl_export') || last.includes('-> ') && last.includes('.stl'))
      sub = 'exporting STL';
    return { running: true, stage: 'blender', substep: sub };
  }
  // Fallback if orchestrate is running but we can't pin it to a substage yet
  return { running: true, stage: 'concept', substep: 'starting up…' };
}

// List running python command lines, cross-platform (Linux workstation = ps).
function _listPythonCmds(cb) {
  if (process.platform === 'win32') {
    cp.exec(
      'powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"name=\'python.exe\'\\\" | Select-Object -ExpandProperty CommandLine"',
      { timeout: 6000 },
      (err, stdout) => cb((err || !stdout) ? '' : stdout));
  } else {
    cp.exec('ps -eo args=', { timeout: 6000, maxBuffer: 8 * 1024 * 1024 },
      (err, stdout) => cb((err || !stdout) ? '' : stdout));
  }
}

function _refreshStage() {
  _listPythonCmds((cmdLines) => {
    const logTail = _readLogTail();
    _stageCache = _deriveSubstage(true, cmdLines, logTail);
  });
}
_refreshStage();
setInterval(_refreshStage, 3000);
function getActiveStage() { return _stageCache; }

// ── RUNS ─────────────────────────────────────────────────────────────────────
function parseRunId(stem) {
  const m = stem.match(/^(\d{8})_(\d{6})_(.+)_[0-9a-f]{6}$/);
  if (!m) return { ts: null, prompt: stem };
  const [, date, time, words] = m;
  const ts = `${date.slice(0,4)}-${date.slice(4,6)}-${date.slice(6,8)} ${time.slice(0,2)}:${time.slice(2,4)}:${time.slice(4,6)}`;
  return { ts, prompt: words.replace(/_/g, ' ') };
}

// Sub-view filename suffixes — these are siblings of the main concept,
// not standalone runs. Exclude them when listing runs.
const SUB_VIEW_SUFFIXES = /_(front|right|back|left|top|mv_grid)\.png$/i;

function getRuns() {
  if (!fs.existsSync(CONCEPTS)) return [];
  const pngs = fs.readdirSync(CONCEPTS)
    .filter(f => f.endsWith('.png') && !SUB_VIEW_SUFFIXES.test(f))
    .map(f => ({ name: f, mtime: fs.statSync(path.join(CONCEPTS, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime)
    .slice(0, 8);

  return pngs.map(({ name }) => {
    const stem = name.replace('.png', '');
    const { ts, prompt } = parseRunId(stem);
    const glbPath = path.join(MESHES, stem + '.glb');
    const hasGlb  = fs.existsSync(glbPath);
    const glbMb   = hasGlb ? (fs.statSync(glbPath).size / 1048576).toFixed(1) : null;

    let step = 1;
    if (hasGlb) step = 3;
    if (fs.existsSync(STL_DIR)) {
      const stls = fs.readdirSync(STL_DIR).filter(f => f.endsWith('.stl'));
      if (stls.length && hasGlb) {
        const newest = stls.map(f => fs.statSync(path.join(STL_DIR, f)).mtimeMs).reduce((a,b)=>Math.max(a,b),0);
        if (newest >= fs.statSync(glbPath).mtimeMs - 1000) step = 4;
      }
    }
    // Multi-view concept detection: {stem}_left/back/right/top.png siblings
    const viewFile = (suffix) => {
      const f = stem + '_' + suffix + '.png';
      return fs.existsSync(path.join(CONCEPTS, f)) ? f : null;
    };
    const conceptViews = {
      front: viewFile('front') || name,
      left:  viewFile('left'),
      back:  viewFile('back'),
      right: viewFile('right'),
      top:   viewFile('top'),
    };
    const hasMv = !!(conceptViews.left || conceptViews.back || conceptViews.right);

    return { run_id: stem, timestamp: ts, prompt, step,
             concept_file: name, has_glb: hasGlb, glb_mb: glbMb,
             glb_file: hasGlb ? stem + '.glb' : null,
             concept_views: conceptViews, has_mv: hasMv };
  });
}

function getSTLs() {
  if (!fs.existsSync(STL_DIR)) return [];
  return fs.readdirSync(STL_DIR)
    .filter(f => f.endsWith('.stl'))
    .map(f => {
      const full = path.join(STL_DIR, f);
      const mb   = (fs.statSync(full).size / 1048576).toFixed(1);
      const cj   = full.replace('.stl', '.cleanup.json');
      let extra  = {};
      if (fs.existsSync(cj)) {
        try { extra = JSON.parse(fs.readFileSync(cj, 'utf8')); } catch {}
      }
      return { name: f, mb, tris: extra.triangles, dims: extra.dims_mm,
               manifold: extra.non_manifold_after === 0 };
    })
    .sort((a, b) => fs.statSync(path.join(STL_DIR, b.name)).mtimeMs
                  - fs.statSync(path.join(STL_DIR, a.name)).mtimeMs)
    .slice(0, 4);
}

// ── TIMING ESTIMATION ────────────────────────────────────────────────────────
// Per-stage estimated duration in seconds. Tuned for RealVisXL + Zero123++ +
// HY3D mv + Blender on RTX 5080 / 16 GB.
const STEP_EST = {
  concept:    45,    // RealVis ~30s + load
  bg_removal: 6,     // rembg/u2net
  multiview:  90,    // Zero123++ 50 steps + split
  mesh:       1800,  // HY3D 2.0 mv (octree 512 · 50 steps)
  blender:    200,   // cleanup + STL export
};
const STAGE_ORDER = ['concept', 'bg_removal', 'multiview', 'mesh', 'blender'];

function calcTiming(current) {
  if (!current || !current.active) return null;
  const stage = current.active_stage;
  if (!stage || stage === 'done' || stage === 'idle') return null;

  // Infer run start from concept file mtime
  const conceptPath = path.join(CONCEPTS, current.concept_file);
  let runStart = Date.now() / 1000;
  try { runStart = fs.statSync(conceptPath).mtimeMs / 1000; } catch {}

  const now          = Date.now() / 1000;
  const totalElapsed = Math.round(now - runStart);

  // How long did steps before current stage take?
  const stageIdx  = STAGE_ORDER.indexOf(stage);
  const prevTotal = STAGE_ORDER.slice(0, stageIdx).reduce((s, k) => s + STEP_EST[k], 0);
  const stepElapsed = Math.max(0, totalElapsed - prevTotal);
  const stepEst     = STEP_EST[stage] || 300;
  const stepPct     = Math.min(99, Math.round(stepElapsed / stepEst * 100));

  const totalEst   = Object.values(STEP_EST).reduce((a, b) => a + b, 0);
  const totalPct   = Math.min(99, Math.round(totalElapsed / totalEst * 100));
  const etaSec     = Math.max(0, totalEst - totalElapsed);

  // Substep label — prefer the live log-derived one (set by _refreshStage)
  let substep = (_stageCache && _stageCache.substep) || '';
  if (!substep) {
    if (stage === 'concept')         substep = 'generating concept image…';
    else if (stage === 'bg_removal') substep = 'removing background (rembg/u2net)';
    else if (stage === 'multiview')  substep = 'generating multi-view (Zero123++)';
    else if (stage === 'mesh') {
      if (stepElapsed < 300)       substep = 'loading HY3D weights (~5 min)';
      else if (stepElapsed < 650)  substep = 'CUDA kernel warmup — first step only';
      else {
        const inferPct = (stepElapsed - 650) / Math.max(1, stepEst - 650);
        const estStep  = Math.min(50, Math.max(2, Math.round(inferPct * 49) + 2));
        substep = `Diffusion sampling ~${estStep} / 50 steps est.`;
      }
    }
    else if (stage === 'blender')   substep = 'mesh repair + STL export…';
  }

  return {
    total_elapsed_s: totalElapsed,
    step_elapsed_s:  stepElapsed,
    step_est_s:      stepEst,
    step_pct:        stepPct,
    total_pct:       totalPct,
    eta_s:           etaSec,
    substep,
  };
}

function getStatus() {
  const gpu   = getGPU();
  const sys   = getSystem();
  const proc  = getActiveStage();
  const runs  = getRuns();
  const stls  = getSTLs();
  const now   = new Date().toISOString().replace('T', ' ').slice(0, 19);

  let current = null;
  if (runs.length) {
    const r = runs[0];
    if (proc.running) {
      const cur = { ...r, active: true, active_stage: proc.stage || 'mesh' };
      cur.timing = calcTiming(cur);
      current = cur;
    } else if (r.step === 4) {
      current = { ...r, active: false, active_stage: 'done' };
    } else {
      current = { ...r, active: false, active_stage: 'idle' };
    }
  }
  // Include latest log file (for the live console tail)
  let latestLog = _activeLog && fs.existsSync(_activeLog) ? path.basename(_activeLog) : null;
  if (!latestLog && fs.existsSync(LOG_DIR)) {
    const logs = fs.readdirSync(LOG_DIR).filter(f => f.endsWith('.log'))
      .map(f => ({ f, m: fs.statSync(path.join(LOG_DIR, f)).mtimeMs }))
      .sort((a, b) => b.m - a.m);
    if (logs.length) latestLog = logs[0].f;
  }
  return { ts: now, gpu, sys, proc, current, history: runs, stls, log: latestLog };
}

// ── LAUNCH / KILL ────────────────────────────────────────────────────────────
function readBody(req) {
  return new Promise((resolve, reject) => {
    let d = '';
    req.on('data', c => d += c);
    req.on('end', () => { try { resolve(JSON.parse(d || '{}')); } catch { resolve({}); } });
    req.on('error', reject);
  });
}

// Save a base64 dataURL image into outputs/concepts/_user_<ts>.png so the
// orchestrator can pick it up via --input-image. Returns the saved path.
function saveUserImage(dataUrl, tag) {
  const m = /^data:(image|video)\/(\w+);base64,(.+)$/.exec(dataUrl || '');
  if (!m) return null;
  const ext = m[2] === 'jpeg' ? 'jpg' : m[2];
  const buf = Buffer.from(m[3], 'base64');
  const ts  = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
  const fn  = `_user_${tag || 'img'}_${ts}.${ext}`;
  const dst = path.join(CONCEPTS, fn);
  fs.writeFileSync(dst, buf);
  return dst;
}

// Generate ONE concept image from a prompt (for the prompt-first wizard flow
// with retry/continue). Synchronous-ish: spawns concept_gen, waits, returns
// the image path. seed varies per call so "retry" gives a new image.
function genConcept(opts) {
  const { prompt, kind = 'mini', seed } = opts || {};
  if (!prompt || !prompt.trim()) return { ok: false, error: 'prompt required' };
  const ts = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
  const fn = `_concept_${ts}.png`;
  const dst = path.join(CONCEPTS, fn);
  const cpy = process.env.GEN3D_PY || 'python';
  const code = `import sys;sys.path.insert(0,r'${path.join(TABLETOP,'pipeline')}');`
    + `import concept_gen;concept_gen.generate(prompt=${JSON.stringify(prompt)},`
    + `out_path=r'${dst}',kind=${JSON.stringify(kind)},`
    + `seed=${seed != null ? Number(seed) : 'None'})`;
  const r = cp.spawnSync(cpy, ['-c', code],
    { timeout: 240000, env: { ...process.env, PYTHONUTF8: '1' } });
  if (r.status !== 0 || !fs.existsSync(dst)) {
    return { ok: false, error: 'concept gen failed',
             detail: String(r.stderr || '').slice(-600) };
  }
  return { ok: true, image: '/api/image/' + encodeURIComponent(fn),
           path: dst, file: fn };
}

function launchRun(opts) {
  const { prompt, kind='terrain', engine='hunyuan', printer='resin',
          seed, mv_engine, concept_image_data_url, input_image_path,
          mv_front_data, mv_left_data, mv_back_data, mv_gif_data,
          mv_reverse } = opts;
  if (_stageCache.running) return { ok: false, error: 'pipeline already running — kill it first' };
  const pipeDir = path.join(TABLETOP, 'pipeline');

  // ---- MULTI-VIEW launch (engine=hunyuan2mv): front/left/back or a GIF ----
  if (engine === 'hunyuan2mv') {
    const slugmv = (prompt || 'multiview').replace(/[^a-z0-9]/gi, '_').slice(0,28).toLowerCase();
    const outStlMv = path.join(STL_DIR, `${slugmv || 'multiview'}.stl`);
    const args = [path.join(pipeDir, 'orchestrate.py'),
      '--engine', 'hunyuan2mv', '--kind', kind, '--printer', printer,
      '--out', outStlMv, '--prompt', prompt || 'multiview'];
    if (seed) args.push('--seed', String(seed));
    if (mv_gif_data) {
      const g = saveUserImage(mv_gif_data, 'gif'); if (!g) return {ok:false,error:'bad gif data'};
      args.push('--mv-gif', g);
    } else {
      const f = saveUserImage(mv_front_data, 'front');
      const l = saveUserImage(mv_left_data,  'left');
      const b = saveUserImage(mv_back_data,  'back');
      if (!f || !l || !b) return { ok:false, error:'multi-view needs front, left and back images (or a GIF)' };
      args.push('--mv-front', f, '--mv-left', l, '--mv-back', b);
    }
    if (mv_reverse) args.push('--mv-reverse');
    return _spawnOrchestrate(args, slugmv, prompt || 'multiview', kind, engine, null);
  }

  // Either a prompt OR an uploaded image is required
  let inputImagePath = null;
  if (concept_image_data_url) {
    inputImagePath = saveUserImage(concept_image_data_url);
    if (!inputImagePath) return { ok: false, error: 'invalid concept image data' };
  } else if (input_image_path) {
    // Caller already has a file on disk inside outputs/ — use it directly
    const safe = path.resolve(input_image_path);
    const outRoot = path.resolve(path.join(TABLETOP, 'outputs'));
    if (!safe.startsWith(outRoot) || !fs.existsSync(safe)) {
      return { ok: false, error: 'input_image_path missing or outside outputs/' };
    }
    inputImagePath = safe;
  }
  if (!inputImagePath && (!prompt || !prompt.trim())) {
    return { ok: false, error: 'prompt or concept image required' };
  }
  const slug    = prompt.replace(/[^a-z0-9]/gi, '_').slice(0, 28).toLowerCase();
  const outStl  = path.join(STL_DIR, `${slug}.stl`);
  const effectivePrompt = prompt && prompt.trim() ? prompt :
    `user image: ${path.basename(inputImagePath || '')}`;
  const args = [path.join(pipeDir, 'orchestrate.py'),
    '--prompt', effectivePrompt, '--kind', kind,
    '--engine', engine, '--printer', printer, '--out', outStl];
  if (seed) args.push('--seed', String(seed));
  if (inputImagePath) args.push('--input-image', inputImagePath);
  return _spawnOrchestrate(args, slug, prompt, kind, engine, mv_engine);
}

function _spawnOrchestrate(args, slug, prompt, kind, engine, mv_engine) {
  const pipeDir = path.join(TABLETOP, 'pipeline');
  const outStl = args[args.indexOf('--out') + 1];

  // Capture stdout+stderr to a log file so the dashboard can tail it
  const ts = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
  const logPath = path.join(LOG_DIR, `${ts}_${slug}.log`);
  const logFd   = fs.openSync(logPath, 'a');
  const mvNote = mv_engine ? ` · mv_engine=${mv_engine}` : '';
  fs.writeSync(logFd, `[${new Date().toISOString()}] launch · prompt="${prompt}" · kind=${kind} · engine=${engine}${mvNote}\n`);
  // Build env — pass MV_ENGINE through so orchestrate.py picks it up
  const childEnv = { ...process.env, PYTHONUTF8: '1' };
  if (mv_engine) childEnv.MV_ENGINE = mv_engine;
  // detached + unref so the orchestrator survives dashboard server restarts.
  // Without this, stopping the dashboard process kills its child (HY3D, etc).
  const child = cp.spawn(process.env.GEN3D_PY || 'python', ['-u', ...args], {
    cwd: pipeDir, detached: true, stdio: ['ignore', logFd, logFd],
    env: childEnv,
    windowsHide: true,
  });
  child.on('exit', (code) => {
    try { fs.writeSync(logFd, `[${new Date().toISOString()}] exit code=${code}\n`); fs.closeSync(logFd); } catch {}
  });
  child.unref();   // sever the lifecycle link
  _activeLog = logPath;
  return { ok: true, pid: child.pid, out_stl: outStl, log: path.basename(logPath), mv_engine: mv_engine || 'zero123' };
}

// Open file in OS file manager. Locked to our outputs dir for safety.
function revealFile(opts) {
  const target = opts && opts.path ? String(opts.path) : '';
  if (!target) return { ok: false, error: 'path required' };
  // Allowlist: must resolve under tabletop/outputs/
  const safe = path.resolve(target);
  const outRoot = path.resolve(path.join(TABLETOP, 'outputs'));
  if (!safe.startsWith(outRoot)) return { ok: false, error: 'path outside outputs dir' };
  if (!fs.existsSync(safe))     return { ok: false, error: 'not found' };
  if (process.platform === 'win32') {
    cp.exec(`explorer.exe /select,"${safe}"`, () => {});
  } else if (process.platform === 'darwin') {
    cp.exec(`open -R "${safe}"`, () => {});
  } else {
    cp.exec(`xdg-open "${path.dirname(safe)}"`, () => {});
  }
  return { ok: true, revealed: safe };
}

// Delete a result file (STL/GLB) + its report sidecars. Allowlisted to the
// outputs result dirs by extension + path resolution.
function deleteOutput(opts) {
  const name = opts && opts.name ? path.basename(String(opts.name)) : '';
  if (!name) return { ok: false, error: 'name required' };
  let dir;
  if (/\.stl$/i.test(name))      dir = STL_DIR;
  else if (/\.glb$/i.test(name)) dir = MESHES;
  else return { ok: false, error: 'only .stl/.glb can be deleted' };
  const target  = path.resolve(path.join(dir, name));
  const outRoot = path.resolve(path.join(TABLETOP, 'outputs'));
  if (!target.startsWith(outRoot)) return { ok: false, error: 'path outside outputs dir' };
  const removed = [];
  try {
    if (fs.existsSync(target)) { fs.unlinkSync(target); removed.push(name); }
    // remove sidecar reports that share the stem
    const stem = name.replace(/\.(stl|glb)$/i, '');
    for (const sfx of ['.cleanup.json', '.validate.json', '.print.json']) {
      const sc = path.join(dir, stem + sfx);
      if (fs.existsSync(sc)) { fs.unlinkSync(sc); removed.push(stem + sfx); }
    }
  } catch (e) {
    return { ok: false, error: String(e && e.message || e) };
  }
  if (!removed.length) return { ok: false, error: 'not found' };
  return { ok: true, removed };
}

function killPipeline() {
  if (process.platform === 'win32') {
    cp.exec(
      'powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"name=\'python.exe\'\\\" | Where-Object {$_.CommandLine -match \'orchestrate|hy3d_infer\'} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force}"',
      { timeout: 8000 }, () => {}
    );
  } else {
    // Linux workstation: pattern-kill the pipeline subprocess tree.
    cp.exec(
      "pkill -f orchestrate.py; pkill -f hy3d_infer; pkill -f _hunyuan; " +
      "pkill -f _triposg_infer; pkill -f finish_mini; pkill -f mesh_gauntlet",
      { timeout: 8000 }, () => {}
    );
  }
  return { ok: true };
}

// Last-resort guards: keep the dashboard alive even if something throws async.
process.on('uncaughtException',  (e) => console.error('[uncaught]', e && e.stack || e));
process.on('unhandledRejection', (e) => console.error('[unhandledRejection]', e));

// ── HTTP ─────────────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  if (req.method === 'POST') {
    const body = await readBody(req);
    let result;
    try {
      if (url === '/api/run')        result = launchRun(body);
      else if (url === '/api/concept') result = genConcept(body);
      else if (url === '/api/kill')  result = killPipeline();
      else if (url === '/api/delete') result = deleteOutput(body);
      else if (url === '/api/reveal') result = revealFile(body);
      else { res.writeHead(404); res.end('not found'); return; }
    } catch (e) {
      // Never let a handler error crash the whole server.
      result = { ok: false, error: String(e && e.message || e) };
    }
    const b = JSON.stringify(result);
    res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(b) });
    res.end(b);
    return;
  }

  if (url === '/api/status') {
    const body = JSON.stringify(getStatus());
    res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) });
    res.end(body);

  } else if (url.startsWith('/api/mesh/')) {
    const fname = decodeURIComponent(url.slice('/api/mesh/'.length));
    const fpath = path.join(MESHES, path.basename(fname));
    if (fs.existsSync(fpath) && /\.glb$/i.test(fname)) {
      const data = fs.readFileSync(fpath);
      res.writeHead(200, { 'Content-Type': 'model/gltf-binary', 'Content-Length': data.length });
      res.end(data);
    } else {
      res.writeHead(404); res.end('not found');
    }

  } else if (url.startsWith('/api/log/')) {
    const fname = decodeURIComponent(url.slice('/api/log/'.length));
    const fpath = path.join(LOG_DIR, path.basename(fname));
    if (fs.existsSync(fpath) && fname.endsWith('.log')) {
      // Return last ~8 KB to keep responses small
      const stat = fs.statSync(fpath);
      const start = Math.max(0, stat.size - 8192);
      const fd = fs.openSync(fpath, 'r');
      const buf = Buffer.alloc(stat.size - start);
      fs.readSync(fd, buf, 0, buf.length, start);
      fs.closeSync(fd);
      const text = buf.toString('utf8');
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8',
        'Content-Length': Buffer.byteLength(text) });
      res.end(text);
    } else {
      res.writeHead(404); res.end('not found');
    }

  } else if (url.startsWith('/api/image/')) {
    const fname = decodeURIComponent(url.slice('/api/image/'.length));
    const fpath = path.join(CONCEPTS, path.basename(fname));
    if (fs.existsSync(fpath) && /\.(png|jpg|jpeg)$/i.test(fname)) {
      const data = fs.readFileSync(fpath);
      res.writeHead(200, { 'Content-Type': 'image/png', 'Content-Length': data.length });
      res.end(data);
    } else {
      res.writeHead(404); res.end('not found');
    }

  } else if (url.startsWith('/api/download/')) {
    // Force-download a result file (STL/GLB/3MF) — works from any machine on
    // the network. Allowlisted to the outputs result dirs by extension.
    const fname = path.basename(decodeURIComponent(url.slice('/api/download/'.length)));
    let fpath = null, ctype = 'application/octet-stream';
    if (/\.stl$/i.test(fname))       { fpath = path.join(STL_DIR, fname); ctype = 'model/stl'; }
    else if (/\.glb$/i.test(fname))  { fpath = path.join(MESHES,  fname); ctype = 'model/gltf-binary'; }
    else if (/\.(3mf|gcode|obj|ply)$/i.test(fname)) { fpath = path.join(STL_DIR, fname); }
    if (fpath && fs.existsSync(fpath)) {
      const data = fs.readFileSync(fpath);
      res.writeHead(200, {
        'Content-Type': ctype,
        'Content-Length': data.length,
        'Content-Disposition': `attachment; filename="${fname}"`,
      });
      res.end(data);
    } else {
      res.writeHead(404); res.end('not found');
    }

  } else if (url === '/favicon.svg' || url === '/favicon.ico') {
    const icoPath = path.join(DASH_DIR, 'favicon.svg');
    if (fs.existsSync(icoPath)) {
      const svg = fs.readFileSync(icoPath);
      res.writeHead(200, { 'Content-Type': 'image/svg+xml',
        'Content-Length': svg.length, 'Cache-Control': 'max-age=86400' });
      res.end(svg);
    } else { res.writeHead(404); res.end('not found'); }

  } else if (url === '/' || url === '/index.html') {
    const html = fs.readFileSync(path.join(DASH_DIR, 'index.html'));
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': html.length });
    res.end(html);

  } else {
    res.writeHead(404); res.end('not found');
  }
});

// Bind all interfaces so the dashboard is reachable from other machines on the
// network (and downloadable results work from any computer).
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Dashboard listening on 0.0.0.0:${PORT} (network-accessible)`);
});
