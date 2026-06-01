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

// ── NETWORK throughput (Linux /proc/net/dev sampling) ───────────────────────
// Sums rx/tx bytes across real interfaces (skips loopback) and diffs against the
// previous sample to get bytes/sec. Returns 0s on non-Linux or if unreadable.
let _lastNet = null;
function getNet() {
  try {
    const raw = fs.readFileSync('/proc/net/dev', 'utf8');
    let rx = 0, tx = 0;
    for (const line of raw.split('\n')) {
      const m = line.match(/^\s*([^:]+):\s*(.*)$/);
      if (!m) continue;
      const iface = m[1].trim();
      if (iface === 'lo' || iface.startsWith('docker') || iface.startsWith('br-')
          || iface.startsWith('veth')) continue;
      const cols = m[2].trim().split(/\s+/).map(Number);
      // /proc/net/dev cols: rx_bytes(0) ... tx_bytes(8)
      if (cols.length >= 9) { rx += cols[0] || 0; tx += cols[8] || 0; }
    }
    const now = Date.now();
    let rxBps = 0, txBps = 0;
    if (_lastNet) {
      const dt = (now - _lastNet.t) / 1000;
      if (dt > 0) {
        rxBps = Math.max(0, (rx - _lastNet.rx) / dt);
        txBps = Math.max(0, (tx - _lastNet.tx) / dt);
      }
    }
    _lastNet = { rx, tx, t: now };
    return { net_rx_bps: Math.round(rxBps), net_tx_bps: Math.round(txBps),
             net_rx_total: rx, net_tx_total: tx };
  } catch {
    return { net_rx_bps: 0, net_tx_bps: 0, net_rx_total: 0, net_tx_total: 0 };
  }
}

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
    ...getNet(),
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
// 5 real stages, in pipeline order:
//   concept    — concept image (Z-Image/FLUX/RealVis) or using an uploaded one
//   bg_removal — rembg cutting background to a clean alpha
//   mesh       — Hunyuan3D 2.1 / 2mv shape generation (load → diffuse → decode)
//   finish     — sharp finish (pymeshfix) / gauntlet / Blender cleanup + detail
//   validate   — slicer validation (Bambu/Prusa) → STL
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
  // Process list gives the gross signal; the run-log tail gives the substage.
  const hasHy     = /_hunyuan(21|2mv)_infer\.py|hy3d_infer\.py/i.test(cmdLines);
  const hasTrellis= /_trellis_infer\.py/i.test(cmdLines);
  const hasHi3d   = /_hi3dgen_infer\.py/i.test(cmdLines);
  const hasTripo  = /_triposg_infer\.py/i.test(cmdLines);
  const hasCraft  = /_craftsman_infer\.py/i.test(cmdLines);
  const hasConcept= /_concept_infer\.py/i.test(cmdLines);
  const hasFinish = /finish_mini\.py|mesh_gauntlet\.py|detail_enhance\.py/i.test(cmdLines);
  const hasValid  = /slicer_validate\.py/i.test(cmdLines);
  const hasOrch   = /orchestrate\.py/i.test(cmdLines);
  const hasNeural = hasHy || hasTrellis || hasHi3d || hasTripo || hasCraft;
  if (!hasOrch && !hasNeural && !hasConcept && !hasFinish && !hasValid) {
    return { running: false, stage: null, substep: '' };
  }
  const lines = (logTail || '').split(/\r?\n/).filter(Boolean);
  const last  = lines.slice(-12).join('\n').toLowerCase();
  // Live progress % from the most recent tqdm bar (e.g. "Sampling ... 50%|").
  const pctM = last.match(/(\d{1,3})%\s*\|/g);
  const pct  = pctM ? pctM[pctM.length - 1].match(/(\d{1,3})%/)[1] + '%' : '';

  // Order matters — latest stage wins.
  if (last.includes('=== done')) {
    return { running: false, stage: 'done', substep: 'complete' };
  }
  // 5) validate
  if (hasValid || /\[4b?\/4\]|\[3\/3\]|slicer validation|slicer_validate/.test(last)) {
    return { running: true, stage: 'validate',
             substep: 'slicing to verify it prints (Bambu / PrusaSlicer)' };
  }
  // 4) finish — sharp finish / gauntlet / blender cleanup / detail enhance
  if (hasFinish ||
      /\[3b?\/4\]|\[2b\/4\]|\[2\/3\]|sharp (mini )?finish|finish_mini|pymeshfix|gauntlet|blender cleanup|detail enhanc|pymeshlab/.test(last)) {
    let sub = 'repairing + sealing the mesh';
    if (/pymeshfix|sharp/.test(last))        sub = 'pymeshfix repair (lossless — keeps detail)';
    else if (/gauntlet|voxel/.test(last))    sub = 'mesh gauntlet (repair + orient + seat)';
    else if (/detail|unsharp|pymeshlab/.test(last)) sub = 'PyMeshLab detail enhancement';
    else if (/blender/.test(last))           sub = 'Blender cleanup + STL export';
    else if (/orient|seat|tweaker/.test(last)) sub = 'auto-orient + seat to bed';
    return { running: true, stage: 'finish', substep: sub };
  }
  // 3) mesh — neural shape generation (engine-aware). Latest matching line wins.
  const meshHit = hasNeural ||
    /\[2\/4\] mesh|\[1\/3\] multi-view|\[(hunyuan2?1?mv?|trellis|hi3dgen|triposg|craftsman)\]|sampling (sparse structure|shape slat|texture slat)|estimating surface normals|shape (diffusion|pipeline)|volume decoding|loading (shape|geometry)? ?pipeline|loading pipeline|flashvdm|full gpu load|octree|o-?voxel|raw mesh|decimat/i.test(last);
  if (meshHit) {
    const p = pct ? ' · ' + pct : '';
    const pn = pct ? (parseInt(pct, 10) || 0) / 100 : 0;  // 0..1 within current bar
    // meshFrac = REAL 0..1 progress through the mesh stage (drives the bar).
    // For 3-stage cascades, sampling spans 0.10..0.80, remesh→0.88, save→0.97.
    const samp3 = (i) => 0.10 + 0.70 * ((i + pn) / 3);    // i = 0,1,2 (sub-stage)
    let sub, meshFrac = null;
    // ---- TRELLIS.2 (4B O-Voxel) ----
    if (hasTrellis || /\[trellis\]|trellis\.2|sampling (sparse structure|shape slat|texture slat)/i.test(last)) {
      sub = 'TRELLIS.2 — starting'; meshFrac = 0.03;
      if (/loading pipeline|downloading|\.safetensors|resolve\/main|dinov|encoder|rmbg|birefnet/i.test(last)) {
        sub = 'TRELLIS.2 — loading / downloading 4B model + encoders (first run is slow)'; meshFrac = 0.06; }
      else if (/sampling sparse structure/i.test(last)) { sub = 'TRELLIS.2 — sampling sparse structure (1/3)' + p; meshFrac = samp3(0); }
      else if (/sampling shape slat/i.test(last))       { sub = 'TRELLIS.2 — sampling shape (2/3)' + p; meshFrac = samp3(1); }
      else if (/sampling texture slat/i.test(last))     { sub = 'TRELLIS.2 — sampling texture (3/3)' + p; meshFrac = samp3(2); }
      else if (/raw mesh|decimat|remesh|geometry remesh/i.test(last)) { sub = 'TRELLIS.2 — remesh + clean mesh (GPU)'; meshFrac = 0.90; }
      else if (/cutlass|forced xformers/i.test(last))   { sub = 'TRELLIS.2 — preparing attention (Blackwell-safe)'; meshFrac = 0.07; }
      else if (/saved/i.test(last))                     { sub = 'TRELLIS.2 — saving mesh'; meshFrac = 0.97; }
    }
    // ---- Hi3DGen (normal-bridging) ----
    else if (hasHi3d || /\[hi3dgen\]/i.test(last)) {
      sub = 'Hi3DGen — starting'; meshFrac = 0.03;
      if (/loading geometry pipeline|downloading|resolve\/main|dinov/i.test(last)) {
        sub = 'Hi3DGen — loading / downloading models (first run is slow)'; meshFrac = 0.06; }
      else if (/stablenormal|estimating surface normals|normal predictor/i.test(last)) { sub = 'Hi3DGen — estimating surface normals'; meshFrac = 0.20; }
      else if (/sampling/i.test(last))  { sub = 'Hi3DGen — normal → geometry diffusion' + p; meshFrac = 0.30 + 0.60 * pn; }
      else if (/saved/i.test(last))     { sub = 'Hi3DGen — saving mesh'; meshFrac = 0.95; }
    }
    // ---- TripoSG (watertight SDF) ----
    else if (hasTripo || /\[triposg\]/i.test(last)) {
      sub = 'TripoSG — diffusion → watertight mesh' + p; meshFrac = 0.15 + 0.75 * pn;
      if (/downloading|loading/i.test(last)) { sub = 'TripoSG — loading model'; meshFrac = 0.06; }
      else if (/saved/i.test(last))          { sub = 'TripoSG — saving mesh'; meshFrac = 0.95; }
    }
    // ---- CraftsMan3D ----
    else if (hasCraft || /\[craftsman\]/i.test(last)) {
      sub = 'CraftsMan3D — coarse 3D + normal refiner' + p; meshFrac = 0.15 + 0.75 * pn;
      if (/downloading|loading/i.test(last)) { sub = 'CraftsMan3D — loading model'; meshFrac = 0.06; }
    }
    // ---- Hunyuan (single / multi-view) ----
    else {
      sub = 'Hunyuan 3D shape generation';
      if (/loading shape pipeline|cpu first|loading.*weights|downloading/.test(last))
        sub = 'loading / downloading Hunyuan model (first run is slow)';
      else if (/full gpu load/.test(last))  sub = 'loading model onto the GPU';
      else if (/flashvdm/.test(last))       sub = 'FlashVDM enabled';
      else if (/preparing image|background/.test(last)) sub = 'preparing input image';
      else if (/shape diffusion|sampling|steps=/.test(last)) sub = 'diffusion sampling (shape)' + p;
      else if (/volume decoding|octree/.test(last)) sub = 'volume decode → mesh (octree 768)';
      else if (/saved/.test(last))          sub = 'saving mesh';
    }
    return { running: true, stage: 'mesh', substep: sub, mesh_frac: meshFrac };
  }
  // 2) bg removal
  if (/background removed|removing background|rembg|\[1b\/4\]/.test(last)) {
    return { running: true, stage: 'bg_removal', substep: 'background removal (rembg / u2net)' };
  }
  // 1) concept
  if (hasConcept || /\[1\/4\]|concept image|concept_gen|user-provided/.test(last)) {
    let sub = 'generating concept image';
    if (/z[_-]?image/.test(last))      sub = 'Z-Image-Turbo concept (8 steps)';
    else if (/flux/.test(last))        sub = 'FLUX concept';
    else if (/realvis/.test(last))     sub = 'RealVisXL concept';
    else if (/dreamshaper/.test(last)) sub = 'DreamShaper XL Turbo concept';
    else if (/user-provided|using user/.test(last)) sub = 'using uploaded image';
    return { running: true, stage: 'concept', substep: sub };
  }
  // Fallback — pin to the most likely current stage from the process list.
  return { running: true,
           stage: hasFinish ? 'finish' : hasHy ? 'mesh' : 'concept',
           substep: 'working…' };
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
// Per-stage estimated duration in seconds (RTX 5090 / 32 GB, octree 768).
// 'mesh' dominates: model load + diffusion (75 steps) + volume decode.
const STEP_EST = {
  concept:    45,    // concept image (instant if uploaded)
  bg_removal: 6,     // rembg/u2net
  mesh:       360,   // HY3D load + diffuse + octree-768 decode (first run +download)
  finish:     120,   // pymeshfix / gauntlet / cleanup + detail
  validate:   60,    // Bambu + PrusaSlicer slice check
};
const STAGE_ORDER = ['concept', 'bg_removal', 'mesh', 'finish', 'validate'];

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
  let   stepPct     = Math.min(99, Math.round(stepElapsed / stepEst * 100));

  const totalEst   = Object.values(STEP_EST).reduce((a, b) => a + b, 0);
  let   totalPct   = Math.min(99, Math.round(totalElapsed / totalEst * 100));
  const etaSec     = Math.max(0, totalEst - totalElapsed);

  // REAL progress override: during the mesh stage TRELLIS/Hi3DGen/TripoSG report
  // an actual fraction (parsed from the live tqdm bars). Use it to drive the bar
  // instead of the time-estimate, so the % tracks what's really happening.
  const mf = _stageCache && typeof _stageCache.mesh_frac === 'number'
    ? _stageCache.mesh_frac : null;
  if (stage === 'mesh' && mf !== null) {
    stepPct = Math.max(0, Math.min(99, Math.round(mf * 100)));
    // Map mesh-stage fraction onto the overall bar: stages before mesh occupy a
    // small slice (uploaded concept is instant), mesh ~10..72%, finish+validate
    // fill the rest (those are fast for these engines).
    totalPct = Math.max(8, Math.min(74, Math.round(8 + 66 * mf)));
  } else if (stage === 'finish') {
    // Mesh handed off at ~74%; keep the bar monotonic through the fast finish.
    totalPct = Math.max(totalPct, 76 + Math.min(13, Math.round(stepElapsed / STEP_EST.finish * 13)));
  } else if (stage === 'validate') {
    totalPct = Math.max(totalPct, 90 + Math.min(9, Math.round(stepElapsed / STEP_EST.validate * 9)));
  }

  // Substep label — prefer the live log-derived one (set by _refreshStage)
  let substep = (_stageCache && _stageCache.substep) || '';
  if (!substep) {
    if (stage === 'concept')         substep = 'generating concept image…';
    else if (stage === 'bg_removal') substep = 'removing background (rembg/u2net)';
    else if (stage === 'mesh') {
      if (stepElapsed < 120)       substep = 'loading / downloading Hunyuan model…';
      else if (stepElapsed < 150)  substep = 'CUDA warmup — first step only';
      else {
        const inferPct = (stepElapsed - 150) / Math.max(1, stepEst - 150);
        const estStep  = Math.min(75, Math.max(2, Math.round(inferPct * 73) + 2));
        substep = `diffusion sampling ~${estStep} / 75 steps`;
      }
    }
    else if (stage === 'finish')    substep = 'mesh repair + finish…';
    else if (stage === 'validate')  substep = 'slicer validation…';
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

// Force-kill: stop the pipeline AND delete the run's data (concept image + its
// views, the mesh/gauntlet GLBs + report sidecars, and the run log). Used to
// abort a bad run and wipe its partial artifacts in one click. Scoped to the
// given run_id (validated) so it can only touch that run's files.
function forceKill(opts) {
  killPipeline();
  const removed = [];
  // Always drop the active log.
  try {
    if (_activeLog && fs.existsSync(_activeLog)) {
      fs.unlinkSync(_activeLog); removed.push(path.basename(_activeLog)); _activeLog = null;
    }
  } catch (e) {}
  const runId = opts && opts.run_id ? String(opts.run_id) : '';
  if (/^\d{8}_\d{6}_.+_[0-9a-f]{6}$/.test(runId)) {
    // concepts: {runId}.png + {runId}_<view>.png
    try {
      for (const f of fs.readdirSync(CONCEPTS)) {
        if (f === runId + '.png' || f.startsWith(runId + '_')) {
          fs.unlinkSync(path.join(CONCEPTS, f)); removed.push(f);
        }
      }
    } catch (e) {}
    // meshes: {runId}.glb, {runId}.gauntlet.glb + .cleanup/.validate json sidecars
    try {
      for (const f of fs.readdirSync(MESHES)) {
        if (f.startsWith(runId + '.') || f.startsWith(runId + '_')) {
          fs.unlinkSync(path.join(MESHES, f)); removed.push(f);
        }
      }
    } catch (e) {}
    // logs that reference this run
    try {
      for (const f of fs.readdirSync(LOG_DIR)) {
        if (f.includes(runId.split('_').slice(2).join('_'))) {
          try { fs.unlinkSync(path.join(LOG_DIR, f)); removed.push(f); } catch (e) {}
        }
      }
    } catch (e) {}
  }
  return { ok: true, removed, count: removed.length };
}

function killPipeline() {
  if (process.platform === 'win32') {
    cp.exec(
      'powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"name=\'python.exe\'\\\" | Where-Object {$_.CommandLine -match \'orchestrate|hy3d_infer\'} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force}"',
      { timeout: 8000 }, () => {}
    );
  } else {
    // Linux workstation: SIGKILL the whole pipeline subprocess tree. -9 because
    // finish-step processes sit in C extensions (pymeshfix/pymeshlab/blender)
    // that ignore SIGTERM for long stretches, so a soft kill "does nothing".
    cp.exec(
      "pkill -9 -f orchestrate.py; pkill -9 -f _hunyuan; pkill -9 -f hy3d_infer; " +
      "pkill -9 -f _trellis_infer; pkill -9 -f _hi3dgen_infer; " +
      "pkill -9 -f _triposg_infer; pkill -9 -f _craftsman_infer; " +
      "pkill -9 -f mesh_gauntlet; pkill -9 -f finish_mini; " +
      "pkill -9 -f blender_cleanup; pkill -9 -f blender; " +
      "pkill -9 -f slicer_validate; pkill -9 -f detail_enhance; " +
      "pkill -9 -f pymeshfix; pkill -9 -f pymeshlab",
      { timeout: 8000 }, () => {}
    );
  }
  return { ok: true };
}

// ── ENGINE HEALTH / SELF-TEST ─────────────────────────────────────────────────
// Runs pipeline/engine_selftest.py (imports every engine's stack in subprocesses)
// so the dashboard can show a real "will it run?" status screen — no surprises
// on a live run. The probe is slow (loads torch), so it runs in the BACKGROUND:
// GET /api/health returns the cached result + a `running` flag; the frontend
// kicks a refresh with ?run=1 and polls until `ts` changes.
let _health = { data: null, running: false, startedAt: 0, finishedAt: 0 };
function startHealth(force) {
  if (_health.running) return;
  // Debounce: don't re-run within 20s unless forced.
  if (!force && _health.finishedAt && Date.now() - _health.finishedAt < 20000) return;
  _health.running = true;
  _health.startedAt = Date.now();
  const py = process.env.GEN3D_PY || 'python';
  const script = path.join(TABLETOP, 'pipeline', 'engine_selftest.py');
  cp.execFile(py, ['-u', script],
    { timeout: 300000, maxBuffer: 8 * 1024 * 1024,
      env: { ...process.env, PYTHONUTF8: '1' } },
    (err, stdout, stderr) => {
      let data;
      try { data = JSON.parse(stdout); }
      catch {
        data = { error: 'self-test did not return JSON',
                 raw: (String(stdout) + '\n' + String(stderr || '') +
                       (err ? '\n' + err.message : '')).slice(-900),
                 engines: [], core: {} };
      }
      data.ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
      data.elapsed_s = Math.round((Date.now() - _health.startedAt) / 1000);
      _health.data = data;
      _health.running = false;
      _health.finishedAt = Date.now();
    });
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
      else if (url === '/api/forcekill') result = forceKill(body);
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

  } else if (url === '/api/health') {
    // ?run=1 forces a fresh self-test; otherwise kick one off lazily if we've
    // never run it. Always respond immediately with whatever we have.
    const wantRun = /[?&]run=1\b/.test(req.url);
    if (wantRun) startHealth(true);
    else if (!_health.data && !_health.running) startHealth(false);
    const payload = JSON.stringify({
      running: _health.running,
      started_at: _health.startedAt ? new Date(_health.startedAt).toISOString() : null,
      report: _health.data,
    });
    res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) });
    res.end(payload);

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
