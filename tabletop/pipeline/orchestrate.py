"""orchestrate - top-level pipeline driver.

Turns a prompt into a printable STL by chaining:
    concept_image (ComfyUI/FLUX) -> mesh (Sparc3D) -> cleanup (Blender) -> STL

Usage:
    python orchestrate.py --prompt "stone watchtower" --kind terrain \\
        --out outputs/stl/watchtower.stl

    python orchestrate.py --check-models

The other modules in this folder (comfy_client.py, sparc3d_client.py,
hunyuan_client.py, blender_cleanup.py, slicer_prep.py) are the per-stage
implementations. This file glues them together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent

# Single shared interpreter for all spawned pipeline subprocesses. On the Linux
# workstation every engine lives in one env, so this is just the running python
# (override with GEN3D_PY if you split envs). Windows multi-venv layout is retired.
PY = os.environ.get("GEN3D_PY") or sys.executable

# Output dirs
OUT_CONCEPTS = TABLETOP_ROOT / "outputs" / "concepts"
OUT_MESHES   = TABLETOP_ROOT / "outputs" / "meshes"
OUT_STL      = TABLETOP_ROOT / "outputs" / "stl"
OUT_PREVIEWS = TABLETOP_ROOT / "outputs" / "previews"
for d in (OUT_CONCEPTS, OUT_MESHES, OUT_STL, OUT_PREVIEWS):
    d.mkdir(parents=True, exist_ok=True)

# Engine registry — extend by importing more clients here
#   parametric -> build123d CAD (terrain: watertight by construction, no AI)
#   hunyuan21  -> Hunyuan3D 2.1 (best neural — minis / organic)
#   triposg/pixal3d/hunyuan/sparc3d -> other neural engines
ENGINES = ("sparc3d", "hunyuan", "hunyuan21", "hunyuan2mv", "pixal3d",
           "triposg", "parametric", "auto", "both")

# Structural keywords that route engine=auto -> parametric CAD
PARAMETRIC_KEYWORDS = ("tower", "wall", "crate", "box", "tile", "floor",
                       "pillar", "column", "keep", "rampart", "battlement")

# Workflow auto-selection: prefer FLUX when weights exist, fall back to SDXL.
def _best_workflow(kind: str) -> str:
    models_dir = TABLETOP_ROOT / "models"
    has_flux = (models_dir / "flux" / "flux1-dev.safetensors").exists()
    prefix = "flux" if has_flux else "sdxl"
    base = "mini" if kind in ("mini", "prop", "scatter") else "terrain"
    return f"{prefix}_{base}_concept.json"

# Minis/props are output at STATUE scale so fine grooves are physically deep
# (a 0.1mm groove on a 32mm mini is 0.5mm on a 160mm statue) — better detail
# and large-format printable. Downsize in the slicer for a tabletop mini.
KIND_DEFAULTS = {
    "mini":     {"scale_mm": 160, "workflow": None},   # statue scale
    "terrain":  {"scale_mm": 100, "workflow": None},
    "prop":     {"scale_mm": 120, "workflow": None},   # statue scale
    "scatter":  {"scale_mm": 80,  "workflow": None},
}


def _new_run_id(prompt: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in prompt.lower())[:40]
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}_{uuid.uuid4().hex[:6]}"


def check_models() -> dict:
    """Report which engines + LoRAs are detected."""
    models_dir = TABLETOP_ROOT / "models"
    report = {
        "models_root": str(models_dir),
        "engines": {},
        "loras": [],
    }
    # FLUX
    flux_main = models_dir / "flux" / "flux1-dev.safetensors"
    report["engines"]["flux"] = {
        "ready": flux_main.exists(),
        "path": str(flux_main),
    }
    # SDXL
    sdxl_main = models_dir / "sdxl" / "sd_xl_base_1.0.safetensors"
    report["engines"]["sdxl"] = {
        "ready": sdxl_main.exists(),
        "path": str(sdxl_main),
    }
    # Sparc3D
    sparc_dir = models_dir / "sparc3d"
    report["engines"]["sparc3d"] = {
        "ready": sparc_dir.exists() and any(sparc_dir.iterdir()),
        "path": str(sparc_dir),
    }
    # Hunyuan3D
    hy_dir = models_dir / "hunyuan3d"
    report["engines"]["hunyuan3d"] = {
        "ready": hy_dir.exists() and any(hy_dir.iterdir()),
        "path": str(hy_dir),
    }
    # Pixal3D
    px_dir = models_dir / "pixal3d"
    report["engines"]["pixal3d"] = {
        "ready": px_dir.exists() and (px_dir / "pipeline.json").exists(),
        "path": str(px_dir),
    }
    # Hunyuan3D 2.1 (the NEW winner — Tencent, most production-mature)
    hy21_dir = TABLETOP_ROOT / "Hunyuan3D-2.1"
    report["engines"]["hunyuan21"] = {
        "ready": hy21_dir.exists()
                 and (hy21_dir / "hy3dshape" / "hy3dshape" / "pipelines.py").exists(),
        "path": str(hy21_dir),
    }
    # Parametric CAD (build123d) — terrain, watertight by construction
    terrain_lib = HERE / "parametric" / "terrain_lib.py"
    report["engines"]["parametric"] = {
        "ready": terrain_lib.exists(),
        "path": str(terrain_lib),
        "templates": list(PARAMETRIC_KEYWORDS),
    }
    # TripoSG (SDF + DMC -> watertight by construction)
    tg_dir = TABLETOP_ROOT / "TripoSG"
    report["engines"]["triposg"] = {
        "ready": tg_dir.exists()
                 and (tg_dir / "triposg" / "inference_utils.py").exists(),
        "path": str(tg_dir),
    }
    # LoRAs
    lora_dir = models_dir / "loras"
    if lora_dir.exists():
        for p in lora_dir.rglob("*.safetensors"):
            report["loras"].append(str(p.relative_to(models_dir)))
    return report


def run(prompt: str, kind: str = "mini", engine: str = "sparc3d",
        printer: str = "resin", scale_mm: float | None = None,
        seed: int | None = None, skip_print_prep: bool = False,
        out_stl: str | None = None,
        input_image: str | None = None,
        mv_front: str | None = None, mv_left: str | None = None,
        mv_back: str | None = None, mv_gif: str | None = None,
        mv_reverse: bool = False) -> dict:
    """Run the full pipeline. Returns a dict of all output paths."""
    if kind not in KIND_DEFAULTS:
        raise ValueError(f"unknown kind: {kind}")
    if engine not in ENGINES:
        raise ValueError(f"unknown engine: {engine}")

    defaults = KIND_DEFAULTS[kind]
    scale_mm = scale_mm or defaults["scale_mm"]
    workflow = _best_workflow(kind)

    # ============================================================
    # MULTI-VIEW TRACK — Hunyuan3D-2mv fuses front/left/back (or a turntable
    # GIF) into one mesh, then the SHARP mini finish. No concept gen.
    # ============================================================
    if engine == "hunyuan2mv":
        import subprocess, json as _json
        run_id = _new_run_id(prompt or "multiview")
        print(f"=== tabletop pipeline run: {run_id} (multi-view) ===")
        mesh_glb = OUT_MESHES / f"{run_id}.glb"
        import hunyuan2mv_engine
        print("  [1/3] multi-view mesh (Hunyuan3D-2mv, front/left/back)...")
        hunyuan2mv_engine.generate(
            out_path=str(mesh_glb),
            front=mv_front, left=mv_left, back=mv_back,
            gif=mv_gif, reverse=mv_reverse,
            seed=seed or 42, steps=30, octree_resolution=512)
        stl_path = Path(out_stl) if out_stl else (OUT_STL / f"{run_id}.stl")
        print("  [2/3] sharp finish (pymeshfix, no resample)...")
        fproc = subprocess.run(
            [PY, str(HERE / "finish_mini.py"),
             "--input", str(mesh_glb), "--output", str(stl_path),
             "--scale-mm", str(scale_mm)],
            capture_output=True, text=True, timeout=600)
        if fproc.returncode != 0:
            raise RuntimeError(f"finish_mini failed:\n{fproc.stderr[-1500:]}")
        print("        " + (fproc.stdout.strip().splitlines() or ["done"])[-1])
        validated = None
        try:
            vproc = subprocess.run(
                [PY, str(HERE / "slicer_validate.py"),
                 "--input", str(stl_path), "--printer", printer],
                capture_output=True, text=True, timeout=400)
            vfile = stl_path.with_suffix(".validate.json")
            if vfile.exists():
                validated = bool(_json.loads(
                    vfile.read_text(encoding="utf-8")).get("PASS"))
            print(f"  [3/3] slicer validation: {'PASS' if validated else 'FAIL'}")
        except Exception as e:
            print(f"  [3/3] validation skipped: {e}")
        result = {"run_id": run_id, "engine": "hunyuan2mv", "kind": kind,
                  "mesh_glb": str(mesh_glb), "stl": str(stl_path),
                  "validated": validated}
        print(f"=== done: {run_id} ===")
        return result

    # engine=auto: route to parametric CAD if the prompt names a structural
    # terrain piece (watertight by construction), else fall back to the best
    # neural engine (hunyuan21).
    if engine == "auto":
        pl = (prompt or "").lower()
        if any(kw in pl for kw in PARAMETRIC_KEYWORDS):
            engine = "parametric"
        else:
            engine = "hunyuan21"

    run_id = _new_run_id(prompt)
    print(f"=== tabletop pipeline run: {run_id} ===")
    print(f"  prompt:  {prompt}")
    print(f"  kind:    {kind}  scale_mm={scale_mm}")
    print(f"  engine:  {engine}  printer={printer}")

    # ============================================================
    # PARAMETRIC TRACK — text -> build123d CAD -> watertight STL.
    # No concept image, no multi-view, no gauntlet. Slicer-validated.
    # ============================================================
    if engine == "parametric":
        import subprocess, json as _json
        terrain_lib = HERE / "parametric" / "terrain_lib.py"
        stl_path = Path(out_stl) if out_stl else (OUT_STL / f"{run_id}.stl")
        print("  [1/2] parametric CAD generation (build123d)...")
        proc = subprocess.run(
            [PY, str(terrain_lib),
             "--prompt", prompt, "--output", str(stl_path),
             "--scale-mm", str(scale_mm)],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"parametric CAD failed (exit {proc.returncode}).\n"
                f"No structural template matched '{prompt}'? "
                f"Try engine=hunyuan21 for organic shapes.\n"
                f"stderr:\n{proc.stderr[-1500:]}")
        print(f"        -> {stl_path.name}")
        # Slicer validation gate
        val = {"PASS": None}
        try:
            vproc = subprocess.run(
                [PY, str(HERE / "slicer_validate.py"),
                 "--input", str(stl_path), "--printer", printer],
                capture_output=True, text=True, timeout=300)
            vfile = stl_path.with_suffix(".validate.json")
            if vfile.exists():
                val = _json.loads(vfile.read_text(encoding="utf-8"))
            print(f"  [2/2] slicer validation: "
                  f"{'PASS' if val.get('PASS') else 'FAIL'}")
        except Exception as e:
            print(f"  [2/2] validation skipped: {e}")
        result = {
            "run_id": run_id, "engine": "parametric",
            "stl": str(stl_path),
            "validated": bool(val.get("PASS")),
            "template": None,
        }
        print(f"=== done: {run_id} ===")
        return result

    # ---- 1. Concept image ----
    # If --input-image is provided, copy it in as the concept and skip
    # generation entirely. Lets the user feed in a photo or a reference
    # image and start the pipeline from the multi-view stage.
    concept_png = OUT_CONCEPTS / f"{run_id}.png"
    used_model  = None
    if input_image and Path(input_image).exists():
        import shutil
        shutil.copy(input_image, concept_png)
        used_model = "user-provided"
        print(f"  [1/4] using user-provided image -> {concept_png.name}")
        # Still run background removal on the user image — they might have a
        # scenic background that HY3D would struggle with.
        try:
            import concept_gen
            concept_gen._remove_bg(str(concept_png))
        except Exception as e:
            print(f"        warn: bg removal failed ({e}); using as-is")
    else:
      try:
        import concept_gen
        if concept_gen.pick_model(kind) is not None:
            print("  [1/4] concept image (concept_gen)...")
            used_model = concept_gen.generate(
                prompt=prompt, out_path=str(concept_png),
                kind=kind, seed=seed,
            )
            print(f"        -> {concept_png.name}  (model: {used_model})")
      except Exception as e:
        print(f"  [1/4] concept_gen failed ({e}); falling back to ComfyUI")
        used_model = None
    if used_model is None:
        import comfy_client
        print("  [1/4] concept image (ComfyUI/SDXL fallback)...")
        comfy_client.generate(
            prompt=prompt, workflow=workflow, kind=kind,
            seed=seed, out_path=str(concept_png),
        )
        print(f"        -> {concept_png.name}")
        # Free ComfyUI VRAM before HY3D loads (SDXL 6.5 GB + HY3D 7 GB > 16 GB)
        comfy_client.free_vram()

    # ---- 1c. Multi-view expansion (engine selectable) ----
    # Pixal3D does direct image→3D with pixel-aligned projection, so it
    # never needs the multi-view stage. Skip it for that engine.
    # MV_ENGINE env var picks which multi-view generator to use:
    #   zero123   (default) — Zero123++ v1.2, 6 views at ±20°/-10° elev
    #   mvadapter            — MV-Adapter, more consistent on complex scenes
    #   single               — skip MV entirely, HY3D 2.1 single-view mode
    mv_paths: dict[str, str] | None = None
    mv_engine = os.environ.get("MV_ENGINE", "zero123").lower()
    if engine in ("pixal3d", "triposg", "hunyuan21"):
        print(f"  [1b/4] engine={engine} - skipping multi-view (direct image-to-3D)")
    elif mv_engine == "single":
        print("  [1b/4] MV_ENGINE=single — skipping multi-view expansion")
    else:
        z123_dir   = TABLETOP_ROOT / "models" / "zero123plus"
        mvadapt_dir= TABLETOP_ROOT / "models" / "mvadapter"
        try:
            import mv_generator
            # Expose the prompt to mv_generator (view-fix pass uses it)
            os.environ["RUN_PROMPT"] = prompt or ""
            if mv_engine == "mvadapter" and mvadapt_dir.exists():
                print("  [1b/4] MV-Adapter multi-view expansion...")
                mv_paths = mv_generator.generate(
                    image_path=str(concept_png), out_dir=str(OUT_CONCEPTS),
                    run_id=run_id, seed=seed, engine="mvadapter")
            elif z123_dir.exists() and any(z123_dir.glob("*.json")):
                print("  [1b/4] Zero123++ multi-view expansion...")
                mv_paths = mv_generator.generate(
                    image_path=str(concept_png), out_dir=str(OUT_CONCEPTS),
                    run_id=run_id, seed=seed)
            else:
                print("  [1b/4] no mv model available; using single-view")
            if mv_paths:
                print(f"        -> {len(mv_paths)} views: {list(mv_paths)}")
        except Exception as e:
            print(f"  [1b/4] WARNING: multi-view failed ({e}); falling back to single-view")
            mv_paths = None

    # ---- 2. Mesh via Sparc3D (printable) and/or Hunyuan3D (preview) ----
    mesh_glb = OUT_MESHES / f"{run_id}.glb"
    preview_png = OUT_PREVIEWS / f"{run_id}.png" if engine in ("hunyuan", "both") else None
    if engine == "pixal3d":
        import pixal3d_engine
        print("  [2/4] mesh generation (Pixal3D, direct image-to-3D)...")
        pixal3d_engine.generate(
            image_path=str(concept_png),
            out_path=str(mesh_glb),
            seed=seed or 42,
            low_vram=True,        # 17 GB models thrash on 16 GB VRAM — low_vram is 6-10x faster here
            manual_fov=0.2,       # skips MoGe
            resolution=1024,      # 1536 also works but slower
        )
        print(f"        -> {mesh_glb.name}")
    if engine == "triposg":
        import triposg_engine
        print("  [2/4] mesh generation (TripoSG, SDF -> watertight DMC)...")
        triposg_engine.generate(
            image_path=str(concept_png),
            out_path=str(mesh_glb),
            seed=seed or 42,
            steps=50,
            faces=400_000,        # decimate from ~2-3M to a slicer-friendly count
        )
        print(f"        -> {mesh_glb.name}")
    if engine == "hunyuan21":
        import hunyuan21_engine
        print("  [2/4] mesh generation (Hunyuan3D 2.1, Tencent, shape only)...")
        hunyuan21_engine.generate(
            image_path=str(concept_png),
            out_path=str(mesh_glb),
            seed=seed or 42,
            steps=int(os.environ.get("HY_STEPS", "75")),     # diffusion refinement
            octree_resolution=int(os.environ.get("HY_OCTREE", "512")),  # 5090/32GB: 512 (~1.8x faces of 384)
        )
        print(f"        -> {mesh_glb.name}")
    if engine in ("sparc3d", "both"):
        import sparc3d_client
        print("  [2/4] mesh generation (Sparc3D, printable)...")
        sparc3d_client.generate(
            image_path=str(concept_png),
            out_path=str(mesh_glb),
            target_scale_mm=scale_mm,
        )
        print(f"        -> {mesh_glb.name}")
    if engine in ("hunyuan", "both"):
        import hunyuan_client
        if engine == "both":
            # Sparc3D already produced mesh_glb; Hunyuan adds a textured preview.
            hy_out = str(preview_png) if preview_png else None
            hy_mesh_only = False
            label = f"preview (Hunyuan3D, textured)"
        else:
            # Hunyuan is the sole 3D engine — output the printable GLB.
            hy_out = str(mesh_glb)
            hy_mesh_only = True
            label = "mesh generation (Hunyuan3D 2.1)"
        print(f"  [2/4] {label}...")
        hy_kwargs = dict(
            image_path=str(concept_png),
            out_path=hy_out,
            mesh_only=hy_mesh_only,
            low_vram=True,
        )
        if mv_paths:
            # multi-view: pass the side views; the 'front' slot is already
            # the original concept image (mv_paths["front"] === concept_png)
            hy_kwargs["image_left"]  = mv_paths.get("left")
            hy_kwargs["image_back"]  = mv_paths.get("back")
            hy_kwargs["image_right"] = mv_paths.get("right")
        hunyuan_client.generate(**hy_kwargs)
        out_name = Path(hy_out).name if hy_out else "(none)"
        print(f"        -> {out_name}")

    # ---- 3. Blender cleanup -> STL ----
    if skip_print_prep:
        print("  [3/4] skip-print-prep set; pipeline done at GLB.")
        result = {
            "run_id": run_id,
            "concept_png": str(concept_png),
            "mesh_glb": str(mesh_glb) if engine != "hunyuan" else None,
            "preview_png": str(preview_png) if preview_png else None,
        }
        if engine == "pixal3d":
            result["mesh_glb"] = str(mesh_glb)
    else:
        stl_path = Path(out_stl) if out_stl else (OUT_STL / f"{run_id}.stl")

        # ============================================================
        # SHARP MINI PATH (hunyuan21 + kind=mini): keep full octree-256
        # detail. pymeshfix repairs topology WITHOUT resampling (the voxel
        # heal smooths fine detail away), then drop tiny floaters, orient,
        # scale, seat, export. Skips the terrain-tuned blender_cleanup that
        # fragments minis. Think: high-detail statue scaled down, details kept.
        # ============================================================
        if engine == "hunyuan21" and kind == "mini" and mesh_glb.exists():
            import subprocess, json as _json
            print("  [3/4] sharp mini finish (pymeshfix, no resample)...")
            fproc = subprocess.run(
                [PY, str(HERE / "finish_mini.py"),
                 "--input", str(mesh_glb), "--output", str(stl_path),
                 "--scale-mm", str(scale_mm)],
                capture_output=True, text=True, timeout=600)
            if fproc.returncode != 0:
                raise RuntimeError(f"finish_mini failed:\n{fproc.stderr[-1500:]}")
            print("        " + (fproc.stdout.strip().splitlines() or ["done"])[-1])
            validated = None
            try:
                vproc = subprocess.run(
                    [PY, str(HERE / "slicer_validate.py"),
                     "--input", str(stl_path), "--printer", printer],
                    capture_output=True, text=True, timeout=400)
                vfile = stl_path.with_suffix(".validate.json")
                if vfile.exists():
                    validated = bool(_json.loads(
                        vfile.read_text(encoding="utf-8")).get("PASS"))
                print(f"  [4/4] slicer validation: "
                      f"{'PASS' if validated else 'FAIL'}")
            except Exception as e:
                print(f"  [4/4] validation skipped: {e}")
            result = {
                "run_id": run_id, "engine": engine, "kind": kind,
                "concept_png": str(concept_png),
                "mesh_glb": str(mesh_glb),
                "stl": str(stl_path), "validated": validated,
            }
            print(f"=== done: {run_id} ===")
            return result

        # ---- 2b. mesh gauntlet (post-engine validation + repair + solidify) ----
        # Runs for SDF-based engines (TripoSG, Pixal3D, Hunyuan when applicable).
        # Pipeline: drop floaters -> pymeshfix repair -> wall thickness check ->
        # Tweaker-3 auto-orient -> seat to bed (z=0) -> [if thin shell]
        # Blender voxel-remesh solidify. Output is a single watertight solid
        # ready for direct slicer ingest — no Blender cleanup heroics needed.
        repaired_glb = OUT_MESHES / f"{run_id}.gauntlet.glb"
        if engine in ("triposg", "pixal3d", "hunyuan21", "hunyuan", "both") and mesh_glb.exists():
            try:
                import subprocess
                gauntlet_py = HERE / "mesh_gauntlet.py"
                # Engine-aware gauntlet mode. Hunyuan3D 2.1 produces clean
                # topology (~10 splits) so we skip the destructive
                # voxel-remesh solidify and orient steps — wall thickness
                # ~0.5mm is fine for resin and we want to keep detail.
                # TripoSG / Pixal3D have hundreds of splits and thin shells
                # → need the full gauntlet.
                if engine == "hunyuan21" and kind == "mini":
                    # minis: fine voxel heal (~scale/180) guarantees manifold +
                    # fuses thin appendages (grounded swords, spears) that come
                    # out as detached components, while keeping tabletop detail.
                    # 0.18mm on a 32mm mini + octree-256 generation keeps ~147K
                    # faces (sharper). Was 0.22mm/96K with octree-192.
                    mini_vox = max(0.16, round(scale_mm / 180.0, 2))
                    gauntlet_args = ["--no-orient", "--force-voxel", str(mini_vox)]
                    print(f"  [2b/4] mesh gauntlet (mini-heal voxel {mini_vox}mm)...")
                elif engine == "hunyuan21":
                    gauntlet_args = ["--no-orient", "--no-solidify"]
                    print(f"  [2b/4] mesh gauntlet (clean engine; drop floaters + repair + seat only)...")
                else:
                    gauntlet_args = []
                    print(f"  [2b/4] mesh gauntlet (repair + orient + seat + solidify)...")
                proc = subprocess.run(
                    [PY, str(gauntlet_py),
                     "--input",    str(mesh_glb),
                     "--output",   str(repaired_glb),
                     "--scale-mm", str(scale_mm), *gauntlet_args],
                    capture_output=True, text=True, timeout=1500)
                if proc.returncode == 0 and repaired_glb.exists():
                    mesh_glb = repaired_glb     # use the gauntlet output downstream
                    print(f"        -> {repaired_glb.name} (gauntleted, printable)")
                else:
                    tail = proc.stderr[-400:] if proc.stderr else proc.stdout[-400:]
                    print(f"  [2b/4] gauntlet failed (exit {proc.returncode}); "
                          f"continuing with raw GLB. Tail:\n{tail}")
            except Exception as e:
                print(f"  [2b/4] gauntlet skipped: {e}")
        # The gauntlet already solidified and oriented; tell blender_cleanup
        # to skip its own Solidify rescue and just do final scale+export.
        os.environ.pop("SOLIDIFY_MODE", None)
        import blender_cleanup
        print("  [3/4] Blender cleanup + STL export...")
        blender_cleanup.run(
            input_mesh=str(mesh_glb),
            out_stl=str(stl_path),
            printer=printer,
            scale_mm=scale_mm,
        )
        print(f"        -> {stl_path.name}")

        # ---- 3b. Detail enhancement (PyMeshLab) ----
        # Sharpens HY3D's soft surface via normal-unsharp + edge-preserving
        # smooth. Same topology, ~2s, makes battlements/runes/creases crisp.
        # SKIP for healed minis: the voxel-heal output is already a clean
        # single watertight solid; unsharp on a voxel mesh re-fragments it
        # (v2 came out as 15 components, not bed-seated). The gauntlet result
        # is final for minis.
        if engine == "hunyuan21" and kind == "mini":
            os.environ["SKIP_DETAIL_ENHANCE"] = "1"
        if os.environ.get("SKIP_DETAIL_ENHANCE", "0") != "1":
            try:
                import subprocess
                inner = HERE / "detail_enhance.py"
                proc = subprocess.run(
                    [PY, str(inner), "--input", str(stl_path)],
                    capture_output=True, text=True, timeout=120)
                if proc.returncode == 0:
                    print("  [3b/4] detail enhancement applied (PyMeshLab)")
                else:
                    print(f"  [3b/4] detail enhance failed (exit {proc.returncode}); "
                          f"continuing without it")
            except Exception as e:
                print(f"  [3b/4] detail enhance skipped: {e}")

        # ---- 4. Slicer prep / orientation ----
        import slicer_prep
        prep = slicer_prep.analyze(str(stl_path), printer=printer)
        manifest_path = stl_path.with_suffix(".print.json")
        manifest_path.write_text(json.dumps(prep, indent=2), encoding="utf-8")
        print(f"  [4/4] orientation advisory -> {manifest_path.name}")

        # ---- 4b. Real-slicer validation gate (Bambu + PrusaSlicer) ----
        # The iteration oracle: does this STL actually print? Records
        # PASS/FAIL + reasons so failures are visible, not silent.
        validated = None
        try:
            import subprocess
            vproc = subprocess.run(
                [PY, str(HERE / "slicer_validate.py"),
                 "--input", str(stl_path), "--printer", printer],
                capture_output=True, text=True, timeout=400)
            vfile = stl_path.with_suffix(".validate.json")
            if vfile.exists():
                vdata = json.loads(vfile.read_text(encoding="utf-8"))
                validated = bool(vdata.get("PASS"))
                tail = (vproc.stdout or "").strip().splitlines()[-2:]
                print(f"  [4b/4] slicer validation: "
                      f"{'PASS' if validated else 'FAIL'}"
                      + (f" — {vdata.get('reasons')}" if not validated else ""))
        except Exception as e:
            print(f"  [4b/4] validation skipped: {e}")

        result = {
            "run_id": run_id,
            "concept_png": str(concept_png),
            "mesh_glb": str(mesh_glb) if engine != "hunyuan" else None,
            "preview_png": str(preview_png) if preview_png else None,
            "stl": str(stl_path),
            "manifest": str(manifest_path),
            "validated": validated,
        }
        if engine == "pixal3d":
            result["mesh_glb"] = str(mesh_glb)

    print(f"=== done: {run_id} ===")
    return result


def main(argv=None) -> int:
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", help="Natural-language description")
    ap.add_argument("--kind", choices=list(KIND_DEFAULTS), default="mini")
    ap.add_argument("--engine", choices=list(ENGINES), default="sparc3d")
    ap.add_argument("--printer", choices=("resin", "fdm"), default="resin")
    ap.add_argument("--scale-mm", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip-print-prep", action="store_true")
    ap.add_argument("--out", default=None, help="STL output path")
    ap.add_argument("--check-models", action="store_true",
                    help="Report detected models then exit")
    ap.add_argument("--input-image", default=None,
                    help="Use this image as the concept instead of generating one")
    # multi-view (engine=hunyuan2mv)
    ap.add_argument("--mv-front", default=None, help="multi-view: front image")
    ap.add_argument("--mv-left",  default=None, help="multi-view: left image")
    ap.add_argument("--mv-back",  default=None, help="multi-view: back image")
    ap.add_argument("--mv-gif",   default=None, help="multi-view: turntable GIF/video")
    ap.add_argument("--mv-reverse", action="store_true",
                    help="multi-view: object spins clockwise")
    args = ap.parse_args(argv)

    if args.check_models:
        print(json.dumps(check_models(), indent=2))
        return 0
    if args.engine != "hunyuan2mv" and not args.prompt and not args.input_image:
        ap.error("either --prompt or --input-image is required")
    if not args.prompt:
        args.prompt = (f"user image: {Path(args.input_image).stem}"
                       if args.input_image else "multiview")
    result = run(
        prompt=args.prompt, kind=args.kind, engine=args.engine,
        printer=args.printer, scale_mm=args.scale_mm, seed=args.seed,
        skip_print_prep=args.skip_print_prep, out_stl=args.out,
        input_image=args.input_image,
        mv_front=args.mv_front, mv_left=args.mv_left, mv_back=args.mv_back,
        mv_gif=args.mv_gif, mv_reverse=args.mv_reverse,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
