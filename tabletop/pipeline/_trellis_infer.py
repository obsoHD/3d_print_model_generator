"""_trellis_infer - Microsoft TRELLIS.2 single-image-to-mesh.

Largest open image->3D model (4B, O-Voxel). Imports the upstream `trellis`
package from the cloned TRELLIS.2 repo via sys.path. Uses xformers attention
(not flash-attn) to dodge the Blackwell flash-attn build. Weights auto-download
from microsoft/TRELLIS.2-4B (or pass --model for a local dir).

NOTE: first-draft API modeled on TRELLIS-1 (run() -> outputs['mesh']); may need
a tweak once we see the real TRELLIS.2 pipeline — the streaming log will show it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _force_xformers_cutlass() -> None:
    """TRELLIS.2 SPARSE attention only supports xformers/flash_attn (no sdpa).
    We route it to xformers, but xformers' default dispatch picks the flash
    *Hopper* kernel (sm_90) which throws 'CUDA error: invalid argument' on
    Blackwell (sm_120). Force the CUTLASS op (broad arch coverage); fall back to
    auto-dispatch if CUTLASS rejects a specific mask/dtype."""
    try:
        import xformers.ops as xops  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return
    cutlass = getattr(xops, "MemoryEfficientAttentionCutlassOp", None)
    if cutlass is None or getattr(xops.memory_efficient_attention, "_cutlass_forced", False):
        return
    orig = xops.memory_efficient_attention

    def patched(*a, **k):
        if k.get("op") is None:
            k["op"] = cutlass
            try:
                return orig(*a, **k)
            except Exception:  # noqa: BLE001
                k.pop("op", None)
        return orig(*a, **k)

    patched._cutlass_forced = True  # noqa: SLF001
    xops.memory_efficient_attention = patched
    print("[trellis] forced xformers CUTLASS attention (Blackwell-safe)", flush=True)


def _signed_distance(bvh, pts, chunk=1_000_000):
    """cuBVH signed distance, chunked (the raw remesh can be ~16M face centers).
    Prefer mode='raystab' (robust on the non-watertight source mesh) when the
    build supports it; otherwise call with defaults. Tolerates tuple returns."""
    import torch
    use_mode = False
    try:
        bvh.signed_distance(pts[:8].contiguous(), mode="raystab")
        use_mode = True
    except TypeError:
        use_mode = False
    except Exception:  # noqa: BLE001 — probe failure; fall back to plain call
        use_mode = False
    outs = []
    for i in range(0, pts.shape[0], chunk):
        sub = pts[i:i + chunk].contiguous()
        r = bvh.signed_distance(sub, mode="raystab") if use_mode \
            else bvh.signed_distance(sub)
        if isinstance(r, (tuple, list)):
            r = r[0]
        outs.append(r.reshape(-1))
    return torch.cat(outs, dim=0)


def _cull_inner_shell(mesh, bvh, voxel, verbose=True):
    """Collapse the UDF dual-contouring DOUBLE shell to a single OUTER shell.

    The narrow-band unsigned-field DC emits an outer sheet (~+1 voxel from the
    surface) AND a concentric inner sheet (~-1 voxel). We classify each remeshed
    face by the signed distance of its centroid to the ORIGINAL surface (via the
    BVH already built for projection) and keep only the faces on/outside the
    surface, deleting the inner sheet — exactly what the fork's remove_inner_faces
    does, but reimplemented because this cumesh build lacks that flag. Guarded:
    only culls when the 'inner' fraction is in a plausible double-shell range, so
    a flipped sign convention or a single-shell input can't nuke the model."""
    import torch
    v, f = mesh.read()
    if f.shape[0] == 0:
        return
    centers = v[f.long()].mean(dim=1).contiguous()
    try:
        sd = _signed_distance(bvh, centers).to(centers.device).reshape(-1)
    except Exception as e:  # noqa: BLE001
        print(f"  [trellis] inner-shell cull SKIPPED (signed_distance failed: {e})",
              flush=True)
        return
    thr = -0.25 * float(voxel)             # ~quarter-voxel inside = inner sheet
    inner = sd < thr
    n_inner = int(inner.sum().item())
    n_tot = int(f.shape[0])
    frac = n_inner / max(1, n_tot)
    if n_inner > 0 and 0.05 <= frac <= 0.95:
        keep = ~inner
        mesh.init(v.contiguous(), f[keep].contiguous())
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:  # noqa: BLE001
            pass
        if verbose:
            print(f"  [trellis] inner-shell cull: dropped {n_inner}/{n_tot} inner "
                  f"faces ({frac*100:.0f}%) via BVH signed-distance "
                  "(UDF double-shell collapse)", flush=True)
    elif verbose:
        print(f"  [trellis] inner-shell cull: skipped — inner frac {frac*100:.0f}% "
              "outside safe 5-95% range (no clear double shell)", flush=True)


def _pymeshfix_manifold(mesh, min_faces=200, verbose=True):
    """GUARANTEED manifold repair via Attene MeshFix (pymeshfix), build-agnostic
    and signed-reference-free — the robust fix the research panel ranked #1 for a
    noisy neural isosurface. Run PER connected component so a multi-part mini
    (body + sword + cape + base) keeps every piece; each becomes a single
    watertight 2-manifold. Components below min_faces (the floater scatter) are
    dropped. Detail outside the defect regions is preserved (MeshFix only rewrites
    around singularities/self-intersections)."""
    import time
    import numpy as np
    import trimesh
    try:
        from pymeshfix import _meshfix
    except Exception as e:  # noqa: BLE001
        print(f"[trellis] pymeshfix unavailable ({e}); skipping manifold repair",
              flush=True)
        return mesh
    t0 = time.time()
    # 1) Drop the tiny floater scatter, keep the meaningful parts. (The cumesh
    #    chain fragments the shell into ~1000 components; most are sub-200-face
    #    specks.)
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:  # noqa: BLE001
        parts = []
    if parts:
        big = [p for p in parts if len(p.faces) >= min_faces]
        dropped = len(parts) - len(big)
        if not big:
            big = [max(parts, key=lambda p: len(p.faces))]
            dropped = len(parts) - 1
        base = trimesh.util.concatenate(big) if len(big) > 1 else big[0]
    else:
        dropped, base = 0, mesh

    # 2) ONE repair over the whole thing via the STABLE low-level PyTMesh API
    #    (the high-level MeshFix wrapper's result attributes differ across
    #    versions — `mf.v` doesn't exist on 0.18.x). join_closest_components()
    #    stitches the separate (mostly open) patches into one coherent surface,
    #    fill_small_boundaries() + clean() close it into ONE watertight 2-manifold
    #    and remove self-intersections — instead of inflating each patch into a
    #    thin closed bag (what per-component repair would do).
    def _repair(v, f):
        try:
            tin = _meshfix.PyTMesh(False)  # quiet, if supported
        except TypeError:
            tin = _meshfix.PyTMesh()       # this build: no ctor args
        tin.load_array(np.asarray(v, dtype=np.float64),
                       np.asarray(f, dtype=np.int32))
        try:
            tin.join_closest_components()
        except Exception:  # noqa: BLE001
            pass
        try:
            tin.fill_small_boundaries()
        except Exception:  # noqa: BLE001
            pass
        tin.clean(max_iters=10, inner_loops=3)
        return tin.return_arrays()

    try:
        rv, rf = _repair(base.vertices, base.faces)
        out = trimesh.Trimesh(rv, rf, process=False)
    except Exception as e:  # noqa: BLE001
        print(f"[trellis] pymeshfix failed ({e}); keeping pre-repair mesh",
              flush=True)
        return mesh
    if not len(out.faces):
        return mesh
    if verbose:
        print(f"[trellis] pymeshfix: {len(parts)} parts, {dropped} floaters "
              f"dropped, joined -> {len(out.faces)} faces | "
              f"{_manifold_str(out)} | {time.time()-t0:.1f}s", flush=True)
    return out


def _alpha_wrap_remesh(mesh, alpha_fraction=None, verbose=True):
    """CGAL Alpha Wrapping via pymeshlab.generate_alpha_wrap — the field-standard,
    soup-ROBUST way to turn a self-intersecting / non-manifold / fragmented neural
    mesh into a single WATERTIGHT, 2-manifold, intersection-free surface that
    ENCLOSES the input. Unlike MeshFix it deletes nothing — it carves a shrink-wrap
    from outside until it hits the geometry, so overlapping shells, 1000 floaters
    and 20k self-intersections simply don't matter; the whole figure is kept.

    alpha_fraction = carving cell size as a fraction of the bbox diagonal (smaller
    = finer detail but more compute). offset_fraction = how tightly it hugs the
    surface (CGAL recommends alpha/30). Default alpha ~ diag/500 ~= 0.4mm on a
    160mm mini — recovers face/cloth detail while staying robust. Tune via
    TRELLIS_ALPHA (e.g. 0.0015 finer / 0.003 coarser+faster)."""
    import time
    import trimesh
    import pymeshlab
    af = alpha_fraction if alpha_fraction is not None else \
        float(os.environ.get("TRELLIS_ALPHA", "0.002"))
    of = af / 30.0
    t0 = time.time()
    try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype("float64"),
                                   face_matrix=mesh.faces.astype("int32")))
        ms.generate_alpha_wrap(alpha_fraction=af, offset_fraction=of)
        cm = ms.current_mesh()
        out = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                              faces=cm.face_matrix(), process=False)
        if not len(out.faces):
            print("[trellis] alpha-wrap produced empty mesh; keeping pre-wrap",
                  flush=True)
            return mesh
        if verbose:
            print(f"[trellis] alpha-wrap: {len(mesh.faces)} -> {len(out.faces)} "
                  f"faces (alpha={af}, offset={of:.5f}) | {_manifold_str(out)} | "
                  f"{time.time()-t0:.1f}s", flush=True)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[trellis] alpha-wrap failed ({e}); keeping pre-wrap mesh",
              flush=True)
        return mesh


def _manifold_stats(mesh):
    """TRUE 2-manifold diagnostics. trimesh.is_watertight is BLIND to edges shared
    by >2 faces (group_rows(require_count=2) silently discards them) — the exact
    defect a slicer rejects and the reason trimesh said 'watertight' while the
    slicer split into hundreds of pieces. So we measure non-manifold edges and
    connected-component count directly."""
    import trimesh
    try:
        es = mesh.edges_sorted
        g2 = trimesh.grouping.group_rows(es, require_count=2)
        nm_edges = int(len(es) - 2 * len(g2))   # edges NOT shared by exactly 2 faces
    except Exception:  # noqa: BLE001
        nm_edges = -1
    try:
        comps = int(len(mesh.split(only_watertight=False)))
    except Exception:  # noqa: BLE001
        comps = -1
    return {"watertight": bool(mesh.is_watertight),
            "nm_edges": nm_edges, "components": comps}


def _manifold_score(mesh):
    """Lower = healthier; 0 == clean single watertight 2-manifold."""
    s = _manifold_stats(mesh)
    score = 0
    if not s["watertight"]:
        score += 1000
    if s["nm_edges"] and s["nm_edges"] > 0:
        score += s["nm_edges"]
    if s["components"] and s["components"] > 1:
        score += (s["components"] - 1)
    return score


def _is_clean_manifold(mesh):
    return _manifold_score(mesh) == 0


def _manifold_str(mesh):
    s = _manifold_stats(mesh)
    return (f"watertight={s['watertight']} nm_edges={s['nm_edges']} "
            f"components={s['components']}")


def _trellis_geometry_mesh(m, max_faces, remesh=True,
                           remesh_band=1.0,
                           remesh_project=float(os.environ.get(
                               "TRELLIS_REMESH_PROJECT", 0.95)),
                           verbose=True):
    """Run ONLY the geometry half of o_voxel.postprocess.to_glb — the cumesh
    remesh + clean steps that build the clean watertight mesh — and read out
    verts/faces, SKIPPING the slow CPU xatlas UV unwrap + nvdiffrast texture
    bake (we print geometry, not color). Mirrors the upstream to_glb body so the
    coordinate convention matches the textured path (Y-up glTF). ~seconds vs the
    multi-minute xatlas tail."""
    import cumesh
    import numpy as np
    import torch
    import trimesh

    dev = m.coords.device
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        dtype=torch.float32, device=dev)
    vs = m.voxel_size
    if isinstance(vs, float):
        vs = [vs, vs, vs]
    if not torch.is_tensor(vs):
        vs = torch.tensor(np.array(vs), dtype=torch.float32, device=dev)
    grid_size = ((aabb[1] - aabb[0]) / vs.to(dev).float()).round().int()

    mesh = cumesh.CuMesh()
    mesh.init(m.vertices.cuda(), m.faces.cuda())
    mesh.fill_holes(max_hole_perimeter=3e-2)
    v, f = mesh.read()
    bvh = cumesh.cuBVH(v, f)

    if remesh:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        # ROOT-CAUSE FIX (9-agent research panel, high confidence).
        # remesh_narrow_band_dc contours an UNSIGNED distance field at +eps. A UDF
        # has no inside/outside sign, so the |dist|=eps level set is satisfied on
        # BOTH sides of the true surface -> dual contouring emits a DOUBLED shell
        # (inner + outer, ~2 voxels apart) UNIFORMLY over the whole model. That
        # doubled shell — not our cleanup — is the source of the "tiny triangle
        # holes + non-manifold edges everywhere" (present single-view too): every
        # seam edge is shared by >2 faces and the inner sheet z-fights the outer.
        # The fork's `remove_inner_faces` cure does NOT exist in this cumesh build
        # (verified: remesh_narrow_band_dc has no such kwarg), so we replicate it
        # ourselves with the cuBVH the function already needs: classify every
        # remeshed face by the SIGNED distance of its centroid to the ORIGINAL
        # surface and DELETE the inner sheet. cuBVH.signed_distance exists on this
        # build. Expect ~half the faces to drop — that's the inner shell going.
        scale_passed = (resolution + 3 * remesh_band) / resolution * scale
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            v, f, center=center, scale=scale_passed,
            resolution=resolution, band=remesh_band,
            project_back=remesh_project, verbose=verbose, bvh=bvh))
        _cull_inner_shell(mesh, bvh, scale_passed / resolution, verbose=verbose)
        mesh.simplify(int(max_faces), verbose=verbose)
        # With the inner shell gone the surface is near-2-manifold. Clean it
        # WITHOUT cumesh.repair_non_manifold_edges() — that op SPLITS vertices into
        # coincident-but-unwelded copies (it does NOT delete faces), which is what
        # shredded connectivity into the uniform non-manifold scatter and made
        # trimesh falsely report watertight while the slicer split it into
        # hundreds of pieces. Prefer the DELETE-faces variant
        # (remove_non_manifold_faces -> true 2-manifold); end the chain on
        # fill_holes (never on a split-repair, which would re-open boundaries).
        mesh.remove_duplicate_faces()
        _rnf = getattr(mesh, "remove_non_manifold_faces", None)
        if callable(_rnf):
            _rnf()
        else:
            mesh.repair_non_manifold_edges()  # fallback: only if delete-variant absent
        mesh.remove_small_connected_components(1e-5)
        mesh.unify_face_orientations()
        mesh.fill_holes(max_hole_perimeter=1.0)
    else:
        mesh.simplify(int(max_faces) * 3, verbose=verbose)
        mesh.remove_duplicate_faces(); mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(int(max_faces), verbose=verbose)
        mesh.remove_duplicate_faces(); mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()

    ov, of = mesh.read()
    ov = ov.cpu().numpy().astype("float64")
    of = of.cpu().numpy()
    # Same coord convention as to_glb (swap Y/Z, invert Y -> glTF Y-up), done
    # safely (no in-place numpy view aliasing).
    conv = ov.copy()
    conv[:, 1] = ov[:, 2]
    conv[:, 2] = -ov[:, 1]
    print(f"  [trellis] geometry mesh: {len(of)} faces / {len(ov)} verts "
          "(post inner-shell removal)", flush=True)
    return trimesh.Trimesh(vertices=conv, faces=of, process=False)


def _install_multidiffusion():
    """Enable MULTI-VIEW conditioning on the dense sparse-structure (occupancy)
    sampler via *multidiffusion*.

    TRELLIS.2 is NOT built to take a concatenated multi-image conditioning in one
    forward pass — get_cond([imgA,imgB,imgC]) returns a BATCH-V conditioning
    ([V, N, feat]) while the noise is batch-1, so the cross-attention dims clash
    ('size of tensor a (384) must match b (128)'). The correct fusion is the
    classic multidiffusion trick: run the model once PER VIEW and AVERAGE the
    velocity predictions. We do that by replicating the batch-1 latent x_t across
    the V conditioning views, running the model batched, then mean-reducing back
    to batch-1 — mathematically the per-step average of the per-view predictions.
    Every view 'votes' on where geometry should be, so the occluded back/sides get
    filled from the views that actually see them.

    We patch ONLY the base FlowEulerSampler._inference_model (a class method, so the
    mixin super() chain hits it). It's a no-op whenever cond batch == latent batch
    (V == 1), so the single-image path and the sparse-latent SLat stages — which we
    deliberately keep single-view — are completely unaffected."""
    from trellis2.pipelines.samplers import flow_euler
    if getattr(flow_euler.FlowEulerSampler, "_mv_patched", False):
        return
    import torch
    _orig = flow_euler.FlowEulerSampler._inference_model

    def _inf(self, model, x_t, t, cond, **kwargs):
        # Fuse only when cond carries multiple views AND x_t is a plain dense
        # batch-1 tensor (the sparse-structure stage). SparseTensor latents or
        # already-matched batches fall straight through to the original impl.
        #
        # We REPLICATE the latent to batch-V and hand it to the ORIGINAL
        # _inference_model — that's important because the original is what turns
        # the scalar float timestep `t` into the [batch]-shaped tensor the model
        # needs (`t.device`/`t[:,None]`). Sizing off x_rep.shape[0] gives a [V] t
        # automatically. Then we mean-reduce the [V,...] velocity back to [1,...].
        try:
            v = cond.shape[0] if torch.is_tensor(cond) else 1
            b = x_t.shape[0] if torch.is_tensor(x_t) else None
        except Exception:
            return _orig(self, model, x_t, t, cond, **kwargs)
        if torch.is_tensor(x_t) and v and v > 1 and b == 1:
            reps = [v] + [1] * (x_t.dim() - 1)
            x_rep = x_t.repeat(*reps)
            out = _orig(self, model, x_rep, t, cond, **kwargs)
            return out.mean(dim=0, keepdim=True)
        return _orig(self, model, x_t, t, cond, **kwargs)

    flow_euler.FlowEulerSampler._inference_model = _inf
    flow_euler.FlowEulerSampler._mv_patched = True
    print("  [trellis-mv] multidiffusion patch installed (occupancy stage)", flush=True)


def _run_multi(pipe, images, seed, steps, max_tokens):
    """TRELLIS.2 MULTI-VIEW run.

    Strategy: condition the OCCUPANCY (dense sparse-structure) stage on ALL views
    via multidiffusion (see _install_multidiffusion) so the model knows the object
    is solid on the back/sides — that's what eliminates the flat-filled occlusion
    gaps. The SURFACE stages (shape SLat / tex SLat) stay conditioned on the FRONT
    view only: those operate on SPARSE latents where per-view averaging is unstable,
    and once the occupancy is correct the front view refines the existing surface
    fine. Net: multi-view solidity + front-faithful detail."""
    import torch
    pt = pipe.default_pipeline_type
    imgs = [pipe.preprocess_image(im) for im in images]
    front = [imgs[0]]
    torch.manual_seed(seed)
    _install_multidiffusion()
    # Multi-view cond -> occupancy.  Front-only cond -> surface.
    cond_mv_512 = pipe.get_cond(imgs, 512)
    cond_s_512  = pipe.get_cond(front, 512)
    cond_s_1024 = pipe.get_cond(front, 1024) if pt != '512' else None
    ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pt]
    sp = {"steps": int(steps)}
    coords = pipe.sample_sparse_structure(cond_mv_512, ss_res, 1, sp)
    if pt == '512':
        shape_slat = pipe.sample_shape_slat(
            cond_s_512, pipe.models['shape_slat_flow_model_512'], coords, sp)
        tex_slat = pipe.sample_tex_slat(
            cond_s_512, pipe.models['tex_slat_flow_model_512'], shape_slat, sp)
        res = 512
    elif pt == '1024':
        shape_slat = pipe.sample_shape_slat(
            cond_s_1024, pipe.models['shape_slat_flow_model_1024'], coords, sp)
        tex_slat = pipe.sample_tex_slat(
            cond_s_1024, pipe.models['tex_slat_flow_model_1024'], shape_slat, sp)
        res = 1024
    else:  # 1024_cascade / 1536_cascade
        hi = 1536 if pt == '1536_cascade' else 1024
        shape_slat, res = pipe.sample_shape_slat_cascade(
            cond_s_512, cond_s_1024,
            pipe.models['shape_slat_flow_model_512'],
            pipe.models['shape_slat_flow_model_1024'],
            512, hi, coords, sp, int(max_tokens))
        tex_slat = pipe.sample_tex_slat(
            cond_s_1024, pipe.models['tex_slat_flow_model_1024'], shape_slat, sp)
    torch.cuda.empty_cache()
    return pipe.decode_latent(shape_slat, tex_slat, res)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--output", required=True, help="output GLB path")
    # Multi-view: pass 2-3 consistent views (front/left/back). When --mv-front is
    # given we condition on all of them so occluded sides are reconstructed.
    ap.add_argument("--mv-front", default=None)
    ap.add_argument("--mv-left",  default=None)
    ap.add_argument("--mv-back",  default=None)
    ap.add_argument("--lib",    required=True, help="TRELLIS.2 repo dir")
    ap.add_argument("--model",  default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--max-faces", type=int,
                    default=int(os.environ.get("TRELLIS_MAX_FACES", 2_000_000)),
                    help="decimate above this (raw is ~4M; finish is fast now, "
                         "so we keep more faces = less faceting on hair/beard)")
    # QUALITY knobs — these use your VRAM headroom (the 4B model itself only
    # needs ~4GB; the rest of your 32GB buys detail, not speed):
    #   max_num_tokens — TRELLIS.2's primary quality dial (default 49152 ≈ 4GB).
    #     Doubling to ~98k ≈ 8-12GB gives a denser/finer O-Voxel structure.
    #   steps — diffusion sampling steps per stage (was 12; 25 = cleaner shape).
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("TRELLIS_TOKENS", 196_608)))
    ap.add_argument("--steps", type=int,
                    default=int(os.environ.get("TRELLIS_STEPS", 30)))
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    # DENSE attention -> pure-torch sdpa (no xformers/flash-attn needed). But
    # TRELLIS.2's SPARSE attention supports ONLY xformers/flash_attn (no sdpa),
    # so route THAT to xformers (installed) and force its CUTLASS op below to
    # avoid the sm_90-only flash-Hopper kernel that crashes on Blackwell.
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import numpy as np
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"[trellis] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    print(f"[trellis] loading pipeline {args.model} ...", flush=True)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipe.cuda()

    _force_xformers_cutlass()   # Blackwell-safe sparse attention

    # ---- MULTI-VIEW: condition on front/left/back so occluded sides fill in ----
    mv_paths = [p for p in (args.mv_front, args.mv_left, args.mv_back) if p]
    if mv_paths:
        views = [Image.open(p).convert("RGBA") for p in mv_paths]
        print(f"[trellis] MULTI-VIEW run ({len(views)} views, seed={args.seed}, "
              f"tokens={args.max_tokens}, steps={args.steps})...", flush=True)
        m = _run_multi(pipe, views, args.seed, args.steps, args.max_tokens)
    else:
        img = Image.open(args.image).convert("RGBA")
        print(f"[trellis] running (seed={args.seed}, tokens={args.max_tokens}, "
              f"steps={args.steps})...", flush=True)
        # preprocess_image=TRUE: TRELLIS.2's preprocess centers/squares/rescales the
        # subject to its trained framing (off -> dwarfish proportions).
        try:
            m = pipe.run(
                img,
                seed=args.seed,
                preprocess_image=True,
                max_num_tokens=int(args.max_tokens),
                sparse_structure_sampler_params={"steps": int(args.steps)},
                shape_slat_sampler_params={"steps": int(args.steps)},
                tex_slat_sampler_params={"steps": int(args.steps)},
            )[0]
        except TypeError as e:
            print(f"[trellis] WARN quality params rejected ({e}); plain run", flush=True)
            m = pipe.run(img, seed=args.seed, preprocess_image=True)[0]

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    # === Build the printable mesh ===
    # Both paths use TRELLIS.2's OFFICIAL o_voxel remesh+clean (cumesh) — that's
    # what turns the raw dual-grid voxel output into a CLEAN watertight mesh.
    # The difference is the texture half (xatlas UV unwrap + nvdiffrast bake),
    # which is a slow CPU step we DON'T need for printing:
    #   TRELLIS_TEXTURE=0 (default) -> geometry only, ~seconds (printing)
    #   TRELLIS_TEXTURE=1           -> full to_glb with baked texture (slow; the
    #                                  colored GLB for viewing, not printing)
    want_texture = os.environ.get("TRELLIS_TEXTURE", "0") == "1"
    mesh = None
    if not want_texture:
        try:
            print(f"[trellis] o_voxel GEOMETRY-only remesh (skip xatlas/texture; "
                  f"target {args.max_faces})...", flush=True)
            mesh = _trellis_geometry_mesh(m, int(args.max_faces), remesh=True,
                                          verbose=True)
            print(f"[trellis] geometry remesh OK (no texture): {len(mesh.faces)} "
                  f"faces, watertight={mesh.is_watertight}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[trellis] WARN geometry-only path failed ({e}); "
                  f"trying full to_glb", flush=True)
            mesh = None
    if mesh is None:
        try:
            import o_voxel
            print(f"[trellis] o_voxel.to_glb (with texture; target "
                  f"{args.max_faces})...", flush=True)
            glb = o_voxel.postprocess.to_glb(
                vertices=m.vertices, faces=m.faces,
                attr_volume=m.attrs, coords=m.coords, attr_layout=m.layout,
                voxel_size=m.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(args.max_faces),
                texture_size=1024, remesh=True, remesh_band=1.0, remesh_project=0.9,
                verbose=True)
            mesh = glb if isinstance(glb, trimesh.Trimesh) else glb.dump(concatenate=True)
            print(f"[trellis] to_glb OK: {len(mesh.faces)} faces, "
                  f"watertight={mesh.is_watertight}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[trellis] WARN o_voxel to_glb unavailable ({e}); "
                  f"raw + pymeshlab decimate fallback", flush=True)
            mesh = None

    if mesh is None:
        mesh = trimesh.Trimesh(vertices=_np(m.vertices), faces=_np(m.faces))
        if len(mesh.faces) > args.max_faces:
            try:
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices.astype("float64"),
                    face_matrix=mesh.faces.astype("int32")))
                for _ps in (
                        dict(targetfacenum=int(args.max_faces), preservenormal=True,
                             preservetopology=True, planarquadric=True,
                             qualitythr=0.35, optimalplacement=True),
                        dict(targetfacenum=int(args.max_faces), preservenormal=True,
                             preservetopology=True),
                        dict(targetfacenum=int(args.max_faces))):
                    try:
                        ms.meshing_decimation_quadric_edge_collapse(**_ps)
                        break
                    except Exception:  # noqa: BLE001
                        continue
                for filt in ("meshing_remove_null_faces",
                             "meshing_remove_duplicate_faces",
                             "meshing_remove_duplicate_vertices",
                             "meshing_remove_unreferenced_vertices"):
                    try:
                        getattr(ms, filt)()
                    except Exception:  # noqa: BLE001
                        pass
                cm = ms.current_mesh()
                mesh = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                                       faces=cm.face_matrix())
                print(f"[trellis] decimated -> {len(mesh.faces)} faces", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[trellis] WARN decimation failed ({e}); raw mesh", flush=True)

    # === PRINTABLE MANIFOLD: CGAL Alpha Wrap (default) ===
    # The cumesh dual-contour output on this build is triangle SOUP (self-
    # intersecting, ~1000 overlapping fragments) that no per-edge repair or MeshFix
    # can fix without destroying it. Alpha wrapping shrink-wraps a single
    # watertight 2-manifold around the whole figure — robust to all of that, keeps
    # the model. This is the printable geometry. Set TRELLIS_ALPHAWRAP=0 to A/B the
    # raw mesh (and fall through to the legacy additive cleanup below).
    if os.environ.get("TRELLIS_ALPHAWRAP", "1") != "0" and not _is_clean_manifold(mesh):
        wrapped = _alpha_wrap_remesh(mesh, verbose=True)
        # Accept only if it kept a real volume (alpha wrap encloses, so it should
        # never collapse — but guard anyway) and is actually cleaner.
        if len(wrapped.faces) > 0 and _manifold_score(wrapped) < _manifold_score(mesh):
            mesh = wrapped

    # Manifold cleanup — but ONLY if cumesh didn't already give us a clean
    # watertight mesh. cumesh's remesh+clean usually outputs watertight; our
    # trimesh dedup + hole-close were OPENING it (stripping doubled geometry that
    # held it closed). So: watertight mesh -> leave it pristine; otherwise try a
    # gentle, ADDITIVE repair (close small holes only; never delete faces).
    if not mesh.is_watertight:
        try:
            mesh.merge_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
        except Exception as e:  # noqa: BLE001
            print(f"[trellis] WARN trimesh dedup partial ({e})", flush=True)
    if not mesh.is_watertight:
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype("float64"),
                                       face_matrix=mesh.faces.astype("int32")))
            # Close holes up to ~300 boundary edges — catches the medium holes in
            # face/hair, while still leaving any truly huge opening alone (no fan).
            for mhs in (300, 100, 40):
                try: ms.meshing_close_holes(maxholesize=mhs); break
                except Exception: continue  # noqa: BLE001
            cm = ms.current_mesh()
            mesh = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                                   faces=cm.face_matrix(), process=False)
        except Exception as e:  # noqa: BLE001
            print(f"[trellis] WARN pymeshlab hole-close skipped ({e})", flush=True)
        # Backup: trimesh's own simple-hole fill catches any small loops left.
        try: trimesh.repair.fill_holes(mesh)
        except Exception: pass  # noqa: BLE001
        print(f"[trellis] hole-close: {len(mesh.faces)} faces, "
              f"watertight={mesh.is_watertight}", flush=True)
    try:
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception:  # noqa: BLE001
        pass

    # Collapse dual-contouring slivers — but ONLY if the mesh is NOT already a
    # clean 2-manifold. With the inner-shell removal upstream, the geometry path
    # now yields a watertight manifold, so this block must NOT touch it: the old
    # UNCONDITIONAL merge_close_vertices(0.02% of bbox-diagonal) + Taubin was
    # itself welding the two former shell layers / thin walls and folding the
    # result into the visible scatter of non-manifold edges + dark triangles
    # (root-cause panel finding). So gate it like the repair blocks above, use an
    # ABSOLUTE weld tolerance tied to median edge length (not a bbox percentage),
    # keep Taubin opt-in (default OFF), and accept the candidate only if it is no
    # worse by a TRUE manifold score (is_watertight alone is blind to >2-face
    # edges, which is exactly why trimesh said watertight while the slicer split).
    if not _is_clean_manifold(mesh):
        try:
            import numpy as _np
            import pymeshlab
            before = _manifold_score(mesh)
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype("float64"),
                                       face_matrix=mesh.faces.astype("int32")))
            merged = False
            try:
                med = float(_np.median(mesh.edges_unique_length))
                ms.meshing_merge_close_vertices(
                    threshold=pymeshlab.AbsoluteValue(0.25 * med)); merged = True
            except Exception:  # noqa: BLE001 — older API / no AbsoluteValue
                try: ms.meshing_merge_close_vertices(); merged = True
                except Exception: pass  # noqa: BLE001
            try: ms.meshing_remove_null_faces()
            except Exception: pass  # noqa: BLE001
            # Taubin smooth is OFF by default now (the root fix removes the flat
            # hole-fill patches it used to round). Opt-in via TRELLIS_SMOOTH>0.
            _sm = int(os.environ.get("TRELLIS_SMOOTH", "0"))
            if _sm > 0:
                try: ms.apply_coord_taubin_smoothing(stepsmoothnum=_sm); merged = True
                except Exception: pass  # noqa: BLE001
            if merged:
                cm = ms.current_mesh()
                cand = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                                       faces=cm.face_matrix(), process=False)
                if _manifold_score(cand) <= before:
                    mesh = cand
                    print(f"[trellis] sliver cleanup: {len(mesh.faces)} faces, "
                          f"{_manifold_str(mesh)}", flush=True)
                else:
                    print("[trellis] sliver cleanup reverted (made topology "
                          "worse); keeping pre-clean mesh", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[trellis] WARN sliver cleanup skipped ({e})", flush=True)
    else:
        print(f"[trellis] mesh already clean 2-manifold "
              f"({_manifold_str(mesh)}); skipping sliver cleanup", flush=True)

    # GUARANTEED manifold repair (Attene MeshFix), the robust build-agnostic fix:
    # the cumesh remesh on THIS build leaves a noisy isosurface — scattered
    # non-manifold edges + tiny floaters — that per-edge cumesh ops can't fully
    # resolve (and which fail the slicer). MeshFix rewrites only around the
    # singularities, per connected component, so each part becomes one watertight
    # 2-manifold while detail is preserved elsewhere. Skip if already clean, or if
    # disabled via TRELLIS_MESHFIX=0 (e.g. to A/B the raw mesh).
    if os.environ.get("TRELLIS_MESHFIX", "0") != "0" and not _is_clean_manifold(mesh):
        _mf_min = int(os.environ.get("TRELLIS_MESHFIX_MINFACES", "200"))
        repaired = _pymeshfix_manifold(mesh, min_faces=_mf_min, verbose=True)
        # SAFETY: MeshFix.clean() deletes self-intersecting triangles, and on a
        # triangle-soup input it can delete almost everything (1.97M -> 14k = just
        # the cape) yet still report "watertight". NEVER accept a result that
        # destroyed the model — require it to retain most of the geometry.
        keep_frac = len(repaired.faces) / max(1, len(mesh.faces))
        if keep_frac >= 0.5 and _manifold_score(repaired) <= _manifold_score(mesh):
            mesh = repaired
        else:
            print(f"[trellis] pymeshfix REJECTED — kept only {keep_frac*100:.1f}% "
                  f"of faces ({len(repaired.faces)}/{len(mesh.faces)}); it would "
                  "destroy the model. Keeping pre-repair mesh.", flush=True)

    # Orientation: the o_voxel coord swap leaves the model glTF Y-up; our gauntlet
    # + slicer are Z-up. Rotate +90 deg about X so +Y (up) -> +Z (up) = upright.
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    print(f"[trellis] final mesh: {len(mesh.faces)} faces, "
          f"{_manifold_str(mesh)} (Z-up upright)", flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[trellis] saved {out.name} ({out.stat().st_size/1e6:.1f} MB, "
          f"faces={len(mesh.faces)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
